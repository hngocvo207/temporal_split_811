"""
Giai đoạn E1+E2 (new_propose.md): hạ tầng train/eval cho inductive_model, kiểm
chứng ở scale bounded bằng corpus Attempt-3 có sẵn (20k train / 5k val /
5k overlap / 5k pure_test) -- so trực tiếp F1(pos)/AUPRC với checkpoint
Attempt-3 gốc (F1(pos)=87.63%, AUPRC overall=0.9155; xem STATUS.md).

Checkpoint/log riêng trong Dataset/inductive_model/output/ -- không đụng
tri_model/output/ (quy ước cô lập thí nghiệm, tiền lệ runs/expanded_test_attempt3/).

Text branch dùng subgraph gcn_adj_*.npz của corpus Attempt-3 KHÔNG được dùng
cho graph branch -- xem docstring data_prep/attempt3_corpus.py: node feature +
neighbor sampling ở đây luôn lấy từ đồ thị full-scale thật (A2), chỉ mượn
docs/labels/split của corpus Attempt-3.

CHƯA CHẠY TRAINING THẬT (dừng theo yêu cầu) -- script này mới được smoke-test
vài step (xem STATUS.md) để verify wiring đúng, số liệu in ra lúc smoke-test
KHÔNG phải kết quả huấn luyện.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pytorch_pretrained_bert.modeling import BertConfig

from data_prep.attempt3_corpus import load_attempt3_examples
from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.fraud_model import FraudClassifier
from model.gnn_encoder import GraphSAGEEncoder
from model.graph_sampling import subgraph_to_data
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import compute_metrics

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
GLOBAL_SEED = 44
D_GRAPH = 128
FUSION_DIM = 256


class ExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch):
    return (
        torch.stack([e.input_ids for e in batch]),
        torch.stack([e.attention_mask for e in batch]),
        torch.tensor([e.label for e in batch], dtype=torch.long),
        torch.tensor([e.global_idx for e in batch], dtype=torch.long),
    )


def build_models(device: torch.device, pretrained_bert: str = "bert-base-uncased"):
    bert_config = BertConfig(
        vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
    )
    graph_encoder = GraphSAGEEncoder(in_channels=23, hidden_channels=128, out_channels=D_GRAPH).to(device)
    classifier = FraudClassifier(
        bert_config, d_graph=D_GRAPH, fusion_dim=FUSION_DIM, pretrained_bert=pretrained_bert
    ).to(device)
    return graph_encoder, classifier


def make_train_loader(train_examples, batch_size: int) -> DataLoader:
    # WeightedRandomSampler theo tần suất nghịch đảo lớp -- cùng ý tưởng
    # tri_model/train1.py:433-436 dùng cho corpus 20k train này (3541 dương /
    # 16459 âm, không cực đoan như full-scale nên WeightedRandomSampler ở
    # CẤP SEED vẫn hợp lý; B2's label-aware budget xử lý CẤP HÀNG XÓM, 2 việc
    # bổ sung nhau chứ không thay thế nhau ở scale nhỏ này).
    labels = np.array([e.label for e in train_examples])
    class_count = np.bincount(labels)
    weight_per_class = 1.0 / class_count
    sample_weight = weight_per_class[labels]
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
    return DataLoader(ExampleDataset(train_examples), batch_size=batch_size, sampler=sampler, collate_fn=collate)


def train_one_epoch(graph_encoder, classifier, loader, sampler: LabelAwareNeighborSampler,
                     node_features: torch.Tensor, optimizer, device, max_steps=None) -> float:
    graph_encoder.train()
    classifier.train()
    total_loss, n_steps = 0.0, 0
    for step, (input_ids, attention_mask, labels, global_idx) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        global_ids, edge_index, edge_weight, seed_local_idx = sampler.sample(global_idx)
        subgraph = subgraph_to_data(global_ids, edge_index, edge_weight, node_features).to(device)
        h_all = graph_encoder(subgraph.x, subgraph.edge_index, subgraph.edge_weight)
        h_graph = h_all[seed_local_idx]

        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        logits = classifier(input_ids, h_graph, attention_mask=attention_mask)
        loss = nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(graph_encoder, classifier, examples, device, split_filter=None, batch_size=16) -> dict:
    """Eval dùng full_graph_forward trên graph_inference (đồ thị đầy đủ, đúng
    nguyên tắc B3/predict_account cho account đã có trong graph) -- tính
    h_graph cho TOÀN BỘ node 1 lần, không sample subgraph lại như lúc train."""
    graph_encoder.eval()
    classifier.eval()

    # full_graph_forward cần ~8GB liền mạch (đo ở B1) CHỈ RIÊNG cho
    # GraphSAGEEncoder; với BERT-base đồng trú (weights+optimizer state+cache
    # còn sót từ các bước train ngay trước) đã chiếm ~7.7GB thật (không phải
    # cache nhàn rỗi -- đã tự thử `torch.cuda.empty_cache()` lúc smoke-test,
    # KHÔNG giải quyết được, xem STATUS.md) thì 7.7+8=15.7GB vượt hẳn 11.63GB
    # của card này. Fix thật: chạy full_graph_forward trên CPU (đo thật: 3s
    # cho 2.97M node/5.35M cạnh, chấp nhận được vì chỉ chạy 1 lần/epoch lúc
    # eval, không phải hot loop) thay vì cố nhét vừa GPU.
    graph_encoder_device = next(graph_encoder.parameters()).device
    graph_encoder.to("cpu")
    graph_inference = load_graph("inference")
    h_graph_all = graph_encoder.full_graph_forward(graph_inference, torch.device("cpu"))  # [N, D_GRAPH], no_grad
    graph_encoder.to(graph_encoder_device)

    eval_examples = [e for e in examples if split_filter is None or e.split == split_filter]
    probs, labels = [], []
    for i in range(0, len(eval_examples), batch_size):
        batch = eval_examples[i:i + batch_size]
        input_ids = torch.stack([e.input_ids for e in batch]).to(device)
        attention_mask = torch.stack([e.attention_mask for e in batch]).to(device)
        global_idx = torch.tensor([e.global_idx for e in batch], dtype=torch.long)
        h_graph = h_graph_all[global_idx].to(device)

        logits = classifier(input_ids, h_graph, attention_mask=attention_mask)
        prob_pos = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(prob_pos.cpu().tolist())
        labels.extend([e.label for e in batch])

    return compute_metrics(np.array(labels), np.array(probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--max-examples", type=int, default=None, help="giới hạn corpus, dùng để smoke-test nhanh")
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading Attempt-3 corpus...")
    examples = load_attempt3_examples(max_examples=args.max_examples)
    train_examples = [e for e in examples if e.split == "train"]
    print(f"total examples={len(examples)} train={len(train_examples)}")

    node_features = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    labels_tensor = torch.zeros(node_features.shape[0], dtype=torch.long)
    for e in train_examples:
        labels_tensor[e.global_idx] = e.label
    sampler = LabelAwareNeighborSampler(adj_train, labels_tensor, seed=GLOBAL_SEED)

    graph_encoder, classifier = build_models(device)
    optimizer = torch.optim.AdamW(
        list(graph_encoder.parameters()) + list(classifier.parameters()), lr=args.lr
    )

    train_loader = make_train_loader(train_examples, args.batch_size)

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            graph_encoder, classifier, train_loader, sampler, node_features, optimizer, device,
            max_steps=args.max_steps_per_epoch,
        )
        dt = time.time() - t0
        val_metrics = evaluate(graph_encoder, classifier, examples, device, split_filter="val")
        print(f"epoch={epoch} train_loss={train_loss:.4f} time={dt:.1f}s val={val_metrics}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        torch.save(
            {"graph_encoder": graph_encoder.state_dict(), "classifier": classifier.state_dict(), "epoch": epoch},
            OUTPUT_DIR / f"e2_checkpoint_epoch{epoch}.pt",
        )

    final_metrics = {
        "val": evaluate(graph_encoder, classifier, examples, device, split_filter="val"),
        "pure_test": evaluate(graph_encoder, classifier, examples, device, split_filter="pure_test"),
        "overlap": evaluate(graph_encoder, classifier, examples, device, split_filter="overlap"),
    }
    print("final metrics:", json.dumps(final_metrics, indent=2))
    with open(OUTPUT_DIR / "e2_final_metrics.json", "w") as f:
        json.dump({"history": history, "final": final_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
