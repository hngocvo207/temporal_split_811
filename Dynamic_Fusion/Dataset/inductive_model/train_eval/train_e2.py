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

Kỷ luật chọn model GIỐNG train_graph_only_hypothesis.py: chọn checkpoint theo
VAL AUPRC tốt nhất (early stopping), chỉ đọc pure_test/overlap SAU KHI đã chọn
xong -- không "nhìn trộm" tập test lúc train. Log trực tiếp wandb (project
fraud_detection_inductive, group "e2_full_fusion").
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
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
WANDB_PROJECT = "fraud_detection_inductive"
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


def make_train_loader(train_examples, batch_size: int, pos_neg_ratio: float = None) -> DataLoader:
    # WeightedRandomSampler theo tần suất nghịch đảo lớp -- cùng ý tưởng
    # tri_model/train1.py:433-436 dùng cho corpus 20k train này (3541 dương /
    # 16459 âm, không cực đoan như full-scale nên WeightedRandomSampler ở
    # CẤP SEED vẫn hợp lý; B2's label-aware budget xử lý CẤP HÀNG XÓM, 2 việc
    # bổ sung nhau chứ không thay thế nhau ở scale nhỏ này).
    #
    # pos_neg_ratio=None (mặc định, dùng ở E2 v1): cân bằng ĐẦY ĐỦ 1:1 kỳ vọng
    # mỗi lần sample -- hợp lý khi n_pos không quá nhỏ (3541 ở Attempt-3).
    # pos_neg_ratio=r (E2 v2, corpus 100k strict-label chỉ có 519 dương): cân
    # bằng đầy đủ 1:1 sẽ lặp mỗi dương ~96 lần/epoch (100000 draw * 0.5 / 519)
    # -- dễ overfit đúng 519 example. Dùng tỷ lệ 1 dương : r âm (vd r=4) để
    # vẫn tăng tần suất dương nhưng không lặp quá mức, giữ đa dạng âm hơn.
    labels = np.array([e.label for e in train_examples])
    class_count = np.bincount(labels)
    if pos_neg_ratio is None:
        weight_per_class = 1.0 / class_count
    else:
        n_neg, n_pos = class_count[0], class_count[1]
        total = 1.0 + pos_neg_ratio
        weight_per_class = np.array([(pos_neg_ratio / total) / n_neg, (1.0 / total) / n_pos])
    sample_weight = weight_per_class[labels]
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
    return DataLoader(ExampleDataset(train_examples), batch_size=batch_size, sampler=sampler, collate_fn=collate)


def train_one_epoch(graph_encoder, classifier, loader, sampler: LabelAwareNeighborSampler,
                     node_features: torch.Tensor, optimizer, device, max_steps=None,
                     pos_weight: float = None) -> float:
    """pos_weight (BERT_debug.md Task 4, mac dinh None = hanh vi CU khong doi):
    them class-weighted CE ben tren WeightedRandomSampler cua make_train_loader
    -- vd pos_weight=2.0 nghia la 1 loi sai tren mau duong bi phat nang gap 2
    loi sai tren mau am, DOC LAP voi ty le sample cua sampler (2 co che khac
    nhau: sampler quyet dinh TAN SUAT xuat hien trong batch, pos_weight quyet
    dinh GRADIENT MAGNITUDE khi da xuat hien)."""
    graph_encoder.train()
    classifier.train()
    ce_weight = None
    if pos_weight is not None:
        ce_weight = torch.tensor([1.0, pos_weight], device=device)
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
        loss = nn.functional.cross_entropy(logits, labels, weight=ce_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def compute_probs_labels(graph_encoder, classifier, examples, device, split_filter=None, batch_size=16):
    """Trả về (probs, labels) thô -- dùng chung cho evaluate() và cho việc
    quét threshold trên val (train_e2_v2.py, không lặp lại full_graph_forward).
    labels lấy `label_strict` nếu Example có (full_test_corpus.py, đã verify
    khớp canonical labels.pkl), fallback về `label` (propagated) cho Example
    cũ không có field này (Attempt-3, attempt3_corpus.py) -- giữ nguyên hành
    vi cũ cho E2 v1, chỉ sửa đúng cho các nguồn dữ liệu có nhãn strict."""
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
        labels.extend([(e.label_strict if getattr(e, "label_strict", None) is not None else e.label) for e in batch])

    return np.array(probs), np.array(labels)


def evaluate(graph_encoder, classifier, examples, device, split_filter=None, batch_size=16,
             threshold: float = 0.5, k_list=None) -> dict:
    """Eval dùng full_graph_forward trên graph_inference (đồ thị đầy đủ, đúng
    nguyên tắc B3/predict_account cho account đã có trong graph) -- tính
    h_graph cho TOÀN BỘ node 1 lần, không sample subgraph lại như lúc train.

    k_list (vd [100, 1000]) -- thêm P@k/Recall@k vào kết quả, dùng để so sánh
    thẳng hàng với GBM baseline (xem train_eval/eval_full_test.py) -- mặc định
    None để KHÔNG đổi hành vi các nơi gọi evaluate() khác."""
    probs, labels = compute_probs_labels(graph_encoder, classifier, examples, device, split_filter, batch_size)
    return compute_metrics(labels, probs, threshold=threshold, k_list=k_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--max-examples", type=int, default=None, help="giới hạn corpus, dùng để smoke-test nhanh")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=3, help="epoch không cải thiện val AUPRC trước khi dừng sớm")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group="e2_full_fusion", name="e2_bert_graphsage_fusion", config=vars(args)
    )

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

    best_val_auprc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            graph_encoder, classifier, train_loader, sampler, node_features, optimizer, device,
            max_steps=args.max_steps_per_epoch,
        )
        dt = time.time() - t0
        val_metrics = evaluate(graph_encoder, classifier, examples, device, split_filter="val")
        val_auprc = val_metrics["auprc"]
        improved = val_auprc > best_val_auprc
        if improved:
            best_val_auprc = val_auprc
            best_state = {
                "graph_encoder": {k: v.detach().cpu().clone() for k, v in graph_encoder.state_dict().items()},
                "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
            }
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(f"epoch={epoch} train_loss={train_loss:.4f} time={dt:.1f}s val={val_metrics}"
              f"{' *best*' if improved else ''}")
        history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt, "val": val_metrics})
        if run is not None:
            run.log({"train_loss": train_loss, "epoch_time_s": dt,
                      **{f"val/{k}": v for k, v in val_metrics.items()},
                      "best_val_auprc_so_far": best_val_auprc}, step=epoch)

        if args.patience and epochs_since_improve >= args.patience:
            print(f"early stop at epoch={epoch} (no val AUPRC improvement for {args.patience} epochs)")
            break

    graph_encoder.load_state_dict(best_state["graph_encoder"])
    classifier.load_state_dict(best_state["classifier"])
    torch.save(best_state, OUTPUT_DIR / "e2_best_checkpoint.pt")

    final_metrics = {
        "val": evaluate(graph_encoder, classifier, examples, device, split_filter="val"),
        "pure_test": evaluate(graph_encoder, classifier, examples, device, split_filter="pure_test"),
        "overlap": evaluate(graph_encoder, classifier, examples, device, split_filter="overlap"),
    }
    print(f"\n=== FINAL RESULT (best val AUPRC checkpoint, epoch {best_epoch}) ===")
    print(json.dumps(final_metrics, indent=2))
    print("\nBaseline Attempt-3 (tri_model, vocab-locked GCN): F1(pos)=87.63%, AUPRC overall=0.9155 "
          "(pure_test F1=0.9038/AUPRC=0.9474, overlap F1=0.8605/AUPRC=0.8846)")

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_auprc"] = best_val_auprc
        for split, m in final_metrics.items():
            for k, v in m.items():
                run.summary[f"final_{split}/{k}"] = v
        run.finish()

    with open(OUTPUT_DIR / "e2_final_metrics.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "final": final_metrics,
                    "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
