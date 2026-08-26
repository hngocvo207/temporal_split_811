"""
Thí nghiệm tối giản để PROVE INDUCTIVE HYPOTHESIS trước khi đầu tư tiếp vào E2
(BERT+fusion đầy đủ) -- theo đúng đề xuất tái sắp xếp thứ tự: "fix experimental
definition -> prove inductive hypothesis -> then build full model", thay vì
chạy thẳng B1->F3 như new_propose.md ban đầu.

GIẢ THUYẾT CẦN CHỨNG MINH: GraphSAGE (B1) + label-aware sampler (B2), tự nó
(KHÔNG cần BERT/fusion), có phân loại được account CHƯA TỪNG có 1 cạnh nào lúc
train hay không -- đây chính xác là dạng test mà kiến trúc cũ (tri_model, GCN
vocab cố định) sập (F1 0.62->0.055, xem new_propose.md mục E3).

ĐỊNH NGHĨA THỰC NGHIỆM ĐàFIX (khác new_propose.md's "vài trăm account giữ
ngoài vocab" mơ hồ) -- dùng ĐÚNG 3 phân vùng đã có sẵn trong split_config.json/
partition.pkl, đã tự verify bằng dữ liệu thật (không suy đoán):
  - train   (1,945,607 acc, 519 dương): dùng để train.
  - val     (216,178 acc, 117 dương): 206,556/216,178 CÓ cạnh trong adj_train
            -- theo dõi tiến độ train (in-distribution, không phải bài test
            inductive).
  - overlap (201,931 acc, 217 dương): 197,351/201,931 CÓ cạnh trong adj_train
            -- "đã thấy 1 phần cấu trúc" (t_min<=T_cutoff<t_max), KHÔNG phải
            bài test inductive nghiêm ngặt, chỉ để đối chiếu.
  - pure_test (609,773 acc, 312 dương): 100% CÓ BẬC = 0 TUYỆT ĐỐI trong
            adj_train (verify thật, không phải theo tài liệu) -- ĐÂY LÀ BÀI
            TEST INDUCTIVE THẬT. Model chưa từng thấy 1 cạnh nào của các
            account này lúc train.

Eval trên pure_test/overlap dùng adj_inference (đồ thị đầy đủ, có cạnh thật
của các account này) qua full_graph_forward -- đúng kịch bản triển khai thật
(dùng đồ thị hiện tại để suy luận, không phải đồ thị đóng băng lúc train).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import PREPROC_DIR, load_labels, load_partition
from model.gnn_encoder import GraphSAGEEncoder
from model.graph_sampling import subgraph_to_data
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import compute_metrics

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
GLOBAL_SEED = 44


def build_partition_indices():
    labels = load_labels()
    partition = np.array(load_partition())
    idx = {}
    for split in ("train", "val", "overlap", "pure_test"):
        mask = partition == split
        idx[f"{split}_pos"] = torch.tensor(np.where(mask & (labels.numpy() == 1))[0], dtype=torch.long)
        idx[f"{split}_neg"] = torch.tensor(np.where(mask & (labels.numpy() == 0))[0], dtype=torch.long)
    return idx, labels


class GraphOnlyClassifier(nn.Module):
    """Không có nhánh text/fusion -- chỉ để kiểm chứng giả thuyết inductive
    của riêng nhánh graph (B1+B2), không phải kiến trúc cuối cùng."""

    def __init__(self, in_channels=23, hidden=128, out=128, dropout=0.2):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels=hidden, out_channels=out, dropout=dropout)
        self.classifier = nn.Linear(out, 2)

    def forward_subgraph(self, subgraph):
        h_all = self.encoder(subgraph.x, subgraph.edge_index, subgraph.edge_weight)
        return h_all  # caller indexes seed_local_idx

    def full_graph_logits(self, data, device):
        h_all = self.encoder.full_graph_forward(data, device)
        return self.classifier(h_all)


def make_epoch_batches(pos_idx: torch.Tensor, neg_idx: torch.Tensor, neg_ratio: int, batch_size: int,
                        rng: torch.Generator):
    n_neg = min(neg_idx.numel(), pos_idx.numel() * neg_ratio)
    neg_sample = neg_idx[torch.randperm(neg_idx.numel(), generator=rng)[:n_neg]]
    combined = torch.cat([pos_idx, neg_sample])
    perm = combined[torch.randperm(combined.numel(), generator=rng)]
    return [perm[i:i + batch_size] for i in range(0, perm.numel(), batch_size)]


def train(args):
    torch.manual_seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    idx, labels = build_partition_indices()
    print({k: v.numel() for k, v in idx.items()})

    node_features = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    sampler = LabelAwareNeighborSampler(adj_train, labels, seed=GLOBAL_SEED)

    model = GraphOnlyClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(GLOBAL_SEED)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        batches = make_epoch_batches(idx["train_pos"], idx["train_neg"], args.neg_ratio, args.batch_size, rng)
        total_loss = 0.0
        for seeds in batches:
            global_ids, edge_index, edge_weight, seed_local_idx = sampler.sample(seeds)
            subgraph = subgraph_to_data(global_ids, edge_index, edge_weight, node_features).to(device)
            h_all = model.forward_subgraph(subgraph)
            logits = model.classifier(h_all[seed_local_idx])
            y = labels[seeds].to(device)
            loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dt = time.time() - t0
        avg_loss = total_loss / len(batches)
        log = {"epoch": epoch, "train_loss": avg_loss, "n_batches": len(batches), "time_s": dt}

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, idx, labels, device)
            log["eval"] = metrics
            print(f"epoch={epoch} loss={avg_loss:.4f} time={dt:.1f}s eval={metrics}")
        else:
            print(f"epoch={epoch} loss={avg_loss:.4f} time={dt:.1f}s")
        history.append(log)

    return model, history, idx, labels


@torch.no_grad()
def evaluate(model: GraphOnlyClassifier, idx: dict, labels: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    graph_inference = load_graph("inference")
    logits_all = model.full_graph_logits(graph_inference, device)  # [N, 2], no_grad, full adj_inference
    probs_all = torch.softmax(logits_all, dim=-1)[:, 1].cpu().numpy()

    results = {}
    for split in ("val", "overlap", "pure_test"):
        pos = idx[f"{split}_pos"]
        neg = idx[f"{split}_neg"]
        split_idx = torch.cat([pos, neg]).numpy()
        y_true = labels[split_idx].numpy()
        y_prob = probs_all[split_idx]
        results[split] = compute_metrics(y_true, y_prob)
    return results


@torch.no_grad()
def naive_baselines(idx: dict, labels: torch.Tensor) -> dict:
    """Sàn tham chiếu: score ngẫu nhiên uniform -- AUPRC kỳ vọng ~ tỉ lệ dương
    (prevalence), F1(pos) của "luôn đoán âm" = 0. Dùng để không ảo tưởng nếu
    con số eval nhìn "cao" nhưng thực ra dễ đạt vì lớp dương siêu hiếm."""
    out = {}
    rng = np.random.default_rng(GLOBAL_SEED)
    for split in ("val", "overlap", "pure_test"):
        pos, neg = idx[f"{split}_pos"], idx[f"{split}_neg"]
        n_pos, n = pos.numel(), pos.numel() + neg.numel()
        y_true = np.concatenate([np.ones(n_pos), np.zeros(neg.numel())])
        y_prob_random = rng.random(n)
        out[split] = {"prevalence": n_pos / n, **compute_metrics(y_true, y_prob_random)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idx, labels = build_partition_indices()
    print("naive baselines (random score, sanity floor):")
    baselines = naive_baselines(idx, labels)
    print(json.dumps(baselines, indent=2))

    model, history, idx, labels = train(args)
    final_eval = evaluate(model, idx, labels, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("\n=== FINAL RESULT ===")
    print(json.dumps(final_eval, indent=2))
    print("\npure_test = bài test inductive THẬT (0 cạnh train-time) -- đây là con số quyết định giả thuyết.")

    with open(OUTPUT_DIR / "graph_only_hypothesis_result.json", "w") as f:
        json.dump({"baselines": baselines, "history": history, "final_eval": final_eval,
                   "args": vars(args)}, f, indent=2)
    torch.save(model.state_dict(), OUTPUT_DIR / "graph_only_hypothesis_model.pt")


if __name__ == "__main__":
    main()
