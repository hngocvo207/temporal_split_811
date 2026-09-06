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

    def __init__(self, in_channels=23, hidden=128, out=128, dropout=0.2, mlp_classifier=False):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels=hidden, out_channels=out, dropout=dropout)
        if mlp_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(out, out // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(out // 2, 2)
            )
        else:
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


def train(args, idx=None, labels=None, node_features=None, adj_train=None, verbose=True, wandb_run=None):
    """args cần: epochs, batch_size, neg_ratio, lr, eval_every, weight_decay,
    hidden, out, dropout, mlp_classifier, patience (early stop theo val AUPRC
    -- CHỌN MODEL BẰNG VAL, KHÔNG BẰNG pure_test, để không "nhìn trộm" tập test
    khi tune -- đúng tinh thần fix experimental definition nghiêm ngặt).

    Trả về model ở checkpoint có val AUPRC TỐT NHẤT (không phải epoch cuối)."""
    torch.manual_seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if idx is None:
        idx, labels = build_partition_indices()
    if node_features is None:
        node_features = load_node_features()
    if adj_train is None:
        adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    sampler = LabelAwareNeighborSampler(adj_train, labels, seed=GLOBAL_SEED)

    model = GraphOnlyClassifier(
        hidden=args.hidden, out=args.out, dropout=args.dropout, mlp_classifier=args.mlp_classifier
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(GLOBAL_SEED)

    best_val_auprc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
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
        log = {"epoch": epoch, "train_loss": avg_loss, "time_s": dt}
        if wandb_run is not None:
            wandb_run.log({"train_loss": avg_loss, "epoch_time_s": dt}, step=epoch)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, idx, labels, device)
            log["eval"] = metrics
            val_auprc = metrics["val"]["auprc"]
            improved = val_auprc > best_val_auprc
            if improved:
                best_val_auprc = val_auprc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += args.eval_every
            if wandb_run is not None:
                flat = {f"{split}/{k}": v for split, m in metrics.items() for k, v in m.items()}
                wandb_run.log({**flat, "best_val_auprc_so_far": best_val_auprc}, step=epoch)
            if verbose:
                print(f"epoch={epoch} loss={avg_loss:.4f} time={dt:.1f}s val_auprc={val_auprc:.4f}"
                      f"{' *best*' if improved else ''}")
            if args.patience and epochs_since_improve >= args.patience:
                if verbose:
                    print(f"early stop at epoch={epoch} (no val AUPRC improvement for {args.patience} epochs)")
                break
        elif verbose:
            print(f"epoch={epoch} loss={avg_loss:.4f} time={dt:.1f}s")
        history.append(log)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, idx, labels, best_epoch, best_val_auprc


_GRAPH_INFERENCE_CACHE = None


def _get_graph_inference():
    # Nạp 1 lần, dùng lại cho mọi lần eval (được gọi rất nhiều lần trong sweep)
    # thay vì đọc lại file .pt 364MB từ đĩa mỗi lần.
    global _GRAPH_INFERENCE_CACHE
    if _GRAPH_INFERENCE_CACHE is None:
        _GRAPH_INFERENCE_CACHE = load_graph("inference")
    return _GRAPH_INFERENCE_CACHE


@torch.no_grad()
def evaluate(model: GraphOnlyClassifier, idx: dict, labels: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    # full_graph_forward ở hidden lớn (vd 256) + model khác đồng thời có thể
    # còn activation/optimizer trên GPU -> OOM thật đã tự bắt được lúc sweep
    # (12GB không đủ). Chạy trên CPU: encoder rất nhỏ (vài trăm nghìn tham số),
    # chuyển qua lại gần như miễn phí, còn forward mất vài giây -- chấp nhận
    # được vì evaluate() không phải hot loop.
    original_device = next(model.parameters()).device
    model.to("cpu")
    graph_inference = _get_graph_inference()
    logits_all = model.full_graph_logits(graph_inference, torch.device("cpu"))
    model.to(original_device)
    probs_all = torch.softmax(logits_all, dim=-1)[:, 1].numpy()

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
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mlp-classifier", action="store_true")
    ap.add_argument("--patience", type=int, default=0, help="0 = tắt early stopping")
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idx, labels = build_partition_indices()
    print("naive baselines (random score, sanity floor):")
    baselines = naive_baselines(idx, labels)
    print(json.dumps(baselines, indent=2))

    model, history, idx, labels, best_epoch, best_val_auprc = train(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_eval = evaluate(model, idx, labels, device)

    print(f"\n=== FINAL RESULT (model = best val AUPRC checkpoint, epoch {best_epoch}) ===")
    print(json.dumps(final_eval, indent=2))
    print("\npure_test = bài test inductive THẬT (0 cạnh train-time) -- đây là con số quyết định giả thuyết.")

    with open(OUTPUT_DIR / "graph_only_hypothesis_result.json", "w") as f:
        json.dump({"baselines": baselines, "history": history, "final_eval": final_eval,
                   "best_epoch": best_epoch, "best_val_auprc": best_val_auprc,
                   "args": vars(args)}, f, indent=2)
    torch.save(model.state_dict(), OUTPUT_DIR / "graph_only_hypothesis_model.pt")


if __name__ == "__main__":
    main()
