"""
Theo yeu cau: "23 dac trung node co phu hop khong cho GraphSAGE?" -- so sanh
AUPRC full pure_test cua nhanh graph-only (GraphSAGEEncoder, cung cau hinh da
thang sweep: lr=0.003/hidden=64/out=64/mlp_classifier=True, xem
graph_only_sweep_v2_fixed_direction_results.json) khi chi dung TOP-K trong 23
dac trung (xep hang theo |correlation| voi nhan tren TRAIN-ONLY, khong ro ri
test) voi K in {3,5,10,15,23}, VA 1 doi chung K=10 CHON NGAU NHIEN (khong
theo importance) de xac nhan xep hang co y nghia that.

Tai su dung nguyen kien truc/optimizer/sampler cua train_graph_only_hypothesis.py
(GraphOnlyClassifier, LabelAwareNeighborSampler, subgraph_to_data) -- chi khac
o_channels dau vao (in_channels=K thay vi 23 co dinh) va node_features/graph
data.x da SLICE truoc theo K cot duoc chon.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import load_labels, load_partition
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import compute_metrics
from train_eval.train_graph_only_hypothesis import (
    GraphOnlyClassifier, build_partition_indices, make_epoch_batches, PREPROC_DIR,
)
import scipy.sparse as sp

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
GLOBAL_SEED = 44
BEST_CFG = dict(lr=0.003, hidden=64, out=64, dropout=0.2, mlp_classifier=True)
EPOCHS, PATIENCE, EVAL_EVERY, BATCH_SIZE, NEG_RATIO = 150, 30, 5, 32, 3


def rank_features_by_corr(node_features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, feature_cols):
    X, y = node_features[train_idx], labels[train_idx]
    scores = []
    for j, c in enumerate(feature_cols):
        xj = X[:, j]
        r = 0.0 if xj.std() < 1e-9 else np.corrcoef(xj, y)[0, 1]
        scores.append((c, j, abs(r) if np.isfinite(r) else 0.0))
    scores.sort(key=lambda t: -t[2])
    return scores  # [(name, orig_col_idx, |corr|), ...] giam dan


def train_and_eval(col_idx: list, node_features_full: torch.Tensor, adj_train, idx, labels, tag: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(GLOBAL_SEED)
    node_features = node_features_full[:, col_idx].contiguous()
    sampler = LabelAwareNeighborSampler(adj_train, labels, seed=GLOBAL_SEED)
    from model.graph_sampling import subgraph_to_data

    model = GraphOnlyClassifier(in_channels=len(col_idx), hidden=BEST_CFG["hidden"], out=BEST_CFG["out"],
                                 dropout=BEST_CFG["dropout"], mlp_classifier=BEST_CFG["mlp_classifier"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_CFG["lr"])
    rng = torch.Generator().manual_seed(GLOBAL_SEED)

    graph_inference_full = load_graph("inference")
    graph_inference_sliced = Data(x=graph_inference_full.x[:, col_idx].contiguous(),
                                   edge_index=graph_inference_full.edge_index,
                                   edge_weight=graph_inference_full.edge_weight)

    best_val_auprc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        batches = make_epoch_batches(idx["train_pos"], idx["train_neg"], NEG_RATIO, BATCH_SIZE, rng)
        for seeds in batches:
            global_ids, edge_index, edge_weight, seed_local_idx = sampler.sample(seeds)
            subgraph = subgraph_to_data(global_ids, edge_index, edge_weight, node_features).to(device)
            h_all = model.forward_subgraph(subgraph)
            logits = model.classifier(h_all[seed_local_idx])
            y = labels[seeds].to(device)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                model.to("cpu")
                logits_all = model.full_graph_logits(graph_inference_sliced, torch.device("cpu"))
                model.to(device)
            probs_all = torch.softmax(logits_all, dim=-1)[:, 1].numpy()
            val_idx = torch.cat([idx["val_pos"], idx["val_neg"]]).numpy()
            val_auprc = compute_metrics(labels[val_idx].numpy(), probs_all[val_idx])["auprc"]
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        model.to("cpu")
        logits_all = model.full_graph_logits(graph_inference_sliced, torch.device("cpu"))
    probs_all = torch.softmax(logits_all, dim=-1)[:, 1].numpy()
    pt_idx = torch.cat([idx["pure_test_pos"], idx["pure_test_neg"]]).numpy()
    m = compute_metrics(labels[pt_idx].numpy(), probs_all[pt_idx], k_list=[100, 1000])
    dt = time.time() - t0
    print(f"[{tag}] K={len(col_idx)} best_epoch={best_epoch} best_val_auprc={best_val_auprc:.4f} "
          f"pure_test: auprc={m['auprc']:.4f} roc_auc={m['roc_auc']:.4f} p_at_100={m['p_at_100']:.3f} "
          f"recall_at_1000={m['recall_at_1000']:.3f} ({dt:.1f}s)")
    return {"tag": tag, "K": len(col_idx), "columns": col_idx, "best_epoch": best_epoch,
            "best_val_auprc": best_val_auprc, "pure_test": m, "time_s": dt}


def main():
    with open(Path(__file__).resolve().parents[1] / "data" / "node_features_all23.meta.json") as f:
        feature_cols = json.load(f)["feature_columns"]

    idx, labels = build_partition_indices()
    node_features_full = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    train_idx = torch.cat([idx["train_pos"], idx["train_neg"]]).numpy()

    ranked = rank_features_by_corr(node_features_full.numpy(), labels.numpy(), train_idx, feature_cols)
    print("Xep hang 23 dac trung theo |corr| voi nhan (TRAIN-only):")
    for name, j, r in ranked:
        print(f"  {name:28s} col={j:2d} |corr|={r:.4f}")

    results = []
    for k in (3, 5, 10, 15, 23):
        top_k_cols = [j for _, j, _ in ranked[:k]]
        results.append(train_and_eval(top_k_cols, node_features_full, adj_train, idx, labels, tag=f"top{k}"))

    rng_np = np.random.default_rng(GLOBAL_SEED)
    random_10_cols = sorted(rng_np.choice(23, size=10, replace=False).tolist())
    results.append(train_and_eval(random_10_cols, node_features_full, adj_train, idx, labels, tag="random10"))

    print("\n=== TONG HOP (pure_test, 609,773 that) ===")
    print(f"{'tag':<10} {'K':>3} {'val_auprc':>10} {'test_auprc':>11} {'roc_auc':>8} {'p@100':>6} {'recall@1000':>12}")
    for r in results:
        m = r["pure_test"]
        print(f"{r['tag']:<10} {r['K']:>3} {r['best_val_auprc']:>10.4f} {m['auprc']:>11.4f} "
              f"{m['roc_auc']:>8.4f} {m['p_at_100']:>6.3f} {m['recall_at_1000']:>12.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "feature_count_ablation_result.json", "w") as f:
        json.dump({"feature_ranking": [(n, j, r) for n, j, r in ranked], "results": results,
                   "best_cfg": BEST_CFG}, f, indent=2)
    print(f"\nSaved -> {OUTPUT_DIR / 'feature_count_ablation_result.json'}")


if __name__ == "__main__":
    main()
