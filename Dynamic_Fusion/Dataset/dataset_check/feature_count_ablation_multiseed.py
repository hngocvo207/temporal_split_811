"""
Ban mo rong cua feature_count_ablation.py (single-seed): chay NHIEU SEED cho
moi K de "chot" so luong dac trung toi uu bang so lieu co y nghia thong ke,
thay vi 1 lan chay co the may-rui (xem STATUS.md: ban truoc co diem trung bat
thuong o K=10, nghi do nhieu seed).

Ky luat chon K* KHONG nhin test: voi moi K, lay TRUNG BINH val AUPRC qua
N_SEEDS lan chay (val subsample partition, khong phai pure_test) -- K* la K
co val AUPRC trung binh cao nhat. Test AUPRC chi de BAO CAO, khong dung de
chon K (dung tinh than "khong nhin trom test" xuyen suot du an).

Cung kien truc/optimizer/sampler voi feature_count_ablation.py (GraphOnlyClassifier,
cau hinh da thang sweep lr=0.003/hidden=64/out=64/mlp_classifier=True) -- chi
khac o CHAY LAP LAI voi seed khac nhau (anh huong model init, sampler RNG,
thu tu batch) cho MOI K, thay vi 1 seed co dinh.
"""
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.data import Data

from data_prep.io_utils import load_graph, load_node_features
from model.label_aware_sampler import LabelAwareNeighborSampler
from model.graph_sampling import subgraph_to_data
from train_eval.metrics import compute_metrics
from train_eval.train_graph_only_hypothesis import (
    GraphOnlyClassifier, build_partition_indices, make_epoch_batches, PREPROC_DIR,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
BASE_SEED = 44
K_GRID = (3, 5, 7, 10, 15, 20, 23)
N_SEEDS = 5
BEST_CFG = dict(lr=0.003, hidden=64, out=64, dropout=0.2, mlp_classifier=True)
EPOCHS, EVAL_EVERY, BATCH_SIZE, NEG_RATIO = 150, 5, 32, 3


def rank_features_by_corr(node_features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, feature_cols):
    X, y = node_features[train_idx], labels[train_idx]
    scores = []
    for j, c in enumerate(feature_cols):
        xj = X[:, j]
        r = 0.0 if xj.std() < 1e-9 else np.corrcoef(xj, y)[0, 1]
        scores.append((c, j, abs(r) if np.isfinite(r) else 0.0))
    scores.sort(key=lambda t: -t[2])
    return scores


def train_and_eval_one_seed(col_idx: list, node_features_full: torch.Tensor, adj_train, idx, labels,
                             seed: int, tag: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    node_features = node_features_full[:, col_idx].contiguous()
    sampler = LabelAwareNeighborSampler(adj_train, labels, seed=seed)

    model = GraphOnlyClassifier(in_channels=len(col_idx), hidden=BEST_CFG["hidden"], out=BEST_CFG["out"],
                                 dropout=BEST_CFG["dropout"], mlp_classifier=BEST_CFG["mlp_classifier"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_CFG["lr"])
    rng = torch.Generator().manual_seed(seed)

    graph_inference_full = load_graph("inference")
    graph_inference_sliced = Data(x=graph_inference_full.x[:, col_idx].contiguous(),
                                   edge_index=graph_inference_full.edge_index,
                                   edge_weight=graph_inference_full.edge_weight)

    best_val_auprc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        batches = make_epoch_batches(idx["train_pos"], idx["train_neg"], NEG_RATIO, BATCH_SIZE, rng)
        for seeds_batch in batches:
            global_ids, edge_index, edge_weight, seed_local_idx = sampler.sample(seeds_batch)
            subgraph = subgraph_to_data(global_ids, edge_index, edge_weight, node_features).to(device)
            h_all = model.forward_subgraph(subgraph)
            logits = model.classifier(h_all[seed_local_idx])
            y = labels[seeds_batch].to(device)
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
    print(f"[{tag}] K={len(col_idx)} seed={seed} best_epoch={best_epoch} "
          f"val_auprc={best_val_auprc:.4f} test_auprc={m['auprc']:.4f} p_at_100={m['p_at_100']:.3f} ({dt:.1f}s)")
    return {"K": len(col_idx), "seed": seed, "best_epoch": best_epoch,
            "val_auprc": best_val_auprc, "test_metrics": m, "time_s": dt}


def main():
    with open(Path(__file__).resolve().parents[1] / "data" / "node_features_all23.meta.json") as f:
        feature_cols = json.load(f)["feature_columns"]

    idx, labels = build_partition_indices()
    node_features_full = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    train_idx = torch.cat([idx["train_pos"], idx["train_neg"]]).numpy()

    ranked = rank_features_by_corr(node_features_full.numpy(), labels.numpy(), train_idx, feature_cols)
    print("Xep hang 23 dac trung theo |corr| voi nhan (TRAIN-only), dung chung cho moi seed:")
    for name, j, r in ranked:
        print(f"  {name:28s} col={j:2d} |corr|={r:.4f}")
    print(f"\nK grid = {K_GRID}, N_SEEDS = {N_SEEDS} -> {len(K_GRID) * N_SEEDS} lan chay tong cong\n")

    all_runs = []
    for k in K_GRID:
        top_k_cols = [j for _, j, _ in ranked[:k]]
        for rep in range(N_SEEDS):
            seed = BASE_SEED + rep
            r = train_and_eval_one_seed(top_k_cols, node_features_full, adj_train, idx, labels,
                                         seed=seed, tag=f"K={k}")
            r["columns"] = top_k_cols
            all_runs.append(r)

    # Tong hop theo K: trung binh + std qua N_SEEDS -- CHON K* THEO VAL AUPRC
    # trung binh (khong nhin test luc chon, dung ky luat cua du an).
    summary = []
    for k in K_GRID:
        runs_k = [r for r in all_runs if r["K"] == k]
        val_aups = np.array([r["val_auprc"] for r in runs_k])
        test_aups = np.array([r["test_metrics"]["auprc"] for r in runs_k])
        p100s = np.array([r["test_metrics"]["p_at_100"] for r in runs_k])
        summary.append({
            "K": k,
            "val_auprc_mean": float(val_aups.mean()), "val_auprc_std": float(val_aups.std(ddof=1)),
            "test_auprc_mean": float(test_aups.mean()), "test_auprc_std": float(test_aups.std(ddof=1)),
            "test_p_at_100_mean": float(p100s.mean()), "test_p_at_100_std": float(p100s.std(ddof=1)),
        })

    k_star = max(summary, key=lambda s: s["val_auprc_mean"])["K"]

    print(f"\n=== TONG HOP {N_SEEDS} SEED/K (chon K* theo VAL AUPRC trung binh, KHONG nhin test) ===")
    print(f"{'K':>3} {'val_auprc':>18} {'test_auprc':>18} {'test_p@100':>16}")
    for s in summary:
        marker = "  <== K*" if s["K"] == k_star else ""
        print(f"{s['K']:>3} {s['val_auprc_mean']:>8.4f} ± {s['val_auprc_std']:.4f}   "
              f"{s['test_auprc_mean']:>8.4f} ± {s['test_auprc_std']:.4f}   "
              f"{s['test_p_at_100_mean']:>6.3f} ± {s['test_p_at_100_std']:.3f}{marker}")
    print(f"\nK* (val AUPRC trung binh cao nhat) = {k_star}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "feature_count_ablation_multiseed_result.json", "w") as f:
        json.dump({"feature_ranking": [(n, j, r) for n, j, r in ranked], "k_grid": K_GRID, "n_seeds": N_SEEDS,
                   "all_runs": all_runs, "summary": summary, "k_star": k_star, "best_cfg": BEST_CFG}, f, indent=2)
    print(f"\nSaved -> {OUTPUT_DIR / 'feature_count_ablation_multiseed_result.json'}")


if __name__ == "__main__":
    main()
