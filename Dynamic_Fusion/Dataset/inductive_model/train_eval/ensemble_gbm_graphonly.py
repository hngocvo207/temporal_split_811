"""
Ensemble GBM (tabular 23 dac trung, tuned) + graph-only GraphSAGE (K*=7, xem
STATUS.md muc "K*=7 ap dung CHINH THUC") -- 2 model doc lap, hoc tu 2 nguon tin
hieu khac nhau (dac trung thu cong vs cau truc do thi hoc duoc) va co diem manh
lech nhau (GBM: P@100 cao hon; graph-only: AUPRC/F1-toi-uu cao hon) -- dau hieu
tin hieu bo sung, dang thu ket hop.

Ky luat: MOI phep chon (trong so ensemble, phuong phap stack) chi duoc fit/chon
tren VAL, KHONG BAO GIO nhin pure_test/overlap luc chon -- dung tinh than
nghiem ngat xuyen suot du an. pure_test/overlap chi dung de BAO CAO 1 lan cuoi.

Khong can train lai GraphSAGE (dung lai checkpoint da luu
output/graph_only_hypothesis_model_k7.pt) -- chi retrain GBM (best-tuned config
da co san tu gbm_baseline.py, ~vai chuc giay) de lay predict_proba tren
val/overlap/pure_test (gbm_baseline.py khong luu san probs, chi luu metric).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data

from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import load_labels, load_partition
from train_eval.metrics import compute_metrics, find_best_f1_threshold
from train_eval.train_graph_only_hypothesis import GraphOnlyClassifier, load_top_k_feature_cols

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
GLOBAL_SEED = 44
# best-tuned config da chon boi 5-fold CV o gbm_baseline.py (khong doi lai o day)
GBM_BEST_CFG = dict(learning_rate=0.3, max_leaf_nodes=63, l2_regularization=1.0, max_iter=300)


def _sample_weight(y):
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    return np.where(y == 1, n_neg / max(n_pos, 1), 1.0)


def get_gbm_probs_full(node_features: np.ndarray, labels: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """Tra ve prob(class=1) cho TOAN BO quan the (N node, cung thu tu global
    index voi node_features/labels/partition) -- de main() tu slice theo
    partition==split, dam bao THANG HANG CHINH XAC voi mang tra ve boi
    get_graphonly_probs_full() (tranh loi 2 model bi lech thu tu account do
    2 ham lay-idx khac nhau -- da phat hien khi viet ban dau, xem lich su sua)."""
    train_idx = np.where(partition == "train")[0]
    X_train, y_train = node_features[train_idx], labels[train_idx]
    clf = HistGradientBoostingClassifier(**GBM_BEST_CFG, random_state=GLOBAL_SEED, early_stopping=True)
    clf.fit(X_train, y_train, sample_weight=_sample_weight(y_train))
    return clf.predict_proba(node_features)[:, 1]


def get_graphonly_probs_full() -> np.ndarray:
    """Tra ve prob(class=1) cho TOAN BO quan the (full_graph_forward tren
    graph_inference, cung thu tu global index voi partition.pkl/labels.pkl)."""
    feature_cols = load_top_k_feature_cols(7)
    model = GraphOnlyClassifier(in_channels=7, hidden=64, out=64, dropout=0.2, mlp_classifier=True)
    model.load_state_dict(torch.load(OUTPUT_DIR / "graph_only_hypothesis_model_k7.pt", map_location="cpu"))
    model.eval()
    graph_inference = load_graph("inference")
    graph_inference = Data(x=graph_inference.x[:, feature_cols].contiguous(),
                            edge_index=graph_inference.edge_index, edge_weight=graph_inference.edge_weight)
    with torch.no_grad():
        logits_all = model.full_graph_logits(graph_inference, torch.device("cpu"))
    return torch.softmax(logits_all, dim=-1)[:, 1].numpy()


def fit_ensemble_methods(gbm_val, graph_val, y_val):
    """Tra ve dict {method_name: predict_fn(gbm_prob, graph_prob) -> ensemble_prob},
    da fit/chon THAM SO tren VAL. predict_fn nhan mang numpy, tra mang numpy."""
    methods = {}

    # 1. Trung binh don gian (khong tham so)
    methods["avg_0.5_0.5"] = lambda g, s: 0.5 * g + 0.5 * s

    # 2. Trung binh co trong so -- grid-search w tren VAL AUPRC (khong nhin test)
    best_w, best_auprc = 0.5, -1.0
    for w in np.arange(0.0, 1.01, 0.05):
        combo = w * gbm_val + (1 - w) * graph_val
        a = compute_metrics(y_val, combo)["auprc"]
        if a > best_auprc:
            best_auprc, best_w = a, w
    methods[f"weighted_avg_w_gbm={best_w:.2f}"] = (lambda g, s, w=best_w: w * g + (1 - w) * s)

    # 3. Logistic-regression stacking tren [gbm_prob, graph_prob] (rank-scale
    # khac nhau giua 2 model duoc logistic tu xu ly qua he so rieng cho tung cot)
    stacker = LogisticRegression(class_weight="balanced", max_iter=1000)
    X_val_stack = np.column_stack([gbm_val, graph_val])
    stacker.fit(X_val_stack, y_val)
    methods["logreg_stack"] = (lambda g, s, m=stacker: m.predict_proba(np.column_stack([g, s]))[:, 1])

    # 4. Trung binh RANK (bat bien voi thang do khac nhau giua 2 model, chi
    # dua vao thu hang tuong doi trong TUNG split)
    def rank_avg(g, s):
        rg = np.argsort(np.argsort(g)) / (len(g) - 1)
        rs = np.argsort(np.argsort(s)) / (len(s) - 1)
        return 0.5 * rg + 0.5 * rs
    methods["rank_avg"] = rank_avg

    return methods


def main():
    t0 = time.time()
    node_features = load_node_features().numpy()
    labels = load_labels().numpy()
    partition = np.array(load_partition())

    print("Training GBM (best-tuned config) tren toan bo train, predict_proba cho toan quan the ...")
    gbm_probs_full = get_gbm_probs_full(node_features, labels, partition)
    print(f"  xong ({time.time()-t0:.1f}s)")

    print("Nap graph-only-K7 checkpoint, full_graph_forward cho toan quan the ...")
    graph_probs_full = get_graphonly_probs_full()

    def split_arrays(split):
        idx = np.where(partition == split)[0]
        return labels[idx], gbm_probs_full[idx], graph_probs_full[idx]

    gbm = {}
    graph = {}
    for split in ("val", "overlap", "pure_test"):
        y_s, g_s, s_s = split_arrays(split)
        gbm[split] = (y_s, g_s)
        graph[split] = (y_s, s_s)

    y_val, gbm_val = gbm["val"]
    _, graph_val = graph["val"]
    print(f"\nVal: n={len(y_val)} n_pos={int(y_val.sum())}")
    print(f"  GBM     val AUPRC={compute_metrics(y_val, gbm_val)['auprc']:.4f}")
    print(f"  Graph7  val AUPRC={compute_metrics(y_val, graph_val)['auprc']:.4f}")

    methods = fit_ensemble_methods(gbm_val, graph_val, y_val)

    print("\n=== Chon phuong phap ensemble THEO VAL AUPRC (khong nhin test) ===")
    val_scores = {}
    for name, fn in methods.items():
        combo_val = fn(gbm_val, graph_val)
        val_scores[name] = compute_metrics(y_val, combo_val)["auprc"]
        print(f"  {name:28s} val_auprc={val_scores[name]:.4f}")
    best_method_name = max(val_scores, key=val_scores.get)
    best_fn = methods[best_method_name]
    print(f"\n==> Chon '{best_method_name}' (val_auprc={val_scores[best_method_name]:.4f})")

    # Threshold F1 toi uu cung quet tren VAL cua chinh ensemble da chon
    combo_val_best = best_fn(gbm_val, graph_val)
    best_threshold = find_best_f1_threshold(y_val, combo_val_best)

    print(f"\n=== BAO CAO CUOI (1 lan duy nhat tren pure_test/overlap that) ===")
    results = {"chosen_method": best_method_name, "val_auprc_by_method": val_scores,
               "best_threshold": best_threshold, "gbm_cfg": GBM_BEST_CFG}
    for split in ("pure_test", "overlap"):
        y_true, gbm_p = gbm[split]
        _, graph_p = graph[split]
        combo = best_fn(gbm_p, graph_p)
        m_default = compute_metrics(y_true, combo, threshold=0.5, k_list=[100, 1000])
        m_best_t = compute_metrics(y_true, combo, threshold=best_threshold, k_list=[100, 1000])
        m_gbm_alone = compute_metrics(y_true, gbm_p, k_list=[100, 1000])
        m_graph_alone = compute_metrics(y_true, graph_p, k_list=[100, 1000])
        results[split] = {"ensemble_default_t": m_default, "ensemble_best_t": m_best_t,
                           "gbm_alone": m_gbm_alone, "graph_only_k7_alone": m_graph_alone}
        print(f"\n--- {split} (n={len(y_true)}, n_pos={int(y_true.sum())}) ---")
        print(f"  {'model':22s} {'AUPRC':>7} {'ROC-AUC':>8} {'P@100':>7} {'P@1000':>8} {'Recall@1000':>12} {'F1@best_t':>10}")
        print(f"  {'GBM alone':22s} {m_gbm_alone['auprc']:7.3f} {m_gbm_alone['roc_auc']:8.3f} "
              f"{m_gbm_alone['p_at_100']:7.3f} {m_gbm_alone['p_at_1000']:8.3f} {m_gbm_alone['recall_at_1000']:12.3f} {'-':>10}")
        print(f"  {'Graph-only-K7 alone':22s} {m_graph_alone['auprc']:7.3f} {m_graph_alone['roc_auc']:8.3f} "
              f"{m_graph_alone['p_at_100']:7.3f} {m_graph_alone['p_at_1000']:8.3f} {m_graph_alone['recall_at_1000']:12.3f} {'-':>10}")
        print(f"  {'ENSEMBLE (' + best_method_name[:10] + ')':22s} {m_best_t['auprc']:7.3f} {m_best_t['roc_auc']:8.3f} "
              f"{m_best_t['p_at_100']:7.3f} {m_best_t['p_at_1000']:8.3f} {m_best_t['recall_at_1000']:12.3f} {m_best_t['f1_pos']:10.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "ensemble_gbm_graphonly_k7_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s")
    print(f"Saved -> {OUTPUT_DIR / 'ensemble_gbm_graphonly_k7_result.json'}")


if __name__ == "__main__":
    main()
