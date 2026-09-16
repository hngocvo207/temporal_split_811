"""
Baseline GBM (HistGradientBoostingClassifier, chi 23 dac trung tabular, KHONG
graph/BERT) -- moc tham chieu bat buoc phai vuot qua truoc khi tin bat ky ket
qua GraphSAGE/BERT/fusion nao (xem STATUS.md muc "PHAT HIEN QUAN TRONG NHAT").

Sweep hyperparameter (learning_rate x max_leaf_nodes x l2_regularization) +
5-fold StratifiedKFold CV tren train partition that, CHON config theo CV mean
AUPRC (KHONG nhin pure_test luc chon -- dung ky luat train/val/test nghiem
ngat xuyen suot du an). Sau khi chon xong, retrain config tot nhat tren TOAN
BO train, eval MOT LAN DUY NHAT tren FULL pure_test.

Ket qua da co (split cu, luu o day de doi chieu):
  default hyperparameter: CV=0.2530, full pure_test AUPRC=0.343
  best-tuned (lr=0.3, max_leaf_nodes=63, l2=1.0): CV=0.3395, full pure_test
    AUPRC=0.347, ROC-AUC=0.959, P@100=0.72, P@1000=0.186, Recall@1000=0.596
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from data_prep.io_utils import load_node_features
from data_prep.labels_io import load_labels, load_partition
from train_eval.metrics import compute_metrics, find_best_f1_threshold

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

LEARNING_RATES = [0.03, 0.1, 0.3]
MAX_LEAF_NODES = [15, 31, 63]
L2_REG = [0.0, 1.0]
N_FOLDS = 5
GLOBAL_SEED = 44


def _sample_weight(y):
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    return np.where(y == 1, n_neg / max(n_pos, 1), 1.0)


def main():
    t0 = time.time()
    node_features = load_node_features().numpy()
    labels = load_labels().numpy()
    partition = np.array(load_partition())

    train_idx = np.where(partition == "train")[0]
    train_y = labels[train_idx]
    X_train = node_features[train_idx]
    print(f"train: n={len(train_idx)} n_pos={train_y.sum()}")

    test_split_name = "test" if "test" in set(partition.tolist()) else "pure_test"
    pt_idx = np.where(partition == test_split_name)[0]
    pt_y = labels[pt_idx]
    X_pt = node_features[pt_idx]
    print(f"{test_split_name} FULL: n={len(pt_idx)} n_pos={pt_y.sum()}\n")

    val_idx = np.where(partition == "val")[0]
    val_y, X_val = labels[val_idx], node_features[val_idx]
    print(f"val (danh cho quet threshold, KHONG dung de tune hyperparameter): "
          f"n={len(val_idx)} n_pos={val_y.sum()}\n")

    grid = list(itertools.product(LEARNING_RATES, MAX_LEAF_NODES, L2_REG))
    print(f"grid: {len(grid)} configs x {N_FOLDS} fold = {len(grid) * N_FOLDS} fits\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
    folds = list(skf.split(X_train, train_y))

    sweep_results = []
    for lr, mln, l2 in grid:
        t_cfg = time.time()
        fold_aucs = []
        for tr_i, va_i in folds:
            y_tr, y_va = train_y[tr_i], train_y[va_i]
            clf = HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=mln, l2_regularization=l2,
                max_iter=300, random_state=GLOBAL_SEED, early_stopping=True,
            )
            clf.fit(X_train[tr_i], y_tr, sample_weight=_sample_weight(y_tr))
            probs_va = clf.predict_proba(X_train[va_i])[:, 1]
            fold_aucs.append(compute_metrics(y_va, probs_va)["auprc"])
        fold_aucs = np.array(fold_aucs)
        mean, std = fold_aucs.mean(), fold_aucs.std(ddof=1)
        print(f"lr={lr:.2f} max_leaf_nodes={mln:2d} l2={l2:.1f}: "
              f"CV AUPRC mean={mean:.4f} std={std:.4f} ({time.time() - t_cfg:.1f}s)")
        sweep_results.append({
            "learning_rate": lr, "max_leaf_nodes": mln, "l2_regularization": l2,
            "cv_fold_auprc": fold_aucs.tolist(), "cv_mean": float(mean), "cv_std": float(std),
        })

    best = max(sweep_results, key=lambda r: r["cv_mean"])
    print(f"\n=== BEST theo CV (khong nhin {test_split_name}): {best} ===")

    print(f"\n=== Retrain BEST tren toan bo train, eval 1 lan tren FULL {test_split_name} ===")
    clf_best = HistGradientBoostingClassifier(
        learning_rate=best["learning_rate"], max_leaf_nodes=best["max_leaf_nodes"],
        l2_regularization=best["l2_regularization"], max_iter=300,
        random_state=GLOBAL_SEED, early_stopping=True,
    )
    clf_best.fit(X_train, train_y, sample_weight=_sample_weight(train_y))
    probs_pt = clf_best.predict_proba(X_pt)[:, 1]
    m_final_default_t = compute_metrics(pt_y, probs_pt, threshold=0.5, k_list=[100, 1000])

    # F1(pos) o threshold TOI UU -- quet tren VAL (khong phai test, tranh nhin
    # trom), roi ap threshold do sang test -- cung phuong phap voi
    # train_e2_v2.py/train_graph_only_hypothesis.py de so sanh cong bang.
    probs_val = clf_best.predict_proba(X_val)[:, 1]
    best_threshold = find_best_f1_threshold(val_y, probs_val)
    m_final_best_t = compute_metrics(pt_y, probs_pt, threshold=best_threshold, k_list=[100, 1000])
    print(f"FULL {test_split_name} ({len(pt_idx):,} that) @ threshold=0.5 (mac dinh): "
          f"{json.dumps(m_final_default_t, indent=2)}")
    print(f"FULL {test_split_name} @ threshold={best_threshold:.4f} (toi uu, quet tren val): "
          f"f1_pos={m_final_best_t['f1_pos']:.4f} precision={m_final_best_t['precision_pos']:.4f} "
          f"recall={m_final_best_t['recall_pos']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "gbm_hparam_sweep_result.json", "w") as f:
        json.dump({
            "test_split_name": test_split_name, "sweep": sweep_results, "best_config": best,
            "final_full_test_metrics": m_final_default_t,
            "best_threshold": best_threshold,
            "final_full_test_metrics_best_threshold": m_final_best_t,
            "total_time_s": time.time() - t0,
        }, f, indent=2)
    print(f"\nTOTAL TIME: {time.time() - t0:.1f}s")
    print(f"Saved -> {OUTPUT_DIR / 'gbm_hparam_sweep_result.json'}")


if __name__ == "__main__":
    main()
