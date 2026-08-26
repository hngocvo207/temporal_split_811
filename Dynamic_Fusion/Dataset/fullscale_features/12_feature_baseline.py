"""
12_feature_baseline.py
======================
Cheapest possible test of whether the graph-feature branch is worth saving:
train a plain classifier on the 23 features ALONE -- no BERT, no GCN, no fusion --
and see how far it gets.

Protocol mirrors the real pipeline exactly, so the numbers are comparable:
  * fit on `mg_partition == 'train'` only (1,945,607 nodes, 519 positives)
  * scaler fit on train only
  * decision threshold calibrated on `val` only (216,178, 117 positives), argmax F1
  * scored on test = overlap u pure_test (811,704 nodes, 529 positives), and on
    the pure_test / overlap slices separately
  * ground truth is `isp` throughout; phisher_accounts.txt is not used

Two comparison points, because the report contains two different test-set
compositions and mixing them up would be meaningless:
  A) FULL test partition, 811,704 nodes, 0.065% positive
     -> compare with full_scale_test_eval_report.md: Attempt 3 = F1(pos) 0.0549
  B) BOUNDED test, 5,000 pure_test + 5,000 overlap with all real positives kept,
     10,000 nodes at 5.29% positive -- the composition of §11's table
     -> compare with Attempt 3 = F1(pos) 0.6227 overall / 0.7406 pure_test

Usage: python3 12_feature_baseline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CSV = BASE / "raw_data/MulDiGraph/features_output_fullscale.csv"
META = {"node", "label", "mg_partition", "n_nodes", "n_edges"}

OLD_TOP10 = ["betweenness_centrality", "clustering_coefficient", "in_degree", "freq_in_long",
             "out_degree", "freq_out_long", "freq_out_short", "max_out_amount",
             "in_degree_centrality", "active_days"]

# Reference numbers from the existing reports, for context in the printout.
REF_FULL = 0.0549     # full_scale_test_eval_report.md §4, Attempt 3 ckpt, full test
REF_BOUNDED = 0.6227  # preprocessing_and_eval_report.md §11, Attempt 3, 10k test


def slog(a):
    """signed log1p — the amount features span 0..800,000 and wreck a linear model."""
    return np.sign(a) * np.log1p(np.abs(a))


def calibrate(y, p):
    """threshold maximising positive-class F1, computed on validation only."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(th) else 0
    return float(th[i]), float(f1[i])


def report(name, y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(
        slice=name, n=len(y), pos=int(y.sum()),
        f1_pos=f1_score(y, yh, pos_label=1, zero_division=0),
        auprc=average_precision_score(y, p),
        auc=roc_auc_score(y, p) if y.sum() else float("nan"),
        precision=(yh[y == 1].sum() / max(yh.sum(), 1)),
        recall=(yh[y == 1].sum() / max(int(y.sum()), 1)),
        n_flagged=int(yh.sum()),
    )


def show(rows, title):
    print(f"\n  {title}")
    print(f"    {'slice':<12}{'n':>9}{'pos':>6}{'F1(pos)':>10}{'AUPRC':>9}{'AUC':>8}"
          f"{'Prec':>8}{'Rec':>8}{'flagged':>10}")
    for r in rows:
        print(f"    {r['slice']:<12}{r['n']:>9,}{r['pos']:>6}{r['f1_pos']:>10.4f}"
              f"{r['auprc']:>9.4f}{r['auc']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['n_flagged']:>10,}")


def main():
    t0 = time.time()
    print(f"[…] reading {CSV.name} …", flush=True)
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in META]
    print(f"[✓] {len(df):,} rows x {len(feats)} features ({time.time()-t0:.0f}s)")

    part = df["mg_partition"].to_numpy()
    y = df["label"].to_numpy(np.int8)
    X = df[feats].to_numpy(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    tr, va = part == "train", part == "val"
    pt, ov = part == "pure_test", part == "overlap"
    te = pt | ov
    for nm, m in [("train", tr), ("val", va), ("overlap", ov), ("pure_test", pt), ("test", te)]:
        print(f"    {nm:<10} {m.sum():>9,} nodes  {int(y[m].sum()):>5} pos  "
              f"{100*y[m].mean():.4f}%")

    # bounded test set with §11's composition: all real positives + random negatives
    rng = np.random.default_rng(20240810)
    def bounded(mask, k):
        idx = np.where(mask)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        neg = rng.choice(neg, size=k - len(pos), replace=False)
        return np.concatenate([pos, neg])
    b_pt, b_ov = bounded(pt, 5000), bounded(ov, 5000)
    b_te = np.concatenate([b_pt, b_ov])

    results = {}
    for featset, cols in [("all 23", feats), ("old top-10", OLD_TOP10)]:
        ci = [feats.index(c) for c in cols]
        Xs = X[:, ci]

        for mname in ["LogReg", "HistGBM"]:
            t = time.time()
            if mname == "LogReg":
                sc = StandardScaler().fit(slog(Xs[tr]))
                Z = sc.transform(slog(Xs))
                clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
                clf.fit(Z[tr], y[tr])
                p = clf.predict_proba(Z)[:, 1]
            else:
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.1, class_weight="balanced",
                    early_stopping=True, validation_fraction=0.1, random_state=0)
                clf.fit(Xs[tr], y[tr])
                p = clf.predict_proba(Xs)[:, 1]

            thr, vf1 = calibrate(y[va], p[va])
            rows_full = [report("pure_test", y[pt], p[pt], thr),
                         report("overlap", y[ov], p[ov], thr),
                         report("overall", y[te], p[te], thr)]
            rows_bnd = [report("pure_test", y[b_pt], p[b_pt], thr),
                        report("overlap", y[b_ov], p[b_ov], thr),
                        report("overall", y[b_te], p[b_te], thr)]

            key = f"{mname} / {featset}"
            results[key] = (rows_full, rows_bnd)
            print(f"\n{'='*94}")
            print(f"  {key}   (fit {time.time()-t:.0f}s)   "
                  f"val-calibrated threshold={thr:.6f} (val F1={vf1:.4f})")
            print("=" * 94)
            show(rows_full, "A) FULL test partition — 811,704 nodes, 0.065% positive")
            show(rows_bnd, "B) BOUNDED test — 10,000 nodes, 5.29% positive (§11 composition)")

    # ── head-to-head against the trained model ──────────────────────────────
    print("\n" + "=" * 94)
    print("  BASELINE vs the full Dynamic-Fusion model (BERT + GCN + features)")
    print("=" * 94)
    print(f"  {'config':<26}{'A) full test F1(pos)':>24}{'B) bounded F1(pos)':>22}")
    print("  " + "─" * 90)
    for k, (rf, rb) in results.items():
        print(f"  {k:<26}{rf[-1]['f1_pos']:>24.4f}{rb[-1]['f1_pos']:>22.4f}")
    print("  " + "─" * 90)
    print(f"  {'Attempt 3 (trained)':<26}{REF_FULL:>24.4f}{REF_BOUNDED:>22.4f}")
    print(f"\n  A) comparable to full_scale_test_eval_report.md §4")
    print(f"  B) comparable to preprocessing_and_eval_report.md §11")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
