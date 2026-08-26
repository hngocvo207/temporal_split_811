"""
11_rerank_features.py
=====================
Re-runs the top-10 feature selection on the CORRECT split at the CORRECT
prevalence, over full coverage, for all 26 features.

Why this exists
---------------
`feature_pipeline.py` picked the current TOP10_FEATURE_NAMES by |Spearman| with
the label, and guarded against leakage by restricting to `split == 'TRAIN'`. Two
problems, both verified:

  1. That `split` column is the OLD ad-hoc split baked into the feature CSV, not
     the T_cutoff split. Of its 4,524 "TRAIN" rows, only 2,815 (62%) are in the
     current `train` partition; 1,048 are in `pure_test`, 343 in `overlap`, 318
     in `val`. So the selection saw held-out data under the split now in force.
  2. It ran on a 5:5-ish sample: 16.0% positive, against a real train-partition
     prevalence of 0.0267% -- a ~600x enrichment -- on 4,524 of 1,945,607 nodes
     (0.23%).

This script fixes both: `mg_partition == 'train'` from partition.pkl, ground
truth from labels.pkl, every node, every feature.

What it reports
---------------
Spearman rho is kept for continuity with the original criterion, but at
0.0267% prevalence a rank correlation is a weak, hard-to-read statistic, so
three additional rankings are produced that behave better under extreme
imbalance:

  * |Spearman rho|            -- the original criterion, for direct comparison
  * AUC (= Mann-Whitney U)    -- P(random positive ranks above random negative);
                                 prevalence-invariant, which is exactly the
                                 property the old 16%-positive sample lacked
  * Average Precision (AUPRC) -- the metric the model is actually judged on
  * separation                -- |AUC - 0.5| * 2, a symmetric readout so that a
                                 strongly NEGATIVE predictor is not ranked last

DIAGNOSTIC ONLY. TOP10_FEATURE_NAMES is deliberately left unchanged; nothing
downstream reads this script's output. It exists so the current top-10 can be
read against a correctly-computed ranking, not to replace it.

Scope note: the run this reads was done with --reduced, so katz_centrality,
closeness_centrality and eigenvector_centrality are absent from the table and
cannot be ranked here. None of the three is in the current top-10.

Usage: python3 11_rerank_features.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CSV = BASE / "raw_data/MulDiGraph/features_output_fullscale.csv"
SPLIT_DIR = BASE / "data/preprocessed/Dataset_MG"
OUT = HERE / "feature_ranking_fullscale.csv"

OLD_TOP10 = ["betweenness_centrality", "clustering_coefficient", "in_degree", "freq_in_long",
             "out_degree", "freq_out_long", "freq_out_short", "max_out_amount",
             "in_degree_centrality", "active_days"]
META = {"node", "label", "mg_partition", "n_nodes", "n_edges"}


def main():
    t0 = time.time()
    print(f"[…] reading {CSV.name} …", flush=True)
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in META]
    print(f"[✓] {len(df):,} rows x {len(feats)} features ({time.time()-t0:.0f}s)")

    train = df["mg_partition"] == "train"
    y = df.loc[train, "label"].to_numpy(np.int8)
    print(f"\n[i] selection set = mg_partition == 'train': {train.sum():,} nodes, "
          f"{y.sum():,} phishing = {100*y.mean():.4f}% positive")
    print(f"    (feature_pipeline.py used 4,524 rows at 16.0% positive)")
    if y.sum() == 0:
        raise SystemExit("[✗] no positives in the train partition — check labels.pkl")

    rows = []
    for c in feats:
        v = df.loc[train, c].to_numpy(np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.std() == 0:
            rows.append(dict(feature=c, rho=0.0, auc=0.5, ap=float(y.mean()),
                             separation=0.0, pct_zero=100.0 * (v == 0).mean()))
            continue
        rho = spearmanr(v, y)[0]
        auc = roc_auc_score(y, v)
        rows.append(dict(
            feature=c,
            rho=0.0 if np.isnan(rho) else rho,
            auc=auc,
            ap=average_precision_score(y, v),
            separation=abs(auc - 0.5) * 2,
            pct_zero=100.0 * (v == 0).mean(),
        ))
        print(f"    {c:<24} rho={rows[-1]['rho']:+.4f}  auc={auc:.4f}  "
              f"ap={rows[-1]['ap']:.5f}", flush=True)

    r = pd.DataFrame(rows)
    r["rank_rho"] = r["rho"].abs().rank(ascending=False).astype(int)
    r["rank_auc"] = r["separation"].rank(ascending=False).astype(int)
    r["rank_ap"] = r["ap"].rank(ascending=False).astype(int)
    r["in_old_top10"] = r["feature"].isin(OLD_TOP10)
    r = r.sort_values("separation", ascending=False).reset_index(drop=True)
    r.to_csv(OUT, index=False)

    print("\n" + "=" * 100)
    print("  FEATURE RANKING — train partition only, real prevalence, full coverage")
    print("=" * 100)
    print(f"  {'#':>3} {'feature':<24} {'|rho|':>8} {'AUC':>8} {'AP':>9} {'sep':>8} "
          f"{'%zero':>7} {'rk_rho':>7} {'rk_ap':>6}  old10")
    print("  " + "─" * 96)
    for i, x in r.iterrows():
        print(f"  {i+1:>3} {x.feature:<24} {abs(x.rho):>8.4f} {x.auc:>8.4f} {x.ap:>9.5f} "
              f"{x.separation:>8.4f} {x.pct_zero:>6.1f}% {x.rank_rho:>7} {x.rank_ap:>6}  "
              f"{'YES' if x.in_old_top10 else ''}")

    print("\n── proposed top-10 by each criterion ──")
    for crit, col, asc in [("|Spearman| (original criterion)", "rho", False),
                           ("AUC separation", "separation", False),
                           ("Average Precision", "ap", False)]:
        key = r[col].abs() if col == "rho" else r[col]
        top = r.assign(_k=key).sort_values("_k", ascending=asc).head(10)["feature"].tolist()
        print(f"\n  {crit}:")
        for i, f in enumerate(top, 1):
            print(f"    {i:>2}. {f:<26}{'' if f in OLD_TOP10 else '   <-- NOT in old top-10'}")
        kept = len(set(top) & set(OLD_TOP10))
        print(f"    overlap with old top-10: {kept}/10")

    print("\n── the old top-10, where each lands now ──")
    for f in OLD_TOP10:
        x = r[r.feature == f]
        if x.empty:
            print(f"    {f:<26} MISSING from the feature table")
            continue
        x = x.iloc[0]
        print(f"    {f:<26} rank_rho {x.rank_rho:>2}/{len(r)}   rank_ap {x.rank_ap:>2}/{len(r)}   "
              f"AUC {x.auc:.4f}   {x.pct_zero:.1f}% zero")

    print(f"\n[✓] saved {OUT}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
