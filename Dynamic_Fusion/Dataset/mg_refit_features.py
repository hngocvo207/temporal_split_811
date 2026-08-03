"""
mg_refit_features.py
=====================
Step 3c — refit node-feature normalization using ONLY the 'train'
partition (new_split.md §B: NOT 'overlap', NOT 'val', NOT 'pure_test' —
train-only, no exceptions), then apply the same transform to every other
partition.

Dataset/feature_pipeline.py already had the right IDEA (fit StandardScaler
on df[df['split']=='TRAIN'] only) but its 'split' column came from the old
per-node BFS-feature CSV's own ad-hoc split, unrelated to the temporal
split fixed in mg_temporal_pipeline.py. This script re-derives the split
label for every node in that CSV from partition.pkl instead, so the
scaler is fit on the CORRECT train set.

Only 5,655 of the 2,973,489 nodes have BFS-depth-2 structural features
extracted (raw_data/MulDiGraph/features_output_split.csv) -- extracting
them for the full graph is a separate, expensive step (select_add_features.py,
BFS per account) out of scope for this session. This script fixes the
normalization LOGIC against the features that do exist; a full-scale run
must first re-run select_add_features.py over the complete node set.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "raw_data/MulDiGraph/features_output_split.csv"
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OUT_CSV = BASE_DIR / "raw_data/MulDiGraph/features_output_top10_MG_fixed.csv"

TOP10_FEATURE_NAMES = [
    "betweenness_centrality", "clustering_coefficient", "in_degree", "freq_in_long",
    "out_degree", "freq_out_long", "freq_out_short", "max_out_amount",
    "in_degree_centrality", "active_days",
]


def sep(t):
    print(f"\n{'─' * 4} {t} {'─' * max(0, 74 - len(t))}")


def main():
    print("=" * 78)
    print("  mg_refit_features.py — train-only feature normalization (Step 3c)")
    print("=" * 78)

    sep("Load features CSV + NEW partition")
    df = pd.read_csv(FEATURES_CSV)
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    df["mg_partition"] = df["node"].map(partition)
    n_matched = df["mg_partition"].notna().sum()
    print(f"  Feature rows: {len(df):,}   matched to a partition: {n_matched:,}")
    print(df["mg_partition"].value_counts(dropna=False).to_string())

    train_mask = df["mg_partition"] == "train"
    train_df = df[train_mask]
    print(f"\n  Fitting on 'train' partition only (no overlap/val/pure_test): {len(train_df):,} rows")
    if len(train_df) < 10:
        print("  [!] Very few rows available for this session's feature-extraction sample; "
              "stats below are illustrative only, not production-grade.")

    sep("Before scaling (full df, all partitions)")
    before = df[TOP10_FEATURE_NAMES].agg(["mean", "std"]).T
    print(before.to_string())

    sep("Fit StandardScaler on train-only, transform everything (Step 3c)")
    mean = train_df[TOP10_FEATURE_NAMES].mean()
    std = train_df[TOP10_FEATURE_NAMES].std().replace(0, 1.0)
    df_scaled = df.copy()
    df_scaled[TOP10_FEATURE_NAMES] = (df[TOP10_FEATURE_NAMES] - mean) / std

    sep("After scaling (train-only mean/std applied to all)")
    after = df_scaled[TOP10_FEATURE_NAMES].agg(["mean", "std"]).T
    print(after.to_string())
    print("\n  (train-partition rows should now have mean~0/std~1; val/test will deviate --")
    print("   that deviation is the honest signal of train/test distribution shift, not a bug.)")

    train_after = df_scaled.loc[train_mask, TOP10_FEATURE_NAMES]
    print("\n  Sanity — train-only rows after scaling:")
    print(train_after.agg(["mean", "std"]).T.to_string())

    out_cols = ["node"] + TOP10_FEATURE_NAMES
    df_scaled[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
