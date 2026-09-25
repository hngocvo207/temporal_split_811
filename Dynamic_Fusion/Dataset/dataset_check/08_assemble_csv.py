"""
08_assemble_csv.py
==================
Merges the exact group 1+2 arrays (02) with the group 3 chunk files (05) into
the final full-scale feature table, in select_add_features.py's own column
schema, for all 2,973,489 nodes.

Writes:
  raw_data/MulDiGraph/features_output_fullscale.csv
        node, label, is_phisher, mg_partition, n_nodes, n_edges + all 26 features
  raw_data/MulDiGraph/features_output_top10_MG_fullscale.csv
        the 10 columns mg_refit_features.py consumes, StandardScaler-normalised
        with mean/std fit on the 'train' partition ONLY (new_split.md §B)

The second file is written here rather than by re-running mg_refit_features.py
because that script hardcodes the 5,655-row CSV path; this keeps its train-only
fitting rule identical while pointing at full coverage. Both the raw and the
scaled table are written so the fit can be audited.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ARR = HERE / "arrays"
CHUNKS = HERE / "group3_chunks"
SPLIT_DIR = BASE / "data/preprocessed/Dataset_MG"

OUT_FULL = BASE / "raw_data/MulDiGraph/features_output_fullscale.csv"
OUT_TOP10 = BASE / "raw_data/MulDiGraph/features_output_top10_MG_fullscale.csv"
OUT_ALL = BASE / "raw_data/MulDiGraph/features_output_all23_MG_fullscale.csv"

G1 = ["out_degree", "in_degree", "direction_ratio", "max_out_amount", "min_out_amount",
      "avg_out_amount", "max_in_amount", "min_in_amount", "avg_in_amount",
      "account_balance", "lifetime_days", "active_days"]
G2 = ["freq_out_short", "freq_in_short", "freq_out_long", "freq_in_long",
      "short_long_out_ratio", "short_long_in_ratio"]
G3 = ["katz_centrality", "betweenness_centrality", "degree_centrality",
      "closeness_centrality", "clustering_coefficient", "eigenvector_centrality",
      "in_degree_centrality", "out_degree_centrality"]

# Dropped by request: katz / closeness / eigenvector (46.8% of group-3 compute).
# None of the three is in TOP10_FEATURE_NAMES, so the model's input is unchanged
# -- the delivered table simply carries 23 features instead of 26.
#
# The 197 chunks computed before the switch DO hold real values for them, which
# would ship these columns populated for ~10% of nodes and NaN for the other 90%.
# A partly-populated column is worse than an absent one: any fit or summary
# statistic that touches it silently uses a biased subsample (and because chunks
# are strided, not even a clean random one). So they are dropped outright.
G3_SKIPPED = ["katz_centrality", "closeness_centrality", "eigenvector_centrality"]
TOP10 = ["betweenness_centrality", "clustering_coefficient", "in_degree", "freq_in_long",
         "out_degree", "freq_out_long", "freq_out_short", "max_out_amount",
         "in_degree_centrality", "active_days"]


def main():
    t0 = time.time()
    nodes = np.load(ARR / "nodes.npy", allow_pickle=True)
    N = len(nodes)

    # ── group 3: stitch chunks, verifying full coverage ──────────────────────
    # chunks are STRIDED (see chunk_ids() in 05): chunk j holds ids j, j+K, j+2K, ...
    g3 = np.full((N, 10), np.nan, dtype=np.float64)
    # Defensive: ignore any stale write-in-progress temp file left by a hard
    # power-off. Current 05 prefixes them with ".tmp_" so they cannot match this
    # glob, but an older run used a ".tmp.npy" suffix that did.
    files = sorted(f for f in CHUNKS.glob("schunk_*.npy")
                   if not f.name.startswith(".tmp") and ".tmp." not in f.name)
    stale = [f for f in CHUNKS.glob("*.npy") if f not in files]
    if stale:
        print(f"[i] ignoring {len(stale)} stale temp file(s) from an interrupted run")
    for f in files:
        parts = f.stem.split("_")
        j, n_chunks = int(parts[1]), int(parts[3])
        ids = np.arange(j, N, n_chunks, dtype=np.int64)
        block = np.load(f)
        assert len(block) == len(ids), f"{f.name}: {len(block)} rows vs {len(ids)} ids"
        g3[ids] = block
    # coverage must be judged on a column that --reduced never blanks; column 0 is
    # katz_centrality, which IS NaN for every reduced chunk. Use n_nodes (col 8),
    # which every mode writes -- otherwise a complete run would look 90% missing.
    missing = np.isnan(g3[:, 8])
    print(f"[i] {len(files):,} chunk files;  rows still missing: {int(missing.sum()):,}")
    if missing.any():
        raise SystemExit(f"[✗] incomplete: {int(missing.sum()):,} nodes have no group-3 row. "
                         f"Re-run 05_group3_centrality.py --reduced (it resumes).")

    # ── group 1+2 ────────────────────────────────────────────────────────────
    z = np.load(HERE / "groups12_full.npz")

    df = pd.DataFrame({"node": nodes})
    for c in G1 + G2:
        df[c] = z[c]
    for i, c in enumerate(G3):
        df[c] = g3[:, i]
    df["n_nodes"] = g3[:, 8].astype(np.int64)
    df["n_edges"] = g3[:, 9].astype(np.int64)

    # ── labels + partition ───────────────────────────────────────────────────
    # Ground truth is the graph's own `isp` attribute (labels.pkl, 1,165 positives).
    #
    # phisher_accounts.txt is deliberately NOT used. It lists 5,480 addresses that
    # are all present in this graph, but it agrees with `isp` on only 963 of them
    # (only-txt 4,517 | only-isp 202) -- two substantially different label sets.
    # Every other stage of this project (mg_temporal_pipeline, mg_propagate_labels,
    # mg_build_examples, train1.py, and every F1 in preprocessing_and_eval_report.md)
    # scores against `isp`, so `isp` is what this table carries.
    with open(SPLIT_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    isp = pd.Series(nodes).map(labels).fillna(0).to_numpy().astype(np.int8)
    df["label"] = isp
    print(f"[i] label = isp from labels.pkl: {int(isp.sum()):,} positives")

    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    df["mg_partition"] = pd.Series(nodes).map(partition).to_numpy()
    print("[i] partition coverage:\n" +
          df["mg_partition"].value_counts(dropna=False).to_string())

    # per-partition positive counts must reproduce split_stats.json exactly
    chk = df.groupby("mg_partition")["label"].sum().to_dict()
    expect = {"train": 519, "val": 117, "overlap": 217, "pure_test": 312}
    ok = all(int(chk.get(k, -1)) == v for k, v in expect.items())
    print(f"[i] positives per partition: "
          f"{ {k: int(chk.get(k, 0)) for k in expect} }  "
          f"expected {expect}  -> {'PASS' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("[✗] label/partition join disagrees with split_stats.json")

    g3_kept = [c for c in G3 if c not in G3_SKIPPED]
    if G3_SKIPPED:
        part = df[G3_SKIPPED].notna().mean()
        print(f"\n[i] dropping {len(G3_SKIPPED)} skipped centralities; populated for only "
              f"{100*part.iloc[0]:.1f}% of nodes: {', '.join(G3_SKIPPED)}")
    else:
        # guard: a NaN here means reduced-mode chunks leaked into a full-set run
        nan_g3 = df[G3].isna().any(axis=1).sum()
        if nan_g3:
            raise SystemExit(f"[✗] {nan_g3:,} rows have NaN in a group-3 column — "
                             f"reduced-mode chunks are mixed in. Delete them and re-run.")
        print(f"\n[i] full feature set: no centrality dropped, no NaN present")

    FEATURES_ALL = G1 + G2 + g3_kept
    cols = ["node", "label", "mg_partition", "n_nodes", "n_edges"] + FEATURES_ALL
    df = df[cols]
    print(f"[i] final feature count: {len(G1)+len(G2)+len(g3_kept)} "
          f"({len(G1)} basic + {len(G2)} temporal + {len(g3_kept)} centrality)")
    print(f"\n[…] writing {OUT_FULL} …", flush=True)
    df.to_csv(OUT_FULL, index=False)
    print(f"[✓] {OUT_FULL}  ({OUT_FULL.stat().st_size/1e6:,.0f} MB)  shape={df.shape}")

    # ── train-only StandardScaler, same rule as mg_refit_features.py ────────
    train_mask = (df["mg_partition"] == "train").to_numpy()
    print(f"\n[i] fitting scaler on 'train' partition only: {int(train_mask.sum()):,} rows "
          f"({100*train_mask.mean():.1f}% of nodes)")
    sub = df.loc[train_mask, TOP10]
    mean, std = sub.mean(), sub.std().replace(0, 1.0)
    scaled = pd.DataFrame({"node": df["node"]})
    scaled[TOP10] = (df[TOP10] - mean) / std
    scaled.to_csv(OUT_TOP10, index=False)
    print(f"[✓] {OUT_TOP10}  ({OUT_TOP10.stat().st_size/1e6:,.0f} MB)")

    # ── same train-only fit, but over ALL delivered features ────────────────
    # The univariate top-10 selection is provably built on a broken basis (§9/§12
    # of the report: old split, 600x-enriched sample, wrong label set), and a
    # 23-feature input is not a dimensionality problem against 1.9M training rows.
    # This file lets the model consume everything and skip the selection step.
    sub_a = df.loc[train_mask, FEATURES_ALL]
    mean_a, std_a = sub_a.mean(), sub_a.std().replace(0, 1.0)
    scaled_a = pd.DataFrame({"node": df["node"]})
    scaled_a[FEATURES_ALL] = (df[FEATURES_ALL] - mean_a) / std_a
    scaled_a.to_csv(OUT_ALL, index=False)
    chk_a = scaled_a.loc[train_mask, FEATURES_ALL]
    print(f"[✓] {OUT_ALL}  ({OUT_ALL.stat().st_size/1e6:,.0f} MB)  "
          f"{len(FEATURES_ALL)} features")
    print(f"  sanity — train rows after scaling: max|mean|={chk_a.mean().abs().max():.2e}  "
          f"std range [{chk_a.std().min():.6f}, {chk_a.std().max():.6f}]")

    print("\n── train-only fit statistics (full coverage) ──")
    stats = pd.DataFrame({"train_mean": mean, "train_std": std,
                          "all_mean_after": scaled[TOP10].mean(),
                          "all_std_after": scaled[TOP10].std()})
    print(stats.to_string())
    chk = scaled.loc[train_mask, TOP10]
    print(f"\n  sanity — train rows after scaling: max|mean|={chk.mean().abs().max():.2e}  "
          f"std range [{chk.std().min():.6f}, {chk.std().max():.6f}]")

    print("\n── per-partition mean of each top-10 feature (raw) ──")
    print(df.groupby("mg_partition", dropna=False)[TOP10].mean().T.to_string())
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
