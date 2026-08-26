"""
04_validate_groups12.py
=======================
Correctness gate for 02_groups12_exact.py.

The vectorised extractor claims to reproduce select_add_features.py's group 1+2
semantics exactly. This checks that claim against the ONLY existing ground truth
-- the 5,655 rows in raw_data/MulDiGraph/features_output_split.csv that the
original networkx script actually produced.

Caveat handled explicitly below: that CSV was produced with `blocked_nodes` =
old-split val+test addresses, so for a seed with a blocked direct neighbour the
original DROPPED that neighbour's edges from the ball and therefore from the
seed's own in/out transaction lists. Those seeds cannot agree with a full-graph
computation and are reported separately rather than silently excused.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ARR = HERE / "arrays"
CSV = BASE / "raw_data/MulDiGraph/features_output_split.csv"
OLD = Path("/home/ngocvo/Desktop/ngocvo/Dynamic_Feature/data/preprocessed/Multigraph")

G12 = ["out_degree", "in_degree", "direction_ratio", "max_out_amount", "min_out_amount",
       "avg_out_amount", "max_in_amount", "min_in_amount", "avg_in_amount",
       "account_balance", "lifetime_days", "active_days",
       "freq_out_short", "freq_in_short", "freq_out_long", "freq_in_long",
       "short_long_out_ratio", "short_long_in_ratio"]


def main():
    nodes = np.load(ARR / "nodes.npy", allow_pickle=True)
    pos = {a: i for i, a in enumerate(nodes)}
    z = np.load(HERE / "groups12_full.npz")

    df = pd.read_csv(CSV)
    df["i"] = df["node"].map(pos)
    assert df["i"].notna().all(), "some CSV nodes are not in the graph"
    idx = df["i"].to_numpy(dtype=np.int64)

    # which of these seeds had a blocked direct neighbour in the original run?
    with open(OLD / "val_addresses.pkl", "rb") as f:
        val_a = pickle.load(f)
    with open(OLD / "test_addresses.pkl", "rb") as f:
        test_a = pickle.load(f)
    blocked = {pos[a] for a in set(val_a) | set(test_a) if a in pos}
    print(f"  original blocked_nodes (old-split val+test): {len(blocked):,}")

    ez = np.load(ARR / "edges.npz")
    src, dst = ez["src"], ez["dst"]
    bmask = np.zeros(len(nodes), dtype=bool)
    bmask[list(blocked)] = True
    touched = np.zeros(len(nodes), dtype=bool)
    touched[src[bmask[dst]]] = True          # has an out-edge to a blocked node
    touched[dst[bmask[src]]] = True          # has an in-edge from a blocked node
    aff = touched[idx]
    print(f"  CSV seeds with >=1 blocked direct neighbour: {aff.sum():,} / {len(df):,}"
          f"  -> compared separately\n")

    print(f"{'feature':<24} {'exact match':>14} {'max |diff|':>14}   (clean seeds only)")
    print("─" * 72)
    ok = True
    for c in G12:
        mine = np.asarray(z[c], dtype=np.float64)[idx]
        theirs = df[c].to_numpy(dtype=np.float64)
        d = np.abs(mine - theirs)
        clean = ~aff
        n_eq = int((d[clean] <= 1e-6).sum())
        n_cl = int(clean.sum())
        flag = "" if n_eq == n_cl else "   <-- MISMATCH"
        if n_eq != n_cl:
            ok = False
        print(f"{c:<24} {n_eq:>7,}/{n_cl:<6,} {d[clean].max():>14.3e}{flag}")

    print("\n  seeds WITH a blocked neighbour (expected to differ — the original")
    print("  dropped those edges from the seed's own transaction list):")
    for c in ["out_degree", "in_degree", "active_days"]:
        mine = np.asarray(z[c], dtype=np.float64)[idx]
        theirs = df[c].to_numpy(dtype=np.float64)
        d = np.abs(mine - theirs)[aff]
        print(f"    {c:<20} differ on {int((d > 1e-6).sum()):,}/{int(aff.sum()):,} "
              f"seeds, max |diff| {d.max():.4g}")

    print("\n" + ("[PASS] group 1+2 reproduce the original exactly on all unblocked seeds."
                  if ok else "[FAIL] see MISMATCH rows above."))


if __name__ == "__main__":
    main()
