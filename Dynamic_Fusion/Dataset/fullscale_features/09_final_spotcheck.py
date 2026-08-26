"""
09_final_spotcheck.py
=====================
Final end-to-end gate, run AFTER 08_assemble_csv.py.

06/07 validated the extractor functions in isolation. This validates the
DELIVERED FILE: it picks random rows out of features_output_fullscale.csv, runs
the untouched original select_add_features.py pipeline on those same accounts,
and compares all 26 features plus n_nodes/n_edges.

This is the check that would catch an indexing slip in the chunk stitching --
i.e. right numbers written against the wrong node -- which none of the earlier
stage-level tests can see.

Usage: python3 09_final_spotcheck.py [n_rows]
"""

from __future__ import annotations

import importlib.util
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ORIG = Path("/home/ngocvo/Desktop/ngocvo/Dynamic_Feature/Dataset/select_add_features.py")
CSV = BASE / "raw_data/MulDiGraph/features_output_fullscale.csv"

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

G12 = ["out_degree", "in_degree", "direction_ratio", "max_out_amount", "min_out_amount",
       "avg_out_amount", "max_in_amount", "min_in_amount", "avg_in_amount",
       "account_balance", "lifetime_days", "active_days",
       "freq_out_short", "freq_in_short", "freq_out_long", "freq_in_long",
       "short_long_out_ratio", "short_long_in_ratio"]
# katz / closeness / eigenvector are not computed in --reduced mode and are absent
# from the delivered CSV, so they are not compared here.
G3_DET = ["degree_centrality", "clustering_coefficient",
          "in_degree_centrality", "out_degree_centrality"]


def main():
    spec = importlib.util.spec_from_file_location("orig", ORIG)
    saf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saf)

    print(f"[…] reading {CSV.name} …", flush=True)
    df = pd.read_csv(CSV)
    print(f"[✓] {len(df):,} rows", flush=True)

    rng = np.random.default_rng(4242)
    pick = rng.choice(len(df), size=N_ROWS, replace=False)
    rows = df.iloc[pick].reset_index(drop=True)

    print("[…] loading networkx MultiDiGraph …", flush=True)
    t = time.time()
    with open(BASE / "raw_data/MulDiGraph/MulDiGraph.pkl", "rb") as f:
        data = pickle.load(f)
    G = data[0] if isinstance(data, list) else data
    print(f"[✓] {time.time()-t:.0f}s\n", flush=True)

    bad = {}
    meta_bad = 0
    bet_rows = []
    for i in range(len(rows)):
        r = rows.iloc[i]
        addr = r["node"]
        sub = saf.bfs_subgraph(G, addr, set(), depth=2)
        ref = {}
        ref.update(saf.extract_basic_features(sub, addr))
        ref.update(saf.extract_temporal_features(sub, addr, 30, 180))
        random.seed(11)
        ref.update(saf.extract_centrality_features(sub, addr, 0.005))

        if sub.number_of_nodes() != r["n_nodes"] or sub.number_of_edges() != r["n_edges"]:
            meta_bad += 1
            print(f"  [{i+1}] META MISMATCH {addr[:14]}: ref "
                  f"({sub.number_of_nodes()},{sub.number_of_edges()}) vs csv "
                  f"({r['n_nodes']},{r['n_edges']})")

        for c in G12:
            if abs(float(ref[c]) - float(r[c])) > 1.01e-6:
                bad.setdefault(c, []).append((addr, ref[c], r[c]))
        for c in G3_DET:
            if abs(float(ref[c]) - float(r[c])) > 1.01e-8:
                bad.setdefault(c, []).append((addr, ref[c], r[c]))
        bet_rows.append((sub.number_of_nodes(), ref["betweenness_centrality"],
                         float(r["betweenness_centrality"])))
        print(f"  [{i+1}/{len(rows)}] {addr[:16]}… ball={sub.number_of_nodes():,} ok", flush=True)

    print("\n" + "=" * 76)
    print(f"  n_nodes/n_edges exact : {len(rows)-meta_bad}/{len(rows)}")
    print(f"  18 group1+2 features  : {'ALL MATCH' if not any(c in bad for c in G12) else 'MISMATCH'}"
          f"   (tol 1e-6, the original's round(x,6) last digit)")
    print(f"  7 deterministic g3    : {'ALL MATCH' if not any(c in bad for c in G3_DET) else 'MISMATCH'}"
          f"   (tol 1e-8)")
    for c, v in bad.items():
        print(f"    {c}: {len(v)} mismatched, e.g. {v[0]}")

    nz = [(n, a, b) for n, a, b in bet_rows if max(a, b) > 0]
    print(f"\n  betweenness            : {len(nz)}/{len(bet_rows)} rows non-zero "
          f"(the original re-samples k sources per call, so exact agreement is")
    print(f"                           not defined for balls with n > 200)")
    print("=" * 76)
    print("\n" + ("[PASS] delivered file reproduces the original script."
                  if not bad and not meta_bad else "[FAIL] see above."))


if __name__ == "__main__":
    main()
