"""
01_benchmark_original.py
========================
Measures the REAL per-seed cost of select_add_features.py's pipeline
(BFS-depth-2 -> subgraph copy -> 26 features) on a uniform random sample of
seeds, so the full-scale (2,973,489-node) extrapolation is based on measurement
rather than guesswork.

Imports the original functions unmodified from Dynamic_Feature/Dataset/
select_add_features.py -- this benchmarks that exact code, not a reimplementation.

Usage: python3 01_benchmark_original.py [n_samples] [timeout_s_per_seed]
"""

from __future__ import annotations

import importlib.util
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
ORIG = Path("/home/ngocvo/Desktop/ngocvo/Dynamic_Feature/Dataset/select_add_features.py")
RAW = BASE / "raw_data/MulDiGraph/MulDiGraph.pkl"

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BUDGET_S = float(sys.argv[2]) if len(sys.argv) > 2 else 900.0


def load_original_module():
    spec = importlib.util.spec_from_file_location("orig_saf", ORIG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    saf = load_original_module()

    print(f"[…] loading graph …", flush=True)
    t = time.time()
    with open(RAW, "rb") as f:
        data = pickle.load(f)
    G = data[0] if isinstance(data, list) else data
    print(f"[✓] {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges "
          f"in {time.time() - t:.0f}s", flush=True)

    rng = random.Random(20240809)
    all_nodes = list(G.nodes())
    seeds = rng.sample(all_nodes, N_SAMPLES)

    blocked = set()  # most favourable case: nothing blocked == fewest early exits

    rows = []
    spent = 0.0
    for i, s in enumerate(seeds):
        t0 = time.time()
        sub = saf.bfs_subgraph(G, s, blocked, depth=2)
        t_bfs = time.time() - t0

        nn, ne = sub.number_of_nodes(), sub.number_of_edges()

        t0 = time.time()
        saf.extract_basic_features(sub, s)
        saf.extract_temporal_features(sub, s, 30, 180)
        t_g12 = time.time() - t0

        t0 = time.time()
        saf.extract_centrality_features(sub, s, 0.005)
        t_g3 = time.time() - t0

        tot = t_bfs + t_g12 + t_g3
        spent += tot
        rows.append((nn, ne, t_bfs, t_g12, t_g3, tot))
        print(f"  [{i+1:3d}/{N_SAMPLES}] ball n={nn:>8,} e={ne:>9,} | "
              f"bfs {t_bfs:7.2f}s  g1+2 {t_g12:6.2f}s  g3 {t_g3:8.2f}s  tot {tot:8.2f}s",
              flush=True)

        if spent > BUDGET_S:
            print(f"\n[!] benchmark budget {BUDGET_S:.0f}s exhausted after "
                  f"{i+1} seeds — extrapolating from those.", flush=True)
            break

    a = np.array(rows, dtype=float)
    nn, ne, t_bfs, t_g12, t_g3, tot = (a[:, k] for k in range(6))
    k = len(a)

    print("\n" + "=" * 78)
    print(f"  MEASURED on {k} uniform-random seeds")
    print("=" * 78)
    for name, v in [("ball nodes", nn), ("ball edges", ne), ("t_bfs (s)", t_bfs),
                    ("t_group1+2 (s)", t_g12), ("t_group3 (s)", t_g3),
                    ("t_total (s)", tot)]:
        print(f"  {name:<16} mean={v.mean():>12.2f}  median={np.median(v):>12.2f}  "
              f"p90={np.percentile(v,90):>12.2f}  max={v.max():>12.2f}")

    N = 2_973_489
    print("\n── extrapolation to all 2,973,489 seeds (single core) ──")
    for name, v in [("BFS+subgraph", t_bfs), ("group1+2", t_g12),
                    ("group3 centrality", t_g3), ("TOTAL", tot)]:
        sec = v.mean() * N
        print(f"  {name:<20} {sec/86400:>12,.1f} core-days   "
              f"({sec/86400/365:>7,.1f} core-years)   [20 cores: {sec/86400/20:>10,.1f} days]")

    print("\n  NOTE: mean is driven by hub-adjacent seeds; the tail is the cost.")
    print("=" * 78)


if __name__ == "__main__":
    main()
