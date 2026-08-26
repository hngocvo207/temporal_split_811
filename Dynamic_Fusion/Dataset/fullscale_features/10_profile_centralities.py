"""
10_profile_centralities.py
==========================
Breaks the group-3 per-seed cost down by individual centrality.

Motivation: the full run measures ~0.7 s/seed with 16 workers against 0.223 s/seed
single-process, i.e. ~31% parallel efficiency -- the workload is memory-bandwidth
bound, not CPU bound. Before accepting a ~44 h ETA it is worth knowing which of
the 8 calls dominates, since some (katz, eigenvector) are pure power iterations
that scipy could do on the sparse matrix directly, while others (betweenness)
are inherently Brandes-in-Python.

Runs single-process on random seeds so the numbers are comparable to the
benchmark in 05 --bench.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g3", HERE / "05_group3_centrality.py")
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def main():
    g3.load_shared()
    rng = np.random.default_rng(20240809)
    seeds = rng.choice(g3.N, size=N_SEEDS, replace=False)

    acc = defaultdict(float)
    balls = []
    t_all = time.time()
    for s in seeds:
        s = int(s)
        t = time.perf_counter()
        ball = g3.ball_depth2(s)
        acc["ball_bfs"] += time.perf_counter() - t

        t = time.perf_counter()
        u, v, w, ne = g3.induced(ball)
        acc["induced_submatrix"] += time.perf_counter() - t

        t = time.perf_counter()
        DG = nx.DiGraph()
        DG.add_edges_from(((int(a), int(b), {"weight": float(c)}) for a, b, c in zip(u, v, w)))
        acc["build_nx_DiGraph"] += time.perf_counter() - t
        n = DG.number_of_nodes()
        balls.append(n)

        if n > 10000:
            k_sample, max_iter, tol = 20, 50, 1e-3
        elif n > 1000:
            k_sample, max_iter, tol = 50, 100, 1e-4
        else:
            k_sample = min(n, 100) if n > 200 else None
            max_iter, tol = 500, 1e-6

        for name, fn in [
            ("katz_centrality", lambda: nx.katz_centrality(DG, alpha=0.005, max_iter=max_iter, tol=tol).get(s, 0.0)),
            ("betweenness_centrality", lambda: nx.betweenness_centrality(DG, k=k_sample, normalized=True, weight="weight", seed=s).get(s, 0.0)),
            ("degree_centrality", lambda: nx.degree_centrality(DG).get(s, 0.0)),
            ("closeness_centrality", lambda: nx.closeness_centrality(DG, u=s)),
            ("clustering_coefficient", lambda: float(nx.clustering(nx.Graph(DG), nodes=s)) if s in DG else 0.0),
            ("eigenvector_centrality", lambda: nx.eigenvector_centrality(DG, max_iter=max_iter, tol=tol, weight="weight").get(s, 0.0)),
            ("in_degree_centrality", lambda: nx.in_degree_centrality(DG).get(s, 0.0)),
            ("out_degree_centrality", lambda: nx.out_degree_centrality(DG).get(s, 0.0)),
        ]:
            t = time.perf_counter()
            g3.safe(fn)
            acc[name] += time.perf_counter() - t

    wall = time.time() - t_all
    total = sum(acc.values())
    print(f"\n── per-seed cost breakdown, {N_SEEDS} random seeds "
          f"(mean ball {np.mean(balls):,.0f} nodes) ──\n")
    print(f"  {'stage':<26} {'ms/seed':>10} {'% of total':>12}")
    print("  " + "─" * 50)
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"  {k:<26} {1000*v/N_SEEDS:>10.1f} {100*v/total:>11.1f}%")
    print("  " + "─" * 50)
    print(f"  {'TOTAL':<26} {1000*total/N_SEEDS:>10.1f} {100.0:>11.1f}%")
    print(f"\n  wall {wall:.0f}s   ({wall/N_SEEDS*1000:.0f} ms/seed incl. overhead)")

    N = 2_973_489
    print(f"\n  full-graph projection at this single-process rate: "
          f"{total/N_SEEDS*N/3600:,.1f} core-hours")


if __name__ == "__main__":
    main()
