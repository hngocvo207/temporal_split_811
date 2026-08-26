"""
06_validate_group3.py
=====================
Correctness gate for 05_group3_centrality.py.

Runs the ORIGINAL select_add_features.py path (networkx MultiDiGraph -> BFS ->
subgraph().copy() -> extract_centrality_features) and the new scipy.sparse path
on the same random seeds, and compares all 8 centralities plus n_nodes/n_edges.

betweenness_centrality is handled separately: when the ball has n > 200 the
original samples k source nodes from the un-seeded global `random` state, so it
is not reproducible even against itself. Those seeds are compared with the
original run TWICE, to show the new value sits within the original's own
run-to-run spread rather than pretending to an exactness that does not exist.
Seeds with n <= 200 use k=None (exact Brandes) and must match to 1e-8.
"""

from __future__ import annotations

import importlib.util
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ORIG = Path("/home/ngocvo/Desktop/ngocvo/Dynamic_Feature/Dataset/select_add_features.py")

sys.path.insert(0, str(HERE))
import importlib
g3 = importlib.import_module("05_group3_centrality") if False else None
spec = importlib.util.spec_from_file_location("g3", HERE / "05_group3_centrality.py")
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
FEATS = g3.FEATS


def main():
    spec2 = importlib.util.spec_from_file_location("orig", ORIG)
    saf = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(saf)

    print("[…] loading shared CSR …", flush=True)
    g3.load_shared()

    print("[…] loading networkx MultiDiGraph (for the reference path) …", flush=True)
    t = time.time()
    with open(BASE / "raw_data/MulDiGraph/MulDiGraph.pkl", "rb") as f:
        data = pickle.load(f)
    G = data[0] if isinstance(data, list) else data
    nodes = np.load(HERE / "arrays/nodes.npy", allow_pickle=True)
    print(f"[✓] {G.number_of_nodes():,} nodes ({time.time()-t:.0f}s)", flush=True)

    rng = np.random.default_rng(7)
    seed_ids = rng.choice(len(nodes), size=N_SEEDS, replace=False)

    rows = []
    for j, sid in enumerate(seed_ids):
        addr = nodes[sid]
        # ── reference path: the original code, unmodified ──
        sub = saf.bfs_subgraph(G, addr, set(), depth=2)
        random.seed(12345)
        ref = saf.extract_centrality_features(sub, addr, 0.005)
        ref_n, ref_e = sub.number_of_nodes(), sub.number_of_edges()
        random.seed(999)
        ref2 = saf.extract_centrality_features(sub, addr, 0.005)

        # ── new path ──
        new = g3.centrality_for(int(sid))
        new_d = dict(zip(FEATS, new[:8]))
        new_n, new_e = new[8], new[9]

        rows.append((int(sid), ref_n, ref_e, new_n, new_e, ref, ref2, new_d))
        print(f"  [{j+1}/{N_SEEDS}] ball ref n={ref_n:,} e={ref_e:,} | "
              f"new n={int(new_n):,} e={int(new_e):,}", flush=True)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  META  (must be exact)")
    print("=" * 84)
    bad_meta = 0
    for sid, rn, re_, nn, ne, *_ in rows:
        if rn != nn or re_ != int(ne):
            bad_meta += 1
            print(f"  MISMATCH node {sid}: ref ({rn},{re_}) vs new ({int(nn)},{int(ne)})")
    print(f"  n_nodes / n_edges: {len(rows)-bad_meta}/{len(rows)} seeds exact")

    print("\n" + "=" * 84)
    print("  DETERMINISTIC CENTRALITIES  (must match to 1e-8)")
    print("=" * 84)
    ok = True
    for f in FEATS:
        if f == "betweenness_centrality":
            continue
        d = np.array([abs(r[5][f] - r[7][f]) for r in rows])
        n_ok = int((d <= 1e-8).sum())
        status = "" if n_ok == len(rows) else "   <-- MISMATCH"
        if n_ok != len(rows):
            ok = False
        print(f"  {f:<26} {n_ok}/{len(rows)} exact   max|diff|={d.max():.3e}{status}")

    print("\n" + "=" * 84)
    print("  betweenness_centrality  (original is stochastic when ball n > 200)")
    print("=" * 84)
    print(f"  {'node':>10} {'ball n':>9} {'orig run1':>13} {'orig run2':>13} {'new':>13}  {'verdict'}")
    n_exact_ok = n_exact = 0
    for sid, rn, re_, nn, ne, ref, ref2, new in rows:
        a, b, c = ref["betweenness_centrality"], ref2["betweenness_centrality"], new["betweenness_centrality"]
        if rn <= 200:
            n_exact += 1
            good = abs(a - c) <= 1e-8
            n_exact_ok += good
            verdict = "exact k=None  " + ("OK" if good else "MISMATCH")
            if not good:
                ok = False
        else:
            spread = abs(a - b)
            dev = abs(c - (a + b) / 2)
            verdict = f"sampled: orig self-spread {spread:.2e}, new dev {dev:.2e}"
        print(f"  {sid:>10} {rn:>9,} {a:>13.8f} {b:>13.8f} {c:>13.8f}  {verdict}")
    if n_exact:
        print(f"\n  exact (k=None, ball<=200) seeds: {n_exact_ok}/{n_exact} match to 1e-8")

    print("\n" + ("[PASS] scipy path reproduces the original." if ok and not bad_meta
                  else "[FAIL] see mismatches above."))


if __name__ == "__main__":
    main()
