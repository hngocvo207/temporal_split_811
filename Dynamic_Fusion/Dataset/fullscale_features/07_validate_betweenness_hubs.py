"""
07_validate_betweenness_hubs.py
===============================
06_validate_group3.py proved the 7 deterministic centralities exact, but every
sampled seed returned betweenness = 0 -- because the median node in this graph
has undirected degree 1, and a leaf lies on no shortest path. So that run did
NOT actually exercise the betweenness code path, and betweenness is one of the
10 features mg_refit_features.py feeds the model.

This re-runs the comparison on seeds chosen to have NON-ZERO betweenness:
high-degree nodes, plus the ground-truth phishing seeds (the rows that matter
most downstream). Also reports how much of the spread is the original's own
un-seeded k-sampling noise, which bounds how exactly any port could agree.
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

spec = importlib.util.spec_from_file_location("g3", HERE / "05_group3_centrality.py")
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)

N_HUB = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N_PHISH = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def main():
    spec2 = importlib.util.spec_from_file_location("orig", ORIG)
    saf = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(saf)

    g3.load_shared()
    nodes = np.load(HERE / "arrays/nodes.npy", allow_pickle=True)
    pos = {a: i for i, a in enumerate(nodes)}
    deg = np.diff(g3.S_ptr)

    # mid-high degree: big enough to be on shortest paths, small enough that the
    # reference networkx run finishes in reasonable time
    cand = np.where((deg >= 20) & (deg <= 200))[0]
    rng = np.random.default_rng(3)
    hubs = rng.choice(cand, size=N_HUB, replace=False)

    # Ground truth from labels.pkl (isp), NOT phisher_accounts.txt -- that file is
    # not sufficiently verified for this dataset and is no longer used anywhere.
    with open(BASE / "data/preprocessed/Dataset_MG/labels.pkl", "rb") as f:
        ph = [a for a, v in pickle.load(f).items() if v == 1]
    ph_ids = np.array([pos[a] for a in ph if a in pos])
    ph_ids = ph_ids[np.argsort(-deg[ph_ids])][:N_PHISH]   # highest-degree phishers

    targets = [("hub", int(s)) for s in hubs] + [("phisher", int(s)) for s in ph_ids]

    print("[…] loading networkx MultiDiGraph …", flush=True)
    t = time.time()
    with open(BASE / "raw_data/MulDiGraph/MulDiGraph.pkl", "rb") as f:
        data = pickle.load(f)
    G = data[0] if isinstance(data, list) else data
    print(f"[✓] {time.time()-t:.0f}s\n", flush=True)

    print(f"{'kind':>8} {'node':>9} {'deg':>6} {'ball n':>9} "
          f"{'orig#1':>12} {'orig#2':>12} {'orig#3':>12} {'new':>12}  {'note'}")
    print("─" * 108)

    nonzero = 0
    det_bad = 0
    for kind, sid in targets:
        addr = nodes[sid]
        sub = saf.bfs_subgraph(G, addr, set(), depth=2)
        runs = []
        for rs in (1, 2, 3):
            random.seed(rs)
            runs.append(saf.extract_centrality_features(sub, addr, 0.005))
        new = dict(zip(g3.FEATS, g3.centrality_for(sid)[:8]))

        b = [r["betweenness_centrality"] for r in runs]
        nb = new["betweenness_centrality"]
        if max(b + [nb]) > 0:
            nonzero += 1
        spread = max(b) - min(b)
        note = (f"orig self-spread {spread:.3e}" if spread > 0
                else ("all zero" if nb == 0 else "orig stable"))
        if sub.number_of_nodes() <= 200:
            note += " | k=None EXACT" + (" OK" if abs(b[0] - nb) < 1e-8 else " MISMATCH")

        # deterministic features must still be exact on these seeds
        for f in g3.FEATS:
            if f == "betweenness_centrality":
                continue
            if abs(runs[0][f] - new[f]) > 1e-8:
                det_bad += 1
                note += f" | {f} MISMATCH"

        print(f"{kind:>8} {sid:>9} {int(deg[sid]):>6} {sub.number_of_nodes():>9,} "
              f"{b[0]:>12.8f} {b[1]:>12.8f} {b[2]:>12.8f} {nb:>12.8f}  {note}")

    print("\n" + "=" * 108)
    print(f"  seeds with non-zero betweenness somewhere: {nonzero}/{len(targets)}"
          f"   (0/N would mean the code path is still untested)")
    print(f"  deterministic-feature mismatches on these seeds: {det_bad}")
    print("=" * 108)


if __name__ == "__main__":
    main()
