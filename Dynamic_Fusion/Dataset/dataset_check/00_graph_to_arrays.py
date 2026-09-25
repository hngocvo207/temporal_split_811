"""
00_graph_to_arrays.py
=====================
Stage 0 of the full-scale (2,973,489-node) run of select_add_features.py.

Loads raw_data/MulDiGraph/MulDiGraph.pkl (networkx MultiDiGraph, 1.25 GB on
disk, ~30 GB resident) exactly ONCE and dumps it to a compact columnar form:

    nodes.npy      (N,)  object array  -- address strings, index i == node id i
    edges.npz            src/dst int32 (M,), ts int64 (M,), val float64 (M,)

Everything downstream (BFS-depth-2 neighbourhoods, degree/amount/temporal
aggregates, centralities) works off these arrays via scipy.sparse instead of
touching networkx again -- a networkx MultiDiGraph re-load costs several
minutes and ~30 GB, which is not something a batched job can afford to repeat.

Run once:  python3 00_graph_to_arrays.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "raw_data/MulDiGraph/MulDiGraph.pkl"
OUT = Path(__file__).resolve().parent / "arrays"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[…] pickle.load('{RAW}') — 1.25 GB, expect several minutes …", flush=True)
    with open(RAW, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        G = data[0]
    elif isinstance(data, dict):
        G = next(iter(data.values()))
    else:
        G = data
    print(f"[✓] loaded in {time.time() - t0:.0f}s: {G.number_of_nodes():,} nodes | "
          f"{G.number_of_edges():,} edges | {type(G).__name__}", flush=True)

    # ── node index ────────────────────────────────────────────────────────────
    t = time.time()
    nodes = np.array(list(G.nodes()), dtype=object)
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    print(f"[✓] node index built ({n:,}) in {time.time() - t:.0f}s", flush=True)
    np.save(OUT / "nodes.npy", nodes, allow_pickle=True)

    # ── edges ─────────────────────────────────────────────────────────────────
    # Preallocate: G.number_of_edges() is exact for a MultiDiGraph.
    m = G.number_of_edges()
    src = np.empty(m, dtype=np.int32)
    dst = np.empty(m, dtype=np.int32)
    ts = np.empty(m, dtype=np.int64)
    val = np.empty(m, dtype=np.float64)

    t = time.time()
    i = 0
    for u, v, d in G.edges(data=True):
        src[i] = idx[u]
        dst[i] = idx[v]
        # same key fallback chain as select_add_features.py's get_node_transactions
        a = d.get("value", d.get("amount", d.get("weight", 0.0)))
        val[i] = float(a) if a is not None else 0.0
        s = d.get("timestamp", d.get("time", d.get("ts")))
        ts[i] = np.int64(s) if s is not None and s != "" else np.int64(-1)
        i += 1
        if i % 2_000_000 == 0:
            print(f"    {i:,}/{m:,} edges  ({time.time() - t:.0f}s)", flush=True)
    assert i == m, (i, m)
    print(f"[✓] edge arrays built in {time.time() - t:.0f}s", flush=True)

    del G, data, idx

    np.savez(OUT / "edges.npz", src=src, dst=dst, ts=ts, val=val)
    print(f"[✓] wrote {OUT}/edges.npz", flush=True)

    # ── quick provenance check against the numbers in the audit report ────────
    print("\n── sanity ───────────────────────────────────────────────")
    print(f"  N nodes          : {n:,}")
    print(f"  M edges          : {m:,}")
    print(f"  ts missing       : {(ts < 0).sum():,}")
    print(f"  ts min/max       : {ts[ts >= 0].min()} / {ts.max()}")
    print(f"  self loops       : {(src == dst).sum():,}")
    print(f"  value min/max    : {val.min():.6g} / {val.max():.6g}")
    print(f"\n[done] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
