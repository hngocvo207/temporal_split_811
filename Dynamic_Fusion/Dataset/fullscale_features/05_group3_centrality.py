"""
05_group3_centrality.py
=======================
Group 3 (8 graph-centrality features) + the n_nodes/n_edges meta columns, for
all 2,973,489 seeds.

Strategy
--------
The expensive part of select_add_features.py is NOT the centrality maths, it is
building a depth-2 ball with networkx: `G.subgraph(...).copy()` on a
MultiDiGraph deep-copies ~25,000 edge dicts per seed, and a networkx
MultiDiGraph of this graph costs ~15 GB resident, which makes 20-way process
parallelism impossible.

So: ball construction and induced-subgraph extraction run on scipy.sparse CSR
arrays (~200 MB total, shared read-only across forked workers), and only the
resulting small simple digraph is handed to networkx -- where the ORIGINAL
centrality calls run unmodified, so the numbers are the script's numbers.

Fidelity notes (each one verified against the original source):
  * DG collapses parallel edges and SUMS their weights -- same as the original's
    `edge_weights` defaultdict.
  * Edge weight uses the original's `... or 1.0` fallback, so a value of 0.0
    becomes 1.0. This hits 9,152,968 of 13,551,303 edges (67.5%) and is NOT the
    same convention groups 1+2 use (`or 0.0`). Precomputed into `w3`.
  * Self-loops are kept in DG (the original's G.subgraph keeps them).
  * The adaptive k_sample / max_iter / tol ladder is copied verbatim.
  * `safe()` swallows convergence failures to 0.0, verbatim.

Deliberate deviation (one, and it is a strict improvement):
  nx.betweenness_centrality(k=...) samples k source nodes from the global
  `random` state, so the original is not reproducible run-to-run. Here it gets
  `seed=<seed node id>` -- deterministic, and independent per node.

Usage:
  python3 05_group3_centrality.py --workers 20 --chunk 5000
  python3 05_group3_centrality.py --bench 200          # timing only, no output
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ARR = HERE / "arrays"
OUT = HERE / "group3_chunks"

FEATS = ["katz_centrality", "betweenness_centrality", "degree_centrality",
         "closeness_centrality", "clustering_coefficient", "eigenvector_centrality",
         "in_degree_centrality", "out_degree_centrality"]
NCOL = len(FEATS) + 2          # + n_nodes, n_edges

# ── globals populated once in the parent, inherited by fork ──────────────────
S_ptr = S_idx = None                    # symmetric adjacency (ball building)
O_ptr = O_idx = O_w3 = O_cnt = None     # directed simple digraph + weights/counts
N = 0
KATZ_ALPHA = 0.005

# --reduced: skip the three centralities that mg_refit_features.py does not consume.
# Measured (10_profile_centralities.py) at 46.8% of total compute:
#   eigenvector 29.2% | closeness 10.3% | katz 7.3%
# They are written as NaN and dropped by 08_assemble_csv.py. The retained set is a
# strict SUBSET of the full set, so chunks already computed under the full set stay
# valid and are simply reused.
SKIP_IN_REDUCED = ("katz_centrality", "closeness_centrality", "eigenvector_centrality")
REDUCED = False


def load_shared():
    global S_ptr, S_idx, O_ptr, O_idx, O_w3, O_cnt, N
    z = np.load(ARR / "csr_sym.npz")
    S_ptr, S_idx = z["indptr"], z["indices"]
    N = int(z["shape"][0])

    e = np.load(ARR / "edges.npz")
    src, dst, val = e["src"], e["dst"], e["val"]
    # the original's `float(value or 1.0)`: a 0.0 value becomes 1.0
    w3 = np.where(val == 0.0, 1.0, val)
    A = sp.coo_matrix((w3, (src, dst)), shape=(N, N)).tocsr()
    A.sum_duplicates()                                   # SUM parallel weights
    C = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(N, N)).tocsr()
    C.sum_duplicates()                                   # COUNT parallel edges
    assert A.nnz == C.nnz and np.array_equal(A.indices, C.indices)
    O_ptr, O_idx, O_w3, O_cnt = A.indptr, A.indices, A.data, C.data


def ball_depth2(s: int) -> np.ndarray:
    """{s} u N(s) u N(N(s)) over the symmetric adjacency -- the original's BFS depth=2."""
    n1 = S_idx[S_ptr[s]:S_ptr[s + 1]]
    if len(n1) == 0:
        return np.array([s], dtype=np.int32)
    starts, ends = S_ptr[n1], S_ptr[n1 + 1]
    lens = ends - starts
    total = int(lens.sum())
    buf = np.empty(total + len(n1) + 1, dtype=np.int32)
    p = 0
    for a, L in zip(starts, lens):
        buf[p:p + L] = S_idx[a:a + L]
        p += L
    buf[p:p + len(n1)] = n1
    buf[p + len(n1)] = s
    return np.unique(buf)


def induced(ball: np.ndarray):
    """Edges (u,v,w) of the simple digraph induced on `ball`, plus multigraph edge count."""
    starts, ends = O_ptr[ball], O_ptr[ball + 1]
    lens = ends - starts
    total = int(lens.sum())
    cols = np.empty(total, dtype=np.int32)
    w = np.empty(total, dtype=np.float64)
    cnt = np.empty(total, dtype=np.float64)
    rows = np.repeat(ball, lens)
    p = 0
    for a, L in zip(starts, lens):
        cols[p:p + L] = O_idx[a:a + L]
        w[p:p + L] = O_w3[a:a + L]
        cnt[p:p + L] = O_cnt[a:a + L]
        p += L
    keep = np.isin(cols, ball, assume_unique=False)
    return rows[keep], cols[keep], w[keep], int(cnt[keep].sum())


def safe(fn, default=0.0):
    try:
        return fn()
    except Exception:
        return default


def centrality_for(seed: int):
    """Verbatim port of extract_centrality_features(), fed a pre-aggregated DG."""
    ball = ball_depth2(seed)
    u, v, w, n_multi_edges = induced(ball)

    DG = nx.DiGraph()
    DG.add_edges_from(((int(a), int(b), {"weight": float(c)}) for a, b, c in zip(u, v, w)))
    n = DG.number_of_nodes()

    # ── adaptive ladder, copied verbatim from the original ──────────────────
    if n > 10000:
        k_sample, max_iter, tol = 20, 50, 1e-3
    elif n > 1000:
        k_sample, max_iter, tol = 50, 100, 1e-4
    else:
        k_sample = min(n, 100) if n > 200 else None
        max_iter, tol = 500, 1e-6

    node = int(seed)
    nan = float("nan")

    if REDUCED:
        katz = clo = eig = nan
    else:
        katz = round(safe(lambda: nx.katz_centrality(
            DG, alpha=KATZ_ALPHA, max_iter=max_iter, tol=tol).get(node, 0.0)), 8)
        clo = round(safe(lambda: nx.closeness_centrality(DG, u=node)), 8)
        eig = round(safe(lambda: nx.eigenvector_centrality(
            DG, max_iter=max_iter, tol=tol, weight="weight").get(node, 0.0)), 8)

    bet = safe(lambda: nx.betweenness_centrality(
        DG, k=k_sample, normalized=True, weight="weight", seed=node).get(node, 0.0))
    deg = safe(lambda: nx.degree_centrality(DG).get(node, 0.0))
    clu = safe(lambda: float(nx.clustering(nx.Graph(DG), nodes=node)) if node in DG else 0.0)
    inc = safe(lambda: nx.in_degree_centrality(DG).get(node, 0.0))
    outc = safe(lambda: nx.out_degree_centrality(DG).get(node, 0.0))

    return (katz, round(bet, 8), round(deg, 8), clo,
            round(clu, 8), eig, round(inc, 8), round(outc, 8),
            float(len(ball)), float(n_multi_edges))


def chunk_ids(j: int, n_chunks: int) -> np.ndarray:
    """
    Node ids handled by chunk j: a STRIDED slice j, j+K, j+2K, ...

    Not a contiguous range. Node ids in this graph are in crawl order, which is
    strongly degree-ordered: ids 0-32,000 average undirected degree 78.5 against
    a global mean of 3.56 (22x), while the last decile averages 1.02. Contiguous
    chunks therefore differ in cost by orders of magnitude -- the first chunks
    would hold every hub in the graph and run for hours while the final chunks
    finish instantly, wrecking both load balance and the ETA. Striding gives
    every chunk the same degree mix, so chunks cost the same and progress is
    linear. The id list is derived from (j, n_chunks), so nothing extra is
    stored and the mapping stays exact at assembly time.
    """
    return np.arange(j, N, n_chunks, dtype=np.int64)


def run_chunk(args):
    j, n_chunks = args
    path = OUT / f"schunk_{j:06d}_of_{n_chunks:06d}.npy"
    if path.exists():
        return j, 0, 0.0, True
    ids = chunk_ids(j, n_chunks)
    t0 = time.time()
    res = np.empty((len(ids), NCOL), dtype=np.float64)
    for i, s in enumerate(ids):
        res[i] = centrality_for(int(s))
    # Write-then-rename so a chunk file is never half-written. The temp name is
    # PREFIXED (".tmp_schunk_...") rather than suffixed (".../schunk_....tmp.npy"),
    # because the latter still matches the "schunk_*.npy" glob that 08 uses to
    # collect chunks: a hard power-off mid-write would leave a stale temp file
    # that 08 would pick up and crash on (int("001983.tmp")). With the prefix the
    # glob cannot see it, so an interrupted run leaves the output set consistent.
    tmp = path.parent / f".tmp_{path.name}"
    np.save(tmp, res)
    os.replace(tmp, path)
    return j, len(ids), time.time() - t0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--bench", type=int, default=0,
                    help="benchmark this many random seeds and exit")
    ap.add_argument("--limit", type=int, default=0, help="only process the first K node ids")
    ap.add_argument("--reduced", action="store_true",
                    help="skip katz/closeness/eigenvector (46.8%% of compute, none consumed "
                         "by mg_refit_features.py); they are written as NaN and dropped at assembly")
    args = ap.parse_args()

    global REDUCED
    REDUCED = args.reduced
    if REDUCED:
        print(f"[i] REDUCED mode: skipping {', '.join(SKIP_IN_REDUCED)}", flush=True)

    print(f"[…] loading shared CSR structures …", flush=True)
    t = time.time()
    load_shared()
    print(f"[✓] N={N:,}  sym_nnz={len(S_idx):,}  out_nnz={len(O_idx):,}  "
          f"({time.time()-t:.0f}s)", flush=True)

    if args.bench:
        rng = np.random.default_rng(20240809)
        seeds = rng.choice(N, size=args.bench, replace=False)
        ts = np.empty(args.bench)
        balls = np.empty(args.bench)
        t0 = time.time()
        for i, s in enumerate(seeds):
            a = time.perf_counter()
            r = centrality_for(int(s))
            ts[i] = time.perf_counter() - a
            balls[i] = r[8]
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{args.bench}  mean {ts[:i+1].mean():.3f}s/seed", flush=True)
        print(f"\n── benchmark on {args.bench} random seeds ──")
        print(f"  per-seed s     mean={ts.mean():.3f} median={np.median(ts):.3f} "
              f"p90={np.percentile(ts,90):.3f} p99={np.percentile(ts,99):.3f} max={ts.max():.3f}")
        print(f"  ball nodes     mean={balls.mean():,.0f} max={balls.max():,.0f}")
        tot = ts.mean() * 2_973_489
        print(f"\n  => full run: {tot/3600:,.1f} core-hours = {tot/86400:,.1f} core-days")
        print(f"  => on {args.workers} workers: {tot/3600/args.workers:,.1f} h wall clock")
        print(f"  (measured wall {time.time()-t0:.0f}s)")
        return

    OUT.mkdir(exist_ok=True)
    n_chunks = (N + args.chunk - 1) // args.chunk
    all_j = list(range(n_chunks))
    todo = [(j, n_chunks) for j in all_j
            if not (OUT / f"schunk_{j:06d}_of_{n_chunks:06d}.npy").exists()]
    print(f"[i] {n_chunks:,} strided chunks of ~{args.chunk:,} nodes;  "
          f"{n_chunks-len(todo):,} already done, {len(todo):,} to go;  "
          f"{args.workers} workers", flush=True)
    if not todo:
        print("[✓] nothing to do.")
        return

    import multiprocessing as mp
    t0 = time.time()
    done_nodes = 0
    total_nodes = sum(len(chunk_ids(j, n_chunks)) for j, _ in todo)
    ctx = mp.get_context("fork")
    orig = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with ctx.Pool(args.workers) as pool:
        signal.signal(signal.SIGINT, orig)
        for k, (j, cnt, dt, skipped) in enumerate(
                pool.imap_unordered(run_chunk, todo, chunksize=1)):
            done_nodes += cnt
            el = time.time() - t0
            rate = done_nodes / el if el > 0 else 0
            eta = (total_nodes - done_nodes) / rate if rate > 0 else float("nan")
            print(f"  [{k+1:,}/{len(todo):,}] chunk {j:,} ({cnt:,} nodes) in {dt:6.1f}s | "
                  f"{done_nodes:,}/{total_nodes:,} nodes | {rate:,.0f} nodes/s | "
                  f"elapsed {el/3600:5.2f}h | ETA {eta/3600:5.2f}h", flush=True)

    print(f"\n[✓] all chunks done in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
