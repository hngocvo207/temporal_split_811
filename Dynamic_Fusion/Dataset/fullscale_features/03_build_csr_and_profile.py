"""
03_build_csr_and_profile.py
===========================
(a) Builds the scipy.sparse CSR/CSC structures every later stage works off, and
(b) profiles the depth-2 ball-size distribution on a large random sample.

(b) matters because the 40-seed networkx benchmark measures a MEAN, and the ball
size |B(s)| on a scale-free transaction graph is heavy-tailed: one seed adjacent
to an exchange hub can pull in millions of nodes and cost more than ten thousand
ordinary seeds combined. Total cost is set by that tail, not by the mean, so we
measure the tail before committing to a full-scale run.

Structures written to arrays/:
  csr_sym.npz   symmetric, de-duplicated simple adjacency  (BFS / ball building)
  csr_out.npz   directed out-adjacency, parallel edges collapsed, summed weight
  csr_in.npz    directed in-adjacency  (same, transposed)

Usage: python3 03_build_csr_and_profile.py [n_profile_samples]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ARR = HERE / "arrays"
N_PROFILE = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


def save_csr(path, M):
    np.savez(path, indptr=M.indptr, indices=M.indices, data=M.data, shape=np.array(M.shape))


def load_csr(path):
    z = np.load(path)
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def main():
    t0 = time.time()
    e = np.load(ARR / "edges.npz")
    src, dst, val = e["src"], e["dst"], e["val"]
    n = len(np.load(ARR / "nodes.npy", allow_pickle=True))
    print(f"[✓] {n:,} nodes / {len(src):,} edges ({time.time()-t0:.0f}s)", flush=True)

    # ── directed simple digraph: parallel edges collapsed, weights summed ─────
    # (this is exactly what extract_centrality_features' `edge_weights` dict does)
    t = time.time()
    A_out = sp.coo_matrix((val, (src, dst)), shape=(n, n)).tocsr()
    A_out.sum_duplicates()
    print(f"[✓] csr_out: nnz={A_out.nnz:,} ({time.time()-t:.0f}s)", flush=True)
    save_csr(ARR / "csr_out.npz", A_out)

    t = time.time()
    A_in = A_out.T.tocsr()
    save_csr(ARR / "csr_in.npz", A_in)
    print(f"[✓] csr_in:  nnz={A_in.nnz:,} ({time.time()-t:.0f}s)", flush=True)

    # ── symmetric simple adjacency for BFS (successors + predecessors) ───────
    t = time.time()
    # float64, NOT int8: coo->csr sums duplicates, and this graph has up to 10,000
    # parallel edges on a single pair. With int8 any count that is a multiple of 256
    # wraps to exactly 0 (15 pairs do), becomes an explicit zero, and is then pruned
    # by the S + S.T / eliminate_zeros() below -- silently deleting real adjacencies
    # and shrinking the depth-2 ball of every seed that needed them.
    S = sp.coo_matrix((np.ones(len(src), np.float64), (src, dst)), shape=(n, n)).tocsr()
    S = S + S.T
    assert (S.data != 0).all(), "unexpected explicit zero in symmetric adjacency"
    S.data[:] = 1
    S.setdiag(0)
    S.eliminate_zeros()
    S.sort_indices()
    print(f"[✓] csr_sym: nnz={S.nnz:,} (undirected degree sum) ({time.time()-t:.0f}s)", flush=True)
    save_csr(ARR / "csr_sym.npz", S)

    deg = np.diff(S.indptr)
    print(f"\n── undirected simple degree ──")
    for q in [50, 90, 99, 99.9, 99.99, 100]:
        print(f"  p{q:<6} {np.percentile(deg, q):>12,.0f}")
    print(f"  mean   {deg.mean():>12,.2f}")
    top = np.argsort(deg)[-10:][::-1]
    print(f"  top-10 degrees: {deg[top].tolist()}")

    # ── profile depth-2 ball sizes on a random sample ────────────────────────
    indptr, indices = S.indptr, S.indices
    rng = np.random.default_rng(20240809)
    seeds = rng.choice(n, size=N_PROFILE, replace=False)

    print(f"\n── profiling |B(s)| on {N_PROFILE:,} random seeds ──", flush=True)
    sizes = np.empty(N_PROFILE, dtype=np.int64)
    times = np.empty(N_PROFILE, dtype=np.float64)
    t = time.time()
    for i, s in enumerate(seeds):
        ts_ = time.perf_counter()
        n1 = indices[indptr[s]:indptr[s + 1]]
        if len(n1) == 0:
            ball = np.array([s], dtype=np.int32)
        else:
            starts, ends = indptr[n1], indptr[n1 + 1]
            total = int((ends - starts).sum())
            buf = np.empty(total + len(n1) + 1, dtype=np.int32)
            p = 0
            for a, b in zip(starts, ends):
                k = b - a
                buf[p:p + k] = indices[a:b]
                p += k
            buf[p:p + len(n1)] = n1
            buf[p + len(n1)] = s
            ball = np.unique(buf)
        sizes[i] = len(ball)
        times[i] = time.perf_counter() - ts_
        if (i + 1) % 500 == 0:
            print(f"    {i+1:,}/{N_PROFILE:,}  elapsed {time.time()-t:.0f}s  "
                  f"mean|B|={sizes[:i+1].mean():,.0f}", flush=True)

    print(f"\n── depth-2 ball size |B(s)| over {N_PROFILE:,} random seeds ──")
    for q in [1, 25, 50, 75, 90, 99, 99.9, 100]:
        print(f"  p{q:<6} {np.percentile(sizes, q):>14,.0f}")
    print(f"  mean   {sizes.mean():>14,.1f}")
    print(f"  sum    {sizes.sum():>14,.0f}  over {N_PROFILE:,} seeds")

    N = 2_973_489
    est_total_ball_nodes = sizes.mean() * N
    print(f"\n  => projected sum over ALL {N:,} seeds: "
          f"{est_total_ball_nodes:,.0f} ball-node-visits ({est_total_ball_nodes/1e12:.2f} x 10^12)")
    print(f"  => ball-construction time alone (numpy, this machine): "
          f"{times.mean()*N/3600:,.1f} core-hours "
          f"({times.mean()*N/3600/20:,.1f} h on 20 cores)")
    print(f"\n  top-10 |B(s)| in sample: {np.sort(sizes)[-10:][::-1].tolist()}")
    print(f"  share of total ball-nodes owned by the top 1% of seeds: "
          f"{np.sort(sizes)[-max(1,N_PROFILE//100):].sum()/sizes.sum()*100:.1f}%")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
