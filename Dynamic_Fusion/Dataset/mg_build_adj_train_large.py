"""
mg_build_adj_train_large.py
============================
Phase 1 (PHASE1_graph_definitions.md muc 5): build them
`adj_train_large.npz` cho Dataset_MG_v3_no_overlap -- cat tai t2 (bien
gioi val/test) thay vi t1 (bien gioi train/val) ma `adj_train.npz` dang
dung. Dung cho Train_large = Train u Val.

Sao chep dung logic build adjacency cua
`mg_temporal_pipeline_v3_no_overlap.py` (cung cong thuc trong so Eq.1-3
qua `mg_graph_weight_formula.build_edge_weights`, cung cach gop coo
matrix), CHI doi T_cutoff tu t1 sang t2 -- khong dung lai toan bo split
(labels/partition/t_first giu nguyen, chi them 1 ma tran adjacency moi).

Output: data/preprocessed/Dataset_MG_v3_no_overlap/adj_train_large.npz
(khong ghi de adj_train.npz/adj_inference.npz da co).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG_v3_no_overlap"
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mg_graph_weight_formula import build_edge_weights  # noqa: E402


def sep(title: str = "") -> None:
    width = 78
    print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}" if title else "─" * width)


def main():
    t0 = time.time()
    print("=" * 78)
    print("  mg_build_adj_train_large.py — adj_train_large.npz (cat tai t2, Train_large = Train u Val)")
    print("=" * 78)

    sep("Load split_config.json / address_to_index.pkl (khong doi split, chi doc)")
    with open(SPLIT_DIR / "split_config.json") as f:
        config = json.load(f)
    t1 = config["t1_train_val_cutoff"]
    t2 = config["t2_val_test_cutoff"]
    print(f"  t1 (train/val)  = {t1}  ({config['t1_human']})")
    print(f"  t2 (val/test)   = {t2}  ({config['t2_human']})  <- T_cutoff moi cho adj_train_large")

    with open(SPLIT_DIR / "address_to_index.pkl", "rb") as f:
        address_to_index = pickle.load(f)
    n = len(address_to_index)
    print(f"  n accounts (index space) = {n:,}")

    sep("Load MulDiGraph.pkl (do thi giao dich tho)")
    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append((u, v, float(d.get("amount", 0.0)), float(d.get("timestamp"))))
    edges_df = pd.DataFrame(rows, columns=["from_addr", "to_addr", "amount", "timestamp"])
    print(f"  edges_df: {len(edges_df):,} ({time.time()-t0:.1f}s)")

    sep("Loc canh <= t2 va tinh trong so Eq.1-3 (build_edge_weights)")
    train_large_edges = edges_df[edges_df["timestamp"] <= t2]
    pct = 100 * len(train_large_edges) / len(edges_df)
    print(f"  train_large-period edges (<=t2): {len(train_large_edges):,} / {len(edges_df):,} ({pct:.2f}%)")
    train_large_agg = build_edge_weights(train_large_edges)
    print(f"  aggregated (from,to) pairs: {len(train_large_agg):,} ({time.time()-t0:.1f}s)")

    sep("Chuyen sang sparse coo matrix, luu adj_train_large.npz")

    def to_sparse(agg, addr2idx, n):
        rows_ = agg["from_addr"].map(addr2idx).to_numpy()
        cols_ = agg["to_addr"].map(addr2idx).to_numpy()
        data_ = agg["weight"].to_numpy(dtype=np.float32)
        return coo_matrix((data_, (rows_, cols_)), shape=(n, n))

    train_large_adj = to_sparse(train_large_agg, address_to_index, n)
    out_path = SPLIT_DIR / "adj_train_large.npz"
    save_npz(out_path, train_large_adj.tocsr())
    print(f"  adj_train_large.npz: shape={train_large_adj.shape} nnz={train_large_adj.nnz:,} ({time.time()-t0:.1f}s)")
    print(f"  Saved: {out_path}")

    sep("Validation checks")
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    part_arr = np.empty(n, dtype=object)
    for addr, idx in address_to_index.items():
        part_arr[idx] = partition.get(addr, "isolated")

    csr = train_large_adj.tocsr()
    out_deg = np.diff(csr.indptr)
    in_deg = np.asarray((csr > 0).sum(axis=0)).flatten()
    deg = out_deg + in_deg

    all_pass = True

    def check(name, ok, detail=""):
        nonlocal all_pass
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    test_idx = np.where(part_arr == "test")[0]
    n_test_nonzero = int((deg[test_idx] > 0).sum())
    check(
        "test co bac = 0 TUYET DOI trong adj_train_large (test van khong duoc train nhin thay)",
        n_test_nonzero == 0, f"{n_test_nonzero}/{len(test_idx)} account co bac > 0",
    )

    val_idx = np.where(part_arr == "val")[0]
    n_val_nonzero = int((deg[val_idx] > 0).sum())
    check(
        "val CO bac > 0 trong adj_train_large (khac adj_train cu, dung cho Train_large)",
        n_val_nonzero > 0, f"{n_val_nonzero:,}/{len(val_idx):,} account co bac > 0",
    )

    old_adj = SPLIT_DIR / "adj_train.npz"
    if old_adj.exists():
        from scipy.sparse import load_npz
        old_csr = load_npz(old_adj)
        check(
            "adj_train_large.nnz >= adj_train.nnz (t2 > t1 nen sieu tap canh)",
            train_large_adj.nnz >= old_csr.nnz,
            f"adj_train_large.nnz={train_large_adj.nnz:,}  adj_train.nnz={old_csr.nnz:,}",
        )

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
