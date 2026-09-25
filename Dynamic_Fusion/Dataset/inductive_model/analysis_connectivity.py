"""
analysis_connectivity.py
=========================
Thuc hien Priority 1 cua MulDiGraph_Temporal_Split_Task_Checklist.md (muc
6.2, 7, 8, 9, 10, 11): trich xuat EXACT directed edge counts tu adjacency
matrix that (khong lam tron), va tinh node-level connectivity cho tung
test node (thay vi chi bao cao aggregate/global).

Dung 3 ma tran da co san (Dataset_MG_v3_no_overlap/):
  - adj_train.npz        -- Train_small graph (cat tai t1)
  - adj_train_large.npz  -- Train_large graph (cat tai t2, Train u Val)
  - adj_inference.npz    -- full graph (dung cho Test connectivity thuc te)

Quy uoc: "edge count" = nnz (so cap (from,to) phan biet), KHONG phai tong
trong so -- khop dung cach dem da dung cho global edge matrix truoc do
(Test->Test = 136,115 la nnz, khong phai sum(weight)).

Output:
  output/connectivity_summary.json   -- moi so exact + thong ke node-level
  output/test_node_connectivity.csv  -- DataFrame 487,855 dong dung dinh
                                         dang muc 8.3 cua checklist
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Dynamic_Fusion/
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG_v3_no_overlap"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def sep(title=""):
    print(f"\n{'─'*4} {title} {'─'*max(0,74-len(title))}" if title else "─"*78)


def load_partition_code(addr_to_idx):
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    n = len(addr_to_idx)
    code = np.full(n, -1, dtype=np.int8)  # -1 = isolated / unknown
    for addr, idx in addr_to_idx.items():
        p = partition.get(addr, "isolated")
        code[idx] = SPLIT_CODE.get(p, -1)
    return code


def directed_crosstab(csr: sp.csr_matrix, part_code: np.ndarray) -> np.ndarray:
    """3x3 exact nnz crosstab, row=from-partition, col=to-partition."""
    coo = csr.tocoo()
    valid = (part_code[coo.row] >= 0) & (part_code[coo.col] >= 0)
    r = part_code[coo.row[valid]]
    c = part_code[coo.col[valid]]
    mat = np.zeros((3, 3), dtype=np.int64)
    key = r * 3 + c
    counts = np.bincount(key, minlength=9)
    mat[:, :] = counts.reshape(3, 3)
    return mat


def per_node_out(csr: sp.csr_matrix, node_idx: np.ndarray, part_code: np.ndarray) -> np.ndarray:
    """Fully vectorized: cho tap node_idx, tra ve mang [len(node_idx), 3] =
    so canh RA (out-degree) toi moi partition (train/val/test). Slice CSR
    theo hang (fancy indexing, O(nnz cua submatrix)), roi bincount 1 lan
    tren toan bo thay vi loop Python tung node (487,855 node se rat cham
    neu loop)."""
    n_nodes = len(node_idx)
    sub = csr[node_idx]  # CSR moi, n_nodes hang, giu nguyen thu tu node_idx
    row_id = np.repeat(np.arange(n_nodes), np.diff(sub.indptr))
    parts = part_code[sub.indices]
    valid = parts >= 0
    key = row_id[valid] * 3 + parts[valid]
    counts = np.bincount(key, minlength=n_nodes * 3)
    return counts.reshape(n_nodes, 3)


def per_node_in(csc: sp.csc_matrix, node_idx: np.ndarray, part_code: np.ndarray) -> np.ndarray:
    """Giong per_node_out nhung tren CSC (cot = node dich) -> so canh VAO
    (in-degree) tu moi partition, fully vectorized."""
    n_nodes = len(node_idx)
    sub = csc[:, node_idx]  # CSC moi, n_nodes cot, giu nguyen thu tu node_idx
    col_id = np.repeat(np.arange(n_nodes), np.diff(sub.indptr))
    parts = part_code[sub.indices]
    valid = parts >= 0
    key = col_id[valid] * 3 + parts[valid]
    counts = np.bincount(key, minlength=n_nodes * 3)
    return counts.reshape(n_nodes, 3)


def stats_row(arr: np.ndarray) -> dict:
    arr = arr.astype(np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "pct_gt0": float((arr > 0).mean() * 100),
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("  analysis_connectivity.py -- Priority 1 (checklist muc 6-11)")
    print("=" * 78)

    sep("Load address_to_index / partition / 3 adjacency matrices")
    with open(SPLIT_DIR / "address_to_index.pkl", "rb") as f:
        addr_to_idx = pickle.load(f)
    part_code = load_partition_code(addr_to_idx)
    n = len(addr_to_idx)
    print(f"  n accounts = {n:,}")
    for lbl, c in [("train", 0), ("val", 1), ("test", 2)]:
        print(f"    {lbl}: {(part_code==c).sum():,}")

    adj_train = sp.load_npz(SPLIT_DIR / "adj_train.npz").tocsr()
    adj_train_large = sp.load_npz(SPLIT_DIR / "adj_train_large.npz").tocsr()
    adj_inf = sp.load_npz(SPLIT_DIR / "adj_inference.npz").tocsr()
    print(f"  adj_train nnz={adj_train.nnz:,}  adj_train_large nnz={adj_train_large.nnz:,}  adj_inference nnz={adj_inf.nnz:,}")

    # ---------------------------------------------------------------
    sep("Muc 6.2 -- EXACT directed crosstab (full inference graph)")
    mat_inf = directed_crosstab(adj_inf, part_code)
    labels = SPLITS
    print("        " + "  ".join(f"{l:>12s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>6s}  " + "  ".join(f"{mat_inf[i,j]:>12,}" for j in range(3)))
    assert mat_inf.sum() == adj_inf.nnz, "crosstab khong khop tong nnz"

    test_to_train_exact = int(mat_inf[2, 0])
    train_to_test_exact = int(mat_inf[0, 2])
    test_to_val_exact = int(mat_inf[2, 1])
    val_to_test_exact = int(mat_inf[1, 2])
    test_to_test_exact = int(mat_inf[2, 2])
    print(f"\n  EXACT Test->Train = {test_to_train_exact:,}   Train->Test = {train_to_test_exact:,}")
    print(f"  EXACT Test->Val   = {test_to_val_exact:,}   Val->Test   = {val_to_test_exact:,}")
    print(f"  EXACT Test->Test  = {test_to_test_exact:,}")
    print(f"  Ratio Train->Test / Test->Train = {train_to_test_exact/test_to_train_exact:.4f}")
    print(f"  Ratio Test->Val / Val->Test     = {test_to_val_exact/val_to_test_exact:.4f}")

    # ---------------------------------------------------------------
    sep("Muc 7-9 -- node-level connectivity cho TUNG test node (full inference graph)")
    test_idx = np.where(part_code == 2)[0]
    n_test = len(test_idx)
    print(f"  n_test nodes = {n_test:,}")

    adj_inf_csc = adj_inf.tocsc()
    t_a = time.time()
    out_counts = per_node_out(adj_inf, test_idx, part_code)  # [n_test,3] -> to train/val/test
    print(f"  per-node OUT done ({time.time()-t_a:.1f}s)")
    t_a = time.time()
    in_counts = per_node_in(adj_inf_csc, test_idx, part_code)  # [n_test,3] -> from train/val/test
    print(f"  per-node IN done ({time.time()-t_a:.1f}s)")

    # sanity: sum over nodes must equal exact global crosstab row/col for test
    assert out_counts[:, 0].sum() == test_to_train_exact
    assert out_counts[:, 1].sum() == test_to_val_exact
    assert out_counts[:, 2].sum() == test_to_test_exact
    assert in_counts[:, 0].sum() == train_to_test_exact
    assert in_counts[:, 1].sum() == val_to_test_exact
    assert in_counts[:, 2].sum() == test_to_test_exact
    print("  [PASS] tong node-level == tong global crosstab (khong mat/thua canh nao)")

    df = pd.DataFrame({
        "test_node_idx": test_idx,
        "test_to_train": out_counts[:, 0],
        "test_to_val": out_counts[:, 1],
        "test_to_test_out": out_counts[:, 2],
        "total_outgoing": out_counts.sum(axis=1),
        "train_to_test": in_counts[:, 0],
        "val_to_test": in_counts[:, 1],
        "test_to_test_in": in_counts[:, 2],
        "total_incoming": in_counts.sum(axis=1),
    })
    # dia chi that (khong chi index) de tra cuu neu can
    idx_to_addr = {v: k for k, v in addr_to_idx.items()}
    df.insert(1, "address", [idx_to_addr[i] for i in df["test_node_idx"]])
    csv_path = OUT_DIR / "test_node_connectivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}  ({len(df):,} rows)")

    # test<->test "undirected" neighbor union (moi test node co it nhat 1
    # test-neighbor theo 1 trong 2 chieu, khong quan tam chieu nao)
    test_test_any = (df["test_to_test_out"] > 0) | (df["test_to_test_in"] > 0)

    node_stats = {
        "test_to_train_out": stats_row(out_counts[:, 0]),
        "test_to_val_out": stats_row(out_counts[:, 1]),
        "test_to_test_out": stats_row(out_counts[:, 2]),
        "total_outgoing": stats_row(out_counts.sum(axis=1)),
        "train_to_test_in": stats_row(in_counts[:, 0]),
        "val_to_test_in": stats_row(in_counts[:, 1]),
        "test_to_test_in": stats_row(in_counts[:, 2]),
        "total_incoming": stats_row(in_counts.sum(axis=1)),
    }
    pct_with_test_out_neighbor = float((out_counts[:, 2] > 0).mean() * 100)
    pct_with_test_in_neighbor = float((in_counts[:, 2] > 0).mean() * 100)
    pct_with_test_any_neighbor = float(test_test_any.mean() * 100)
    pct_with_train_out_neighbor = float((out_counts[:, 0] > 0).mean() * 100)
    pct_of_out_edges_to_train = float(out_counts[:, 0].sum() / out_counts.sum() * 100)

    print(f"\n  % test nodes co >=1 test-neighbor (out-degree test->test>0)  : {pct_with_test_out_neighbor:.2f}%")
    print(f"  % test nodes co >=1 test-neighbor (in-degree test<-test>0)    : {pct_with_test_in_neighbor:.2f}%")
    print(f"  % test nodes co >=1 test-neighbor (out HOAC in, undirected)  : {pct_with_test_any_neighbor:.2f}%")
    print(f"  % test nodes co >=1 train-neighbor (out-degree test->train>0): {pct_with_train_out_neighbor:.2f}%")
    print(f"  % TONG test outgoing edges di toi Train                     : {pct_of_out_edges_to_train:.2f}%")
    print(f"  Median test_to_train (out) = {node_stats['test_to_train_out']['median']}  P90 = {node_stats['test_to_train_out']['p90']}")
    print(f"  Median test_to_test  (out) = {node_stats['test_to_test_out']['median']}  P90 = {node_stats['test_to_test_out']['p90']}")

    # ---------------------------------------------------------------
    sep("Muc 10 -- Train_small vs Train_large: Test<->Train-region connectivity")
    # E(Test -> Train_small) = test_to_train_exact (da tinh o tren, tu adj_inference,
    # nhung "Train region" o day la TAP HOP node train, khong phai do thi train_small
    # -- dung dung crosstab tren adj_inference, cot=train vs cot=(train+val))
    e_test_to_train_small = test_to_train_exact
    e_test_to_train_large = test_to_train_exact + test_to_val_exact
    d_test = e_test_to_train_large - e_test_to_train_small
    pct_test = d_test / e_test_to_train_small * 100

    e_train_small_to_test = train_to_test_exact
    e_train_large_to_test = train_to_test_exact + val_to_test_exact
    d_train = e_train_large_to_test - e_train_small_to_test
    pct_train = d_train / e_train_small_to_test * 100

    print(f"  E(Test -> Train_small)        = {e_test_to_train_small:,}")
    print(f"  E(Test -> Train_large)        = {e_test_to_train_large:,}  (= Test->Train + Test->Val)")
    print(f"  Delta                          = {d_test:,}   (+{pct_test:.2f}%)")
    print(f"  E(Train_small -> Test)        = {e_train_small_to_test:,}")
    print(f"  E(Train_large -> Test)        = {e_train_large_to_test:,}  (= Train->Test + Val->Test)")
    print(f"  Delta                          = {d_train:,}   (+{pct_train:.2f}%)")

    train_large_table = {
        "Test -> Train region": {
            "train_small": e_test_to_train_small, "train_large": e_test_to_train_large,
            "delta": d_test, "pct_increase": pct_test,
        },
        "Train region -> Test": {
            "train_small": e_train_small_to_test, "train_large": e_train_large_to_test,
            "delta": d_train, "pct_increase": pct_train,
        },
    }

    # ---------------------------------------------------------------
    sep("Doi chieu adj_train / adj_train_large (edges CHUA DEN test, dung de xac nhan 0)")
    mat_train_small = directed_crosstab(adj_train, part_code)
    mat_train_large = directed_crosstab(adj_train_large, part_code)
    print("  adj_train (Train_small graph) crosstab:")
    print(mat_train_small)
    print("  adj_train_large (Train_large graph) crosstab:")
    print(mat_train_large)
    assert mat_train_small[:, 2].sum() == 0 and mat_train_small[2, :].sum() == 0
    assert mat_train_large[:, 2].sum() == 0 and mat_train_large[2, :].sum() == 0
    print("  [PASS] Test bac=0 tuyet doi trong CA adj_train VA adj_train_large (dung nhu ky vong)")

    # ---------------------------------------------------------------
    sep("Save summary JSON")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "n_accounts": int(n),
        "n_by_split": {s: int((part_code == SPLIT_CODE[s]).sum()) for s in SPLITS},
        "exact_directed_crosstab_full_graph": {
            "labels": labels, "matrix_row_from_col_to": mat_inf.tolist(), "total_edges": int(mat_inf.sum()),
        },
        "exact_directed_crosstab_adj_train": mat_train_small.tolist(),
        "exact_directed_crosstab_adj_train_large": mat_train_large.tolist(),
        "exact_pairs": {
            "test_to_train": test_to_train_exact, "train_to_test": train_to_test_exact,
            "test_to_val": test_to_val_exact, "val_to_test": val_to_test_exact,
            "test_to_test": test_to_test_exact,
            "ratio_train_to_test_over_test_to_train": train_to_test_exact / test_to_train_exact,
            "ratio_test_to_val_over_val_to_test": test_to_val_exact / val_to_test_exact,
        },
        "node_level_stats_per_test_node": node_stats,
        "pct_test_nodes_with_test_neighbor_out": pct_with_test_out_neighbor,
        "pct_test_nodes_with_test_neighbor_in": pct_with_test_in_neighbor,
        "pct_test_nodes_with_test_neighbor_any_direction": pct_with_test_any_neighbor,
        "pct_test_nodes_with_train_neighbor_out": pct_with_train_out_neighbor,
        "pct_of_test_outgoing_edges_to_train": pct_of_out_edges_to_train,
        "train_small_vs_train_large": train_large_table,
        "csv_path": str(csv_path),
    }
    json_path = OUT_DIR / "connectivity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
