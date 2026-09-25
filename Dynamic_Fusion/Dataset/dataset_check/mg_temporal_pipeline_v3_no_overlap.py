"""
mg_temporal_pipeline_v3_no_overlap.py
=======================================
SUA LAI v2 (mg_temporal_pipeline_v2_quantile.py) theo yeu cau: KHONG con
"overlap" -- phan loai MOI account CHI theo t_first (giong het
Dataset/temporal_pyg.py::build_splits(), khong dung (t_first, t_last)
straddle-check cua mg_temporal_pipeline.py ban goc nua).

3 category DUY NHAT, dung dung 3 window cua temporal_pyg.py:
  train : t_first <  t1
  val   : t1 <= t_first < t2
  test  : t_first >= t2
(t1 = quantile 0.65, t2 = quantile 0.80, tren t_first cua 1,165 phisher --
giong het temporal_pyg.py va mg_temporal_pipeline_v2_quantile.py).

adj_train.npz: T_cutoff = t1 (KHONG phai t2 nhu v2) -- khop dung voi chinh
dinh nghia "train window" cua temporal_pyg.py (train subgraph tich luy DEN
t1). adj_inference.npz khong doi (toan bo giao dich).

Output: data/preprocessed/Dataset_MG_v3_no_overlap/ (khong ghi de v2/ban goc).
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

BASE_DIR = Path(__file__).resolve().parent.parent
OLD_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OUT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG_v3_no_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
PHISHER_LIST_PATH = BASE_DIR / "raw_data/MulDiGraph/phisher_account_muldi.txt"

sys.path.insert(0, str(BASE_DIR / "Dataset"))
from mg_graph_weight_formula import build_edge_weights  # noqa: E402

Q_TRAIN_VAL = 0.65
Q_VAL_TEST = 0.80
MIN_PHISHING_PER_SPLIT = 5


def sep(title: str = "") -> None:
    width = 78
    print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}" if title else "─" * width)


def ts_to_str(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    t0 = time.time()
    print("=" * 78)
    print("  mg_temporal_pipeline_v3_no_overlap.py — 3-way split theo t_first (temporal_pyg.py)")
    print("=" * 78)

    sep("Load labels.pkl / t_first.pkl (da co san)")
    with open(OLD_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(OLD_DIR / "t_first.pkl", "rb") as f:
        t_first = pickle.load(f)
    with open(OLD_DIR / "address_to_index.pkl", "rb") as f:
        address_to_index = pickle.load(f)
    print(f"  labels={len(labels):,} t_first={len(t_first):,} address_to_index={len(address_to_index):,}")

    sep(f"Moc cat quantile tren t_first cua 1,165 phisher (q={Q_TRAIN_VAL}/{Q_VAL_TEST})")
    with open(PHISHER_LIST_PATH) as f:
        phisher_addrs = [line.strip().lower() for line in f if line.strip()]
    t_first_pos = np.array([t_first[a] for a in phisher_addrs])
    t1 = float(np.quantile(t_first_pos, Q_TRAIN_VAL))
    t2 = float(np.quantile(t_first_pos, Q_VAL_TEST))
    print(f"  t1 (train/val) = {t1:.0f}  ({ts_to_str(t1)})")
    print(f"  t2 (val/test)  = {t2:.0f}  ({ts_to_str(t2)})")

    sep("Phan loai 3-way CHI theo t_first (train/val/test, KHONG overlap)")
    partition = {}
    n_no_tfirst = 0
    for addr in address_to_index:
        tf = t_first.get(addr)
        if tf is None:
            partition[addr] = "isolated"
            n_no_tfirst += 1
            continue
        if tf < t1:
            partition[addr] = "train"
        elif tf < t2:
            partition[addr] = "val"
        else:
            partition[addr] = "test"
    counts = {}
    for v in partition.values():
        counts[v] = counts.get(v, 0) + 1
    for k in ("train", "val", "test", "isolated"):
        print(f"  {k:<10}: {counts.get(k, 0):,}")
    print(f"  (accounts khong co t_first -> isolated: {n_no_tfirst:,})")

    sep("Phishing-distribution diagnostic")
    warnings = []
    stats = {}
    for b in ("train", "val", "test", "isolated"):
        addrs = [a for a, p in partition.items() if p == b]
        n_total = len(addrs)
        n_pos = sum(labels[a] for a in addrs)
        pct = 100 * n_pos / n_total if n_total else 0.0
        stats[b] = {"total": n_total, "phishing_n": n_pos, "phishing_pct": pct}
        print(f"  {b:<10}: total={n_total:>9,}  phishing={n_pos:>5,}  ({pct:.4f}%)  "
              f"[{100*n_pos/1165:.1f}% cua 1,165 tong]")
        if b != "isolated" and n_pos < MIN_PHISHING_PER_SPLIT:
            warnings.append(f"WARNING: split '{b}' chi co {n_pos} phishing")
    for w in warnings:
        print(f"  [!] {w}")
    if not warnings:
        print("  [OK] Moi split deu co so luong phishing dang ke.")

    sep("Save labels.pkl / t_first.pkl / partition.pkl / address_to_index.pkl")

    def save_pkl(obj, name):
        p = OUT_DIR / name
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {p}")

    save_pkl(labels, "labels.pkl")
    save_pkl(t_first, "t_first.pkl")
    save_pkl(partition, "partition.pkl")
    shutil.copy(OLD_DIR / "address_to_index.pkl", OUT_DIR / "address_to_index.pkl")
    print(f"  Copied: address_to_index.pkl (khong doi index space)")

    sep("Build adj_train.npz voi T_cutoff = t1 (khop dung 'train window' cua temporal_pyg.py)")
    print("  loading MulDiGraph.pkl...")
    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append((u, v, float(d.get("amount", 0.0)), float(d.get("timestamp"))))
    edges_df = pd.DataFrame(rows, columns=["from_addr", "to_addr", "amount", "timestamp"])
    print(f"  edges_df: {len(edges_df):,} ({time.time()-t0:.1f}s)")

    train_edges = edges_df[edges_df["timestamp"] <= t1]
    print(f"  train-period edges (<=t1): {len(train_edges):,} / {len(edges_df):,} "
          f"({100*len(train_edges)/len(edges_df):.2f}%)")
    train_agg = build_edge_weights(train_edges)

    def to_sparse(agg, addr2idx, n):
        rows_ = agg["from_addr"].map(addr2idx).to_numpy()
        cols_ = agg["to_addr"].map(addr2idx).to_numpy()
        data_ = agg["weight"].to_numpy(dtype=np.float32)
        return coo_matrix((data_, (rows_, cols_)), shape=(n, n))

    n = len(address_to_index)
    train_adj = to_sparse(train_agg, address_to_index, n)
    save_npz(OUT_DIR / "adj_train.npz", train_adj.tocsr())
    print(f"  adj_train.npz: shape={train_adj.shape} nnz={train_adj.nnz:,} ({time.time()-t0:.1f}s)")

    shutil.copy(OLD_DIR / "adj_inference.npz", OUT_DIR / "adj_inference.npz")
    print(f"  Copied adj_inference.npz tu ban goc (khong doi -- toan bo giao dich)")

    sep("Validation checks")
    all_pass = True

    def check(name, ok, detail=""):
        nonlocal all_pass
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    train_set = {a for a, p in partition.items() if p == "train"}
    val_set = {a for a, p in partition.items() if p == "val"}
    test_set = {a for a, p in partition.items() if p == "test"}
    isolated_set = {a for a, p in partition.items() if p == "isolated"}
    check("train/val/test/isolated doi mot roi nhau",
          train_set.isdisjoint(val_set) and train_set.isdisjoint(test_set) and val_set.isdisjoint(test_set))
    check("hop 4 nhom = toan bo account",
          (train_set | val_set | test_set | isolated_set) == set(partition.keys()))
    check("Phishing coverage: khong mat account nao",
          sum(labels.values()) == sum(labels[a] for a in partition.keys()),
          f"original={sum(labels.values())} covered={sum(labels[a] for a in partition.keys())}")
    max_train_tfirst = max((t_first[a] for a in train_set), default=-np.inf)
    min_val_tfirst = min((t_first[a] for a in val_set), default=np.inf)
    min_test_tfirst = min((t_first[a] for a in test_set), default=np.inf)
    check("Boundary: max(t_first, train) < t1 <= min(t_first, val)",
          bool(max_train_tfirst < t1 <= min_val_tfirst),
          f"max_train_tfirst={max_train_tfirst:.0f} t1={t1:.0f} min_val_tfirst={min_val_tfirst:.0f}")
    check("Boundary: max(t_first, val) < t2 <= min(t_first, test)",
          bool(max((t_first[a] for a in val_set), default=-np.inf) < t2 <= min_test_tfirst),
          f"t2={t2:.0f} min_test_tfirst={min_test_tfirst:.0f}")

    # invariant nghiem ngat nhat xuyen suot du an: test phai co bac=0 tuyet doi trong adj_train
    idx_test = np.array([address_to_index[a] for a in test_set])
    out_deg = np.diff(train_adj.tocsr().indptr)[idx_test]
    in_deg = np.asarray((train_adj.tocsr()[:, idx_test] > 0).sum(axis=0)).flatten()
    n_nonzero_deg = int(((out_deg + in_deg) > 0).sum())
    check("test co bac = 0 TUYET DOI trong adj_train.npz (invariant nghiem ngat nhat)",
          n_nonzero_deg == 0, f"{n_nonzero_deg}/{len(idx_test)} account co bac > 0")

    for name, addrs in [("val", list(val_set)), ("test", list(test_set))]:
        if not addrs:
            continue
        rate = sum(labels[a] for a in addrs) / len(addrs)
        check(f"Phan bo thuc te ({name} khong bi can bang nhan tao)",
              not (0.3 < rate < 0.7), f"phishing_rate={rate:.6f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")

    config = {
        "t1_train_val_cutoff": t1, "t1_human": ts_to_str(t1),
        "t2_val_test_cutoff": t2, "t2_human": ts_to_str(t2),
        "cutoff_method": "quantile tren t_first cua 1,165 phisher (q_train_val=0.65, q_val_test=0.80), "
                         "GIONG HET Dataset/temporal_pyg.py",
        "partition_rule": "CHI theo t_first (KHONG dung t_last, KHONG overlap): "
                           "train: t_first<t1 | val: t1<=t_first<t2 | test: t_first>=t2",
        "adj_train_cutoff": "t1 (khop dung train window cua temporal_pyg.py, KHAC v2 dung t2)",
        "label_scheme": "confirmed_phishing_only (isp==1), tai su dung labels.pkl goc",
        "counts": counts,
        "total_phishing_original": sum(labels.values()),
        "total_phishing_across_partitions": sum(labels[a] for a in partition.keys()),
        "supersedes": "data/preprocessed/Dataset_MG_v2_quantile (co 'overlap', dung (t_first,t_last) -- "
                       "khong con dung theo yeu cau moi nhat)",
    }
    with open(OUT_DIR / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'split_config.json'}")
    with open(OUT_DIR / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'split_stats.json'}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
