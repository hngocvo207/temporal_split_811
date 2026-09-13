"""
mg_temporal_pipeline_v2_quantile.py
====================================
Bien the CUA mg_temporal_pipeline.py -- GIU NGUYEN kien truc du lieu cu (4
category train/val/overlap/pure_test, phan loai tung account theo (t_first,
t_last) so voi cac moc cat -- 1 do thi suy luan chung `adj_inference.npz`
dung cho moi split, dung logic cu tri_model/inductive_model dang phu thuoc)
-- CHI THAY CACH TINH 2 MOC CAT, theo de xuat trong Dataset/temporal_pyg.py
(quantile tren CHINH t_first cua 1,165 tai khoan phishing, khong phai
percentile tren TOAN BO giao dich nhu ban goc).

THAY DOI DUY NHAT so voi mg_temporal_pipeline.py:
  1. T_cutoff (moc phan tach pure_train_candidates/overlap/pure_test) =
     QUANTILE q_val_test=0.80 tren t_first cua 1,165 phisher (thay vi
     percentile 80% tren TOAN BO 13,551,303 timestamp giao dich).
  2. Moc phan tach train/val = QUANTILE q_train_val=0.65 tren cung tap
     t_first do (thay vi "latest 10% cua pure_train_candidates theo
     t_last" cua ban goc) -- val = pure_train_candidates co t_last > t1.

Ly do dung ca (t_first, t_last) cho MOI account (khong chi t_first nhu
temporal_pyg.py don gian hoa) -- giu dung tinh nghiem ngat "0 canh train-time"
cho pure_test da duoc kiem chung ky luong xuyen suot du an (xem STATUS.md).

Output: KHONG ghi de data/preprocessed/Dataset_MG/ -- luu rieng vao
data/preprocessed/Dataset_MG_v2_quantile/ de doi chieu truoc khi quyet dinh
thay the ban goc.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OLD_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OUT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG_v2_quantile"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHISHER_LIST_PATH = BASE_DIR / "raw_data/MulDiGraph/phisher_account_muldi.txt"

Q_TRAIN_VAL = 0.65  # moc t1: bien gioi train/val
Q_VAL_TEST = 0.80   # moc t2: bien gioi val/(overlap+pure_test), dong vai tro T_cutoff cu
MIN_PHISHING_PER_SPLIT = 5


def sep(title: str = "") -> None:
    width = 78
    print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}" if title else "─" * width)


def ts_to_str(ts: float) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_existing_timelines_and_labels():
    sep("Load labels.pkl / t_first.pkl / t_last.pkl da co san (khong doc lai MulDiGraph.pkl)")
    with open(OLD_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(OLD_DIR / "t_first.pkl", "rb") as f:
        t_first = pickle.load(f)
    with open(OLD_DIR / "t_last.pkl", "rb") as f:
        t_last = pickle.load(f)
    print(f"  labels={len(labels):,}  t_first={len(t_first):,}  t_last={len(t_last):,}")
    return labels, t_first, t_last


def compute_quantile_cutoffs(labels: dict, t_first: dict) -> tuple[float, float]:
    sep(f"Moc cat theo QUANTILE tren t_first cua 1,165 phisher (q={Q_TRAIN_VAL}/{Q_VAL_TEST})")
    with open(PHISHER_LIST_PATH) as f:
        phisher_addrs = [line.strip().lower() for line in f if line.strip()]
    t_first_pos = np.array([t_first[a] for a in phisher_addrs if a in t_first])
    print(f"  n_phishing_voi_t_first={len(t_first_pos):,} / {len(phisher_addrs)}")
    t1 = float(np.quantile(t_first_pos, Q_TRAIN_VAL))
    t2 = float(np.quantile(t_first_pos, Q_VAL_TEST))
    print(f"  t1 (train/val)  = {t1:.0f}  ({ts_to_str(t1)})")
    print(f"  t2 (val/test)   = {t2:.0f}  ({ts_to_str(t2)})  <- dong vai tro T_cutoff cu")
    return t1, t2


def classify_partitions(t_first: dict, t_last: dict, all_node_ids: list, t1: float, t2: float) -> dict:
    """GIONG HET logic cu (classify_partitions trong mg_temporal_pipeline.py)
    nhung T_cutoff = t2 (thay vi percentile-80-tat-ca-giao-dich)."""
    sep("pure_train_candidates / overlap / pure_test (T_cutoff = t2 moi)")
    partition = {}
    for addr in all_node_ids:
        tmin = t_first.get(addr)
        if tmin is None:
            partition[addr] = "isolated"
            continue
        tmax = t_last[addr]
        if tmax <= t2:
            partition[addr] = "pure_train_candidates"
        elif tmin <= t2 < tmax:
            partition[addr] = "overlap"
        else:
            partition[addr] = "pure_test"
    counts = {}
    for v in partition.values():
        counts[v] = counts.get(v, 0) + 1
    for k in ("pure_train_candidates", "overlap", "pure_test", "isolated"):
        print(f"  {k:<24}: {counts.get(k, 0):,}")
    return partition


def split_train_val_by_t1(partition: dict, t_last: dict, t1: float) -> dict:
    """THAY THE "latest 10% by t_last" cua ban goc bang moc t1 quantile:
    val = pure_train_candidates co t_last > t1 (hoat dong keo dai qua t1,
    tuc "gan T_cutoff hon"); train = t_last <= t1."""
    sep(f"Carve val = pure_train_candidates co t_last > t1 ({ts_to_str(t1)})")
    candidates = [a for a, p in partition.items() if p == "pure_train_candidates"]
    final = dict(partition)
    n_val = 0
    for a in candidates:
        if t_last[a] > t1:
            final[a] = "val"
            n_val += 1
        else:
            final[a] = "train"
    print(f"  pure_train_candidates: {len(candidates):,}   -> train: {len(candidates) - n_val:,}   val: {n_val:,}")
    return final


def phishing_diagnostic(partition: dict, labels: dict) -> dict:
    sep("Phishing-distribution diagnostic")
    buckets = ["train", "val", "overlap", "pure_test", "isolated"]
    stats = {}
    warnings = []
    for b in buckets:
        addrs = [a for a, p in partition.items() if p == b]
        n_total = len(addrs)
        n_pos = sum(labels[a] for a in addrs)
        pct = 100 * n_pos / n_total if n_total else 0.0
        stats[b] = {"total": n_total, "phishing_n": n_pos, "phishing_pct": pct}
        print(f"  {b:<12}: total={n_total:>9,}  phishing={n_pos:>5,}  ({pct:.4f}%)  "
              f"[{100*n_pos/1165:.1f}% cua 1,165 tong]")
        if b != "isolated" and n_pos < MIN_PHISHING_PER_SPLIT:
            warnings.append(f"WARNING: split '{b}' chi co {n_pos} phishing (< {MIN_PHISHING_PER_SPLIT})")
    test_addrs = [a for a, p in partition.items() if p in ("overlap", "pure_test")]
    n_test_pos = sum(labels[a] for a in test_addrs)
    stats["test_(overlap+pure_test)"] = {
        "total": len(test_addrs), "phishing_n": n_test_pos,
        "phishing_pct": 100 * n_test_pos / len(test_addrs) if test_addrs else 0.0,
    }
    print(f"  {'test(ovl+pt)':<12}: total={len(test_addrs):>9,}  phishing={n_test_pos:>5,}  "
          f"({stats['test_(overlap+pure_test)']['phishing_pct']:.4f}%)")
    for w in warnings:
        print(f"  [!] {w}")
    if not warnings:
        print("  [OK] Moi split deu co so luong phishing dang ke.")
    stats["_warnings"] = warnings
    return stats


def step8_validation(labels, t_first, t_last, partition, t1, t2) -> bool:
    sep("Validation checks (giong het bo assertion cua ban goc)")
    all_pass = True

    def check(name, ok, detail=""):
        nonlocal all_pass
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    train_set = {a for a, p in partition.items() if p == "train"}
    val_set = {a for a, p in partition.items() if p == "val"}
    overlap_set = {a for a, p in partition.items() if p == "overlap"}
    pure_test_set = {a for a, p in partition.items() if p == "pure_test"}
    isolated_set = {a for a, p in partition.items() if p == "isolated"}

    max_train_val_tlast = max((t_last[a] for a in train_set | val_set), default=-np.inf)
    min_test_tfirst = min((t_first[a] for a in pure_test_set), default=np.inf)
    check(
        "Boundary consistency: max(t_last, train+val) <= t2 < min(t_first, pure_test)",
        bool(max_train_val_tlast <= t2 < min_test_tfirst),
        f"max_train_val_tlast={max_train_val_tlast:.0f} t2={t2:.0f} min_pure_test_tfirst={min_test_tfirst:.0f}",
    )
    max_train_tlast = max((t_last[a] for a in train_set), default=-np.inf)
    check(
        "train/val boundary: max(t_last, train) <= t1 < ... (val co t_last > t1)",
        bool(max_train_tlast <= t1),
        f"max_train_tlast={max_train_tlast:.0f} t1={t1:.0f}",
    )
    disjoint = train_set.isdisjoint(val_set) and train_set.isdisjoint(overlap_set) and \
        train_set.isdisjoint(pure_test_set) and val_set.isdisjoint(overlap_set) and \
        val_set.isdisjoint(pure_test_set) and overlap_set.isdisjoint(pure_test_set)
    check("train/val/overlap/pure_test doi mot roi nhau", disjoint)
    covers_all = (train_set | val_set | overlap_set | pure_test_set | isolated_set) == set(partition.keys())
    check("hop 5 nhom = toan bo account", covers_all)

    total_original = sum(labels.values())
    total_covered = sum(labels[a] for a in partition.keys())
    check("Phishing coverage: khong mat account nao", total_original == total_covered,
          f"original={total_original} covered={total_covered}")

    for name, addrs in [("val", list(val_set)), ("pure_test", list(pure_test_set))]:
        if not addrs:
            continue
        rate = sum(labels[a] for a in addrs) / len(addrs)
        suspiciously_balanced = 0.3 < rate < 0.7
        check(f"Phan bo thuc te ({name} khong bi can bang nhan tao)", not suspiciously_balanced,
              f"phishing_rate={rate:.6f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def save_outputs(labels, t_first, t_last, partition, t1, t2, phishing_stats):
    sep("Save outputs")

    def save_pkl(obj, name):
        p = OUT_DIR / name
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {p}")

    save_pkl(labels, "labels.pkl")
    save_pkl(t_first, "t_first.pkl")
    save_pkl(t_last, "t_last.pkl")
    save_pkl(partition, "partition.pkl")

    counts = {}
    for v in partition.values():
        counts[v] = counts.get(v, 0) + 1

    config = {
        "t1_train_val_cutoff": t1, "t1_human": ts_to_str(t1),
        "t2_val_test_cutoff": t2, "t2_human": ts_to_str(t2),
        "cutoff_method": "quantile tren t_first cua 1,165 phisher (q_train_val=0.65, q_val_test=0.80) "
                         "-- xem Dataset/temporal_pyg.py -- THAY vi percentile-80-tat-ca-giao-dich "
                         "cua mg_temporal_pipeline.py ban goc",
        "split_criterion": "t2 dong vai tro T_cutoff cu (bien gioi pure_train_candidates/overlap/pure_test); "
                            "val = pure_train_candidates co t_last > t1 (thay vi latest-10%-by-t_last)",
        "partition_rule": "train: t_last<=t1 | val: t1<t_last<=t2 (trong pure_train_candidates) | "
                           "overlap: t_first<=t2<t_last | pure_test: t_first>t2",
        "label_scheme": "confirmed_phishing_only (isp==1), tai su dung labels.pkl goc khong doi",
        "counts": counts,
        "total_phishing_original": sum(labels.values()),
        "total_phishing_across_partitions": sum(labels[a] for a in partition.keys()),
        "not_yet_regenerated": [
            "adj_train.npz (can build lai voi T_cutoff=t2 tu MulDiGraph.pkl)",
            "adj_inference.npz (khong doi -- van la toan bo giao dich, giong ban goc)",
            "address_to_index.pkl (giu nguyen -- KHONG doi index space, chi doi nhan partition)",
        ],
    }
    with open(OUT_DIR / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'split_config.json'}")
    with open(OUT_DIR / "split_stats.json", "w") as f:
        json.dump(phishing_stats, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'split_stats.json'}")


def main():
    print("=" * 78)
    print("  mg_temporal_pipeline_v2_quantile.py — quantile cutoff, kien truc du lieu CU")
    print("=" * 78)
    labels, t_first, t_last = load_existing_timelines_and_labels()
    t1, t2 = compute_quantile_cutoffs(labels, t_first)
    all_node_ids = list(t_first.keys() | t_last.keys() | labels.keys())
    # dam bao co ca node isolated (co trong labels/address_to_index nhung khong co t_first)
    with open(OLD_DIR / "address_to_index.pkl", "rb") as f:
        addr2idx = pickle.load(f)
    all_node_ids = list(addr2idx.keys())
    partition = classify_partitions(t_first, t_last, all_node_ids, t1, t2)
    partition = split_train_val_by_t1(partition, t_last, t1)
    phishing_stats = phishing_diagnostic(partition, labels)
    save_outputs(labels, t_first, t_last, partition, t1, t2, phishing_stats)
    all_pass = step8_validation(labels, t_first, t_last, partition, t1, t2)
    sep("DONE")
    if not all_pass:
        print("  Validation FAILED — kiem tra lai truoc khi dung.")
        sys.exit(1)


if __name__ == "__main__":
    main()
