"""
mg_temporal_pipeline.py
========================
Leakage-free temporal train/val/test split for the MulDiGraph dataset
(Chen et al. 2020, "Phishing Scams Detection in Ethereum Transaction
Network", XBlock — 2,973,489 nodes / 13,551,303 edges / 1,165 phishing
nodes).
  1. T_cutoff = the 80th percentile of ALL edge timestamps in the graph
     (every transaction counts once, not one T_first per account).
     Pre-cutoff graph:  timestamp <= T_cutoff   (~80% of transactions)
     Post-cutoff graph: timestamp >  T_cutoff

  2. Every account is classified by its FULL activity range (t_first,
     t_last) relative to T_cutoff:
       pure_train_candidates : t_last  <= T_cutoff   (only txns before cutoff)
       overlap                : t_first <= T_cutoff < t_last (spans the cutoff)
       pure_test              : t_first  >  T_cutoff  (starts after cutoff)
     Accounts with zero timestamped edges cannot be temporally placed and
     are kept in a separate 'isolated' bucket rather than silently dropped.

  3. Validation = the 10% of pure_train_candidates with the LATEST last-
     activity timestamp before T_cutoff (i.e. sorted by t_last descending,
     top decile). The remaining 90% becomes the final 'train' partition.

  4. Test = overlap UNION pure_test (kept as two distinct partition labels
     — 'overlap' and 'pure_test' — so downstream scripts can still report
     the pure_test / overlap / full-test breakdown).

  5. Label-leakage boundary: only 'train' contributes to the training
     LOSS. 'overlap' occupies a slot in the GCN vocabulary/adjacency for
     message passing but is masked out of the loss (handled in
     mg_build_examples.py) — see module docstring there.

Outputs -> data/preprocessed/Dataset_MG/:
  labels.pkl        {addr: 0/1}                 (confirmed phishing only, isp==1)
  t_first.pkl       {addr: float}                (first_transaction_timestamp)
  t_last.pkl        {addr: float}
  partition.pkl     {addr: 'train'|'val'|'overlap'|'pure_test'|'isolated'}
  split_config.json boundaries + counts
  split_stats.json  per-partition label distribution
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
OUT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_CUTOFF_PERCENTILE = 80.0
VAL_FRAC_OF_PURE_TRAIN = 0.10
MIN_PHISHING_PER_SPLIT = 5  # below this -> WARNING in the diagnostic step


def sep(title: str = "") -> None:
    width = 78
    print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}" if title else "─" * width)


def ts_to_str(ts: float) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_graph(path: Path):
    sep("Load MulDiGraph.pkl")
    print(f"  Loading {path} ...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"  Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}")
    return G


def compute_labels(G) -> dict[str, int]:
    sep("Labels (confirmed phishing only, isp==1)")
    labels = {addr: int(d.get("isp", 0) == 1) for addr, d in G.nodes(data=True)}
    n_pos = sum(labels.values())
    print(f"  Phishing (isp=1): {n_pos:,}   Normal (isp=0): {len(labels) - n_pos:,}")
    return labels


def compute_timelines_and_cutoff(G) -> tuple[dict, dict, float, int]:
    """Vectorized: per-node t_first/t_last over ALL edges, AND the global
    T_cutoff = 80th percentile of every edge timestamp (transaction-level,
    not one T_first per account)."""
    sep("Per-node t_first / t_last + global T_cutoff (80th pct of ALL edge timestamps)")
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    n_edges = G.number_of_edges()
    src = np.empty(n_edges, dtype=np.int64)
    dst = np.empty(n_edges, dtype=np.int64)
    ts = np.empty(n_edges, dtype=np.float64)
    k = 0
    n_missing = 0
    for u, v, d in G.edges(data=True):
        t = d.get("timestamp")
        if t is None:
            n_missing += 1
            continue
        src[k] = idx[u]
        dst[k] = idx[v]
        ts[k] = float(t)
        k += 1
    src, dst, ts = src[:k], dst[:k], ts[:k]
    print(f"  Edges with timestamp: {k:,}   (missing: {n_missing:,})")

    T_cutoff = float(np.percentile(ts, T_CUTOFF_PERCENTILE))
    frac_pre = float((ts <= T_cutoff).mean())
    print(f"  T_cutoff ({T_CUTOFF_PERCENTILE:.0f}th pct of {k:,} edge timestamps) = {T_cutoff:.0f}  ({ts_to_str(T_cutoff)})")
    print(f"  Fraction of transactions with timestamp <= T_cutoff: {100 * frac_pre:.4f}%")

    t_first = np.full(n, np.inf, dtype=np.float64)
    t_last = np.full(n, -np.inf, dtype=np.float64)
    np.minimum.at(t_first, src, ts)
    np.minimum.at(t_first, dst, ts)
    np.maximum.at(t_last, src, ts)
    np.maximum.at(t_last, dst, ts)

    t_first_d = {nodes[i]: float(t_first[i]) for i in range(n) if np.isfinite(t_first[i])}
    t_last_d = {nodes[i]: float(t_last[i]) for i in range(n) if np.isfinite(t_last[i])}
    n_isolated = n - len(t_first_d)
    print(f"  Nodes with >=1 timestamped edge: {len(t_first_d):,}   Isolated (no edges): {n_isolated:,}")
    return t_first_d, t_last_d, T_cutoff, k, frac_pre


def classify_partitions(
    t_first: dict[str, float], t_last: dict[str, float], all_node_ids: list, T_cutoff: float
) -> dict[str, str]:
    """pure_train_candidates / overlap / pure_test by FULL activity range
    (t_min, t_max) relative to the single T_cutoff boundary, then carve
    'val' out of pure_train_candidates (see split_train_val)."""
    sep("pure_train_candidates / overlap / pure_test classification")
    partition: dict[str, str] = {}
    for addr in all_node_ids:
        tmin = t_first.get(addr)
        if tmin is None:
            partition[addr] = "isolated"
            continue
        tmax = t_last[addr]
        if tmax <= T_cutoff:
            partition[addr] = "pure_train_candidates"
        elif tmin <= T_cutoff < tmax:
            partition[addr] = "overlap"
        else:  # tmin > T_cutoff
            partition[addr] = "pure_test"

    counts = {}
    for v in partition.values():
        counts[v] = counts.get(v, 0) + 1
    for k in ("pure_train_candidates", "overlap", "pure_test", "isolated"):
        print(f"  {k:<24}: {counts.get(k, 0):,}")
    return partition


def split_train_val(partition: dict[str, str], t_last: dict[str, float]) -> dict[str, str]:
    """Validation = the VAL_FRAC_OF_PURE_TRAIN latest-activity accounts
    within pure_train_candidates (sorted by t_last, i.e. most recent
    activity before T_cutoff, descending). Remainder -> 'train'."""
    sep(f"Carve val = latest {VAL_FRAC_OF_PURE_TRAIN:.0%} of pure_train_candidates by t_last")
    candidates = [a for a, p in partition.items() if p == "pure_train_candidates"]
    candidates.sort(key=lambda a: t_last[a], reverse=True)  # latest first
    n_val = int(round(len(candidates) * VAL_FRAC_OF_PURE_TRAIN))
    val_accounts = set(candidates[:n_val])

    final = dict(partition)
    for a in candidates:
        final[a] = "val" if a in val_accounts else "train"

    n_train = len(candidates) - len(val_accounts)
    print(f"  pure_train_candidates: {len(candidates):,}   -> train: {n_train:,}   val: {len(val_accounts):,}")
    if val_accounts:
        val_t_last = [t_last[a] for a in val_accounts]
        print(f"  val t_last range: [{min(val_t_last):.0f}, {max(val_t_last):.0f}]  (all <= T_cutoff by construction)")
    return final


def phishing_diagnostic(partition: dict[str, str], labels: dict[str, int]) -> dict:
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
        print(f"  {b:<12}: total={n_total:>9,}  phishing={n_pos:>5,}  ({pct:.4f}%)")
        if b != "isolated" and n_pos < MIN_PHISHING_PER_SPLIT:
            warnings.append(
                f"WARNING: split '{b}' has only {n_pos} phishing accounts "
                f"(< {MIN_PHISHING_PER_SPLIT}) — boundary may need manual adjustment."
            )
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
        print("  [OK] Every split has a non-trivial phishing count.")
    stats["_warnings"] = warnings
    return stats


def save_outputs(
    labels: dict, t_first: dict, t_last: dict, partition: dict, T_cutoff: float,
    n_edges_timestamped: int, phishing_stats: dict,
) -> None:
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
        "T_cutoff": T_cutoff,
        "T_cutoff_human": ts_to_str(T_cutoff),
        "T_cutoff_percentile": T_CUTOFF_PERCENTILE,
        "n_edges_timestamped": n_edges_timestamped,
        "val_frac_of_pure_train": VAL_FRAC_OF_PURE_TRAIN,
        "split_criterion": "global T_cutoff = 80th percentile of ALL edge timestamps; "
                            "val = latest 10% of pure-train accounts by t_last",
        "partition_rule": "train/val (from pure_train_candidates: t_max<=T_cutoff) | "
                           "overlap: t_min<=T_cutoff<t_max | pure_test: t_min>T_cutoff",
        "label_scheme": "confirmed_phishing_only (isp==1); no potential-fraud relabeling here "
                         "(see mg_propagate_labels.py for the separate propagated_1hop signal)",
        "counts": counts,
        "total_phishing_original": sum(labels.values()),
        "total_phishing_across_partitions": sum(labels[a] for a in partition.keys()),
    }
    with open(OUT_DIR / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'split_config.json'}")

    with open(OUT_DIR / "split_stats.json", "w") as f:
        json.dump(phishing_stats, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'split_stats.json'}")


def step8_validation(
    labels: dict, t_first: dict, t_last: dict, partition: dict, T_cutoff: float,
    n_edges_timestamped: int, frac_pre: float,
) -> bool:
    sep("Validation checks (assertions, new_split.md §4)")
    all_pass = True

    def check(name, ok, detail=""):
        nonlocal all_pass
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    # 1. T_cutoff really is the 80th percentile of ALL edge timestamps (+-0.1%),
    # i.e. the fraction of transactions with timestamp <= T_cutoff (computed
    # per-edge in compute_timelines_and_cutoff) must be within 0.1 percentage
    # points of 80%.
    target_frac = T_CUTOFF_PERCENTILE / 100.0
    check(
        f"T_cutoff is the {T_CUTOFF_PERCENTILE:.0f}th percentile of all edge timestamps (+-0.1%)",
        bool(abs(frac_pre - target_frac) <= 0.001),
        f"fraction of {n_edges_timestamped:,} timestamped edges with timestamp<=T_cutoff = {100*frac_pre:.4f}% "
        f"(target {T_CUTOFF_PERCENTILE:.0f}.0000% +-0.1%)",
    )

    # 2. Boundary consistency: pure_train/val ends at/before T_cutoff, pure_test starts after it.
    pure_train_like = [a for a, p in partition.items() if p in ("train", "val")]
    overlap_addrs = [a for a, p in partition.items() if p == "overlap"]
    pure_test_addrs = [a for a, p in partition.items() if p == "pure_test"]
    max_train_tlast = max((t_last[a] for a in pure_train_like), default=-np.inf)
    min_test_tfirst = min((t_first[a] for a in pure_test_addrs), default=np.inf)
    check(
        "Boundary consistency: max(t_last, train+val) <= T_cutoff < min(t_first, pure_test)",
        bool(max_train_tlast <= T_cutoff < min_test_tfirst),
        f"max_train_val_tlast={max_train_tlast:.0f} T_cutoff={T_cutoff:.0f} min_pure_test_tfirst={min_test_tfirst:.0f}",
    )

    # 3. pure_train_candidates(train+val) / overlap / pure_test pairwise disjoint & cover all non-isolated accounts
    train_val_set = set(pure_train_like)
    overlap_set = set(overlap_addrs)
    pure_test_set = set(pure_test_addrs)
    isolated_set = {a for a, p in partition.items() if p == "isolated"}
    disjoint = (
        train_val_set.isdisjoint(overlap_set)
        and train_val_set.isdisjoint(pure_test_set)
        and overlap_set.isdisjoint(pure_test_set)
    )
    covers_all = (train_val_set | overlap_set | pure_test_set | isolated_set) == set(partition.keys())
    check("pure-train/overlap/pure-test pairwise disjoint", disjoint)
    check("pure-train ∪ overlap ∪ pure-test ∪ isolated == all accounts", covers_all)

    # 4. train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅
    train_set = {a for a, p in partition.items() if p == "train"}
    val_set = {a for a, p in partition.items() if p == "val"}
    test_set = overlap_set | pure_test_set
    check("train ∩ val = ∅", train_set.isdisjoint(val_set))
    check("train ∩ test(overlap∪pure_test) = ∅", train_set.isdisjoint(test_set))
    check("val ∩ test(overlap∪pure_test) = ∅", val_set.isdisjoint(test_set))

    # 4. Phishing coverage — nothing silently dropped
    total_original = sum(labels.values())
    total_covered = sum(labels[a] for a in partition.keys())
    check(
        "Phishing coverage: sum(labels) across all partitions (incl. isolated) == original total",
        total_original == total_covered,
        f"original={total_original} covered={total_covered}",
    )

    # 5. Real distribution check (not artificially balanced)
    for name, addrs in [("val", list(val_set)), ("pure_test", pure_test_addrs)]:
        if not addrs:
            continue
        rate = sum(labels[a] for a in addrs) / len(addrs)
        suspiciously_balanced = 0.3 < rate < 0.7 or abs(rate - 0.1) < 0.01
        check(
            f"Real distribution check ({name} not artificially balanced)",
            not suspiciously_balanced,
            f"phishing_rate={rate:.6f}",
        )

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def main():
    print("=" * 78)
    print("  mg_temporal_pipeline.py — 80% T_cutoff temporal split for MulDiGraph")
    print("=" * 78)

    G = load_graph(MG_PATH)
    labels = compute_labels(G)
    t_first, t_last, T_cutoff, n_edges_timestamped, frac_pre = compute_timelines_and_cutoff(G)
    partition = classify_partitions(t_first, t_last, list(G.nodes()), T_cutoff)
    partition = split_train_val(partition, t_last)
    phishing_stats = phishing_diagnostic(partition, labels)
    save_outputs(labels, t_first, t_last, partition, T_cutoff, n_edges_timestamped, phishing_stats)
    all_pass = step8_validation(labels, t_first, t_last, partition, T_cutoff, n_edges_timestamped, frac_pre)

    sep("DONE")
    if not all_pass:
        print("  Validation FAILED — inspect the checks above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
