"""
mg_propagate_labels.py
=======================
Controlled 1-hop label propagation from the 1,165 ground-truth phishing
seeds (isp==1) in MulDiGraph.pkl, producing a SEPARATE "suspected fraud"
signal (label_source == "propagated_1hop") used only to augment the TRAIN
partition. Ground truth (`isp`) is never modified and is always what
train1.py reports benchmark numbers against.

Safety constraints implemented (new_split.md §C):

  1. Directionality — only OUT-edges from a seed (money flowing OUT of a
     phishing account, i.e. towards likely mules/launderers) are
     propagated. IN-edges to a seed (senders = victims) are never turned
     into positives; they are tagged `is_victim=1` for analysis only.

  2. Hub / exchange guard — before propagating, degree (in+out) is
     computed for every node. Any node above the 99th-percentile degree
     (hot wallet / exchange) is excluded from BOTH sides of propagation:
     a hub SEED does not propagate outward (its out-neighborhood is
     essentially "everyone"), and a hub CANDIDATE is never labeled
     propagated_1hop (it's an exchange, not a mule). Both exclusion counts
     are logged separately.

  3. Label separation — `label_source` in {"ground_truth", "propagated_1hop",
     "benign"}. `isp` (original benchmark ground truth) is left untouched.
     `isp_augmented` = isp OR (propagated_1hop, only if propagation is
     enabled) — used for TRAINING ONLY, never for benchmark reporting.

  4. No leakage into val/test — propagated_1hop is only ever assigned to
     accounts in the `train` partition. Assertion:
     propagated_1hop_accounts ∩ (val ∪ overlap ∪ pure_test) == ∅.

  5. Soft label, not hard label — propagated_1hop accounts get a
     `propagation_weight` < 1.0 (default 0.5, tunable), consumed downstream
     as a per-example loss weight (mg_build_examples.py / train1.py),
     never treated as equal-confidence to ground truth.

  6. Logging — seed count, pre-filter candidate count, hub-excluded counts,
     final propagated count, positive-class rate before/after, all printed
     and saved to propagation_stats.json.

Outputs -> data/preprocessed/Dataset_MG/:
  label_source.pkl        {addr: 'ground_truth'|'propagated_1hop'|'benign'}
  is_victim.pkl            {addr: 1}                (only accounts flagged as victims)
  isp_augmented.pkl        {addr: 0/1}
  propagation_weight.pkl   {addr: float}             (1.0 unless propagated_1hop)
  propagation_stats.json   full stats block (also embedded in the eval report)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"

DEFAULT_HUB_PERCENTILE = 99.0
DEFAULT_PROPAGATION_WEIGHT = 0.5
TEST_PARTITIONS = ("overlap", "pure_test")  # per mg_temporal_pipeline.py naming


def sep(t: str = "") -> None:
    width = 78
    print(f"\n{'─' * 4} {t} {'─' * max(0, width - len(t) - 6)}" if t else "─" * width)


def load_graph(path: Path):
    sep("Load MulDiGraph.pkl")
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"  Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}")
    return G


def compute_degrees(G) -> dict[str, int]:
    sep("Compute degree (in+out) for every node")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    degree = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}
    vals = np.array(list(degree.values()), dtype=np.int64)
    print(f"  degree stats: min={vals.min()} median={int(np.median(vals))} "
          f"mean={vals.mean():.2f} max={vals.max()}")
    return degree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable-label-propagation", action="store_true",
                     help="Actually assign propagated_1hop / fold it into isp_augmented. "
                          "Off by default: candidate stats are still computed and logged, "
                          "but label_source stays ground_truth/benign only and "
                          "isp_augmented == isp.")
    ap.add_argument("--propagation-weight", type=float, default=DEFAULT_PROPAGATION_WEIGHT,
                     help="Soft-label sample weight for propagated_1hop examples in the loss "
                          "(< 1.0, never equal to ground-truth confidence).")
    ap.add_argument("--hub-percentile", type=float, default=DEFAULT_HUB_PERCENTILE,
                     help="Degree percentile above which a node is treated as a hub/exchange "
                          "and excluded from propagation (both as source and as candidate).")
    args = ap.parse_args()

    print("=" * 78)
    print("  mg_propagate_labels.py — controlled 1-hop phishing label propagation")
    print("=" * 78)
    print(f"  enable_label_propagation = {args.enable_label_propagation}")
    print(f"  propagation_weight       = {args.propagation_weight}")
    print(f"  hub_percentile           = {args.hub_percentile}")
    if args.propagation_weight >= 1.0:
        raise ValueError("--propagation-weight must be < 1.0 (soft label, never equal to ground truth)")

    sep("Load split outputs from mg_temporal_pipeline.py")
    with open(SPLIT_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    n_seeds_original = sum(labels.values())
    print(f"  Ground-truth phishing seeds (isp==1): {n_seeds_original:,}")

    G = load_graph(MG_PATH)
    degree = compute_degrees(G)

    seeds = [a for a, v in labels.items() if v == 1]
    assert len(seeds) == n_seeds_original

    hub_threshold = float(np.percentile(list(degree.values()), args.hub_percentile))
    print(f"\n  Hub threshold (degree > {args.hub_percentile:.0f}th pct) = {hub_threshold:.1f}")

    # ---- Hub guard on SEEDS: a hub-like seed does not propagate outward ----
    sep("1. Hub guard on seeds (do not propagate through hot-wallet-like phishing accounts)")
    seeds_set = set(seeds)
    hub_seeds = {a for a in seeds if degree.get(a, 0) > hub_threshold}
    propagating_seeds = [a for a in seeds if a not in hub_seeds]
    print(f"  Seeds excluded as hubs (do not propagate): {len(hub_seeds):,}")
    print(f"  Seeds used as propagation sources: {len(propagating_seeds):,}")

    # ---- Directionality: OUT-edges only from propagating seeds ----------
    sep("2. Collect 1-hop OUT-neighbors (candidates) + IN-neighbors (victims)")
    candidates_raw: set[str] = set()
    for s in propagating_seeds:
        for _, v in G.out_edges(s):
            if v not in seeds_set:
                candidates_raw.add(v)
    n_candidates_before_filter = len(candidates_raw)
    print(f"  Candidates before hub filter (OUT-neighbors of non-hub seeds, seeds excluded): "
          f"{n_candidates_before_filter:,}")

    victims: set[str] = set()
    for s in seeds:  # victim tagging is informational; not gated by the hub guard on seeds
        for u, _ in G.in_edges(s):
            if u not in seeds_set:
                victims.add(u)
    print(f"  Victims (senders TO any seed, IN-edge direction, never used as positive): {len(victims):,}")

    # ---- Hub guard on CANDIDATES: exclude exchange-like out-neighbors ----
    sep("3. Hub guard on candidates (exclude hot-wallet/exchange out-neighbors)")
    hub_candidates = {a for a in candidates_raw if degree.get(a, 0) > hub_threshold}
    candidates_after_hub_filter = candidates_raw - hub_candidates
    print(f"  Candidates excluded as hubs: {len(hub_candidates):,}")
    print(f"  Candidates remaining after hub filter: {len(candidates_after_hub_filter):,}")

    # ---- Leakage guard: propagated_1hop ONLY applies to 'train' partition
    sep("4. Restrict to 'train' partition only (no val/overlap/pure_test leakage)")
    train_accounts = {a for a, p in partition.items() if p == "train"}
    candidates_in_train = candidates_after_hub_filter & train_accounts
    n_dropped_not_train = len(candidates_after_hub_filter) - len(candidates_in_train)
    print(f"  Candidates dropped (not in 'train' partition): {n_dropped_not_train:,}")
    print(f"  Final propagated_1hop candidates (train-only): {len(candidates_in_train):,}")

    test_like = {a for a, p in partition.items() if p in TEST_PARTITIONS}
    val_like = {a for a, p in partition.items() if p == "val"}
    leak_check = candidates_in_train & (val_like | test_like)
    assert not leak_check, f"LEAKAGE: {len(leak_check)} propagated_1hop accounts fell into val/test: {list(leak_check)[:5]}"
    print("  [PASS] propagated_1hop_accounts ∩ (val ∪ overlap ∪ pure_test) == ∅")

    # ---- Build label_source / isp_augmented / propagation_weight ---------
    sep("5. Build label_source / isp_augmented / propagation_weight (soft label)")
    propagated_final = candidates_in_train if args.enable_label_propagation else set()

    label_source: dict[str, str] = {}
    isp_augmented: dict[str, int] = {}
    propagation_weight: dict[str, float] = {}
    for a in G.nodes():
        if labels.get(a, 0) == 1:
            label_source[a] = "ground_truth"
            isp_augmented[a] = 1
            propagation_weight[a] = 1.0
        elif a in propagated_final:
            label_source[a] = "propagated_1hop"
            isp_augmented[a] = 1
            propagation_weight[a] = args.propagation_weight
        else:
            label_source[a] = "benign"
            isp_augmented[a] = labels.get(a, 0)  # == 0, kept explicit for clarity
            propagation_weight[a] = 1.0

    is_victim = {a: 1 for a in victims if label_source.get(a) != "ground_truth"}

    # ---- Assertions (new_split.md §4) ------------------------------------
    sep("6. Assertions")
    gt_addrs = {a for a, s in label_source.items() if s == "ground_truth"}
    prop_addrs = {a for a, s in label_source.items() if s == "propagated_1hop"}
    assert gt_addrs.isdisjoint(prop_addrs), "an address is both ground_truth and propagated_1hop"
    print("  [PASS] no address is both label_source=ground_truth and label_source=propagated_1hop")
    assert prop_addrs.isdisjoint(val_like | test_like), "propagated_1hop leaked into val/test"
    print("  [PASS] propagated_1hop ∩ (val ∪ test) == ∅")
    assert len(gt_addrs) == n_seeds_original, f"seed count changed: {len(gt_addrs)} != {n_seeds_original}"
    print(f"  [PASS] ground-truth seed count preserved: {len(gt_addrs):,} == {n_seeds_original:,}")

    # ---- Stats (mục 6, mandatory logging) ---------------------------------
    sep("7. Summary stats")
    train_pos_before = sum(labels.get(a, 0) for a in train_accounts)
    train_pos_after = sum(isp_augmented.get(a, 0) for a in train_accounts)
    n_train = len(train_accounts)
    rate_before = train_pos_before / n_train if n_train else 0.0
    rate_after = train_pos_after / n_train if n_train else 0.0

    stats = {
        "enable_label_propagation": args.enable_label_propagation,
        "propagation_weight": args.propagation_weight,
        "hub_percentile": args.hub_percentile,
        "hub_degree_threshold": hub_threshold,
        "n_seeds_original": n_seeds_original,
        "n_seeds_hub_excluded": len(hub_seeds),
        "n_seeds_propagating": len(propagating_seeds),
        "n_candidates_before_hub_filter": n_candidates_before_filter,
        "n_candidates_hub_excluded": len(hub_candidates),
        "n_candidates_after_hub_filter": len(candidates_after_hub_filter),
        "n_candidates_dropped_not_train_partition": n_dropped_not_train,
        "n_final_propagated_1hop": len(propagated_final),
        "n_victims": len(is_victim),
        "train_partition_size": n_train,
        "train_positive_count_before": train_pos_before,
        "train_positive_count_after": train_pos_after,
        "train_positive_rate_before": rate_before,
        "train_positive_rate_after": rate_after,
        "train_positive_rate_increase_factor": (rate_after / rate_before) if rate_before else None,
    }
    for k, v in stats.items():
        print(f"  {k:<42}: {v}")

    sep("Save outputs")

    def save_pkl(obj, name):
        p = SPLIT_DIR / name
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {p}")

    save_pkl(label_source, "label_source.pkl")
    save_pkl(is_victim, "is_victim.pkl")
    save_pkl(isp_augmented, "isp_augmented.pkl")
    save_pkl(propagation_weight, "propagation_weight.pkl")
    with open(SPLIT_DIR / "propagation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {SPLIT_DIR / 'propagation_stats.json'}")

    sep("DONE")


if __name__ == "__main__":
    main()
