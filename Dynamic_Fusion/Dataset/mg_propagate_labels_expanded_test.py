"""
mg_propagate_labels_expanded_test.py
=====================================
Basic 1-hop label propagation from the 1,165 verified phishing seeds
(isp==1) in MulDiGraph.pkl. Finds every account 1 hop away (out-neighbor)
from a seed, labels it as propagated fraud, and lets it fall into whatever
partition (train/val/overlap/pure_test) it already belongs to -- including
the test set, on purpose, to grow the number of fraud-labeled accounts
available there.

Ground truth (`isp`) is not modified. The propagated result is a separate
column, `isp_expanded`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"

PARTITIONS = ("train", "val", "overlap", "pure_test")


def main():
    with open(SPLIT_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)

    seeds = {a for a, v in labels.items() if v == 1}
    print(f"Verified phishing seeds: {len(seeds):,}")

    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"Graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    # 1-hop propagation: every out-neighbor of a seed, not already a seed
    propagated = set()
    for s in seeds:
        for _, v in G.out_edges(s):
            if v not in seeds:
                propagated.add(v)
    print(f"Propagated (1-hop) accounts: {len(propagated):,}")

    isp_expanded = dict(labels)
    for a in propagated:
        isp_expanded[a] = 1

    # split into partitions, including test
    stats = {}
    for p in PARTITIONS:
        accounts_p = [a for a, pp in partition.items() if pp == p]
        n = len(accounts_p)
        before = sum(labels.get(a, 0) for a in accounts_p)
        after = sum(isp_expanded.get(a, 0) for a in accounts_p)
        stats[p] = {"n_accounts": n, "fraud_before": before, "fraud_after": after}

    test_before = stats["overlap"]["fraud_before"] + stats["pure_test"]["fraud_before"]
    test_after = stats["overlap"]["fraud_after"] + stats["pure_test"]["fraud_after"]
    test_n = stats["overlap"]["n_accounts"] + stats["pure_test"]["n_accounts"]

    print("\nPartition | accounts | fraud before | fraud after")
    for p in PARTITIONS:
        s = stats[p]
        print(f"{p:10} | {s['n_accounts']:>9,} | {s['fraud_before']:>13,} | {s['fraud_after']:>12,}")
    print(f"{'test':10} | {test_n:>9,} | {test_before:>13,} | {test_after:>12,}")
    print(f"{'total':10} | {len(labels):>9,} | {sum(labels.values()):>13,} | {sum(isp_expanded.values()):>12,}")

    stats["test_combined"] = {"n_accounts": test_n, "fraud_before": test_before, "fraud_after": test_after}
    stats["total"] = {"n_accounts": len(labels), "fraud_before": sum(labels.values()), "fraud_after": sum(isp_expanded.values())}

    with open(SPLIT_DIR / "isp_expanded.pkl", "wb") as f:
        pickle.dump(isp_expanded, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(SPLIT_DIR / "propagation_expanded_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {SPLIT_DIR / 'isp_expanded.pkl'}")
    print(f"Saved: {SPLIT_DIR / 'propagation_expanded_stats.json'}")


if __name__ == "__main__":
    main()
