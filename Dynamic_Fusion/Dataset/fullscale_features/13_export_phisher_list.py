"""
13_export_phisher_list.py
=========================
Exports the project's ground-truth phishing accounts to a plain text file:

    raw_data/MulDiGraph/phisher_account_muldi.txt

Source is `labels.pkl` (built by mg_temporal_pipeline.py from MulDiGraph.pkl's
own `isp` node attribute), i.e. the 1,165 confirmed phishing accounts that every
stage of this project trains and scores against.

This REPLACES `phisher_accounts.txt` (5,480 addresses), which is not sufficiently
verified for this dataset and is no longer read by any code path here. The two
disagree substantially -- overlap 963, only-txt 4,517, only-isp 202 -- and using
the .txt put a label column in the feature table that contradicted the labels the
model is trained and scored against. See fullscale_feature_extraction_report.md §12.

Format matches the old file exactly: one lowercase address per line, LF, no
header, so anything that consumed phisher_accounts.txt can consume this.

Usage: python3 13_export_phisher_list.py
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SPLIT_DIR = BASE / "data/preprocessed/Dataset_MG"
OUT = BASE / "raw_data/MulDiGraph/phisher_account_muldi.txt"


def main():
    with open(SPLIT_DIR / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)

    addrs = sorted(a for a, v in labels.items() if v == 1)
    print(f"[i] labels.pkl: {len(labels):,} accounts, {len(addrs):,} with isp == 1")

    # every exported address must actually exist as a node in the graph
    nodes = set(np.load(HERE / "arrays/nodes.npy", allow_pickle=True))
    missing = [a for a in addrs if a not in nodes]
    if missing:
        raise SystemExit(f"[✗] {len(missing)} exported addresses are not graph nodes")
    print(f"[✓] all {len(addrs):,} present as nodes in MulDiGraph.pkl")

    if len(addrs) != 1165:
        raise SystemExit(f"[✗] expected 1,165 phishing accounts, got {len(addrs):,}")

    # partition breakdown must reproduce split_stats.json
    pc = Counter(partition.get(a) for a in addrs)
    expect = {"train": 519, "val": 117, "overlap": 217, "pure_test": 312}
    got = {k: pc.get(k, 0) for k in expect}
    print(f"[i] per partition: {got}")
    if got != expect:
        raise SystemExit(f"[✗] partition breakdown {got} != split_stats.json {expect}")
    print(f"[✓] partition breakdown matches split_stats.json")

    # sanity: lowercase, no duplicates, no blanks
    assert all(a == a.lower() and a.strip() == a and a for a in addrs), "malformed address"
    assert len(set(addrs)) == len(addrs), "duplicate address"

    OUT.write_text("\n".join(addrs) + "\n")
    print(f"\n[✓] wrote {OUT}")
    print(f"    {len(addrs):,} lines, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
