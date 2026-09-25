"""
Random node-based 7:1:2 train/val/test split for the MulDiGraph dataset,
reimplemented from the description in Section VI-A of the LMAE4Eth paper:

    "We adopt a random node-based splitting strategy on the transaction
    graph, dividing the accounts into training, validation, and test sets
    with a ratio of 7:1:2."

The official LMAE4Eth code release (github.com/lmae4eth/LMAE4Eth) references
this split via `datasets.lc_sampler` in finetune.py, but that module is not
published in the repo (verified against the live repo tree, not just this
local checkout), so the split logic itself is reconstructed here directly
from raw_data/MulDiGraph/MulDiGraph.pkl.

MulDiGraph.pkl is a networkx.MultiDiGraph with 2,973,489 nodes and
13,551,303 edges (matches Table II of the paper): each node carries an
"isp" attribute (1 = phisher, 0 = normal account) and each edge carries
"amount" and "timestamp", i.e. exactly the (v_i, d_i, tau_i) transaction
tuple of Eq. (1) (direction is implied by edge orientation: u -> v is an
outflow for u and an inflow for v).

Usage:
    python Data/split_muldigraph.py \
        --pkl raw_data/MulDiGraph/MulDiGraph.pkl \
        --out raw_data/MulDiGraph/split_idx.pt
"""
import argparse
import pickle

import numpy as np
import torch


def load_muldigraph(pkl_path):
    with open(pkl_path, "rb") as f:
        g = pickle.load(f)  # networkx.MultiDiGraph, node attr "isp" in {0, 1}
    return g


def build_node_table(g):
    """Deterministic address -> integer-id mapping plus aligned label array."""
    addresses = sorted(g.nodes())
    labels = np.array([g.nodes[addr].get("isp", 0) for addr in addresses], dtype=np.int64)
    return addresses, labels


def random_node_split(num_nodes, ratios=(0.7, 0.1, 0.2), seed=42):
    """Random node-based split, ratio train:val:test = 7:1:2 (paper default)."""
    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios must sum to 1"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)

    n_train = int(round(num_nodes * ratios[0]))
    n_val = int(round(num_nodes * ratios[1]))
    # whatever remains goes to test, so rounding never drops a node
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return {
        "train": torch.from_numpy(train_idx.copy()).long(),
        "valid": torch.from_numpy(val_idx.copy()).long(),
        "test": torch.from_numpy(test_idx.copy()).long(),
    }


def stratified_node_split(labels, ratios=(0.7, 0.1, 0.2), seed=42):
    """Random split done independently within each label class, so every
    split keeps (approximately) the same phisher/normal ratio as the full
    dataset. The paper's text does not mention stratification (it only says
    "random node-based"), but with only 1,165 phishers out of ~2.97M nodes a
    plain global shuffle can noticeably over/under-represent the positive
    class in val/test across seeds -- this is an alternative for a more
    stable evaluation split."""
    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios must sum to 1"

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    splits = {"train": [], "valid": [], "test": []}

    for c in np.unique(labels):
        class_idx = np.flatnonzero(labels == c)
        perm = rng.permutation(class_idx)

        n_train = int(round(len(perm) * ratios[0]))
        n_val = int(round(len(perm) * ratios[1]))

        splits["train"].append(perm[:n_train])
        splits["valid"].append(perm[n_train:n_train + n_val])
        splits["test"].append(perm[n_train + n_val:])

    return {
        name: torch.from_numpy(rng.permutation(np.concatenate(parts))).long()
        for name, parts in splits.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default="raw_data/MulDiGraph/MulDiGraph.pkl")
    parser.add_argument("--out", default="raw_data/MulDiGraph/split_idx.pt")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2),
                         metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true",
                         help="split within each label class so the phisher ratio "
                              "is preserved across train/val/test (paper itself "
                              "only specifies a plain random node split)")
    args = parser.parse_args()

    g = load_muldigraph(args.pkl)
    addresses, labels = build_node_table(g)
    if args.stratified:
        split_idx = stratified_node_split(labels, ratios=tuple(args.ratios), seed=args.seed)
    else:
        split_idx = random_node_split(len(addresses), ratios=tuple(args.ratios), seed=args.seed)

    torch.save({
        "addresses": addresses,           # index i -> Ethereum address (str)
        "labels": torch.from_numpy(labels).long(),  # index i -> 0/1 (phisher)
        "split_idx": split_idx,           # {"train"/"valid"/"test": LongTensor of node ids}
    }, args.out)

    print(f"total nodes: {len(addresses)}")
    for name in ("train", "valid", "test"):
        idx = split_idx[name].numpy()
        n_pos = int(labels[idx].sum())
        print(f"{name:5s}: {len(idx):>8d} nodes ({len(idx) / len(addresses):.1%})  phishers={n_pos}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
