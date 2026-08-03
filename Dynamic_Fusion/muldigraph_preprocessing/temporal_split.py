"""
temporal_split.py  —  Temporal train/test split for MulDiGraph (PyG output).

Two split modes
---------------
overlap       : bridge nodes (T_first ≤ T_cutoff < T_last) → both masks True
zero_leakage  : bridge nodes → test only  (eliminates future-label leakage)

Two PyG Data objects
---------------------
train_data     : edges with ts ≤ T_cutoff   |  y = static fraud label (masked to train nodes)
inference_data : ALL edges (full graph)      |  y = static fraud label (masked to all nodes)

Label definition
----------------
fraud_nodes (tag=1) : always fraud — both confirmed (isp=1) and potential (isp=0, tag=1).
                      Label is static; no temporal activation threshold.
normal_nodes (tag=0): never fraud.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# Config & Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalSplitConfig:
    cutoff_ts: float                         # Unix timestamp for train/test cut
    mode: Literal["overlap", "zero_leakage"]


@dataclass
class SplitResult:
    train_data: Data
    inference_data: Data
    node_to_idx: dict[str, int]             # shared address → index map
    meta: dict


# ─────────────────────────────────────────────────────────────────────────────
# Internal — edge collection
# ─────────────────────────────────────────────────────────────────────────────

def _collect_edges(G, node_to_idx: dict
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single pass over G → pre-allocated numpy arrays (no Python-list overhead).
    Edges without a timestamp are skipped.
    Returns: src [E, int64], dst [E, int64], ts [E, float64], amt [E, float32]
    """
    n_max   = G.number_of_edges()
    src_buf = np.empty(n_max, dtype=np.int64)
    dst_buf = np.empty(n_max, dtype=np.int64)
    ts_buf  = np.empty(n_max, dtype=np.float64)
    amt_buf = np.empty(n_max, dtype=np.float32)

    k = 0
    for u, v, _, d in G.edges(data=True, keys=True):
        ts = d.get("timestamp") or d.get("time")
        if ts is None:
            continue
        src_buf[k] = node_to_idx[u]
        dst_buf[k] = node_to_idx[v]
        ts_buf[k]  = float(ts)
        amt_buf[k] = float(d.get("amount", 0.0))
        k += 1

    return src_buf[:k], dst_buf[:k], ts_buf[:k], amt_buf[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Internal — vectorized timeline, masks, labels
# ─────────────────────────────────────────────────────────────────────────────

def _node_timelines(src: np.ndarray, dst: np.ndarray,
                    ts: np.ndarray, cutoff: float, n: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-node first/last timestamps using numpy scatter operations.

    Returns:
        t_first      [N] — earliest ts over all edges   (inf  for isolated nodes)
        t_last       [N] — latest  ts over all edges   (-inf for isolated nodes)
        t_last_train [N] — latest  ts for ts ≤ cutoff  (-inf if no train edge)
    """
    t_first      = np.full(n, np.inf,  dtype=np.float64)
    t_last       = np.full(n, -np.inf, dtype=np.float64)
    t_last_train = np.full(n, -np.inf, dtype=np.float64)

    np.minimum.at(t_first, src, ts)
    np.minimum.at(t_first, dst, ts)
    np.maximum.at(t_last,  src, ts)
    np.maximum.at(t_last,  dst, ts)

    tr = ts <= cutoff
    np.maximum.at(t_last_train, src[tr], ts[tr])
    np.maximum.at(t_last_train, dst[tr], ts[tr])

    return t_first, t_last, t_last_train


def _assign_masks(t_first: np.ndarray, t_last: np.ndarray,
                  mode: str, cutoff: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    overlap:
        train_mask = T_first ≤ cutoff    (node has ≥1 edge in train period)
        test_mask  = T_last  > cutoff    (node has ≥1 edge in test period)
        → bridge nodes appear in BOTH masks

    zero_leakage:
        train_mask = T_last  ≤ cutoff    (TH1 — fully past nodes only)
        test_mask  = T_last  > cutoff    (TH2 + TH3 — future or bridge)
        → no overlap by construction

    Isolated nodes (t_last = -inf): excluded from both masks in both modes.
    """
    has_edge = t_last > -np.inf
    if mode == "overlap":
        return (t_first <= cutoff), (t_last > cutoff)
    else:
        return (t_last <= cutoff) & has_edge, (t_last > cutoff)


def _assign_labels(t_last: np.ndarray, t_last_train: np.ndarray,
                   fraud_arr: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Static label assignment.

    y_train[n]     = fraud_arr[n]  if n has ≥1 train edge,  else 0
    y_inference[n] = fraud_arr[n]  if n has ≥1 edge at all, else 0

    fraud_arr : bool[N] — True iff node has tag=1 (confirmed or potential fraud)
    """
    in_train = t_last_train > -np.inf
    has_edge = t_last       > -np.inf
    y_train     = (fraud_arr & in_train).astype(np.int64)
    y_inference = (fraud_arr & has_edge).astype(np.int64)
    return y_train, y_inference


def _make_data(src: np.ndarray, dst: np.ndarray,
               ts: np.ndarray, amt: np.ndarray,
               n_nodes: int, y: np.ndarray,
               train_mask: np.ndarray, test_mask: np.ndarray) -> Data:
    return Data(
        x          = torch.ones(n_nodes, 1, dtype=torch.float32),
        edge_index = torch.from_numpy(np.stack([src, dst])),
        edge_attr  = torch.from_numpy(np.stack([ts.astype(np.float32), amt], axis=1)),
        y          = torch.from_numpy(y),
        train_mask = torch.from_numpy(train_mask),
        test_mask  = torch.from_numpy(test_mask),
        num_nodes  = n_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_temporal_split(
    G,
    fraud_nodes: set,
    config: TemporalSplitConfig,
) -> SplitResult:
    """
    Parameters
    ----------
    G           : networkx MultiDiGraph
    fraud_nodes : set of node addresses with tag=1 (confirmed + potential fraud)
    config      : TemporalSplitConfig

    Returns
    -------
    SplitResult with train_data, inference_data, node_to_idx, meta
    """
    cutoff = config.cutoff_ts

    # 1. Stable node index — shared between both Data objects
    all_nodes   = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes     = len(all_nodes)
    print(f"Nodes: {n_nodes:,}")

    # 2. Collect all edges → typed numpy arrays (single pass)
    print("Collecting edges ...")
    src, dst, ts_all, amt = _collect_edges(G, node_to_idx)
    print(f"Edges with timestamps: {len(ts_all):,}")

    # 3. Per-node timelines (vectorized scatter)
    print("Computing timelines ...")
    t_first, t_last, t_last_train = _node_timelines(src, dst, ts_all, cutoff, n_nodes)

    # 4. Node masks
    print(f"Assigning masks (mode={config.mode!r}) ...")
    train_mask, test_mask = _assign_masks(t_first, t_last, config.mode, cutoff)

    # 5. Build fraud label array indexed by node_to_idx
    fraud_arr = np.zeros(n_nodes, dtype=bool)
    for node in fraud_nodes:
        if node in node_to_idx:
            fraud_arr[node_to_idx[node]] = True

    # 6. Labels (static: fraud_arr masked to nodes present in each period)
    y_train, y_inference = _assign_labels(t_last, t_last_train, fraud_arr)

    # 7. Build PyG Data objects
    print("Building PyG Data objects ...")
    tr_sel = ts_all <= cutoff

    train_data = _make_data(
        src[tr_sel], dst[tr_sel], ts_all[tr_sel], amt[tr_sel],
        n_nodes, y_train,
        train_mask, np.zeros(n_nodes, dtype=bool),
    )
    inference_data = _make_data(
        src, dst, ts_all, amt,
        n_nodes, y_inference,
        train_mask, test_mask,
    )

    # 8. Meta statistics
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    meta = {
        "mode":             config.mode,
        "cutoff_ts":        cutoff,
        "n_nodes":          n_nodes,
        "n_train":          n_tr,
        "n_test":           n_te,
        "n_overlap":        int((train_mask & test_mask).sum()),
        "n_train_fraud":    int(y_train[train_mask].sum()),
        "n_test_fraud":     int(y_inference[test_mask].sum()),
        "n_train_edges":    int(tr_sel.sum()),
        "n_total_edges":    len(ts_all),
        "fraud_rate_train": int(y_train[train_mask].sum()) / n_tr if n_tr else 0.0,
        "fraud_rate_test":  int(y_inference[test_mask].sum()) / n_te if n_te else 0.0,
    }

    return SplitResult(
        train_data     = train_data,
        inference_data = inference_data,
        node_to_idx    = node_to_idx,
        meta           = meta,
    )


def print_meta(result: SplitResult) -> None:
    m = result.meta
    print(f"Mode          : {m['mode']}")
    print(f"Nodes total   : {m['n_nodes']:,}")
    print(f"Train nodes   : {m['n_train']:,}  ({m['n_train_fraud']:,} fraud,"
          f" {m['fraud_rate_train']*100:.4f}%)")
    print(f"Test  nodes   : {m['n_test']:,}   ({m['n_test_fraud']:,} fraud,"
          f" {m['fraud_rate_test']*100:.4f}%)")
    print(f"Overlap nodes : {m['n_overlap']:,}")
    print(f"Train edges   : {m['n_train_edges']:,}")
    print(f"Total edges   : {m['n_total_edges']:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    MG_PATH   = "/home/thegreatestrang/FraudGT-re/FraudGT/FraudGT_pipeline/data/MG/MulDiGraph.pkl"
    CORR_PATH = "/home/thegreatestrang/FraudGT-re/FraudGT/FraudGT_pipeline/data/MG/transactions7_corrected.pkl"
    CUTOFF    = 1_527_988_904.0   # 2018-06-03 01:21:44 UTC  (80/20 edge-count split)

    print("Loading MulDiGraph ...")
    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)

    print("Loading transactions7_corrected.pkl ...")
    with open(CORR_PATH, "rb") as f:
        accounts = pickle.load(f)
    fraud_nodes = frozenset(addr for addr, dl in accounts.items() if dl[0].get("tag", 0) == 1)
    del accounts
    print(f"Fraud nodes: {len(fraud_nodes):,}")

    for mode in ("overlap", "zero_leakage"):
        print(f"\n{'='*60}")
        config = TemporalSplitConfig(cutoff_ts=CUTOFF, mode=mode)
        result = create_temporal_split(G, fraud_nodes, config)
        print_meta(result)
        print(f"train_data     : {result.train_data}")
        print(f"inference_data : {result.inference_data}")
