"""
mg_build_adjacency.py
======================
Step 3 — sparse adjacency matrix construction for MulDiGraph, leakage-fixed.

Replaces Dataset/adjust_matrix.py's approach (dense np.zeros((N,N)) built
over the FULL account set with no train/test boundary — infeasible anyway
at MulDiGraph's 2,973,489-node scale: a dense N x N float32 matrix would
need ~35 TB of RAM). Builds directly as a sparse matrix instead, and adds
the missing train/inference boundary:

  3a. TRAIN adjacency — edges with timestamp <= T_cutoff (the global 80th-
      percentile cutoff from mg_temporal_pipeline.py) ONLY. This includes
      the pre-cutoff edges of 'overlap' accounts too (they're valid
      message-passing context, just not loss-bearing — see
      mg_build_examples.py). This is what the model may use for GCN
      message passing during training.
  3b. INFERENCE adjacency — the FULL graph (all edges, no time filter).
      This is intentional and correct under the transductive setting the
      brief specifies (Step 2e/3b): the GNN is allowed to see the full
      graph STRUCTURE at inference time. What must never happen is an
      overlap/val/pure_test account's LABEL leaking into the training
      loss — that is handled separately in mg_build_examples.py (only
      'train'-partition accounts get an InputExample in train_examples)
      and is a different concern from the adjacency matrix.

Both matrices reuse the exact Eq.1-3 weight formula (mg_graph_weight_formula.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

from mg_graph_weight_formula import build_edge_weights

BASE_DIR = Path(__file__).resolve().parent.parent
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OUT_DIR = SPLIT_DIR


def sep(title: str) -> None:
    print(f"\n{'─' * 4} {title} {'─' * max(0, 74 - len(title))}")


def load_edges_and_index() -> tuple[pd.DataFrame, dict]:
    sep("Load MulDiGraph.pkl -> edge DataFrame + address_to_index")
    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    address_to_index = {n: i for i, n in enumerate(nodes)}
    print(f"  Nodes: {len(nodes):,}")

    rows = []
    for u, v, d in G.edges(data=True):
        rows.append((u, v, float(d.get("amount", 0.0)), float(d.get("timestamp"))))
    edges_df = pd.DataFrame(rows, columns=["from_addr", "to_addr", "amount", "timestamp"])
    print(f"  Edges: {len(edges_df):,}")
    return edges_df, address_to_index


def to_sparse(agg: pd.DataFrame, address_to_index: dict, n: int) -> coo_matrix:
    rows = agg["from_addr"].map(address_to_index).to_numpy()
    cols = agg["to_addr"].map(address_to_index).to_numpy()
    data = agg["weight"].to_numpy(dtype=np.float32)
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def main():
    print("=" * 78)
    print("  mg_build_adjacency.py — sparse, leakage-fixed adjacency (Step 3)")
    print("=" * 78)

    with open(SPLIT_DIR / "split_config.json") as f:
        import json

        config = json.load(f)
    T_cutoff = config["T_cutoff"]
    print(f"  T_cutoff (80th pct, train/val boundary) = {T_cutoff:.0f}  ({config['T_cutoff_human']})")

    edges_df, address_to_index = load_edges_and_index()
    n = len(address_to_index)

    with open(OUT_DIR / "address_to_index.pkl", "wb") as f:
        pickle.dump(address_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {OUT_DIR / 'address_to_index.pkl'}")

    # ---- 3a. TRAIN adjacency: edges with timestamp <= T_cutoff only -------
    sep("3a. TRAIN adjacency (timestamp <= T_cutoff only)")
    train_edges = edges_df[edges_df["timestamp"] <= T_cutoff]
    print(f"  Train-period edges: {len(train_edges):,} / {len(edges_df):,} "
          f"({100 * len(train_edges) / len(edges_df):.2f}%)")
    train_agg = build_edge_weights(train_edges)
    train_adj = to_sparse(train_agg, address_to_index, n)
    save_npz(OUT_DIR / "adj_train.npz", train_adj.tocsr())
    print(f"  Train adjacency: shape={train_adj.shape}  nnz={train_adj.nnz:,}")
    print(f"  Saved: {OUT_DIR / 'adj_train.npz'}")

    # ---- 3b. INFERENCE adjacency: full graph, all edges --------------------
    sep("3b. INFERENCE adjacency (full graph, transductive — all edges)")
    full_agg = build_edge_weights(edges_df)
    full_adj = to_sparse(full_agg, address_to_index, n)
    save_npz(OUT_DIR / "adj_inference.npz", full_adj.tocsr())
    print(f"  Inference adjacency: shape={full_adj.shape}  nnz={full_adj.nnz:,}")
    print(f"  Saved: {OUT_DIR / 'adj_inference.npz'}")
    print(
        "\n  NOTE: adj_inference intentionally includes post-T_cutoff structure — this is\n"
        "  expected under the transductive GNN setting (Step 3b). The leakage this\n"
        "  fix targets is LABELS/LOSS crossing the boundary (handled in\n"
        "  mg_build_examples.py), not the GNN seeing later graph structure."
    )

    sep("Spot-check")
    print(f"  Train nnz / Inference nnz = {train_adj.nnz:,} / {full_adj.nnz:,} "
          f"({100 * train_adj.nnz / full_adj.nnz:.2f}%)")


if __name__ == "__main__":
    main()
