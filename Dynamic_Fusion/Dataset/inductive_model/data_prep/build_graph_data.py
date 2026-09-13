"""
Task A2 (new_propose.md giai đoạn A): chuyển adj_train.npz / adj_inference.npz
(scipy CSR) sang torch_geometric.data.Data -- chỉ đổi định dạng lưu trữ, giữ
nguyên phân biệt train-time graph (<= T_cutoff) vs inference-time graph
(đầy đủ). Không đối xứng hoá / không lọc lại cạnh ở bước này.

Yêu cầu thư viện: torch_geometric (xem A3_library_decision.md để biết lý do
chọn PyG thay vì DGL).
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Dynamic_Fusion
PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "Dataset_MG"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ADJ_FILES = {
    "train": PREPROC_DIR / "adj_train.npz",
    "inference": PREPROC_DIR / "adj_inference.npz",
}
NODE_FEATURES_PT = DATA_DIR / "node_features_all23.pt"


def csr_to_pyg_data(csr: sp.csr_matrix, x: torch.Tensor) -> Data:
    """QUY UOC HUONG CANH (bug phat hien + sua 2026-09-06, xem graph_data.meta.json
    field 'edge_direction_convention' de biet ly do day du -- DA SUA LAI LAN 2,
    dao nguoc quyet dinh lan 1): adj_csr[i,j] != 0 nghia la "i gui tien cho j"
    (row=from_addr, col=to_addr -- xem mg_build_adjacency.py). Do thi CO HUONG,
    dung quy uoc INCOMING: moi target t aggregate tu cac source s tro VAO t
    (s->t, dung [coo.row, coo.col] KHONG doi -- giu nguyen ban goc). PyG
    MessagePassing mac dinh: edge_index[0]=nguon (x_j trong message()),
    edge_index[1]=dich (noi aggregate) -- [coo.row, coo.col] cho dung y nghia
    nay (t=col nhan tin tu s=row, tuc nhan tu nguoi da gui tien cho no).
    model/graph_sampling.py::sample_union_subgraph (B2/B3/pretrain) duoc SUA O
    PHIA GOI (LabelAwareNeighborSampler, build_ego_subgraph, build_case_b_subgraph,
    pretrain_graph_encoder.py deu transpose adj_csr truoc khi dua vao) de khop
    dung quy uoc incoming nay, KHONG sua o day nua."""
    coo = csr.tocoo()
    edge_index = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    edge_weight = torch.from_numpy(coo.data.astype(np.float32))
    n_nodes = csr.shape[0]
    assert x.shape[0] == n_nodes, f"feature tensor has {x.shape[0]} rows, adjacency has {n_nodes} nodes"
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=n_nodes)


def main():
    if not NODE_FEATURES_PT.exists():
        raise FileNotFoundError(
            f"{NODE_FEATURES_PT} not found -- run build_node_features.py (A1) first."
        )
    x = torch.load(NODE_FEATURES_PT)
    print(f"Loaded node features {tuple(x.shape)} from {NODE_FEATURES_PT.name}")

    stats = {}
    for split, path in ADJ_FILES.items():
        csr = sp.load_npz(path)
        print(f"[{split}] loaded {path.name}: shape={csr.shape}, nnz={csr.nnz}")
        data = csr_to_pyg_data(csr, x)

        out_path = DATA_DIR / f"graph_{split}.pt"
        torch.save(data, out_path)
        print(f"[{split}] saved PyG Data -> {out_path}")
        print(f"[{split}] {data}")

        stats[split] = {
            "num_nodes": data.num_nodes,
            "num_edges": int(data.edge_index.shape[1]),
            "edge_weight_min": float(data.edge_weight.min()),
            "edge_weight_max": float(data.edge_weight.max()),
            "source_npz": str(path.relative_to(REPO_ROOT)),
        }

    # Sanity check preserved from source data: train edges must be a subset
    # of inference edges (train-time graph is a prefix-in-time of the full
    # inference-time graph).
    train_csr = sp.load_npz(ADJ_FILES["train"]).tocoo()
    infer_csr = sp.load_npz(ADJ_FILES["inference"]).tocoo()
    train_pairs = set(zip(train_csr.row.tolist(), train_csr.col.tolist()))
    infer_pairs = set(zip(infer_csr.row.tolist(), infer_csr.col.tolist()))
    is_subset = train_pairs.issubset(infer_pairs)
    print(f"train edges subset of inference edges: {is_subset}")
    if not is_subset:
        raise ValueError(
            "adj_train is not a subset of adj_inference -- train/inference "
            "time-cutoff distinction from A2 would be silently broken."
        )

    meta = {
        "index_convention": "same node index space as address_to_index.pkl / node_features_all23.pt",
        "directed": True,
        "edge_weight_note": (
            "raw weights carried over unchanged from mg_graph_weight_formula.py "
            "output; wide dynamic range (0 to ~6.8e12) -- B1 must normalize/"
            "log-transform before using as GNN edge weights, not done here."
        ),
        "edge_direction_convention": {
            "fixed_date": "2026-09-06 (superseded same-day revision -- see 'revision_history')",
            "raw_adjacency_semantic": (
                "adj_csr[i,j] != 0 means account i SENT to account j "
                "(row=from_addr, col=to_addr -- mg_build_adjacency.py)."
            ),
            "pyg_edge_index_semantic": (
                "edge_index = [coo.row, coo.col] (the RAW, unflipped adjacency -- "
                "final, current convention). PyG MessagePassing default: "
                "edge_index[0]=source (x_j in message()), edge_index[1]=destination "
                "(aggregation target). Node v (appearing as edge_index[1]) aggregates "
                "from u=edge_index[0] wherever adj_csr[u,v]!=0, i.e. from accounts "
                "that SENT MONEY TO v -- v's PAYERS / true in-neighbors. This is the "
                "standard 'incoming aggregation' GNN convention: target t aggregates "
                "from source s across edges s->t (Hu et al. 2020, Heterogeneous Graph "
                "Transformer, Definition 2 / Eq.1: H[t] <- Aggregate_{s in N(t)}(...), "
                "N(t) = source nodes of t)."
            ),
            "why_this_direction": (
                "Standard GNN/HGT convention (see pyg_edge_index_semantic) -- target "
                "aggregates from incoming sources. A prior same-day revision had "
                "instead flipped this file to match model/graph_sampling.py's "
                "then-existing (payee/outgoing) convention, reasoning it would avoid "
                "retraining already-trained checkpoints; that reasoning was overridden "
                "in favor of the principled, standard convention once flagged -- "
                "model/graph_sampling.py's CALLERS were fixed instead (see "
                "'train_side_fix' below), not this file. Every checkpoint trained "
                "before 2026-09-06 (GraphMAE pretrain, E2/E2v2/E2v3, graph-only "
                "hypothesis, ablation, sweep) was trained under the OLD payee "
                "convention and needs RETRAINING to be validly evaluated under this "
                "(final) incoming/payer convention -- their existing results are "
                "stale with respect to graph direction."
            ),
            "train_side_fix": (
                "model/graph_sampling.py::_neighbors()/sample_union_subgraph() always "
                "read adj_csr's ROW for a center node (out-neighbors/payees) -- "
                "renaming edge_src/edge_dst variables alone cannot change this, since "
                "the neighbor SET fetched never changes, only which role (source vs "
                "destination) it's assigned. The actual fix: every caller now passes "
                "the TRANSPOSED adjacency (adj_csr.T.tocsr()) so the row lookup "
                "returns the center's true in-neighbors/payers instead -- see "
                "model/label_aware_sampler.py (LabelAwareNeighborSampler.__init__), "
                "model/ego_subgraph.py (build_ego_subgraph), "
                "model/case_b_subgraph.py (build_case_b_subgraph), and "
                "train_eval/pretrain_graph_encoder.py."
            ),
            "future_extension_note": (
                "If both directions (payer AND payee signal) are wanted simultaneously "
                "later, the HGT-consistent way (per the same paper, Sec 5.2: 'all "
                "relation phi have a reverse relation type phi^-1') is to add the "
                "reverse relation as an EXPLICIT second edge type, not to merge/flip "
                "within a single relation -- out of scope for this fix."
            ),
            "regression_check": (
                "data_prep/verify_edge_direction_consistency.py -- calls the real "
                "model/graph_sampling.py::_neighbors() on the transposed adjacency and "
                "asserts, for 100 random nodes (train and inference graphs), that its "
                "output exactly equals full_graph_forward's per-node source set "
                "(edge_index[0] where edge_index[1]==node). PASS 100/100 both splits "
                "as of this fix. Run again after any change to either path."
            ),
        },
        "train_subset_of_inference": is_subset,
        "splits": stats,
    }
    out_meta = DATA_DIR / "graph_data.meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata -> {out_meta}")


if __name__ == "__main__":
    main()
