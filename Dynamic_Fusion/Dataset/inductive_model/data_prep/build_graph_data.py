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
        "train_subset_of_inference": is_subset,
        "splits": stats,
    }
    out_meta = DATA_DIR / "graph_data.meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata -> {out_meta}")


if __name__ == "__main__":
    main()
