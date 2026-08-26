"""
Task A1 (new_propose.md giai đoạn A): chuyển
features_output_all23_MG_fullscale.csv -> tensor [N, 23] float32, index hoá
theo đúng không gian address_to_index.pkl (chung với adj_train.npz /
adj_inference.npz) để làm node feature input cho GraphSAGE/GAT ở giai đoạn B.

Không giả định thứ tự dòng CSV trùng address_to_index (dù đã kiểm tra thủ
công là trùng) -- reindex tường minh theo địa chỉ để không lặp lại lớp lỗi
vocab-order đã từng gây sập F1 ở tri_model.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Dynamic_Fusion
RAW_CSV = REPO_ROOT / "raw_data" / "MulDiGraph" / "features_output_all23_MG_fullscale.csv"
ADDR2IDX_PKL = REPO_ROOT / "data" / "preprocessed" / "Dataset_MG" / "address_to_index.pkl"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ADDR2IDX_PKL, "rb") as f:
        addr_to_idx: dict = pickle.load(f)
    n_nodes = len(addr_to_idx)
    print(f"address_to_index: {n_nodes} accounts")

    df = pd.read_csv(RAW_CSV)
    feature_cols = [c for c in df.columns if c != "node"]
    assert len(feature_cols) == 23, f"expected 23 feature columns, got {len(feature_cols)}"
    print(f"CSV: {len(df)} rows, {len(feature_cols)} feature columns")

    row_idx = df["node"].map(addr_to_idx)
    n_missing = row_idx.isna().sum()
    if n_missing:
        raise ValueError(
            f"{n_missing} addresses in the feature CSV are not present in "
            f"address_to_index.pkl -- feature/adjacency vocab mismatch."
        )
    row_idx = row_idx.astype(np.int64).to_numpy()

    if len(row_idx) != n_nodes or set(row_idx.tolist()) != set(range(n_nodes)):
        raise ValueError(
            "CSV addresses do not form a full permutation of address_to_index "
            "indices (missing accounts or duplicates) -- cannot build a dense "
            "[N, 23] feature tensor safely."
        )

    values = df[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite (NaN/Inf) values found in feature CSV.")

    feature_matrix = np.empty((n_nodes, len(feature_cols)), dtype=np.float32)
    feature_matrix[row_idx] = values

    feature_tensor = torch.from_numpy(feature_matrix)
    out_pt = OUT_DIR / "node_features_all23.pt"
    torch.save(feature_tensor, out_pt)

    meta = {
        "num_nodes": n_nodes,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "source_csv": str(RAW_CSV.relative_to(REPO_ROOT)),
        "index_map_source": str(ADDR2IDX_PKL.relative_to(REPO_ROOT)),
        "index_convention": (
            "row i of node_features_all23.pt corresponds to the account whose "
            "address_to_index.pkl value is i -- same index space as "
            "adj_train.npz / adj_inference.npz / labels.pkl / partition.pkl."
        ),
    }
    out_meta = OUT_DIR / "node_features_all23.meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved tensor {tuple(feature_tensor.shape)} -> {out_pt}")
    print(f"Saved metadata -> {out_meta}")
    print("Per-column min/max (sanity check):")
    for j, col in enumerate(feature_cols):
        print(f"  {col:28s} min={feature_matrix[:, j].min():.4f} max={feature_matrix[:, j].max():.4f}")


if __name__ == "__main__":
    main()
