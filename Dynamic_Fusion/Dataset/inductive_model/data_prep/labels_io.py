"""
Helper dùng chung: load labels.pkl / partition.pkl (keyed theo address) và
reindex về đúng không gian address_to_index.pkl (cùng index space với
node_features_all23.pt / graph_*.pt -- xem io_utils.py).
"""
import pickle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Dynamic_Fusion
PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "Dataset_MG"


def _load_addr_to_idx() -> dict:
    with open(PREPROC_DIR / "address_to_index.pkl", "rb") as f:
        return pickle.load(f)


def load_labels() -> torch.Tensor:
    """LongTensor [N]; 1 = phisher xác nhận (isp==1), 0 = còn lại -- đúng
    label_scheme 'confirmed_phishing_only' ghi trong split_config.json."""
    addr_to_idx = _load_addr_to_idx()
    with open(PREPROC_DIR / "labels.pkl", "rb") as f:
        labels_by_addr = pickle.load(f)
    n = len(addr_to_idx)
    labels = torch.zeros(n, dtype=torch.long)
    for addr, lbl in labels_by_addr.items():
        labels[addr_to_idx[addr]] = int(lbl)
    return labels


def load_partition() -> list:
    """list[str] độ dài N, partition[i] in {'train','val','overlap','pure_test'}."""
    addr_to_idx = _load_addr_to_idx()
    with open(PREPROC_DIR / "partition.pkl", "rb") as f:
        partition_by_addr = pickle.load(f)
    n = len(addr_to_idx)
    partition = [None] * n
    for addr, split in partition_by_addr.items():
        partition[addr_to_idx[addr]] = split
    return partition
