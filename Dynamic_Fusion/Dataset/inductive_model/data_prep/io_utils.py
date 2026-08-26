"""
Shared loaders for artifacts produced by A1 (build_node_features.py) and A2
(build_graph_data.py). torch>=2.6 defaults torch.load(weights_only=True),
which rejects torch_geometric.data.Data pickles -- centralize the
weights_only=False loads here instead of repeating the gotcha in every
downstream script (B1/B3/E*).
"""
from pathlib import Path

import torch
from torch_geometric.data import Data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_node_features() -> torch.Tensor:
    return torch.load(DATA_DIR / "node_features_all23.pt", weights_only=False)


def load_graph(split: str) -> Data:
    assert split in ("train", "inference"), split
    return torch.load(DATA_DIR / f"graph_{split}.pt", weights_only=False)
