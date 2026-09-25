"""
Lõi dùng chung cho B2 (LabelAwareNeighborSampler, train) và B3 (ego-subgraph cho
predict_account, infer): mở rộng seed_nodes ra num_layers tầng theo 1 hàm
budget_fn(center, hop) -> (budget, allow_oversample), gộp tất cả cạnh đã thăm
thành 1 subgraph union (local-reindexed) rồi forward 1 lần qua encoder N lớp
(xem gnn_encoder.py và label_aware_sampler.py về lý do union-subgraph là đủ,
không cần block song phương riêng từng tầng kiểu DGL).
"""
from typing import Callable, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

BudgetFn = Callable[[int, int], Tuple[int, bool]]


def _neighbors(adj_csr: sp.csr_matrix, node: int) -> Tuple[np.ndarray, np.ndarray]:
    start, end = adj_csr.indptr[node], adj_csr.indptr[node + 1]
    return adj_csr.indices[start:end], adj_csr.data[start:end]


def sample_union_subgraph(
    adj_csr: sp.csr_matrix,
    seed_nodes: Sequence[int],
    num_layers: int,
    budget_fn: BudgetFn,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trả về (global_ids, edge_index[local], edge_weight, seed_local_idx)."""
    visited: dict = {}

    def get_local(g: int) -> int:
        local = visited.get(g)
        if local is None:
            local = len(visited)
            visited[g] = local
        return local

    seed_list = list(seed_nodes)
    for g in seed_list:
        get_local(g)

    edge_src, edge_dst, edge_w = [], [], []
    frontier = list(seed_list)
    for hop in range(num_layers):
        next_frontier = []
        for center in frontier:
            nbr_ids, nbr_w = _neighbors(adj_csr, center)
            if nbr_ids.size == 0:
                continue
            budget, allow_oversample = budget_fn(center, hop)

            if nbr_ids.size > budget:
                pos = rng.choice(nbr_ids.size, size=budget, replace=False)
            elif allow_oversample and nbr_ids.size < budget:
                pos = rng.choice(nbr_ids.size, size=budget, replace=True)
            else:
                pos = np.arange(nbr_ids.size)

            dst_local = get_local(center)
            for p in pos:
                src_global = int(nbr_ids[p])
                src_local = get_local(src_global)
                edge_src.append(src_local)
                edge_dst.append(dst_local)
                edge_w.append(float(nbr_w[p]))
                next_frontier.append(src_global)
        frontier = next_frontier

    global_ids = torch.tensor(list(visited.keys()), dtype=torch.long)
    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_weight = torch.tensor(edge_w, dtype=torch.float32) if edge_w else torch.zeros(0, dtype=torch.float32)
    seed_local_idx = torch.tensor([visited[g] for g in seed_list], dtype=torch.long)
    return global_ids, edge_index, edge_weight, seed_local_idx


def subgraph_to_data(global_ids: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                      node_features: torch.Tensor) -> Data:
    x = node_features[global_ids]
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=global_ids.numel())
    data.global_ids = global_ids
    return data
