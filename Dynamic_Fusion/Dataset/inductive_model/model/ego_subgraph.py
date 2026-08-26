"""
Task B3 (new_propose.md giai đoạn B): predict_account(addr) cho account ĐÃ có
trong graph (Case A) -- hệ quả tự nhiên của A2 (đồ thị inference-time,
adj_inference.npz) + B1 (encoder), không cần rebuild toàn model như cách làm cũ
của tri_model.

Sampler ở đây UNIFORM, không label-aware như B2: tại lúc infer, nhãn thật của
CHÍNH account đang hỏi là ẩn số cần dự đoán nên "budget theo nhãn của center"
không áp dụng được cho seed; và không được lệch sampling theo nhãn hàng xóm vì
sẽ làm rò rỉ tín hiệu nhãn vào receptive field của node đang test (data leakage
kiểu khác với B2, nhưng cùng bản chất: không được dùng thông tin nhãn theo cách
model sẽ không có ở production).
"""
from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from model.graph_sampling import sample_union_subgraph, subgraph_to_data

DEFAULT_FANOUT = (15, 10)  # khớp budget đề xuất ở B1


def _uniform_budget_fn(fanout: Sequence[int]):
    def fn(center: int, hop: int) -> Tuple[int, bool]:
        return fanout[hop], False  # không oversample -- Case A luôn có ít nhất seed node thật
    return fn


def build_ego_subgraph(
    adj_inference_csr: sp.csr_matrix,
    seed_node: int,
    node_features: torch.Tensor,
    fanout: Sequence[int] = DEFAULT_FANOUT,
    seed: int = 44,
) -> Tuple[Data, int, torch.Tensor]:
    """Trả về (subgraph, seed_local_idx, global_ids). global_ids dùng ở
    predict_account.py để tính cờ độ tin cậy (bao nhiêu node trong subgraph
    không có lịch sử train-time)."""
    rng = np.random.default_rng(seed)
    global_ids, edge_index, edge_weight, seed_local_idx = sample_union_subgraph(
        adj_inference_csr, [seed_node], len(fanout), _uniform_budget_fn(fanout), rng
    )
    data = subgraph_to_data(global_ids, edge_index, edge_weight, node_features)
    return data, int(seed_local_idx.item()), global_ids
