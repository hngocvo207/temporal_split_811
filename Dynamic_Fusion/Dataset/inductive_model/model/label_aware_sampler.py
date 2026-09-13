"""
Task B2 (new_propose.md giai đoạn B): label-aware neighbor sampling kiểu PC-GNN,
gắn vào sampler ở B1 -- thay thế WeightedRandomSampler cấp-mẫu hiện tại của
tri_model (chỉ cân bằng tần suất *seed* dương/âm mỗi batch, không cân bằng được
số lượng hàng xóm mỗi seed kéo vào receptive field).

Biến thể đơn giản hoá so với PC-GNN gốc: không có similarity-aware learnable
neighbor selector (module riêng, tốn công huấn luyện thêm) -- ở đây ngân sách
hàng xóm mỗi tầng phụ thuộc NHÃN CỦA CHÍNH NODE đang mở rộng (phần lớn hàng xóm
trong đồ thị MulDiGraph này không có nhãn nên không thể chấm điểm similarity theo
nhãn hàng xóm như bản gốc):
  - node dương (phisher xác nhận, isp==1): budget hàng xóm LỚN HƠN, lấy mẫu CÓ
    hoàn lại nếu bậc thật < budget (oversample) -- receptive field giàu hơn quanh
    node hiếm.
  - node âm/chưa nhãn: budget NHỎ HƠN, lấy mẫu KHÔNG hoàn lại, cắt bớt nếu bậc
    thật > budget (undersample) -- giảm pha loãng tín hiệu dương khi aggregate.

Chỉ dùng adj_train (train-time CSR) khi train -- không được rò rỉ cạnh
inference-time (đúng ranh giới đã giữ nguyên ở A2). Thuật toán mở rộng
neighbor/gộp subgraph dùng chung với B3 -- xem model/graph_sampling.py.
"""
from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from model.graph_sampling import sample_union_subgraph, subgraph_to_data


class LabelAwareNeighborSampler:
    def __init__(
        self,
        adj_csr: sp.csr_matrix,
        labels: torch.Tensor,
        budgets_pos: Sequence[int] = (30, 20),
        budgets_neg: Sequence[int] = (15, 10),
        seed: int = 44,
    ):
        assert len(budgets_pos) == len(budgets_neg)
        # QUY UOC HUONG CANH (sua bug 2026-09-06, xem data/graph_data.meta.json
        # field 'edge_direction_convention'): _neighbors() trong graph_sampling.py
        # luon doc HANG cua center trong CSR (adj_csr.indptr[center]:...+1]).
        # Doi ten bien edge_src/edge_dst trong sample_union_subgraph KHONG du de
        # doi ngu nghia -- van la HANG cua center (payee/out-neighbor), du gan
        # vao vai tro nao. De center thuc su aggregate tu PAYER (nguoi da gui
        # tien CHO center, dung quy uoc "incoming" cua full_graph_forward/
        # graph_train.pt) thi phai TRANSPOSE ma tran truoc: hang cua center
        # trong adj_csr.T = cot cua center trong adj_csr goc = payer that.
        self.adj_csr = adj_csr.T.tocsr()
        self.labels = labels
        self.budgets_pos = tuple(budgets_pos)
        self.budgets_neg = tuple(budgets_neg)
        self.num_layers = len(budgets_pos)
        self.rng = np.random.default_rng(seed)

    def _budget_fn(self, center: int, hop: int) -> Tuple[int, bool]:
        is_positive = int(self.labels[center]) == 1
        budget = self.budgets_pos[hop] if is_positive else self.budgets_neg[hop]
        return budget, is_positive  # oversample (allow replacement) only for positive centers

    def sample(self, seed_nodes: torch.Tensor):
        return sample_union_subgraph(
            self.adj_csr, seed_nodes.tolist(), self.num_layers, self._budget_fn, self.rng
        )


def build_subgraph_batch(
    sampler: LabelAwareNeighborSampler, seed_nodes: torch.Tensor, node_features: torch.Tensor
) -> Tuple[Data, torch.Tensor]:
    global_ids, edge_index, edge_weight, seed_local_idx = sampler.sample(seed_nodes)
    data = subgraph_to_data(global_ids, edge_index, edge_weight, node_features)
    return data, seed_local_idx
