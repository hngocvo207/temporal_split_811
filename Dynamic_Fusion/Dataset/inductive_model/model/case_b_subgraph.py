"""
Task B4 (new_propose.md giai đoạn B), phần ghép đồ thị: nối 1 account hoàn toàn
mới (chưa từng là node) vào bản sao/slice của đồ thị inference-time, KHÔNG sửa
đổi adj_inference.npz gốc (một CSR 2,973,489 x 2,973,489 quá tốn để rebuild mỗi
lần gọi predict_account cho 1 account -- xem lý do trong docstring
model/gnn_encoder.py về ngân sách bộ nhớ). Thay vào đó: tự tay nối cạnh
(counterparty -> new_node) rồi mở rộng (num_layers - 1) tầng còn lại từ các
counterparty đã biết bằng đúng hạ tầng sampling union-subgraph dùng chung với
B2/B3 (model/graph_sampling.py).

Chỉ cần cạnh MỘT CHIỀU (counterparty -> new_node): new_node là seed duy nhất cần
đọc h_graph ở cuối, các counterparty chỉ đóng vai trò node trong receptive field
(không cần embedding của chính chúng thay đổi để phản ánh cạnh mới).

Counterparty nào KHÔNG có trong address_to_index (bản thân cũng là account lạ,
ngoài snapshot dữ liệu) bị bỏ qua -- không đệ quy tạo thêm node ảo (ngoài phạm vi
B4). Tỉ lệ counterparty bị bỏ qua là một tín hiệu độ tin cậy, trả về cùng
kết quả để predict_account.py quyết định cờ low_confidence.
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from model.graph_sampling import sample_union_subgraph, subgraph_to_data
from model.new_account_features import RawTx

DEFAULT_FANOUT = (15, 10)


def resolve_counterparties(
    address: str, txs: Sequence[RawTx], address_to_idx: Dict[str, int]
) -> Tuple[List[int], List[float], int]:
    """Gộp trọng số (tổng ETH đã giao dịch) theo từng counterparty đã resolve
    được sang global id. Trả về (global_ids, weights, num_unresolved)."""
    weight_by_global: Dict[int, float] = defaultdict(float)
    num_unresolved = 0
    seen_unresolved = set()
    addr_lower = address.lower()

    for t in txs:
        other = t.to_addr if t.from_addr.lower() == addr_lower else t.from_addr
        other_lower = other.lower()
        g = address_to_idx.get(other_lower, address_to_idx.get(other))
        if g is None:
            if other_lower not in seen_unresolved:
                seen_unresolved.add(other_lower)
                num_unresolved += 1
            continue
        weight_by_global[g] += max(t.value_eth, 0.0)

    global_ids = list(weight_by_global.keys())
    weights = [weight_by_global[g] for g in global_ids]
    return global_ids, weights, num_unresolved


def build_case_b_subgraph(
    adj_inference_csr: sp.csr_matrix,
    node_features: torch.Tensor,
    new_node_feature_vec: np.ndarray,
    counterparty_global_ids: Sequence[int],
    counterparty_weights: Sequence[float],
    fanout: Sequence[int] = DEFAULT_FANOUT,
    seed: int = 44,
) -> Tuple[Data, int, torch.Tensor]:
    """Trả về (subgraph, new_node_local_idx, resolved_global_ids)."""
    rng = np.random.default_rng(seed)
    num_layers = len(fanout)

    if not counterparty_global_ids:
        # Account mới hoàn toàn không có counterparty nào resolve được (hoặc
        # chưa từng giao dịch) -- subgraph chỉ có đúng 1 node cô lập.
        x = torch.from_numpy(new_node_feature_vec).unsqueeze(0)
        data = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_weight=torch.zeros(0, dtype=torch.float32), num_nodes=1)
        data.global_ids = torch.tensor([-1], dtype=torch.long)  # -1 = node ảo, không có trong address_to_index
        return data, 0, data.global_ids

    def uniform_budget_fn(fo):
        def fn(center: int, hop: int) -> Tuple[int, bool]:
            return fo[hop], False
        return fn

    if num_layers == 1:
        # 1 lớp: chỉ cần chính các counterparty (không mở rộng thêm tầng nào).
        cap = fanout[0]
        if len(counterparty_global_ids) > cap:
            idx = rng.choice(len(counterparty_global_ids), size=cap, replace=False)
            counterparty_global_ids = [counterparty_global_ids[i] for i in idx]
            counterparty_weights = [counterparty_weights[i] for i in idx]
        global_ids = torch.tensor(counterparty_global_ids, dtype=torch.long)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
    else:
        remaining_fanout = fanout[1:]
        seeds_for_expansion = counterparty_global_ids
        cap = fanout[0]
        if len(seeds_for_expansion) > cap:
            idx = rng.choice(len(seeds_for_expansion), size=cap, replace=False)
            seeds_for_expansion = [seeds_for_expansion[i] for i in idx]
            counterparty_weights = [counterparty_weights[i] for i in idx]
            counterparty_global_ids = seeds_for_expansion
        global_ids, edge_index, edge_weight, _ = sample_union_subgraph(
            adj_inference_csr, seeds_for_expansion, len(remaining_fanout),
            uniform_budget_fn(remaining_fanout), rng,
        )

    global_to_local = {int(g): i for i, g in enumerate(global_ids.tolist())}
    new_node_local = global_ids.numel()

    extra_src, extra_dst, extra_w = [], [], []
    for g, w in zip(counterparty_global_ids, counterparty_weights):
        local = global_to_local.get(int(g))
        if local is not None:
            extra_src.append(local)
            extra_dst.append(new_node_local)
            extra_w.append(float(w))

    full_edge_index = torch.cat([edge_index, torch.tensor([extra_src, extra_dst], dtype=torch.long)], dim=1) \
        if extra_src else edge_index
    full_edge_weight = torch.cat([edge_weight, torch.tensor(extra_w, dtype=torch.float32)]) \
        if extra_w else edge_weight

    x_existing = node_features[global_ids]
    x_new = torch.from_numpy(new_node_feature_vec).unsqueeze(0)
    x_full = torch.cat([x_existing, x_new], dim=0)

    data = Data(x=x_full, edge_index=full_edge_index, edge_weight=full_edge_weight, num_nodes=new_node_local + 1)
    data.global_ids = torch.cat([global_ids, torch.tensor([-1], dtype=torch.long)])
    return data, new_node_local, data.global_ids
