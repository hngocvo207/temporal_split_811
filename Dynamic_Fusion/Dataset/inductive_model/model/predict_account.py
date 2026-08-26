"""
Task B3 + B4 (new_propose.md giai đoạn B): API dự đoán cho 1 địa chỉ, gộp cả
Case A (account đã có trong graph, B3) và Case B (account hoàn toàn mới, B4)
thành 1 hàm predict_account() duy nhất -- đúng tinh thần "hệ quả tự nhiên của
A2+B1, không cần rebuild toàn model" của B3.

Độ tin cậy (B4 yêu cầu "gắn nhãn độ tin cậy GCN thấp nếu hàng xóm cũng là node
mới") được đo bằng tỉ lệ node trong receptive field (trừ chính seed) có bậc = 0
trong adj_train (train_degree) -- tức những node model chưa từng thấy giao dịch
nào trước T_cutoff lúc train, nên GNN gần như không biết gì có ích về chúng.
Case B cộng thêm tỉ lệ counterparty không resolve được (ngoài snapshot đồ thị)
vào cùng cờ.

CHƯA CHẠY END-TO-END THẬT với Case B (cần ETHERSCAN_API_KEY + mạng, không có
trong sandbox này) -- xem smoke_test_stage_b4_predict.py để biết phần đã kiểm
chứng được (feature formulas + graph splicing với dữ liệu giả lập từ node thật)
và etherscan_client.py::TODO cho phần còn thiếu.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from model.case_b_subgraph import build_case_b_subgraph, resolve_counterparties
from model.ego_subgraph import DEFAULT_FANOUT, build_ego_subgraph
from model.etherscan_client import fetch_raw_transactions
from model.new_account_features import build_new_account_feature_vector

LOW_CONFIDENCE_THRESHOLD = 0.5  # >50% receptive field không có lịch sử train-time


@dataclass
class PredictionResult:
    address: str
    case: str  # "A" (đã có trong graph) | "B" (account mới)
    h_graph: torch.Tensor
    low_confidence: bool
    low_history_fraction: float
    num_unresolved_counterparties: int = 0
    notes: List[str] = field(default_factory=list)


def _train_degree_array(adj_train_csr: sp.csr_matrix) -> np.ndarray:
    out_deg = np.diff(adj_train_csr.indptr)
    in_deg = np.asarray(adj_train_csr.sum(axis=0)).flatten()
    # sum(axis=0) trên ma trận trọng số ra tổng weight chứ không phải bậc đếm
    # cạnh -- ép về nhị phân trước khi sum để lấy đúng bậc (đếm cạnh).
    in_deg_count = np.asarray((adj_train_csr > 0).sum(axis=0)).flatten()
    return out_deg + in_deg_count


def predict_account(
    address: str,
    encoder: torch.nn.Module,
    node_features: torch.Tensor,
    adj_inference_csr: sp.csr_matrix,
    address_to_idx: Dict[str, int],
    device: torch.device,
    train_degree: Optional[np.ndarray] = None,
    adj_train_csr: Optional[sp.csr_matrix] = None,
    fanout=DEFAULT_FANOUT,
    etherscan_api_key: Optional[str] = None,
    seed: int = 44,
) -> PredictionResult:
    if train_degree is None:
        if adj_train_csr is None:
            raise ValueError("cần truyền train_degree (đã precompute) hoặc adj_train_csr để tự tính")
        train_degree = _train_degree_array(adj_train_csr)

    addr_norm = address.lower()
    notes: List[str] = []
    num_unresolved = 0

    if addr_norm in address_to_idx:
        case = "A"
        node_idx = address_to_idx[addr_norm]
        data, seed_local_idx, global_ids = build_ego_subgraph(
            adj_inference_csr, node_idx, node_features, fanout=fanout, seed=seed
        )
    else:
        case = "B"
        notes.append("account không có trong snapshot đồ thị hiện tại -- dùng Etherscan để lấy giao dịch thô (B4)")
        txs = fetch_raw_transactions(addr_norm, api_key=etherscan_api_key)
        if not txs:
            notes.append("địa chỉ chưa từng có giao dịch on-chain nào tìm thấy qua Etherscan")
        feat_vec = build_new_account_feature_vector(addr_norm, txs)
        counterparty_ids, weights, num_unresolved = resolve_counterparties(addr_norm, txs, address_to_idx)
        if num_unresolved:
            notes.append(f"{num_unresolved} counterparty không resolve được (ngoài snapshot đồ thị)")
        data, seed_local_idx, global_ids = build_case_b_subgraph(
            adj_inference_csr, node_features, feat_vec, counterparty_ids, weights, fanout=fanout, seed=seed
        )

    data = data.to(device)
    with torch.no_grad():
        h_all = encoder(data.x, data.edge_index, data.edge_weight)
    h_graph = h_all[seed_local_idx]

    other_global_ids = [g for i, g in enumerate(global_ids.tolist()) if i != seed_local_idx and g >= 0]
    if other_global_ids:
        low_history_fraction = float(np.mean(train_degree[other_global_ids] == 0))
    else:
        low_history_fraction = 1.0  # không có hàng xóm nào resolve được -- tin cậy thấp nhất
        notes.append("receptive field không có hàng xóm nào (node cô lập hoặc mọi counterparty đều unresolved)")

    low_confidence = low_history_fraction > LOW_CONFIDENCE_THRESHOLD or case == "B"
    if case == "B":
        notes.append("Case B (account mới hoàn toàn): luôn đánh dấu low_confidence theo yêu cầu B4")

    return PredictionResult(
        address=addr_norm,
        case=case,
        h_graph=h_graph.cpu(),
        low_confidence=low_confidence,
        low_history_fraction=low_history_fraction,
        num_unresolved_counterparties=num_unresolved,
        notes=notes,
    )
