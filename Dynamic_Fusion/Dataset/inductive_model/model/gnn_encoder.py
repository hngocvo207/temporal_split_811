"""
Task B1 (new_propose.md giai đoạn B): encoder inductive cho h_graph(x0).

Tham khảo cấu trúc từ LMAE4Eth (không copy code, khác thư viện -- xem
A3_library_decision.md):
  - model/sage/model.py (SAGE): mean aggregator, N lớp, inference theo tầng
    không lặp lại tính toán.
  - model/gat.py (GAT): multi-head attention, dropout riêng cho feature/attention.

Khác biệt có chủ đích so với LMAE4Eth:
  - GraphSAGEEncoder dùng WeightedSAGEConv (mean **có trọng số**) thay vì mean đều,
    để tận dụng edge_weight (khối lượng giao dịch) thay vì bỏ qua nó -- xem
    STATUS.md, ghi chú "edge_weight ... B1 phải xử lý trước khi đưa vào GNN".
  - full_graph_forward() thay cho SAGE.inference() theo tầng/theo batch của DGL:
    đã ĐO THẬT (smoke_test_stage_b.py) trên graph_train.pt (2,973,489 node,
    4,162,599 cạnh, hidden=out=128): 0.3s, peak 8.0GB VRAM trên RTX 3060 12GB --
    khớp ngân sách nhưng SÁT trần, và số đo này là torch.no_grad() thuần suy luận.
    KHÔNG dùng full_graph_forward() trong vòng lặp backward (không đủ VRAM cho
    activation cache của autograd + optimizer state + BERT đồng trú -- xem F1).
    Train luôn đi qua LabelAwareNeighborSampler (B2, model/label_aware_sampler.py)
    -- subgraph vài trăm node mỗi batch, không phải full-graph. full_graph_forward
    chỉ dùng để suy luận hàng loạt (B3 batch, E2/E3 eval, F-scale sau này). Nếu
    graph phình to hơn đáng kể và 8GB không còn đủ ở torch.no_grad(), quay lại
    pattern layer-wise NeighborLoader (num_neighbors=[-1]) là refactor cục bộ,
    không đổi API bên ngoài.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter


def normalize_edge_weight(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """log1p(raw weight) rồi chuẩn hoá theo tổng trọng số vào mỗi node đích
    (softmax-sum-to-1 kiểu mean), để WeightedSAGEConv làm weighted-mean thay vì
    mean đều. raw weight dải giá trị rất rộng (0..~6.8e12, xem A2) nên bắt buộc
    log1p trước khi normalize -- nếu không một cạnh outlier sẽ nuốt hết trọng số.
    """
    w = torch.log1p(edge_weight.clamp(min=0))
    dst = edge_index[1]
    denom = scatter(w, dst, dim=0, dim_size=num_nodes, reduce="sum")
    denom = denom.clamp(min=1e-12)
    return w / denom[dst]


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE mean-aggregator, có trọng số theo edge_weight_norm (đã chuẩn hoá
    sum-to-1 theo node đích ở normalize_edge_weight)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight_norm: torch.Tensor) -> torch.Tensor:
        neigh = self.propagate(edge_index, x=x, edge_weight=edge_weight_norm, size=(x.size(0), x.size(0)))
        return self.lin_self(x) + self.lin_neigh(neigh)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.view(-1, 1)


class GraphSAGEEncoder(nn.Module):
    """2 lớp mặc định (budget hàng xóm [15, 10] áp ở NeighborLoader khi train,
    xem model/minibatch.py)."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList([WeightedSAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.dropout = dropout
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        edge_weight_norm = normalize_edge_weight(edge_index, edge_weight, x.size(0))
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight_norm)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    @torch.no_grad()
    def full_graph_forward(self, data: Data, device: torch.device) -> torch.Tensor:
        """h_graph cho MỌI node trong data, một lần forward toàn đồ thị (xem
        docstring module về lý do không chia theo tầng như LMAE4Eth)."""
        self.eval()
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device)
        return self.forward(x, edge_index, edge_weight)


class GATEncoder(nn.Module):
    """Lựa chọn thay thế cho GraphSAGEEncoder (B1 cho phép GraphSAGE HOẶC GAT).
    Dùng edge_dim=1 để attention nhìn thấy log1p(edge_weight) thay vì bỏ qua.

    ĐÃ ĐO THẬT full_graph_forward trên graph_train.pt, khác hẳn GraphSAGEEncoder:
      - default (hidden=64, out=128, heads=4): OOM thật trên RTX 3060 12GB (cần
        6.8GB thêm trong khi đã dùng 6.26GB, multi-head attention tốn bộ nhớ
        trung gian hơn nhiều so với WeightedSAGEConv ở cùng cạnh số).
      - hidden=16, out=32, heads=2: chạy được, peak 3.5GB, 0.28s.
      Nếu dùng GAT cho full-graph forward (B3 batch/E2/E3), PHẢI hạ hidden/heads
      xuống mức nhỏ như trên hoặc quay lại pattern layer-wise NeighborLoader --
      không dùng default constructor argument cho việc này."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 128,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.out_channels = out_channels
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, edge_dim=1, dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, edge_dim=1, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=1, dropout=dropout)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        edge_attr = torch.log1p(edge_weight.clamp(min=0)).view(-1, 1)
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr=edge_attr)
            if i != len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    @torch.no_grad()
    def full_graph_forward(self, data: Data, device: torch.device) -> torch.Tensor:
        self.eval()
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device)
        return self.forward(x, edge_index, edge_weight)


def build_encoder(kind: str, in_channels: int, **kwargs) -> nn.Module:
    if kind == "sage":
        return GraphSAGEEncoder(in_channels, **kwargs)
    if kind == "gat":
        return GATEncoder(in_channels, **kwargs)
    raise ValueError(f"unknown encoder kind: {kind!r} (expected 'sage' or 'gat')")
