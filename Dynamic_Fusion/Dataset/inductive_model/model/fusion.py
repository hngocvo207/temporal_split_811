"""
Task D1 (new_propose.md giai đoạn D): module fusion account-level, bắt đầu bằng
gated-fusion (D2 cross-attention để sau, nếu D1 chưa đủ mạnh).

So với DynamicFusionLayer cũ (tri_model/ETH_GBert.py, xem A3_library_decision.md
và STATUS.md về nguyên tắc không đụng tri_model/) -- token-level, 3 luồng
(bert, gcn-enhanced, feature) áp cho mỗi vị trí trong seq_len:
  - Ở đây chỉ còn 2 luồng account-level: h_text(x0) [B, d_text] và
    h_graph(x0) [B, d_graph] -- mỗi account 1 vector, không còn chiều seq_len.
    23 đặc trưng không còn là luồng thứ 3 riêng (đã thành input của GNN ở A1/B1).
  - Rẻ hơn, ít tham số hơn cross-attention -- phù hợp tập dương nhỏ
    (519-3,541 mẫu).

diff_softmax được viết lại cục bộ (không import từ tri_model/ETH_GBert.py) để
giữ đúng nguyên tắc cô lập 2 pipeline: inductive_model/ không phụ thuộc vào
tri_model/ dù chỉ 1 hàm tiện ích, tránh 1 thay đổi ở tri_model/ (pipeline khác,
đang chạy thí nghiệm riêng) âm thầm ảnh hưởng tới pipeline này.
"""
import torch
import torch.nn as nn


def diff_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> torch.Tensor:
    y_soft = (logits / tau).softmax(dim)
    if not hard:
        return y_soft
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class GatedFusion(nn.Module):
    """fused = gate_text * proj(h_text) + gate_graph * proj(h_graph), gate học
    được từ chính 2 vector (DiffSoftmax 2 lối thay vì 4 cổng + luồng mixed như
    bản cũ -- không có luồng "mixed" vì chỉ còn 2 luồng, không phải 3)."""

    def __init__(self, d_text: int, d_graph: int, fusion_dim: int, tau: float = 1.0,
                 hard_gate: bool = False, dropout: float = 0.1):
        super().__init__()
        self.proj_text = nn.Linear(d_text, fusion_dim)
        self.proj_graph = nn.Linear(d_graph, fusion_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )
        self.tau = tau
        self.hard_gate = hard_gate

    def forward(self, h_text: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        h_text_p = self.proj_text(h_text)      # [B, fusion_dim]
        h_graph_p = self.proj_graph(h_graph)    # [B, fusion_dim]
        gate_logits = self.gate_network(torch.cat([h_text_p, h_graph_p], dim=-1))  # [B, 2]
        gate = diff_softmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)
        gate_text, gate_graph = gate[:, 0:1], gate[:, 1:2]
        return gate_text * h_text_p + gate_graph * h_graph_p  # [B, fusion_dim]
