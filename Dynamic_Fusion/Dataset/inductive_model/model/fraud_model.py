"""
Ghép B1 (h_graph) + C1 (h_text) + D1 (gated-fusion) thành 1 model duy nhất, đúng
sơ đồ kiến trúc v2 ở new_propose.md mục 1: P(fraud | x0) = classifier(fusion(
h_text(x0), h_graph(x0))).

h_graph KHÔNG được tính bên trong forward() -- cố ý tách rời: cách tính h_graph
khác nhau tuỳ ngữ cảnh (train dùng subgraph nhỏ từ LabelAwareNeighborSampler ở
B2, eval/inference dùng full_graph_forward của B1 hoặc ego-subgraph của B3/B4),
còn TxBertEncoder (C1) luôn nhận input_ids trực tiếp theo batch. Người gọi
(vòng lặp train ở E1, hoặc predict_account.py) tính h_graph bằng cách phù hợp
rồi truyền vào đây -- FraudClassifier chỉ lo phần fusion + classifier (D1),
không lo nguồn gốc h_graph.
"""
import torch
import torch.nn as nn

from model.fusion import GatedFusion
from model.txbert import TxBertEncoder


class FraudClassifier(nn.Module):
    def __init__(self, bert_config, d_graph: int, fusion_dim: int = None,
                 num_labels: int = 2, tau: float = 1.0, hard_gate: bool = False, dropout: float = 0.1,
                 pretrained_bert: str = None):
        """pretrained_bert: nếu set (vd 'bert-base-uncased'), warm-start
        text_encoder từ pretrained thay vì random-init -- xem
        TxBertEncoder.from_pretrained(); cần cho E2 so sánh công bằng với
        Attempt-3 (cũng warm-start cùng bert-base-uncased). bert_config vẫn cần
        truyền vào để biết d_text=hidden_size (dùng config thật của
        pretrained_bert khi 2 tham số này không khớp sẽ assert lỗi sớm thay vì
        lỗi khó hiểu ở forward)."""
        super().__init__()
        if pretrained_bert:
            self.text_encoder = TxBertEncoder.from_pretrained(pretrained_bert)
            assert self.text_encoder.bert.config.hidden_size == bert_config.hidden_size, (
                f"bert_config.hidden_size={bert_config.hidden_size} không khớp "
                f"pretrained_bert={pretrained_bert} (hidden_size="
                f"{self.text_encoder.bert.config.hidden_size})"
            )
        else:
            self.text_encoder = TxBertEncoder(bert_config)
        d_text = bert_config.hidden_size
        fusion_dim = fusion_dim or d_text
        self.fusion = GatedFusion(d_text, d_graph, fusion_dim, tau=tau, hard_gate=hard_gate, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        h_graph: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        h_text = self.text_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        fused = self.fusion(h_text, h_graph)
        logits = self.classifier(self.dropout(fused))
        return logits
