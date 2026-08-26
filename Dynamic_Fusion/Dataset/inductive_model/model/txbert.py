"""
Task C1 (new_propose.md giai đoạn C): tách nhánh BERT khỏi mọi tensor liên quan
tới GCN.

So với tri_model/ETH_GBert.py (ETH_GBertEmbeddings/ETH_GBertModel):
  - Bỏ hoàn toàn gcn_vocab_ids, VocabGraphConvolution, FeatureProjector,
    DynamicFusionLayer khỏi nhánh này -- 23 đặc trưng giờ là input trực tiếp
    của GNN (A1/B1), không còn broadcast vào token embedding.
  - Không cần subclass BertEmbeddings riêng nữa (không còn tensor phụ nào phải
    luồn qua embedding layer) -- dùng thẳng BertModel chuẩn của
    pytorch_pretrained_bert (cùng thư viện 0.6.2 API mà tri_model đang dùng).
  - forward() trả về pooled_output (h_text(x0)) thay vì logits: phân loại
    chuyển sang sau fusion (Giai đoạn D), không còn nằm trong nhánh BERT như
    kiến trúc cũ.

Checkpoint cũ (tri_model/output/*.pt, bi_model/output/*.pt) KHÔNG tương thích
thẳng: state_dict cũ có thêm các khoá `embeddings.vocab_gcn.*`,
`embeddings.feature_projector.*`, `embeddings.dynamic_fusion_layer.*`,
`classifier.*` mà TxBertEncoder không có. Warm-start từ checkpoint cũ (nếu cần)
đòi hỏi partial-load chỉ phần `bert.embeddings.word_embeddings.*` /
`bert.encoder.*` / `bert.pooler.*` -- không nằm trong phạm vi C1, để E1 quyết
định.
"""
import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertModel


class TxBertEncoder(nn.Module):
    """h_text(x0) = pooled_output của BERT chuẩn -- chỉ nhận input_ids
    (+ token_type_ids/attention_mask), không còn nhánh GCN/feature nào khác."""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)  # BertModel.__init__ đã tự apply init_bert_weights

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "TxBertEncoder":
        """Warm-start từ BERT pretrained thật (vd 'bert-base-uncased', cùng
        checkpoint tri_model dùng -- xem data_prep/attempt3_corpus.py) thay vì
        random-init -- cần cho E2 so sánh công bằng với Attempt-3 (Attempt-3
        cũng warm-start từ cùng bert-base-uncased, không train from scratch)."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.bert = BertModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return obj

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        _, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False,
        )
        return pooled_output  # [B, hidden_size]
