"""Smoke test cho C1 (TxBertEncoder): forward pass hợp lệ, chỉ nhận input_ids,
không còn tham số nào liên quan GCN (gcn_vocab_ids, graph_features, vocab_adj_list)."""
import inspect

import torch

from model.txbert import TxBertEncoder
from pytorch_pretrained_bert.modeling import BertConfig


def main():
    config = BertConfig(
        vocab_size_or_config_json_file=1000,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = TxBertEncoder(config)
    model.eval()

    sig = inspect.signature(model.forward)
    param_names = list(sig.parameters.keys())
    print(f"TxBertEncoder.forward params: {param_names}")
    forbidden = {"gcn_vocab_ids", "graph_features", "vocab_adj_list", "precomputed_H_vh"}
    leaked = forbidden & set(param_names)
    assert not leaked, f"C1 phải bỏ hết tensor GCN khỏi nhánh BERT, còn sót: {leaked}"

    B, L = 4, 20
    input_ids = torch.randint(0, 1000, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    token_type_ids = torch.zeros(B, L, dtype=torch.long)

    with torch.no_grad():
        h_text = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    print(f"h_text shape={tuple(h_text.shape)}")
    assert h_text.shape == (B, 32)
    assert torch.isfinite(h_text).all()

    print("C1 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
