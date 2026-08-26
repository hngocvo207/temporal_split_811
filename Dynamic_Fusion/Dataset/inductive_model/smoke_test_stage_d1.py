"""
Smoke test D1: ghép B1 (GraphSAGEEncoder, subgraph THẬT từ B2's sampler) + C1
(TxBertEncoder, input_ids GIẢ LẬP -- dữ liệu tokenized giao dịch thật thuộc
phạm vi E1, chưa làm) + D1 (GatedFusion) + classifier thành 1 forward/backward
pass hoàn chỉnh. Kiểm tra không chỉ shape mà cả gradient có chảy vào ĐỦ CẢ BA
nhánh (text encoder, graph encoder, fusion+classifier) hay không -- 1 module
wiring sai (vd quên .detach() nhầm chỗ, hoặc quên bật requires_grad) thường chỉ
lộ ra ở bước backward, không lộ ở forward.
"""
import scipy.sparse as sp
import torch
import torch.nn as nn

from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR, load_labels
from model.fraud_model import FraudClassifier
from model.gnn_encoder import GraphSAGEEncoder
from model.label_aware_sampler import LabelAwareNeighborSampler, build_subgraph_batch
from pytorch_pretrained_bert.modeling import BertConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = load_node_features()
    labels = load_labels()
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    seed_nodes = torch.cat([pos_idx[:4], neg_idx[:4]])
    B = seed_nodes.numel()

    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    sampler = LabelAwareNeighborSampler(adj_train, labels)
    subgraph, seed_local_idx = build_subgraph_batch(sampler, seed_nodes, x)
    subgraph = subgraph.to(device)

    graph_encoder = GraphSAGEEncoder(in_channels=23, hidden_channels=32, out_channels=24).to(device)
    h_all = graph_encoder(subgraph.x, subgraph.edge_index, subgraph.edge_weight)
    h_graph = h_all[seed_local_idx]  # [B, 24], KHÔNG detach -- muốn gradient chảy ngược vào graph_encoder
    print(f"h_graph: {tuple(h_graph.shape)}, requires_grad={h_graph.requires_grad}")

    bert_config = BertConfig(
        vocab_size_or_config_json_file=1000, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
    )
    model = FraudClassifier(bert_config, d_graph=24, fusion_dim=16, num_labels=2).to(device)

    L = 20
    input_ids = torch.randint(0, 1000, (B, L), device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    y = labels[seed_nodes].to(device)

    logits = model(input_ids, h_graph, attention_mask=attention_mask)
    print(f"logits: {tuple(logits.shape)}")
    assert logits.shape == (B, 2)
    assert torch.isfinite(logits).all()

    loss = nn.functional.cross_entropy(logits, y)
    print(f"loss: {loss.item():.4f}")
    loss.backward()

    def has_grad(module, name):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        ok = len(grads) > 0 and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
        print(f"  {name}: has_grad={ok}")
        return ok

    print("gradient check:")
    assert has_grad(model.text_encoder, "text_encoder (C1)")
    assert has_grad(model.fusion, "fusion (D1)")
    assert has_grad(model.classifier, "classifier (D1)")
    assert has_grad(graph_encoder, "graph_encoder (B1, qua h_graph không detach)")

    print("\nD1 END-TO-END WIRING (forward + backward) PASSED")


if __name__ == "__main__":
    main()
