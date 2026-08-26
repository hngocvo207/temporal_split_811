"""
Smoke test cho B1 (encoder) + B2 (label-aware sampler) trên dữ liệu thật đã
tạo ở giai đoạn A. Không phải bộ test chính thức của dự án -- chạy 1 lần để
xác nhận shape/luồng dữ liệu đúng trước khi viết B3/B4, và để verify tuyên bố
"full-graph forward không cần chia batch" trong gnn_encoder.py bằng số đo thật
thay vì suy đoán.
"""
import time

import scipy.sparse as sp
import torch

from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import PREPROC_DIR, load_labels
from model.gnn_encoder import GraphSAGEEncoder, GATEncoder
from model.label_aware_sampler import LabelAwareNeighborSampler, build_subgraph_batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x = load_node_features()
    labels = load_labels()
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    print(f"num_nodes={x.shape[0]} num_features={x.shape[1]} num_positive={pos_idx.numel()}")

    adj_train_csr = sp.load_npz(PREPROC_DIR / "adj_train.npz")

    seed_nodes = torch.cat([pos_idx[:4], neg_idx[:4]])
    print(f"seed_nodes (4 pos + 4 neg): {seed_nodes.tolist()}")

    sampler = LabelAwareNeighborSampler(adj_train_csr, labels)
    subgraph, seed_local_idx = build_subgraph_batch(sampler, seed_nodes, x)
    print(f"subgraph: {subgraph}")
    print(f"seed_local_idx: {seed_local_idx.tolist()}")
    assert subgraph.edge_index.max().item() < subgraph.num_nodes
    assert seed_local_idx.numel() == seed_nodes.numel()

    for name, enc in [
        ("GraphSAGEEncoder", GraphSAGEEncoder(in_channels=23, hidden_channels=32, out_channels=16)),
        ("GATEncoder", GATEncoder(in_channels=23, hidden_channels=8, out_channels=16, heads=2)),
    ]:
        enc.eval()
        with torch.no_grad():
            h_all = enc(subgraph.x, subgraph.edge_index, subgraph.edge_weight)
        h_seed = h_all[seed_local_idx]
        print(f"[{name}] h_all={tuple(h_all.shape)} h_seed={tuple(h_seed.shape)}")
        assert h_seed.shape == (seed_nodes.numel(), 16)
        assert torch.isfinite(h_seed).all()

    # Full-graph forward timing/memory check on the actual train graph (not a toy).
    print("\n--- full_graph_forward on graph_train.pt (real scale) ---")
    graph_train = load_graph("train")
    enc = GraphSAGEEncoder(in_channels=23, hidden_channels=128, out_channels=128).to(device)
    t0 = time.time()
    h_full = enc.full_graph_forward(graph_train, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak GPU memory: {peak_mem_gb:.2f} GB")
    dt = time.time() - t0
    print(f"full_graph_forward: out={tuple(h_full.shape)} took {dt:.2f}s")
    assert h_full.shape == (graph_train.num_nodes, 128)
    assert torch.isfinite(h_full).all()

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
