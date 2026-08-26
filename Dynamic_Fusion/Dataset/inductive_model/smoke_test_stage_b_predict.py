"""
Smoke test cho B3 (Case A, predict_account với account đã có trong graph) và
B4 (Case B, ghép graph cho account mới -- dùng giao dịch GIẢ LẬP vì không có
ETHERSCAN_API_KEY/mạng trong sandbox này, xem etherscan_client.py::TODO).
"""
import numpy as np
import scipy.sparse as sp
import torch

from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR, _load_addr_to_idx
from model.case_b_subgraph import build_case_b_subgraph, resolve_counterparties
from model.gnn_encoder import GraphSAGEEncoder
from model.new_account_features import RawTx, build_new_account_feature_vector
from model.predict_account import _train_degree_array, predict_account


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = load_node_features()
    addr_to_idx = _load_addr_to_idx()
    idx_to_addr = {v: k for k, v in addr_to_idx.items()}

    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    adj_inference = sp.load_npz(PREPROC_DIR / "adj_inference.npz")
    train_degree = _train_degree_array(adj_train)

    encoder = GraphSAGEEncoder(in_channels=23, hidden_channels=32, out_channels=16).to(device)

    print("--- Case A: account đã có trong graph ---")
    known_idx = 12345
    known_addr = idx_to_addr[known_idx]
    result_a = predict_account(
        known_addr, encoder, x, adj_inference, addr_to_idx, device, train_degree=train_degree,
    )
    print(result_a)
    assert result_a.case == "A"
    assert result_a.h_graph.shape == (16,)
    assert torch.isfinite(result_a.h_graph).all()

    print("\n--- Case B: account mới hoàn toàn (giao dịch giả lập) ---")
    fake_addr = "0x" + "f" * 40
    assert fake_addr not in addr_to_idx
    # 5 counterparty thật lấy từ đồ thị (đóng vai "đối tác giao dịch" của account giả),
    # cộng 1 counterparty KHÔNG có trong snapshot để test đường unresolved.
    real_counterparties = [idx_to_addr[i] for i in [0, 1, 2, 100, 500000]]
    unresolved_counterparty = "0x" + "9" * 40
    txs = [RawTx(from_addr=fake_addr, to_addr=addr, value_eth=1.5, timestamp=1_600_000_000 + i * 1000)
           for i, addr in enumerate(real_counterparties)]
    txs.append(RawTx(from_addr=unresolved_counterparty, to_addr=fake_addr, value_eth=0.3, timestamp=1_600_100_000))

    feat_vec = build_new_account_feature_vector(fake_addr, txs)
    print(f"feat_vec shape={feat_vec.shape} finite={np.isfinite(feat_vec).all()}")
    assert feat_vec.shape == (23,)
    assert np.isfinite(feat_vec).all()

    counterparty_ids, weights, num_unresolved = resolve_counterparties(fake_addr, txs, addr_to_idx)
    print(f"resolved={len(counterparty_ids)} unresolved={num_unresolved}")
    assert len(counterparty_ids) == 5
    assert num_unresolved == 1

    subgraph, new_node_local, global_ids = build_case_b_subgraph(
        adj_inference, x, feat_vec, counterparty_ids, weights,
    )
    print(f"subgraph={subgraph} new_node_local={new_node_local}")
    subgraph = subgraph.to(device)
    with torch.no_grad():
        h_all = encoder(subgraph.x, subgraph.edge_index, subgraph.edge_weight)
    h_new = h_all[new_node_local]
    print(f"h_new shape={tuple(h_new.shape)} finite={torch.isfinite(h_new).all().item()}")
    assert h_new.shape == (16,)
    assert torch.isfinite(h_new).all()

    print("\n--- predict_account Case B tanpa API key phải raise lỗi rõ ràng ---")
    try:
        predict_account(fake_addr, encoder, x, adj_inference, addr_to_idx, device, train_degree=train_degree)
        raise SystemExit("expected RuntimeError (no ETHERSCAN_API_KEY) but predict_account succeeded")
    except RuntimeError as e:
        print(f"OK, raised as expected: {e}")

    print("\nALL PREDICT_ACCOUNT SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
