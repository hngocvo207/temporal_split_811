"""
Validate B4's compute_g1_g2_raw (model/new_account_features.py) against REAL
data: lấy vài node thật đã có trong đồ thị, dựng lại đúng danh sách giao dịch
thô của nó từ fullscale_features/arrays/edges.npz (cùng công thức nguồn dùng để
tạo features_output_fullscale.csv), chạy qua compute_g1_g2_raw() y như thể đó
là 1 account hoàn toàn mới, rồi so trực tiếp với giá trị thật trong
features_output_fullscale.csv (RAW, chưa scale) cho đúng node đó.

Đây là cách kiểm chứng B4 duy nhất khả thi trong sandbox này (không có
ETHERSCAN_API_KEY / không có mạng để gọi thật model/etherscan_client.py) --
nhưng nó kiểm chứng đúng phần quan trọng nhất: công thức feature có khớp nguồn
gốc hay không.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from model.new_account_features import RawTx, compute_g1_g2_raw, G1_COLS, G2_COLS

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Dynamic_Fusion
ARR = REPO_ROOT / "Dataset" / "fullscale_features" / "arrays"
RAW_UNSCALED_CSV = REPO_ROOT / "raw_data" / "MulDiGraph" / "features_output_fullscale.csv"


def main():
    e = np.load(ARR / "edges.npz")
    src, dst, ts, val = e["src"], e["dst"], e["ts"], e["val"]
    nodes = np.load(ARR / "nodes.npy", allow_pickle=True)

    df = pd.read_csv(RAW_UNSCALED_CSV)
    assert (df["node"].to_numpy() == nodes).all(), "row order mismatch nodes.npy vs raw CSV"

    out_deg_count = np.bincount(src, minlength=len(nodes))
    in_deg_count = np.bincount(dst, minlength=len(nodes))
    total_deg = out_deg_count + in_deg_count
    candidates = np.where((total_deg >= 3) & (total_deg <= 40))[0]
    rng = np.random.default_rng(0)
    test_nodes = rng.choice(candidates, size=8, replace=False)

    max_diff_seen = 0.0
    for node_idx in test_nodes:
        addr = nodes[node_idx]
        out_mask = src == node_idx
        in_mask = dst == node_idx
        txs = []
        for i in np.where(out_mask)[0]:
            txs.append(RawTx(from_addr=addr, to_addr=str(dst[i]), value_eth=float(val[i]), timestamp=int(ts[i])))
        for i in np.where(in_mask)[0]:
            txs.append(RawTx(from_addr=str(src[i]), to_addr=addr, value_eth=float(val[i]), timestamp=int(ts[i])))

        computed = compute_g1_g2_raw(addr, txs)
        actual_row = df.iloc[int(node_idx)]

        row_max_diff = 0.0
        for col in G1_COLS + G2_COLS:
            a, b = computed[col], float(actual_row[col])
            diff = abs(a - b)
            row_max_diff = max(row_max_diff, diff)
            if diff > 1e-4:
                print(f"  [MISMATCH] node={node_idx} addr={addr} col={col} computed={a} actual={b}")
        max_diff_seen = max(max_diff_seen, row_max_diff)
        print(f"node={node_idx} deg={total_deg[node_idx]} max_col_diff={row_max_diff:.6f}")

    print(f"\nmax diff across all test nodes/cols: {max_diff_seen:.6f}")
    if max_diff_seen > 1e-4:
        raise SystemExit("B4 feature formulas DO NOT match source pipeline -- fix before trusting predict_account for Case B")
    print("B4 G1/G2 FEATURE FORMULAS VALIDATED AGAINST REAL DATA")


if __name__ == "__main__":
    main()
