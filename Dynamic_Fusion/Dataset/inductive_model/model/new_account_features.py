"""
Task B4 (new_propose.md giai đoạn B), phần feature: tính vector 23 chiều cho 1
account HOÀN TOÀN MỚI (chưa từng là node) từ danh sách giao dịch thô, đúng công
thức Nhóm 1 (basic, 12 cột) + Nhóm 2 (temporal frequency, 6 cột) đã dùng để tạo
features_output_fullscale.csv -- xem fullscale_features/02_groups12_exact.py
dòng 115-202 (đã đọc trực tiếp source, không suy đoán công thức).

5 cột Nhóm 3 (centrality: betweenness_centrality, degree_centrality,
clustering_coefficient, in_degree_centrality, out_degree_centrality) CẦN TOÀN BỘ
đồ thị để tính đúng (betweenness đặc biệt tốn kém ở scale 2.97M node) -- không thể
tính đúng cho 1 node cô lập chưa thực sự nối vào đồ thị. Gán train_mean cho 5 cột
này (=> z-score = 0 sau khi scale, tức "trung tính") và LUÔN trả về
low_confidence=True kèm lý do -- đúng yêu cầu B4 "gắn nhãn độ tin cậy GCN thấp".

Scaler dùng data/node_features_all23.scaler.json (đã recompute + verify ở A1
addendum, xem data_prep/compute_feature_scaler.py).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

SCALER_PATH = Path(__file__).resolve().parent.parent / "data" / "node_features_all23.scaler.json"

G1_COLS = [
    "out_degree", "in_degree", "direction_ratio", "max_out_amount", "min_out_amount",
    "avg_out_amount", "max_in_amount", "min_in_amount", "avg_in_amount", "account_balance",
    "lifetime_days", "active_days",
]
G2_COLS = [
    "freq_out_short", "freq_in_short", "freq_out_long", "freq_in_long",
    "short_long_out_ratio", "short_long_in_ratio",
]
G3_COLS = [
    "betweenness_centrality", "degree_centrality", "clustering_coefficient",
    "in_degree_centrality", "out_degree_centrality",
]
DAY_SECONDS = 86400
SHORT_WINDOW_DAYS = 30
LONG_WINDOW_DAYS = 180


@dataclass
class RawTx:
    from_addr: str
    to_addr: str
    value_eth: float
    timestamp: int  # unix seconds


def compute_g1_g2_raw(address: str, txs: Sequence[RawTx]) -> dict:
    """Công thức y hệt 02_groups12_exact.py, áp cho 1 node duy nhất thay vì
    vector hoá toàn đồ thị (n=1 ở đây)."""
    out_vals = [t.value_eth for t in txs if t.from_addr.lower() == address.lower()]
    in_vals = [t.value_eth for t in txs if t.to_addr.lower() == address.lower()]
    out_ts = [t.timestamp for t in txs if t.from_addr.lower() == address.lower()]
    in_ts = [t.timestamp for t in txs if t.to_addr.lower() == address.lower()]
    all_ts = out_ts + in_ts

    out_degree = len(out_vals)
    in_degree = len(in_vals)
    direction_ratio = round(out_degree / (out_degree + in_degree + 1e-9), 6)

    def _max(v):
        return max(v) if v else 0.0

    def _min(v):
        return min(v) if v else 0.0

    def _avg(v):
        return (sum(v) / len(v)) if v else 0.0

    max_out_amount, min_out_amount, avg_out_amount = _max(out_vals), _min(out_vals), _avg(out_vals)
    max_in_amount, min_in_amount, avg_in_amount = _max(in_vals), _min(in_vals), _avg(in_vals)
    account_balance = round(sum(in_vals) - sum(out_vals), 6)

    incident = out_degree + in_degree
    if incident >= 2:
        lifetime_days = (max(all_ts) - min(all_ts)) // DAY_SECONDS
    else:
        lifetime_days = 0
    active_days = len({ts // DAY_SECONDS for ts in all_ts})

    ref = max(all_ts) if all_ts else -1  # "now" của chính node = giao dịch gần nhất của nó, đúng công thức gốc (ref = mx)
    short_cut = ref - SHORT_WINDOW_DAYS * DAY_SECONDS
    long_cut = ref - LONG_WINDOW_DAYS * DAY_SECONDS
    out_short = sum(1 for ts in out_ts if ts >= short_cut)
    out_long = sum(1 for ts in out_ts if ts >= long_cut)
    in_short = sum(1 for ts in in_ts if ts >= short_cut)
    in_long = sum(1 for ts in in_ts if ts >= long_cut)

    # Ratios dùng giá trị freq CHƯA làm tròn (khớp đúng thứ tự phép tính gốc:
    # 02_groups12_exact.py chỉ np.round() khi ghi vào cols dict, còn tỉ lệ
    # short_long_*_ratio được tính từ freq_* thô -- làm tròn freq trước rồi mới
    # chia sẽ lệch ở chữ số thập phân thứ 5-6, như đã bắt được lúc validate).
    freq_out_short_raw = out_short / SHORT_WINDOW_DAYS
    freq_in_short_raw = in_short / SHORT_WINDOW_DAYS
    freq_out_long_raw = out_long / LONG_WINDOW_DAYS
    freq_in_long_raw = in_long / LONG_WINDOW_DAYS
    freq_out_short = round(freq_out_short_raw, 6)
    freq_in_short = round(freq_in_short_raw, 6)
    freq_out_long = round(freq_out_long_raw, 6)
    freq_in_long = round(freq_in_long_raw, 6)

    return {
        "out_degree": float(out_degree),
        "in_degree": float(in_degree),
        "direction_ratio": direction_ratio,
        "max_out_amount": round(max_out_amount, 6),
        "min_out_amount": round(min_out_amount, 6),
        "avg_out_amount": round(avg_out_amount, 6),
        "max_in_amount": round(max_in_amount, 6),
        "min_in_amount": round(min_in_amount, 6),
        "avg_in_amount": round(avg_in_amount, 6),
        "account_balance": account_balance,
        "lifetime_days": float(lifetime_days),
        "active_days": float(active_days),
        "freq_out_short": freq_out_short,
        "freq_in_short": freq_in_short,
        "freq_out_long": freq_out_long,
        "freq_in_long": freq_in_long,
        "short_long_out_ratio": round(freq_out_short_raw / (freq_out_long_raw + 1e-9), 6),
        "short_long_in_ratio": round(freq_in_short_raw / (freq_in_long_raw + 1e-9), 6),
    }


def build_new_account_feature_vector(address: str, txs: Sequence[RawTx]) -> np.ndarray:
    with open(SCALER_PATH) as f:
        scaler = json.load(f)
    feature_cols: List[str] = scaler["feature_columns"]
    mean = np.array(scaler["train_mean"], dtype=np.float64)
    std = np.array(scaler["train_std"], dtype=np.float64)

    raw_g1_g2 = compute_g1_g2_raw(address, txs)
    raw_vec = np.zeros(len(feature_cols), dtype=np.float64)
    for i, col in enumerate(feature_cols):
        if col in raw_g1_g2:
            raw_vec[i] = raw_g1_g2[col]
        elif col in G3_COLS:
            raw_vec[i] = mean[i]  # -> z=0 sau khi scale, xem docstring module
        else:
            raise ValueError(f"unexpected feature column not in G1/G2/G3: {col}")

    z = (raw_vec - mean) / std
    return z.astype(np.float32)
