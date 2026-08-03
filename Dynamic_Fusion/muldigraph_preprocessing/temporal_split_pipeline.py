"""
temporal_split_pipeline.py — Temporal train/val/test split cho MulDiGraph blockchain.

Luồng xử lý:
  1. Load & kiểm tra MulDiGraph.pkl
  2. Gán nhãn đúng (confirmed fraud + potential fraud)
  3. Tính T_first cho mỗi node (timestamp giao dịch đầu tiên)
  4. Chia temporal: Train(80%) / Val(10%) / Test(10%) theo khoảng thời gian
  5. Xử lý overlap (zero_leakage: node thuộc tập sớm nhất theo T_first)
  6. Thống kê & cảnh báo nếu lệch > 5pp
  7. Lưu kết quả ra data/preprocessed/Dataset/

Chạy:
    python temporal_split_pipeline.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path("/home/ngocvo/Desktop/ngocvo/Dynamic_Fusion")
MG_PATH     = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
OUT_DIR     = BASE_DIR / "data/preprocessed/Dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80   # 80% khoảng thời gian đầu
VAL_RATIO   = 0.10   # 10% tiếp theo
TEST_RATIO  = 0.10   # 10% còn lại
IMBALANCE_WARN_PP = 5.0   # cảnh báo nếu chênh lệch tỉ lệ illicit > 5 điểm %

OVERLAP_STRATEGY = "zero_leakage"
# "zero_leakage" : node thuộc tập sớm nhất (theo T_first) → không có overlap
# "strict_first"  : giống zero_leakage nhưng tên rõ ràng hơn

# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích
# ─────────────────────────────────────────────────────────────────────────────

def sep(title: str = "") -> None:
    width = 70
    if title:
        print(f"\n{'─'*5} {title} {'─'*(width - len(title) - 7)}")
    else:
        print("─" * width)


def pct(num: int, total: int) -> str:
    return f"{100 * num / total:.2f}%" if total else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1 — Load & kiểm tra
# ─────────────────────────────────────────────────────────────────────────────

def step1_load_and_inspect(path: Path):
    sep("BƯỚC 1 — Load & kiểm tra MulDiGraph")

    print(f"Đang load: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  Tổng số node : {n_nodes:,}")
    print(f"  Tổng số edge : {n_edges:,}")

    # Kiểm tra node attributes (lấy mẫu 5 node đầu)
    sample_nodes = list(G.nodes(data=True))[:5]
    node_attr_keys = set()
    for _, d in sample_nodes:
        node_attr_keys.update(d.keys())
    print(f"  Node attributes: {sorted(node_attr_keys)}")
    print(f"  Node samples   :")
    for addr, d in sample_nodes:
        print(f"    {addr[:20]}...  {d}")

    # Kiểm tra edge attributes (lấy mẫu 5 edge đầu)
    sample_edges = list(G.edges(data=True))[:5]
    edge_attr_keys = set()
    for _, _, d in sample_edges:
        edge_attr_keys.update(d.keys())
    print(f"  Edge attributes: {sorted(edge_attr_keys)}")
    print(f"  Edge samples   :")
    for u, v, d in sample_edges:
        print(f"    {u[:12]}→{v[:12]}  {d}")

    # Phân bố nhãn isp
    isp_counts = {0: 0, 1: 0, "missing": 0}
    for _, d in G.nodes(data=True):
        isp = d.get("isp")
        if isp == 1:
            isp_counts[1] += 1
        elif isp == 0:
            isp_counts[0] += 1
        else:
            isp_counts["missing"] += 1

    total_labeled = isp_counts[0] + isp_counts[1]
    print(f"\n  Phân bố nhãn isp (ground truth):")
    print(f"    Normal  (isp=0): {isp_counts[0]:>8,}  ({pct(isp_counts[0], n_nodes)})")
    print(f"    Illicit (isp=1): {isp_counts[1]:>8,}  ({pct(isp_counts[1], n_nodes)})")
    print(f"    Missing        : {isp_counts['missing']:>8,}")

    # Khoảng thời gian
    ts_values = []
    n_missing_ts = 0
    for u, v, d in G.edges(data=True):
        ts = d.get("timestamp") or d.get("time")
        if ts is not None:
            ts_values.append(float(ts))
        else:
            n_missing_ts += 1

    if not ts_values:
        print("  [LỖI] Không tìm thấy edge nào có timestamp!")
        sys.exit(1)

    ts_arr = np.array(ts_values, dtype=np.float64)
    ts_min, ts_max = ts_arr.min(), ts_arr.max()
    print(f"\n  Khoảng thời gian edge timestamps:")
    print(f"    Min : {ts_min:.0f}  ({_ts_to_str(ts_min)})")
    print(f"    Max : {ts_max:.0f}  ({_ts_to_str(ts_max)})")
    print(f"    Edge thiếu timestamp: {n_missing_ts:,}")

    if n_missing_ts > 0:
        print(f"  [CẢNH BÁO] Có {n_missing_ts:,} edge thiếu timestamp — sẽ bị bỏ qua.")

    return G, ts_arr, ts_min, ts_max


def _ts_to_str(ts: float) -> str:
    """Chuyển unix timestamp sang chuỗi UTC, bỏ qua lỗi nếu có."""
    try:
        import datetime
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2a — Gán nhãn (confirmed + potential fraud)
# ─────────────────────────────────────────────────────────────────────────────

def step2a_assign_labels(G) -> dict[str, int]:
    """
    Trả về dict {node_addr: label} với:
      - label=1 : confirmed fraud (isp=1) HOẶC potential fraud (isp=0, nhận từ isp=1)
      - label=0 : pure normal

    Lý do dùng potential fraud: paper phát biểu "even if only one transaction
    in the account is related to fraud, the account itself may be potentially risky."
    """
    sep("BƯỚC 2a — Gán nhãn (confirmed + potential fraud)")

    # Confirmed fraud
    fraud_confirmed = {
        addr for addr, d in G.nodes(data=True)
        if d.get("isp", 0) == 1
    }
    print(f"  Confirmed fraud (isp=1)          : {len(fraud_confirmed):,}")

    # Potential fraud: isp=0 VÀ nhận tiền từ ≥1 node isp=1
    potential_fraud: set[str] = set()
    for fa in fraud_confirmed:
        for succ in G.successors(fa):
            if G.nodes[succ].get("isp", 0) == 0:
                potential_fraud.add(succ)
    potential_fraud -= fraud_confirmed
    print(f"  Potential fraud (isp=0←fraud)    : {len(potential_fraud):,}")

    all_fraud = fraud_confirmed | potential_fraud
    total_nodes = G.number_of_nodes()
    pure_normal = total_nodes - len(all_fraud)
    print(f"  Total positive (tag=1)           : {len(all_fraud):,}  ({pct(len(all_fraud), total_nodes)})")
    print(f"  Total negative (tag=0)           : {pure_normal:,}  ({pct(pure_normal, total_nodes)})")

    labels: dict[str, int] = {}
    for addr in G.nodes():
        labels[addr] = 1 if addr in all_fraud else 0

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2b — Tính T_first cho mỗi node
# ─────────────────────────────────────────────────────────────────────────────

def step2b_compute_t_first(G) -> dict[str, float]:
    """
    T_first[node] = timestamp nhỏ nhất trong tất cả các edge mà node tham gia.

    Lý do chọn T_first thay vì T_last:
      - T_first phản ánh thời điểm node XUẤT HIỆN LẦN ĐẦU trong đồ thị.
      - Chia theo T_first → đảm bảo val/test chỉ chứa node mới (chưa thấy
        trong training) → inductive evaluation không bị data leakage.
      - T_last sẽ khiến nhiều node "cũ" (đã có giao dịch từ rất sớm) bị
        phân vào val/test vì giao dịch cuối của họ xảy ra muộn — đây là
        transductive setting và không phản ánh khả năng generalize.
    """
    sep("BƯỚC 2b — Tính T_first cho từng node")
    print("  Đang duyệt edge để tính T_first ...")

    t_first: dict[str, float] = {}

    for u, v, d in G.edges(data=True):
        ts = d.get("timestamp") or d.get("time")
        if ts is None:
            continue
        ts = float(ts)
        if u not in t_first or ts < t_first[u]:
            t_first[u] = ts
        if v not in t_first or ts < t_first[v]:
            t_first[v] = ts

    n_with_ts = len(t_first)
    n_isolated = G.number_of_nodes() - n_with_ts
    print(f"  Node có ít nhất 1 edge có ts   : {n_with_ts:,}")
    print(f"  Node isolated (không có edge)  : {n_isolated:,}")

    return t_first


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2c — Xác định t1, t2 & chia node vào train/val/test
# ─────────────────────────────────────────────────────────────────────────────

def step2c_temporal_split(
    t_first: dict[str, float],
    ts_min: float,
    ts_max: float,
    labels: dict[str, int],
):
    """
    Chia node theo khoảng thời gian (không theo phần trăm số node):
      t1 = ts_min + 0.80 * (ts_max - ts_min)  →  80% khoảng thời gian đầu
      t2 = ts_min + 0.90 * (ts_max - ts_min)  →  val chiếm 10% tiếp theo

    Gán theo T_first (zero_leakage):
      T_first ≤ t1          → train
      t1 < T_first ≤ t2     → val
      T_first > t2           → test
      Không có T_first       → bỏ qua (isolated node)

    Không có overlap vì mỗi node được gán vào DUY NHẤT một tập dựa trên
    lần đầu xuất hiện — đây là zero_leakage theo định nghĩa.
    """
    sep("BƯỚC 2c & 3 — Xác định t1/t2 và chia temporal (zero_leakage)")

    span = ts_max - ts_min
    t1 = ts_min + TRAIN_RATIO * span
    t2 = ts_min + (TRAIN_RATIO + VAL_RATIO) * span

    print(f"  ts_min : {ts_min:.0f}  ({_ts_to_str(ts_min)})")
    print(f"  ts_max : {ts_max:.0f}  ({_ts_to_str(ts_max)})")
    print(f"  span   : {span:.0f} giây  ≈ {span/86400:.1f} ngày")
    print(f"  t1     : {t1:.0f}  ({_ts_to_str(t1)})  [80% khoảng TG đầu]")
    print(f"  t2     : {t2:.0f}  ({_ts_to_str(t2)})  [90% khoảng TG đầu]")

    train_nodes: list[str] = []
    val_nodes:   list[str] = []
    test_nodes:  list[str] = []
    isolated_nodes: list[str] = []

    for node in labels:   # duyệt TẤT CẢ node trong G, kể cả isolated
        tf = t_first.get(node)
        if tf is None:
            isolated_nodes.append(node)
            continue
        if tf <= t1:
            train_nodes.append(node)
        elif tf <= t2:
            val_nodes.append(node)
        else:
            test_nodes.append(node)

    total_active = len(train_nodes) + len(val_nodes) + len(test_nodes)
    print(f"\n  Chiến lược overlap : {OVERLAP_STRATEGY}")
    print(f"  Node isolated bị loại         : {len(isolated_nodes):,}")
    print(f"  Node trong train              : {len(train_nodes):,}  ({pct(len(train_nodes), total_active)})")
    print(f"  Node trong val                : {len(val_nodes):,}  ({pct(len(val_nodes), total_active)})")
    print(f"  Node trong test               : {len(test_nodes):,}  ({pct(len(test_nodes), total_active)})")

    return train_nodes, val_nodes, test_nodes, isolated_nodes, t1, t2


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4 — Thống kê phân bố nhãn
# ─────────────────────────────────────────────────────────────────────────────

def step4_label_stats(
    train_nodes: list[str],
    val_nodes:   list[str],
    test_nodes:  list[str],
    labels: dict[str, int],
) -> dict:
    sep("BƯỚC 4 — Thống kê phân bố nhãn")

    rows = []
    all_nodes = train_nodes + val_nodes + test_nodes
    splits = [
        ("Train", train_nodes),
        ("Val",   val_nodes),
        ("Test",  test_nodes),
        ("Total", all_nodes),
    ]

    illicit_rates = []
    for name, nodes in splits[:-1]:   # không tính Total
        n_ill = sum(labels[n] for n in nodes)
        n_lic = len(nodes) - n_ill
        rate  = 100 * n_ill / len(nodes) if nodes else 0.0
        illicit_rates.append(rate)
        rows.append({
            "Tập": name,
            "Tổng node": len(nodes),
            "Illicit (n)": n_ill,
            "Illicit (%)": f"{rate:.2f}%",
            "Licit (n)": n_lic,
            "Licit (%)": f"{100-rate:.2f}%",
        })

    # Total row
    total_nodes_list = all_nodes
    total_ill = sum(labels[n] for n in total_nodes_list)
    total_lic = len(total_nodes_list) - total_ill
    total_rate = 100 * total_ill / len(total_nodes_list) if total_nodes_list else 0.0
    rows.append({
        "Tập": "Total",
        "Tổng node": len(total_nodes_list),
        "Illicit (n)": total_ill,
        "Illicit (%)": f"{total_rate:.2f}%",
        "Licit (n)": total_lic,
        "Licit (%)": f"{100-total_rate:.2f}%",
    })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Cảnh báo lệch
    max_diff = max(illicit_rates) - min(illicit_rates)
    print(f"\n  Chênh lệch max illicit% giữa các tập: {max_diff:.2f} điểm %")
    if max_diff > IMBALANCE_WARN_PP:
        print(f"  [CẢNH BÁO] Chênh lệch > {IMBALANCE_WARN_PP}pp !")
        print("  Đề xuất: xem Bước 5 — điều chỉnh t1/t2 hoặc stratified temporal split.")
    else:
        print(f"  [OK] Phân bố cân bằng (ngưỡng {IMBALANCE_WARN_PP}pp).")

    stats_json = {
        split["Tập"]: {
            "total":      split["Tổng node"],
            "illicit_n":  split["Illicit (n)"],
            "illicit_pct":float(split["Illicit (%)"].rstrip("%")),
            "licit_n":    split["Licit (n)"],
            "licit_pct":  float(split["Licit (%)"].rstrip("%")),
        }
        for split in rows
    }
    return stats_json, max_diff, illicit_rates


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5 — Cân bằng nếu cần (điều chỉnh t1/t2)
# ─────────────────────────────────────────────────────────────────────────────

def step5_balance_if_needed(
    max_diff: float,
    t_first: dict[str, float],
    labels: dict[str, int],
    ts_min: float,
    ts_max: float,
) -> tuple[list, list, list, float, float] | None:
    """
    Nếu max_diff > ngưỡng: thử dịch chuyển t1 ±5% để cân bằng phân bố.
    Trả về (train, val, test, t1_new, t2_new) nếu cải thiện, else None.
    """
    if max_diff <= IMBALANCE_WARN_PP:
        return None

    sep("BƯỚC 5 — Điều chỉnh t1/t2 để cân bằng phân bố")
    span = ts_max - ts_min

    best_result = None
    best_diff   = max_diff

    # Thử các tỉ lệ khác nhau xung quanh 80% ±10%
    for train_frac in np.arange(0.70, 0.91, 0.02):
        t1_try = ts_min + train_frac * span
        t2_try = ts_min + (train_frac + VAL_RATIO) * span

        tr, vl, te = [], [], []
        for node, tf in t_first.items():
            if tf <= t1_try:
                tr.append(node)
            elif tf <= t2_try:
                vl.append(node)
            else:
                te.append(node)

        if not vl or not te:
            continue

        rates = []
        for grp in [tr, vl, te]:
            n_ill = sum(labels[n] for n in grp)
            rates.append(100 * n_ill / len(grp) if grp else 0.0)
        diff = max(rates) - min(rates)

        if diff < best_diff:
            best_diff   = diff
            best_result = (tr, vl, te, t1_try, t2_try, train_frac)

    if best_result and best_diff < max_diff:
        tr, vl, te, t1_new, t2_new, frac = best_result
        print(f"  Tìm thấy t1/t2 tốt hơn (train_ratio={frac:.2f}):")
        print(f"    t1 mới = {t1_new:.0f}  ({_ts_to_str(t1_new)})")
        print(f"    t2 mới = {t2_new:.0f}  ({_ts_to_str(t2_new)})")
        print(f"    Chênh lệch mới: {best_diff:.2f}pp  (cũ: {max_diff:.2f}pp)")
        return tr, vl, te, t1_new, t2_new
    else:
        print("  Không tìm thấy cấu hình tốt hơn — giữ nguyên t1/t2 ban đầu.")
        print("  Đề xuất thêm: oversampling illicit trong val/test khi huấn luyện.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6 — Lưu kết quả
# ─────────────────────────────────────────────────────────────────────────────

def step6_save(
    train_nodes: list[str],
    val_nodes:   list[str],
    test_nodes:  list[str],
    stats_json:  dict,
    t1: float,
    t2: float,
    ts_min: float,
    ts_max: float,
):
    sep("BƯỚC 6 — Lưu kết quả")

    def save_pkl(obj, name):
        path = OUT_DIR / name
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Đã lưu: {path}  ({len(obj):,} items)")

    save_pkl(train_nodes, "train_nodes.pkl")
    save_pkl(val_nodes,   "val_nodes.pkl")
    save_pkl(test_nodes,  "test_nodes.pkl")

    # split_stats.json
    stats_path = OUT_DIR / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu: {stats_path}")

    # split_config.json
    config = {
        "ts_min": ts_min,
        "ts_max": ts_max,
        "t1": t1,
        "t2": t2,
        "t1_human": _ts_to_str(t1),
        "t2_human": _ts_to_str(t2),
        "train_ratio": TRAIN_RATIO,
        "val_ratio":   VAL_RATIO,
        "test_ratio":  TEST_RATIO,
        "overlap_strategy": OVERLAP_STRATEGY,
        "label_scheme": "confirmed_fraud(isp=1) + potential_fraud(isp=0_received_from_fraud)",
        "split_criterion": "T_first (earliest edge timestamp per node)",
        "n_train": len(train_nodes),
        "n_val":   len(val_nodes),
        "n_test":  len(test_nodes),
        "random_seed": 42,
    }
    config_path = OUT_DIR / "split_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu: {config_path}")

    print(f"\n  Output directory: {OUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TEMPORAL SPLIT PIPELINE — Blockchain MulDiGraph")
    print("=" * 70)

    # Bước 1
    G, ts_arr, ts_min, ts_max = step1_load_and_inspect(MG_PATH)

    # Bước 2a — gán nhãn
    labels = step2a_assign_labels(G)

    # Bước 2b — T_first
    t_first = step2b_compute_t_first(G)

    # Bước 2c + 3 — chia temporal & xử lý overlap
    train_nodes, val_nodes, test_nodes, isolated_nodes, t1, t2 = \
        step2c_temporal_split(t_first, ts_min, ts_max, labels)

    # Bước 4 — thống kê
    stats_json, max_diff, illicit_rates = \
        step4_label_stats(train_nodes, val_nodes, test_nodes, labels)

    # Bước 5 — cân bằng nếu lệch
    result5 = step5_balance_if_needed(
        max_diff, t_first, labels, ts_min, ts_max
    )
    if result5 is not None:
        train_nodes, val_nodes, test_nodes, t1, t2 = result5
        sep("BƯỚC 5 — Thống kê sau điều chỉnh")
        stats_json, max_diff, illicit_rates = \
            step4_label_stats(train_nodes, val_nodes, test_nodes, labels)

    # Bước 6 — lưu
    step6_save(
        train_nodes, val_nodes, test_nodes,
        stats_json, t1, t2, ts_min, ts_max,
    )

    sep("HOÀN THÀNH")
    print(f"  Train : {len(train_nodes):,} nodes")
    print(f"  Val   : {len(val_nodes):,} nodes")
    print(f"  Test  : {len(test_nodes):,} nodes")
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
