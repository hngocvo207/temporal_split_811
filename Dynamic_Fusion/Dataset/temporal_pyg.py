"""
Temporal + inductive split cho bộ MulDiGraph (Ethereum phishing) dùng PyTorch
Geometric, thiết kế riêng cho phân bố t_first LỆCH MẠNH: 67.5% số node
phishing (786/1,165) rơi vào 6 tháng đầu 2018, đỉnh điểm T5/2018 (193 node,
16.6% toàn tập). Gần như không có gì trước T7/2017.

Ý TƯỞNG CỐT LÕI (đã kiểm chứng bằng mô phỏng số):
  - KHÔNG cắt mốc train/val/test theo % độ dài lịch (vd: 65% khoảng
    Aug/2015-Jan/2019). Vì độ lệch quá mạnh, cách này khiến train chỉ nhận
    được ~5% số node dương, val nuốt tới ~60%, test ~35% — méo hoàn toàn.
  - Thay vào đó cắt theo PHÂN VỊ (quantile) trên chính 1,165 giá trị t_first.
    Ngưỡng q=0.65 / q=0.80 cho ra train≈65%, val≈15%, test≈20% số node dương,
    đúng như thiết kế mong muốn, bất kể mật độ thời gian không đều.
  - Đồ thị dùng theo kiểu "growing graph" (tích lũy): subgraph của val chứa
    toàn bộ lịch sử subgraph train + phần mới; subgraph test chứa toàn bộ
    lịch sử trước đó + phần mới nhất. Đây là cách chuẩn để GraphSAGE inductive
    suy luận trên node MỚI mà vẫn tận dụng được cấu trúc/ngữ cảnh lịch sử.
  - Riêng nhãn (y) trong mỗi Data object CHỈ gán cho các node "mới xuất hiện"
    trong đúng cửa sổ thời gian của split đó (train: mọi node đến t1; val:
    chỉ node mới trong [t1, t2); test: chỉ node mới trong [t2, t_end]) —
    tránh đánh giá lại trên node đã dùng để train.

TRƯỚC KHI CHẠY - bạn cần thay phần load_raw_data() bằng dữ liệu thật:
  - node_first_time : np.ndarray [N_total]  first-seen timestamp của MỌI node
                       (không chỉ 1,165 phishing) - cần để giới hạn pool âm
                       đúng cửa sổ thời gian. Nếu chưa có, có thể tạm dùng
                       timestamp của giao dịch (edge) sớm nhất liên quan node đó.
  - edge_index       : LongTensor [2, E]     chỉ số node theo cặp (src, dst)
  - edge_time        : LongTensor [E]        timestamp của từng cạnh (khớp cột)
  - node_feat        : FloatTensor [N_total, F]  đặc trưng node (có thể để None
                       và tự khởi tạo feature rỗng nếu bạn dùng feature học được)
  - phishing_idx     : np.ndarray [1165]     chỉ số (index) của 1,165 node phishing
  - t_first_pos      : np.ndarray [1165]     first-seen timestamp tương ứng
                       (chính là nội dung t_first.pkl đã lọc riêng cho phishing)
"""

from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# 1. Cutoff theo QUANTILE (không theo % lịch) - đã kiểm chứng bằng mô phỏng
# ---------------------------------------------------------------------------
def compute_quantile_cutoffs(t_first_pos: np.ndarray,
                              q_train_val: float = 0.65,
                              q_val_test: float = 0.80) -> Tuple[int, int]:
    t_sorted = np.sort(t_first_pos)
    t1 = int(np.quantile(t_sorted, q_train_val))
    t2 = int(np.quantile(t_sorted, q_val_test))
    return t1, t2


# ---------------------------------------------------------------------------
# 2. Negative sampling CHỈ trong đúng cửa sổ thời gian của split
# ---------------------------------------------------------------------------
def sample_negatives_in_window(all_node_idx: np.ndarray,
                                node_first_time: np.ndarray,
                                phishing_idx: np.ndarray,
                                t_lo: int, t_hi: int,
                                n_pos_in_window: int,
                                ratio: int = 10,
                                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Chỉ chọn node "âm" có first-seen timestamp nằm trong [t_lo, t_hi) - tức
    cùng "sinh ra" trong đúng giai đoạn với các node dương của split này.
    Tránh lấy negative từ tương lai (leak) hoặc quá khứ quá xa (không thực tế).
    """
    rng = rng or np.random.default_rng(42)
    in_window = (node_first_time >= t_lo) & (node_first_time < t_hi)
    mask = in_window.copy()
    mask[phishing_idx] = False  # loại toàn bộ 1,165 phishing khỏi pool âm
    candidates = all_node_idx[mask]

    n_neg = min(len(candidates), n_pos_in_window * ratio)
    if n_neg == 0:
        return np.array([], dtype=candidates.dtype)
    return rng.choice(candidates, size=n_neg, replace=False)


# ---------------------------------------------------------------------------
# 3. Subgraph tích lũy theo thời gian (growing graph) cho GraphSAGE inductive
# ---------------------------------------------------------------------------
def build_temporal_subgraph(edge_index: torch.Tensor,
                             edge_time: torch.Tensor,
                             node_feat: torch.Tensor,
                             node_first_time: np.ndarray,
                             t_cutoff: int) -> Data:
    """
    Giữ node có first_time < t_cutoff và cạnh có time <= t_cutoff.
    Đây là subgraph "tại thời điểm t_cutoff" - dùng chung cho train (t1),
    val (t2), test (t_end): mỗi cái là một lát cắt tích lũy lớn dần.
    """
    node_mask_np = node_first_time < t_cutoff
    keep_idx = np.nonzero(node_mask_np)[0]
    node_mask = torch.as_tensor(node_mask_np)

    edge_time_mask = edge_time <= t_cutoff
    ei = edge_index[:, edge_time_mask]
    both_kept = node_mask[ei[0]] & node_mask[ei[1]]
    ei = ei[:, both_kept]

    # remap global index -> local index trong subgraph
    remap = -np.ones(node_feat.shape[0], dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    ei_local = torch.as_tensor(remap)[ei]

    data = Data(x=node_feat[keep_idx], edge_index=ei_local)
    data.global_idx = torch.as_tensor(keep_idx)  # map ngược ra chỉ số gốc
    data._remap = remap  # lưu tạm để gán nhãn ở bước sau (không phải field chuẩn của PyG)
    return data


# ---------------------------------------------------------------------------
# 4. Lắp ráp toàn bộ pipeline: 3 Data object train/val/test
# ---------------------------------------------------------------------------
def build_splits(edge_index: torch.Tensor,
                  edge_time: torch.Tensor,
                  node_feat: torch.Tensor,
                  node_first_time: np.ndarray,
                  phishing_idx: np.ndarray,
                  t_first_pos: np.ndarray,
                  neg_ratio: int = 10,
                  q_train_val: float = 0.65,
                  q_val_test: float = 0.80) -> Dict[str, Data]:

    t1, t2 = compute_quantile_cutoffs(t_first_pos, q_train_val, q_val_test)
    t_end = int(node_first_time.max()) + 1
    print(f"Cutoff train/val = {t1} | Cutoff val/test = {t2}")

    all_node_idx = np.arange(node_feat.shape[0])
    windows = {
        "train": (node_first_time.min(), t1, t1),
        "val":   (t1, t2, t2),
        "test":  (t2, t_end, t_end),
    }

    splits: Dict[str, Data] = {}
    for name, (lo, hi, t_cutoff) in windows.items():
        # (a) node dương MỚI trong cửa sổ này
        pos_mask = (t_first_pos >= lo) & (t_first_pos < hi)
        pos_idx = phishing_idx[pos_mask]

        # (b) node âm lấy mẫu MỚI trong cùng cửa sổ
        neg_idx = sample_negatives_in_window(
            all_node_idx, node_first_time, phishing_idx,
            lo, hi, n_pos_in_window=len(pos_idx), ratio=neg_ratio,
        )

        # (c) subgraph tích lũy tại t_cutoff (chứa toàn bộ lịch sử trước đó)
        data = build_temporal_subgraph(edge_index, edge_time, node_feat,
                                        node_first_time, t_cutoff)

        # (d) gán nhãn: chỉ node mới của cửa sổ này mới có y xác định
        y = torch.full((data.num_nodes,), fill_value=-1, dtype=torch.long)
        labeled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        for idx, label in zip(pos_idx, [1] * len(pos_idx)):
            local = data._remap[idx]
            if local >= 0:
                y[local] = label
                labeled_mask[local] = True
        for idx in neg_idx:
            local = data._remap[idx]
            if local >= 0:
                y[local] = 0
                labeled_mask[local] = True

        data.y = y
        data.labeled_mask = labeled_mask
        del data._remap  # không cần giữ lại, chỉ dùng nội bộ

        splits[name] = data
        print(f"[{name}] duong_moi={len(pos_idx)}, am_moi={len(neg_idx)}, "
              f"tong_node_subgraph={data.num_nodes}, tong_canh={data.num_edges}")

    return splits


# ---------------------------------------------------------------------------
# 5. Load dữ liệu THẬT (2026-09-07) -- dùng ĐÚNG index space voi
#    address_to_index.pkl / node_features_all23.pt cua pipeline
#    inductive_model hien co, de 2 huong split co the doi chieu truc tiep.
# ---------------------------------------------------------------------------
import pickle
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # .../Dynamic_Fusion
PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "Dataset_MG"
MG_PATH = REPO_ROOT / "raw_data" / "MulDiGraph" / "MulDiGraph.pkl"
PHISHER_LIST_PATH = REPO_ROOT / "raw_data" / "MulDiGraph" / "phisher_account_muldi.txt"
NODE_FEATURES_PT = REPO_ROOT / "Dataset" / "inductive_model" / "data" / "node_features_all23.pt"


def load_raw_data():
    """Doc that tu MulDiGraph.pkl + t_first.pkl + address_to_index.pkl +
    node_features_all23.pt (CUNG index space voi toan bo pipeline
    inductive_model hien co -- global_idx o day == global_idx cac noi khac
    trong du an, cho phep doi chieu truc tiep 2 cach split)."""
    t0 = time.time()
    print("loading address_to_index.pkl...")
    with open(PREPROC_DIR / "address_to_index.pkl", "rb") as f:
        addr2idx = pickle.load(f)
    n_total = len(addr2idx)
    print(f"  n_total={n_total} ({time.time()-t0:.1f}s)")

    print("loading t_first.pkl (first-seen timestamp, MOI node)...")
    with open(PREPROC_DIR / "t_first.pkl", "rb") as f:
        t_first_map = pickle.load(f)
    node_first_time = np.empty(n_total, dtype=np.int64)
    for addr, idx in addr2idx.items():
        node_first_time[idx] = int(t_first_map[addr])
    print(f"  done ({time.time()-t0:.1f}s)")

    print("loading node_features_all23.pt...")
    node_feat = torch.load(NODE_FEATURES_PT, weights_only=False)
    assert node_feat.shape[0] == n_total
    print(f"  shape={tuple(node_feat.shape)} ({time.time()-t0:.1f}s)")

    print(f"loading {PHISHER_LIST_PATH.name} (1,165 dia chi phishing xac nhan)...")
    with open(PHISHER_LIST_PATH) as f:
        phisher_addrs = [line.strip().lower() for line in f if line.strip()]
    phishing_idx = np.array([addr2idx[a] for a in phisher_addrs], dtype=np.int64)
    t_first_pos = node_first_time[phishing_idx]
    print(f"  n_phishing={len(phishing_idx)} ({time.time()-t0:.1f}s)")

    print("loading MulDiGraph.pkl (do thi giao dich tho, ~1.26GB)...")
    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"  n_nodes={G.number_of_nodes()} n_edges={G.number_of_edges()} ({time.time()-t0:.1f}s)")

    print("dung edge_index/edge_time tu MulDiGraph (map ve global index)...")
    n_edges = G.number_of_edges()
    src = np.empty(n_edges, dtype=np.int64)
    dst = np.empty(n_edges, dtype=np.int64)
    etime = np.empty(n_edges, dtype=np.int64)
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        src[i] = addr2idx[u]
        dst[i] = addr2idx[v]
        etime[i] = int(d["timestamp"])
    edge_index = torch.from_numpy(np.stack([src, dst]))
    edge_time = torch.from_numpy(etime)
    print(f"  edge_index={tuple(edge_index.shape)} ({time.time()-t0:.1f}s)")

    print(f"load_raw_data() xong, tong thoi gian {time.time()-t0:.1f}s")
    return edge_index, edge_time, node_feat, node_first_time, phishing_idx, t_first_pos


if __name__ == "__main__":
    (edge_index, edge_time, node_feat,
     node_first_time, phishing_idx, t_first_pos) = load_raw_data()

    splits = build_splits(
        edge_index, edge_time, node_feat,
        node_first_time, phishing_idx, t_first_pos,
        neg_ratio=10,          # bắt đầu với 1:10, có thể tăng dần lên 1:20+
        q_train_val=0.65,
        q_val_test=0.80,
    )

    # Ví dụ dùng với GraphSAGE của PyG:
    #
    # from torch_geometric.nn import SAGEConv
    # from torch_geometric.loader import NeighborLoader
    #
    # train_loader = NeighborLoader(
    #     splits["train"], num_neighbors=[25, 10], batch_size=256,
    #     input_nodes=splits["train"].labeled_mask,
    # )
    # ... huấn luyện trên train_loader, đánh giá inductive trên splits["val"]/["test"]
    # bằng cách forward toàn bộ subgraph tương ứng và chỉ tính loss/metric trên
    # labeled_mask của subgraph đó.