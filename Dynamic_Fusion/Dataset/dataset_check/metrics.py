"""F1(pos)/AUPRC -- cùng 2 chỉ số Attempt-3 baseline dùng để báo cáo, xem STATUS.md
(Attempt-3: F1(pos)=87.63%, AUPRC overall=0.9155, pure_test F1=0.9038/AUPRC=0.9474,
overlap F1=0.8605/AUPRC=0.8846). Bổ sung precision/recall/roc_auc cho E2 v2
(xem STATUS.md) -- AUPRC vẫn là tiêu chí CHÍNH để chọn checkpoint (không phụ
thuộc threshold), các chỉ số threshold-dependent chỉ để theo dõi/báo cáo."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def p_at_k(y_true: np.ndarray, y_prob_pos: np.ndarray, k: int) -> float:
    """Precision trong top-k rank cao nhat (k bi cap o len(y_true) neu nho hon)."""
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    order = np.argsort(-y_prob_pos)
    return float(y_true[order[:k]].sum()) / k


def recall_at_k(y_true: np.ndarray, y_prob_pos: np.ndarray, k: int) -> float:
    """Ty le duong THAT lot vao top-k rank cao nhat (tren TOAN BO duong that co trong y_true)."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-y_prob_pos)
    return float(y_true[order[:k]].sum()) / n_pos


def compute_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float = 0.5,
                     k_list=None) -> dict:
    n = int(len(y_true))
    if n == 0:
        # split rong (chi xay ra khi smoke-test voi --max-examples nho, cat
        # truoc khi cham toi split do -- full run luon non-empty) -- tra ve
        # gia tri trung lap thay vi crash sklearn (doi hoi >=1 sample).
        out = {"f1_pos": 0.0, "precision_pos": 0.0, "recall_pos": 0.0, "auprc": float("nan"),
               "roc_auc": float("nan"), "threshold": float(threshold), "n": 0, "n_pos": 0}
        for k in (k_list or []):
            out[f"p_at_{k}"] = 0.0
            out[f"recall_at_{k}"] = float("nan")
        return out
    y_pred = (y_prob_pos >= threshold).astype(int)
    n_pos = int(y_true.sum())
    out = {
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_prob_pos)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_pos)) if 0 < n_pos < n else float("nan"),
        "threshold": float(threshold),
        "n": n,
        "n_pos": n_pos,
    }
    # k_list (vd [1000]) -- them P@k / Recall@k, dung cho bao cao thuc chien
    # (vd "duyet top-1000 tai khoan bat duoc bao nhieu % phisher that") --
    # mac dinh None de KHONG doi hanh vi cac noi da dung compute_metrics truoc gio.
    for k in (k_list or []):
        out[f"p_at_{k}"] = p_at_k(y_true, y_prob_pos, k)
        out[f"recall_at_{k}"] = recall_at_k(y_true, y_prob_pos, k)
    return out


def find_best_f1_threshold(y_true: np.ndarray, y_prob_pos: np.ndarray) -> float:
    """Quét threshold trên các giá trị prob THỰC TẾ xuất hiện (không phải lưới
    cố định) -- chọn threshold cho F1(pos) cao nhất. CHỈ được gọi trên VAL,
    không bao giờ gọi trên pure_test/overlap (sẽ là nhìn trộm test)."""
    candidates = np.unique(y_prob_pos)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        y_pred = (y_prob_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
