"""F1(pos)/AUPRC -- cùng 2 chỉ số Attempt-3 baseline dùng để báo cáo, xem STATUS.md
(Attempt-3: F1(pos)=87.63%, AUPRC overall=0.9155, pure_test F1=0.9038/AUPRC=0.9474,
overlap F1=0.8605/AUPRC=0.8846)."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def compute_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob_pos >= threshold).astype(int)
    return {
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_prob_pos)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
