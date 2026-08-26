"""
Addendum phát hiện khi làm B4 (predict cho account hoàn toàn mới): 23 cột feature
trong features_output_all23_MG_fullscale.csv đã được StandardScaler hoá (train-only
fit, xem fullscale_features/08_assemble_csv.py dòng 160-185: OUT_ALL fit trên
partition 'train'), nhưng mean/std dùng để fit KHÔNG được lưu ra đĩa -- script gốc
chỉ in ra console rồi thoát. Không có mean/std thì không thể đưa feature thô của 1
account mới (Case B, B4) vào đúng không gian đã chuẩn hoá mà GNN được train trên.

Script này recompute mean/std từ file RAW CHƯA SCALE
(raw_data/MulDiGraph/features_output_fullscale.csv == OUT_FULL trong
08_assemble_csv.py, có sẵn cột mg_partition) -- KHÔNG phải từ
features_output_all23_MG_fullscale.csv (đó là OUT_ALL, đã scale rồi, dùng lại nó ở
đây sẽ tuần hoàn/vô nghĩa). Sau khi tính xong, verify bằng cách áp (raw-mean)/std
lên vài dòng rồi so trực tiếp với giá trị thật trong OUT_ALL.
"""
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Dynamic_Fusion
RAW_UNSCALED_CSV = REPO_ROOT / "raw_data" / "MulDiGraph" / "features_output_fullscale.csv"
SCALED_ALL23_CSV = REPO_ROOT / "raw_data" / "MulDiGraph" / "features_output_all23_MG_fullscale.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    df = pd.read_csv(RAW_UNSCALED_CSV)
    feature_cols = [c for c in df.columns if c not in ("node", "label", "mg_partition", "n_nodes", "n_edges")]
    assert len(feature_cols) == 23, f"expected 23 feature columns, got {len(feature_cols)}: {feature_cols}"

    train_mask = (df["mg_partition"] == "train").to_numpy()
    print(f"train rows: {train_mask.sum()} / {len(df)} ({100*train_mask.mean():.1f}%)")

    sub = df.loc[train_mask, feature_cols]
    mean = sub.mean()
    std = sub.std().replace(0, 1.0)

    # Real correctness check (not circular): apply the recomputed transform to
    # the raw rows and compare against the actual pre-scaled all23 CSV values.
    scaled_check = pd.read_csv(SCALED_ALL23_CSV, nrows=2000)
    raw_check = df.iloc[:2000]
    assert (raw_check["node"].to_numpy() == scaled_check["node"].to_numpy()).all(), \
        "row order mismatch between raw and scaled CSVs -- cannot validate scaler"
    recomputed = (raw_check[feature_cols] - mean) / std
    max_abs_diff = (recomputed.to_numpy() - scaled_check[feature_cols].to_numpy()).__abs__().max()
    print(f"validation vs actual OUT_ALL (first 2000 rows): max abs diff = {max_abs_diff:.6e}")
    if max_abs_diff > 1e-3:
        raise ValueError(
            f"recomputed scaler does not reproduce features_output_all23_MG_fullscale.csv "
            f"(max abs diff {max_abs_diff:.4e}) -- do not trust this scaler for B4."
        )

    out = {
        "feature_columns": feature_cols,
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "note": (
            "z = (raw - train_mean) / train_std, fit on mg_partition=='train' rows of "
            "features_output_fullscale.csv (OUT_FULL, unscaled), reproducing OUT_ALL's "
            "scaler (verified against features_output_all23_MG_fullscale.csv, see "
            "validation printed at generation time). node_features_all23.pt already has "
            "this transform applied -- this file exists only so new (never-seen) "
            "accounts (B4) can be normalized into the same space at inference time."
        ),
    }
    out_path = OUT_DIR / "node_features_all23.scaler.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
