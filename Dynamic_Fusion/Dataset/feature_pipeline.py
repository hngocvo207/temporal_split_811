"""
Feature Selection using Spearman Correlation
Dataset : features_output1.csv  (is_phisher ratio 5:5 — balanced)
Target  : is_phisher  True → 1 / False → 0

Chạy  : python spearman_feature_selection.py
Output: top10_features.csv  |  top10_spearman_barchart.png

Yêu cầu: pip install pandas scipy matplotlib
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# CONFIG
DATA_PATH    = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output_split.csv"
OUTPUT_CSV   = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output_top20.csv"
OUTPUT_CHART = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/top20_spearman_barchart.png"
TOP_N        = 20

def select_and_scale_top_features(input_csv, output_csv, top_k=20):
    print("=" * 65)
    print("   SPEARMAN CORRELATION – FEATURE SELECTION  (Chống Data Leakage)")
    print("=" * 65)
    
    print(f"Đang đọc dữ liệu từ: {input_csv} ...")
    df = pd.read_csv(input_csv)

    # 1. LỌC DỮ LIỆU: Chỉ lấy tập TRAIN để phân tích
    train_df = df[df['split'] == 'TRAIN'].copy()
    
    n_p = int(train_df['label'].sum())
    n_np = int(len(train_df) - n_p)
    print(f"Số lượng Node trong tập TRAIN: {len(train_df)} (Phisher: {n_p} | Normal: {n_np})")

    # 2. XÁC ĐỊNH CÁC CỘT ĐẶC TRƯNG
    exclude_cols = ['node', 'sample_idx', 'split', 'label', 'is_phisher', 'n_nodes', 'n_edges']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Có tổng cộng {len(feature_cols)} đặc trưng tham gia đánh giá Spearman.")

    # 3. TÍNH SPEARMAN CORRELATION CHỈ TRÊN TẬP TRAIN
    correlations = {}
    for col in feature_cols:
        coef, p_value = spearmanr(train_df[col], train_df['label'])
        # Sửa lỗi: Lưu giá trị GỐC (có âm dương) để vẽ biểu đồ
        correlations[col] = coef if not np.isnan(coef) else 0

    # Sửa lỗi: Sắp xếp theo TRỊ TUYỆT ĐỐI của hệ số tương quan
    sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_k_list = sorted_features[:top_k]
    top_k_features = [f[0] for f in top_k_list]

    print(f"\n--- TOP {top_k} ĐẶC TRƯNG TỐT NHẤT (Không Leakage) ---")
    for i, (feat, score) in enumerate(top_k_list, 1):
        print(f"{i:2d}. {feat:<25}: {score:+.4f}")

    # Tạo DataFrame tóm tắt để truyền sang hàm vẽ biểu đồ
    plot_df = pd.DataFrame([{"feature": f, "spearman_corr": c} for f, c in top_k_list])

    # 4. CHUẨN HÓA DỮ LIỆU (Standardization)
    print("\nĐang chuẩn hóa dữ liệu ...")
    scaler = StandardScaler()
    
    # Chỉ fit trên tập TRAIN
    scaler.fit(train_df[top_k_features])
    
    # Áp dụng (transform) cho TOÀN BỘ dữ liệu (Train, Val, Test)
    df_scaled = df.copy()
    df_scaled[top_k_features] = scaler.transform(df[top_k_features])
    output_cols = ['node'] + top_k_features

    # 5. XUẤT RA FILE CUỐI CÙNG CHO MÔ HÌNH
    output_cols = ['node'] + top_k_features
    final_df = df_scaled[output_cols]
    final_df.to_csv(output_csv, index=False)
    
    print(f"[✓] Đã xuất thành công Top {top_k} đặc trưng ra file: {output_csv}")
    print(f"    (Dữ liệu có kích thước: {final_df.shape[0]} nodes, {final_df.shape[1]} cột)")
    
    return plot_df, n_p, n_np

def visualize_top_spearman(top_df, n_p, n_np):
    # Dữ liệu truyền vào hàm này giờ là plot_df (chứa cột 'feature' và 'spearman_corr')
    plot_df    = top_df.sort_values('spearman_corr')
    corr_vals  = plot_df['spearman_corr'].values
    feat_names = plot_df['feature'].values
    bar_colors = ['#ff6b6b' if v < 0 else '#4fc3f7' for v in corr_vals]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    bars = ax.barh(feat_names, corr_vals, color=bar_colors,
                   edgecolor='#30363d', linewidth=0.6, height=0.62)

    x_max = max(abs(corr_vals)); pad = x_max * 0.02
    for bar, val in zip(bars, corr_vals):
        ha  = 'left'  if val >= 0 else 'right'
        pos = val + pad if val >= 0 else val - pad
        ax.text(pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.4f}', va='center', ha=ha,
                color='white', fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#0d1117', ec='none', alpha=0.6))

    ax.axvline(0, color='#8b949e', linewidth=1.0, linestyle='--', alpha=0.6)
    xlim = ax.get_xlim()
    if xlim[0] < 0: ax.axvspan(xlim[0], 0, alpha=0.04, color='#ff6b6b')
    if xlim[1] > 0: ax.axvspan(0, xlim[1], alpha=0.04, color='#4fc3f7')

    ax.set_xlabel('Spearman Correlation Coefficient', color='#c9d1d9', fontsize=12, labelpad=10)
    ax.set_title(
        f'Top {TOP_N} Features — Spearman Correlation with is_phisher\n'
        f'(Train dataset: {n_p} phishers / {n_np} non-phishers)',
        color='#e6edf3', fontsize=13, fontweight='bold', pad=14)
    
    ax.tick_params(axis='both', colors='#c9d1d9', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#30363d'); spine.set_linewidth(0.8)
    ax.grid(axis='x', color='#21262d', linewidth=0.6, linestyle='--')

    pos_patch = mpatches.Patch(color='#4fc3f7', label='Positive correlation (↑ phisher)')
    neg_patch = mpatches.Patch(color='#ff6b6b', label='Negative correlation (↓ phisher)')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right',
              facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=10, framealpha=0.9)
    
    fig.text(0.5, -0.02,
             'Sorted by absolute Spearman correlation | Chỉ đánh giá trên tập TRAIN để tránh Data Leakage',
             ha='center', color='#8b949e', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n[✓] Bar chart → {OUTPUT_CHART}")
    print("\n" + "=" * 65)
    print("   HOÀN THÀNH!")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc Top 20 đặc trưng bằng Spearman")
    parser.add_argument("--input", default=DATA_PATH, help="File CSV chứa toàn bộ đặc trưng đã extract")
    parser.add_argument("--output", default=OUTPUT_CSV, help="File output dùng cho training (train1.py)")
    parser.add_argument("--k", type=int, default=TOP_N, help="Số lượng đặc trưng cần lấy")
    
    args = parser.parse_args()
    
    # Lấy ra plot_df đã được fix
    plot_df, n_p, n_np = select_and_scale_top_features(args.input, args.output, args.k)
    
    # Đẩy plot_df vào hàm vẽ
    visualize_top_spearman(plot_df, n_p=n_p, n_np=n_np)
