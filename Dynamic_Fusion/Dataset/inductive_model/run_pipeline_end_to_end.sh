#!/bin/bash
# =============================================================================
# run_pipeline_end_to_end.sh
# =============================================================================
# Mo ta (khong phai "chay mu tu tren xuong duoi") toan bo pipeline end-to-end
# cua inductive_model, DUNG THU TU THAT da thuc hien trong du an, tinh den
# 2026-09. Moi giai doan la 1 ham rieng, COMMENT RO thoi gian thuc te da do +
# trang thai (DA XONG / CHUA LAM) -- doc truoc, chon giai doan can chay, dung
# copy-paste ca file roi bam Enter, vi nhieu buoc ton VAI GIO DEN VAI CHUC GIO.
#
# Quy uoc thu muc lam viec:
#   - Cac script mg_*.py / temporal_pyg.py       chay tu   Dynamic_Fusion/Dataset/
#   - Cac script data_prep/*, model/*, train_eval/* chay tu Dynamic_Fusion/Dataset/inductive_model/
#     (dung "python3 -m package.module ..." tu day, KHONG "python3 path/to/file.py")
#
# Trang thai tong quan (xem STATUS.md de biet chi tiet/ly do):
#   [DA XONG]  A1-A3, B1-B4, C1, D1  (ha tang co ban)
#   [DA XONG]  Bug huong canh da phat hien + sua (2026-09-06)
#   [DA XONG]  Split moi (quantile, khong overlap) da build, moi wire duoc
#              cho nhanh graph-only, CHUA wire cho nhanh BERT/fusion
#   [DANG CHAY / CHUA XONG] E2 v2 retrain voi bug-fix + optimizer moi
#   [CHUA LAM] Rebuild corpus BERT cho split moi; train full 1.9M account
# =============================================================================
set -euo pipefail
DYNAMIC_FUSION_ROOT="/home/ngocvo/Desktop/ngocvo/Dynamic_Fusion"
INDUCTIVE_MODEL_DIR="$DYNAMIC_FUSION_ROOT/Dataset/inductive_model"

# -----------------------------------------------------------------------------
# GIAI DOAN 0 -- Du lieu tho (chi can co san, khong can chay lai)
# -----------------------------------------------------------------------------
# raw_data/MulDiGraph/MulDiGraph.pkl              2,973,489 node / 13,551,303 canh
# raw_data/MulDiGraph/phisher_account_muldi.txt   1,165 dia chi phishing xac nhan
# raw_data/MulDiGraph/features_output_fullscale.csv   23 dac trung tabular THO (chua scale)
stage0_check_raw_data() {
    cd "$DYNAMIC_FUSION_ROOT"
    ls -la raw_data/MulDiGraph/MulDiGraph.pkl raw_data/MulDiGraph/phisher_account_muldi.txt
}

# -----------------------------------------------------------------------------
# GIAI DOAN 1 -- Temporal split + adjacency + node features [DA XONG]
# -----------------------------------------------------------------------------
# 1a. Split "CU" (dang dung cho MOI ket qua da train tinh den nay):
#     T_cutoff = percentile 80% tren TOAN BO giao dich -> data/preprocessed/Dataset_MG/
#     Thoi gian thuc te: ~1 phut.
stage1a_temporal_split_old() {
    cd "$DYNAMIC_FUSION_ROOT"
    python3 Dataset/mg_temporal_pipeline.py
    python3 Dataset/mg_build_adjacency.py   # tao adj_train.npz / adj_inference.npz / address_to_index.pkl
}

# 1b. Split "MOI" (quantile tren t_first cua 1,165 phisher, KHONG overlap,
#     train/val/test = 65/15/20%) -> data/preprocessed/Dataset_MG_v3_no_overlap/
#     CHI dung duoc day du cho nhanh graph-only ngay bay gio (xem STATUS.md
#     muc "Da wire split v3" -- nhanh BERT/fusion CAN build lai corpus o
#     Giai doan 3b truoc khi dung split nay). Thoi gian: ~58 giay.
stage1b_temporal_split_new_quantile() {
    cd "$DYNAMIC_FUSION_ROOT"
    python3 Dataset/mg_temporal_pipeline_v3_no_overlap.py
}

# 1c. Dac trung node (23 cot) -> data/node_features_all23.pt + scaler
stage1c_node_features() {
    cd "$INDUCTIVE_MODEL_DIR"
    python3 -m data_prep.build_node_features
    python3 -m data_prep.compute_feature_scaler
    python3 -m data_prep.build_graph_data   # -> data/graph_train.pt, data/graph_inference.pt
}

# 1d. Kiem tra huong canh nhat quan train/eval [BAT BUOC chay lai sau MOI lan
#     dung adj_train/inference moi hoac sua model/graph_sampling.py] -- ~2 giay.
#     PASS 100/100 node ca 2 do thi la dieu kien can truoc khi train bat ky gi.
stage1d_verify_edge_direction() {
    cd "$INDUCTIVE_MODEL_DIR"
    python3 -m data_prep.verify_edge_direction_consistency
}

# -----------------------------------------------------------------------------
# GIAI DOAN 2 -- Baseline GBM (tabular, KHONG graph/BERT) [DA XONG, ~1 PHUT]
# -----------------------------------------------------------------------------
# Luon chay truoc tien: RE, nhanh (vai chuc giay), cho 1 moc tham chieu
# "tran thap nhat can vuot" truoc khi dau tu hang gio vao GraphSAGE/BERT.
# Ket qua da co: AUPRC=0.347 (da tune), full pure_test that 609,773 account.
stage2_gbm_baseline() {
    cd "$INDUCTIVE_MODEL_DIR"
    # sweep hyperparameter (18 config) + 5-fold CV, chon theo CV (khong nhin
    # test luc chon) -- ~vai phut. Tu dong nhan dien split cu ('pure_test')
    # hay split moi ('test') tu chinh partition.pkl dang duoc PREPROC_DIR
    # (data_prep/labels_io.py) tro toi.
    python3 -m train_eval.gbm_baseline
}

# -----------------------------------------------------------------------------
# GIAI DOAN 3 -- Corpus van ban BERT [3a DA XONG cho split CU; 3b CHUA LAM]
# -----------------------------------------------------------------------------
# 3a. Corpus train 100,000 account (519 duong that + 99,481 am mau, nhan
#     strict) + corpus test day du (811,704 = 609,773 pure_test + 201,931
#     overlap) -- CHO SPLIT CU. Thoi gian: build corpus 100k ~vai phut,
#     full_test_corpus ~lau hon (811k doc).
stage3a_bert_corpus_old_split() {
    cd "$DYNAMIC_FUSION_ROOT"
    python3 Dataset/mg_build_examples.py --cap_train 100000 --labels_source labels.pkl \
        --out_dir runs/inductive_e2_corpus_100k/corpus
    python3 Dataset/mg_build_full_test_eval.py   # -> runs/expanded_test_attempt3/full_test_corpus/
}

# 3b. [CHUA LAM] Build lai corpus tuong tu NHUNG cho split MOI (v3_no_overlap)
#     -- CAN THIET truoc khi E2 v2/v3 co the dung split moi. Chi phi: tuong
#     duong 3a nhung tren tap account khac (2,217,282 train candidates thay vi
#     1,945,607) -- CHUA UOC LUONG THOI GIAN THAT, can do truoc khi cam ket.
stage3b_bert_corpus_new_split_TODO() {
    echo "CHUA LAM -- can sua mg_build_examples.py de doc tu"
    echo "data/preprocessed/Dataset_MG_v3_no_overlap/ (partition co 'test' thay 'pure_test'/'overlap')"
    echo "truoc khi chay buoc nay. Xem STATUS.md muc 'eval_full_test.py -- KHONG doi'."
}

# -----------------------------------------------------------------------------
# GIAI DOAN 4 -- Graph-only hypothesis test (chan doan RE, khong BERT) [DA XONG]
# -----------------------------------------------------------------------------
# Kiem chung rieng nhanh GraphSAGE truoc khi dau tu BERT+fusion (dat qua nhat
# sau khi sua bug huong canh: AUPRC pure_test = 0.303, gan bang GBM).
stage4_graph_only_hypothesis() {
    cd "$INDUCTIVE_MODEL_DIR"
    # split CU (mac dinh, tuong thich nguoc):
    python3 -m train_eval.train_graph_only_hypothesis --epochs 150 --patience 30 \
        --lr 0.003 --hidden 64 --out 64 --mlp-classifier
    # HOAC split MOI (--split-dir, xem Giai doan 1b):
    # python3 -m train_eval.train_graph_only_hypothesis --epochs 150 --patience 30 \
    #     --lr 0.003 --hidden 64 --out 64 --mlp-classifier \
    #     --split-dir "$DYNAMIC_FUSION_ROOT/data/preprocessed/Dataset_MG_v3_no_overlap" \
    #     --output-suffix "_v3"
}

# -----------------------------------------------------------------------------
# GIAI DOAN 5 -- E2 v2: BERT + GraphSAGE fusion, train that [DANG CHAY]
# -----------------------------------------------------------------------------
# Corpus 100k (Giai doan 3a) + split CU + bug huong canh DA SUA + optimizer
# rieng LR cho tung nhom tham so (BERT 2e-5, graph/fusion/classifier 1e-3).
# Thoi gian THAT DA DO: ~91.5 phut/epoch, 15 epoch toi da (patience=4) ->
# tong ~14 gio wall-clock. CHAY TRONG TMUX (khong chay truc tiep, se mat neu
# session ngat):
stage5_train_e2v2_fusion() {
    cd "$INDUCTIVE_MODEL_DIR"
    tmux new-session -d -s e2_v2_retrain \
        'python3 -u -m train_eval.train_e2_v2 2>&1 | tee output/e2_v2_retrain.log; echo TRAIN_EXIT_CODE=$?'
    echo "Theo doi: tmux attach -t e2_v2_retrain  (Ctrl+B roi D de thoat ma khong dung)"
}

# -----------------------------------------------------------------------------
# GIAI DOAN 6 -- Eval full-scale + so sanh voi GBM [SAN SANG, CHO GIAI DOAN 5 XONG]
# -----------------------------------------------------------------------------
# Eval TREN TOAN BO pure_test that (609,773) + overlap that (201,931), khong
# phai mau nho -- tu in bang so sanh voi GBM da tune (Giai doan 2).
# Thoi gian thuc te da do: ~3.1h (pure_test) + ~0.9h (overlap) o ~54-60 acc/s.
stage6_eval_full_and_compare_gbm() {
    cd "$INDUCTIVE_MODEL_DIR"
    python3 -u -m train_eval.eval_full_test \
        --checkpoint output/e2_v2_best_checkpoint.pt \
        --out-json e2_v2_full_eval_result.json \
        --group e2_v2_full_eval
}

# -----------------------------------------------------------------------------
# Cho phep chay 1 giai doan cu the tu dong lenh, vd:
#   ./run_pipeline_end_to_end.sh stage2_gbm_baseline
#   ./run_pipeline_end_to_end.sh stage6_eval_full_and_compare_gbm
# Khong truyen gi -> chi in danh sach giai doan, KHONG tu dong chay gi ca.
# -----------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "Cac giai doan co san (xem comment trong file de biet chi tiet + trang thai):"
    grep -oP '^stage\w+(?=\(\))' "$0"
    echo
    echo "Chay:  ./run_pipeline_end_to_end.sh <ten_giai_doan>"
else
    "$1"
fi
