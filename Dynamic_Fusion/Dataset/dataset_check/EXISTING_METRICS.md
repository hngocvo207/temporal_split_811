# Bảng metric hiện có — mức lệch giữa các split/slice

Gộp lại **mọi kết quả đã chạy thật** (không phải dự đoán) trong dự án tính đến
nay, để thấy được mức lệch giữa `pure_test` (chưa từng thấy hoàn toàn) vs
`overlap`/`val` (còn dính một phần cấu trúc) vs split cũ vs split v3. Mỗi
dòng ghi rõ **file nguồn** để tự đối chiếu lại.

## A. BERT+GCN — `bi_model` / `tri_model` (split cũ, T_cutoff = percentile 80% toàn giao dịch)

### A1. Bounded-scale (20,000 train / 5,000 val / 10,000 test, gcn_vocab=35,000)

| Model | Slice | n | pos | F1(pos) | AUPRC |
|---|---|---:|---:|---:|---:|
| tri_model (Attempt 3) | pure_test | 5,000 | 312 | 0.7406 | 0.8194 |
| tri_model (Attempt 3) | overlap | 5,000 | 217 | 0.5107 | 0.6285 |
| tri_model (Attempt 3) | **overall** | 10,000 | 529 | **0.6227** | 0.6904 |
| bi_model (Attempt 3-parity) | pure_test | 5,000 | 312 | 0.8209 | 0.8751 |
| bi_model (Attempt 3-parity) | overlap | 5,000 | 217 | 0.5007 | 0.5172 |
| bi_model (Attempt 3-parity) | **overall** | 10,000 | 529 | **0.6476** | 0.6595 |

Nguồn: `preprocessing_and_eval_report.md` §11, `bi_modal_run_report.md` §9.

### A2. Full-scale evaluation-only (toàn bộ 811,704 test thật — checkpoint bounded ở trên, KHÔNG train lại)

| Model | Slice | n | pos | F1(pos) | AUPRC |
|---|---|---:|---:|---:|---:|
| tri_model | pure_test | 609,773 | 312 | 0.0840 | 0.1176 |
| tri_model | overlap | 201,931 | 217 | 0.0371 | 0.1793 |
| tri_model | **overall** | 811,704 | 529 | **0.0549** | 0.1241 |
| bi_model | pure_test | 609,773 | 312 | 0.0858 | 0.1159 |
| bi_model | overlap | 201,931 | 217 | 0.0293 | 0.0566 |
| bi_model | **overall** | 811,704 | 529 | **0.0474** | 0.0801 |

Nguồn: `full_scale_test_eval_report.md` §4/§6.

**Mức lệch A1 → A2 (nguyên nhân đã xác định)**: F1(pos) sụp 0.62-0.65 →
0.047-0.055 — vì `VocabGraphConvolution.W0_vh` transductive (1 hàng embedding
học riêng/account cụ thể), 98.8% tài khoản test thật (801,704/811,704) chưa
từng nằm trong vocab 35,000 lúc train.

## B. Graph-only (GraphSAGE, không BERT) — `inductive_model`, split cũ vs split v3

| Config | Split | Slice | AUPRC | F1(pos)@0.5 | F1(pos)@threshold tối ưu |
|---|---|---|---:|---:|---:|
| 23 features, hidden=64/64, mlp | Split cũ (T_cutoff) | val | 0.3118 | 0.0768 | — |
| 23 features, hidden=64/64, mlp | Split cũ | pure_test | 0.3025 | 0.0861 | — |
| 23 features, hidden=64/64, mlp | Split cũ | overlap | 0.1704 | 0.0266 | — |
| **top-7 features**, patience=100 | Split cũ | val | 0.3013 | 0.0765 | — |
| **top-7 features**, patience=100 | Split cũ | pure_test | **0.3409** | 0.0918 | — |
| **top-7 features**, patience=100 | Split cũ | overlap | 0.1511 | 0.0206 | — |
| 23 features | **Split v3 (no_overlap)** | val | 0.3344 | 0.0786 | — |
| 23 features | **Split v3** | **test** | **0.3986** | 0.0736 | 0.4619 (P=0.484, R=0.442) |
| **top-7 features** | **Split v3** | val | 0.4564 | 0.1583 | — |
| **top-7 features** | **Split v3** | **test** | **0.5383** | 0.1416 | **0.6024** (P=0.667, R=0.549) |

Nguồn: `output/graph_only_sweep_v2_fixed_direction_results.json`,
`output/graph_only_hypothesis_result_k7_pat100.json`,
`output/graph_only_hypothesis_result_v3.json`,
`output/graph_only_hypothesis_result_v3_k7.json`.

**Mức lệch quan trọng nhất bảng này**: split v3 (nghiêm ngặt hơn — val CŨNG
bậc=0 trong `adj_train`, không chỉ test) cho AUPRC **cao hơn** split cũ ở mọi
cấu hình tương ứng (0.3025→0.3986, 0.3409→0.5383) — ngược với trực giác
"nghiêm ngặt hơn phải khó hơn". Diễn giải: split cũ có nhóm `overlap` khiến
`val` vẫn dính một phần cấu trúc train-time, có thể đã làm early-stopping
chọn nhầm checkpoint kém tổng quát hơn — split v3 buộc val phản ánh đúng chế
độ inductive nên tín hiệu chọn checkpoint đáng tin hơn.

## C. BERT+GraphSAGE fusion — `inductive_model` E2 v2 (split cũ, full-scale 811,704 test)

| Slice | n | pos | F1(pos)@threshold tối ưu (val) | AUPRC | Recall@100 |
|---|---:|---:|---:|---:|---:|
| pure_test | 609,773 | 312 | 0.2011 (P=0.140, R=0.356) | 0.1436 | 0.115 |
| overlap | 201,931 | 217 | 0.1073 (P=0.061, R=0.442) | 0.0783 | 0.083 |

Nguồn: `output/e2_v2_full_eval_result_v2.json`.

**Mức lệch B → C**: fusion (BERT+GraphSAGE) AUPRC pure_test (0.144) **thấp
hơn hẳn** graph-only riêng lẻ ở cùng split cũ (0.303-0.341) — nhánh BERT khi
ghép vào đang **kéo tụt** hiệu năng thay vì cộng hưởng, ở scale full 811,704.
Đây là lý do `BERT_debug.md` (Task 1-5) được viết để sửa root cause phía text
(self-address bị lặp lại gây overfit, xem `TASK_CHECKLIST_RESULTS.md`/README
ở đây mục "no_overlap" cho bối cảnh liên quan).

## D. Baseline tabular thuần (GBM, không graph/BERT) — split cũ

| Config | Slice | AUPRC | F1(pos)@0.5 |
|---|---|---:|---:|
| default hyperparameter | full pure_test (609,773) | 0.3471 | 0.2099 |
| tuned (CV-selected) | 5-fold CV | 0.3395 | — |

Nguồn: `output/e2_v2_full_eval_result_v2.json` (`gbm_tuned_reference`),
docstring `gbm_baseline.py`.

**Đọc chéo B/C/D ở cùng slice `pure_test`, split cũ, full-scale-tương-đương**:
GBM tabular thuần (0.347) ≈ graph-only tốt nhất (0.341, top-7) > BERT+GraphSAGE
fusion full-scale (0.144). Graph/tabular đang mang phần lớn tín hiệu; nhánh
BERT (ở trạng thái hiện tại, chưa áp Task 1-5 của `BERT_debug.md`) chưa chứng
minh được đóng góp dương ở scale full.

## Ghi chú phạm vi

Bảng này gộp kết quả từ nhiều thời điểm/phiên làm việc khác nhau của dự án —
không phải mọi cấu hình đều test full-scale (một số chỉ có bounded-scale, đã
ghi rõ ở cột "Split"/tiêu đề mục). Chưa có kết quả full-scale cho: E2 trên
split v3 (chưa build corpus BERT cho v3, xem `run_pipeline_end_to_end.sh`
giai đoạn 3b), và GBM/E2 trên `Train_large`. Muốn số liệu mới hơn, chạy lại
đúng script tương ứng đã liệt kê trong thư mục này và trỏ `--split-dir`/
`PREPROC_DIR` vào `Dataset_MG_v3_no_overlap`.
