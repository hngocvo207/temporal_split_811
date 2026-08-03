# Session Changelog — Dynamic_Fusion Pipeline Overhaul

Tổng hợp toàn bộ nhiệm vụ đã thực hiện trong phiên làm việc này, theo thứ tự thời gian. Đây là bản
tóm tắt dạng changelog; chi tiết đầy đủ về số liệu/phương pháp luận nằm trong
`preprocessing_and_eval_report.md` (đã được cập nhật song song trong suốt phiên).

**Trạng thái tại thời điểm ghi file này**: pipeline dữ liệu đã hoàn thiện và chạy full-scale thành
công; fix kiến trúc mở rộng quy mô đã verify; lần chạy training lớn (20,000 tài khoản, đã sửa lỗi
chồng trọng số) đang chạy nền trong tmux `train1_large_v3` (wandb run `unzkxl3r`), chưa có kết quả
cuối cùng.

---

## 1. Refactor pipeline phân chia dữ liệu — Temporal Split 80% T_cutoff

**File**: `Dataset/mg_temporal_pipeline.py` (viết lại hoàn toàn)

- Thay logic split cũ (position-based 60/20/20 theo `T_first`) bằng **T_cutoff = bách phân vị thứ 80
  của TOÀN BỘ timestamp giao dịch** (không phải theo `T_first` từng tài khoản).
- Phân loại tài khoản theo `(t_first, t_last)` so với T_cutoff: `pure_train_candidates` / `overlap` /
  `pure_test`.
- **Validation** = 10% tài khoản có `t_last` muộn nhất trong `pure_train_candidates`. Phần còn lại →
  `train`.
- Chạy full-scale thật trên `MulDiGraph.pkl` (2,973,489 nodes / 13,551,303 edges): T_cutoff =
  `1527988904` (2018-06-03 01:21:44 UTC), đúng 80.0000% (verify bằng assertion cứng ±0.1%).
- Kết quả: train=1,945,607 (519 phishing) / val=216,178 (117) / overlap=201,931 (217) /
  pure_test=609,773 (312). Tổng 1,165 phishing được bảo toàn. Tất cả assertion PASS.

## 2. Mở rộng nhãn phishing qua lan truyền 1-hop có kiểm soát

**File**: `Dataset/mg_propagate_labels.py` (mới)

- Chỉ lan truyền theo **out-edge** từ seed phishing (in-edge → gắn `is_victim`, không dùng làm positive).
- Loại trừ **hub node** (degree > percentile 99) khỏi cả hai phía lan truyền (seed và candidate).
- Cột `label_source` (`ground_truth` / `propagated_1hop` / `benign`) tách biệt khỏi `isp` gốc.
  `isp_augmented = isp OR propagated_1hop`, chỉ dùng khi train, không dùng khi báo cáo benchmark.
- **Soft label**: `propagation_weight` mặc định 0.5 (< 1.0), không coi ngang bằng ground-truth.
- Ràng buộc leakage: `propagated_1hop` chỉ áp dụng cho tài khoản thuộc partition `train`, có assertion
  kiểm tra `propagated_1hop ∩ (val ∪ test) == ∅`.
- Chạy full-scale thật: 1,165 seed → 929 seed dùng để lan truyền (236 bị loại vì là hub) → 1,743 ứng
  viên → 1,319 sau lọc hub → **519 tài khoản cuối cùng** được gán `propagated_1hop` (đều thuộc
  `train`). Positive rate trong `train` tăng từ 0.0267% → 0.0534% (2×).

## 3. Cập nhật các file pipeline còn lại theo chuẩn mới

- `Dataset/mg_build_adjacency.py`: đổi từ ranh giới `T1` cũ sang `T_cutoff`.
- `Dataset/mg_refit_features.py`: StandardScaler giờ fit **chỉ trên `train`** (trước đây fit trên
  `pure_train + overlap`).
- `Dataset/mg_build_examples.py`: đổi tên partition (`train`/`val`/`overlap`/`pure_test`), wire thêm
  `isp_augmented` / `propagation_weight` / `label_source` từ bước 2, xuất cả `train_y` (isp gốc) và
  `train_y_augmented` (isp_augmented) + `train_sample_weight`; sửa lỗi import `utils` bị gãy khi chạy
  từ `Dataset/`.

## 4. Wiring nhãn augmented + soft-weight vào model

- `Dataset/tri_model/utils.py`: `InputExample`/`InputFeatures`/`CorpusDataset` mang thêm
  `sample_weight` xuyên suốt tới batch đã gộp.
- `Dataset/tri_model/train1.py`: thêm flag `--use_isp_augmented` (train bằng `isp` gốc hoặc
  `isp_augmented`); **báo cáo kết quả luôn dùng `isp` gốc** bất kể flag; focal loss nhân thêm
  `sample_weight` để nhãn lan truyền không có trọng số ngang ground-truth.

## 5. Nâng cấp hạ tầng thực thi (theo yêu cầu, tham khảo `bi_model/train_origin.py`)

- WandB run-id được lưu/khôi phục để resume vào đúng run cũ thay vì tạo run mới mỗi lần restart.
- Checkpoint resume theo từng bước (mỗi 250 bước + mỗi epoch), không chỉ lưu "best".
- Kết quả Step 5/6/8 cuối cùng (threshold hiệu chỉnh, breakdown pure_test/overlap) được đẩy lên WandB,
  trước đây chỉ có log epoch-level.
- **Không đụng vào kiến trúc gốc của mô hình** — chỉ phần hạ tầng train/checkpoint/logging.

## 6. Rà soát toàn bộ pipeline + chạy thử end-to-end

- Grep toàn repo tìm tên cũ (`pure_train`, `T1`, `T2`) — đã dọn sạch trong pipeline đang dùng; 2 chỗ
  còn sót trong `bi_model/train_origin.py` (biến thể model khác, ngoài phạm vi) được ghi nhận, không
  sửa.
- Chạy thật toàn bộ chuỗi `mg_temporal_pipeline.py → mg_propagate_labels.py → mg_build_adjacency.py →
  mg_refit_features.py → mg_build_examples.py → train1.py` trên dữ liệu thật, full-scale — phát hiện
  và sửa 1 lỗi runtime thật (import `utils` gãy) + 1 xung đột checkpoint (tên file trùng giữa các lần
  chạy khác vocab size — xử lý bằng cách backup, không xoá).

## 7. Verification run (corpus 3,000 tài khoản) — kết quả hội tụ thật

- Build corpus nhỏ (train=3,000/val=800/overlap=800/pure_test=800) trên split thật.
- Train đến early-stopping (epoch 3, best = epoch 0): Valid F1=87.48%, Test F1(pos) overall=0.6504,
  pure_test=0.8157, overlap=0.5127 — tái xác nhận pattern "overlap khó hơn pure_test" từ report gốc.
- Bổ sung tính F1(weighted) song song F1(pos) cho từng lát cắt để minh hoạ rõ khoảng cách giữa hai
  chỉ số dưới mất cân bằng nặng.

## 8. Fix kiến trúc: Sparse Gather/Embedding-Lookup thay one-hot dày

**Files**: `Dataset/tri_model/utils.py` (`CorpusDataset.pad()`), `Dataset/tri_model/ETH_GBert.py`
(`VocabGraphConvolution`, `ETH_GBertEmbeddings`, `ETH_GBertModel`)

- Thay `gcn_swop_eye` (ma trận one-hot dày `[batch, vocab_size, seq_len]`) bằng `gcn_vocab_ids` (tensor
  chỉ số) + phép **gather** vào bảng `[vocab_size, hid_dim]` thay vì scatter vào ma trận dày.
- Về toán học **tương đương chính xác** với công thức cũ — verify bằng test số học độc lập
  (sai lệch tối đa 2.4e-7) và loss bước đầu tiên giống hệt bit-for-bit trước/sau fix.
- Benchmark bộ nhớ thật ở **full-scale V=2,973,489** (dùng `adj_train.npz` thật): peak GPU
  **7,362 MB** cho forward+backward — trong khi cách cũ chỉ riêng tensor one-hot đã cần ~39.6 GB
  (không phải ~39.6 TB như báo cáo gốc ghi nhầm đơn vị — đã sửa lại trong report).
- Kết luận: gỡ được nút thắt **bộ nhớ** cho vocab full-scale; nút thắt **thời gian** (số bước training)
  thì không đổi.

## 9. Chạy corpus lớn (20,000 tài khoản) — phát hiện & sửa 2 vấn đề thật

**Vấn đề 1 — model chọn epoch 0 làm "best" và không bao giờ cải thiện:**
- Lần chạy đầu (BertAdam, LR=8e-6, chọn checkpoint theo weighted F1): early-stop ở epoch 5, best vẫn
  là epoch 0. Test F1(pos) overall chỉ 0.5035 dù F1(weighted)=95.61% (minh chứng rõ việc weighted F1
  bị lệch bởi lớp đa số).
- **Chẩn đoán**: tiêu chí chọn best-checkpoint (`perform_metrics`) là weighted F1 trên validation chỉ
  ~2.3% dương — đúng chỉ số đã chứng minh là gây hiểu lầm.
- **Sửa (`train1.py`)**: 4 thay đổi — (a) bật `WeightedRandomSampler` cho `train_dataloader` (code có
  sẵn nhưng chưa từng dùng), (b) đổi tiêu chí chọn best/early-stopping sang **F1(pos)**, (c) báo cáo
  F1(pos) như số liệu chính (không chỉ nằm trong `classification_report`), (d) tăng LR mặc định lên
  2e-5 (bỏ override cứng 8e-6), đổi optimizer từ `BertAdam` sang `torch.optim.Adam` +
  `CosineAnnealingLR`.

**Vấn đề 2 — chồng hai cơ chế cân bằng lớp, model sụp đổ về đoán "phishing" cho mọi mẫu:**
- Lần chạy lại với 4 thay đổi trên: `Accuracy` trên valid/test khớp gần như chính xác với tỷ lệ
  phishing thật của tập đó (2.34%/5.29%) — dấu hiệu model luôn đoán positive. Lặp lại y hệt suốt 4
  epoch, không nhúc nhích.
- **Chẩn đoán**: `WeightedRandomSampler` (mới bật) đã cân bằng batch ~50/50, nhưng `focal_alpha` (tính
  từ tần suất lớp GỐC, `[0.026, 0.974]`) vẫn nhân thêm hệ số ~37x cho lớp dương lên TRÊN batch đã cân
  bằng — chồng hai lần correction.
- **Sửa**: `focal_alpha` chuyển về trung lập `[0.5, 0.5]` (giữ nguyên phần focusing `(1-pt)^γ`), vì
  sampler đã lo phần cân bằng lớp rồi.
- Đã dừng job lỗi, backup checkpoint (không xoá), sửa code, smoke-test sạch, chạy lại — **đang chạy
  nền, chưa có kết quả cuối** (xem `preprocessing_and_eval_report.md` §11 để cập nhật khi xong).

## 10. Danh sách file đã thay đổi/tạo mới

| File | Loại thay đổi |
|---|---|
| `Dataset/mg_temporal_pipeline.py` | Viết lại (split 80% T_cutoff) |
| `Dataset/mg_propagate_labels.py` | **Mới** (lan truyền nhãn 1-hop) |
| `Dataset/mg_build_adjacency.py` | Cập nhật (T_cutoff) |
| `Dataset/mg_refit_features.py` | Cập nhật (fit scaler chỉ trên `train`) |
| `Dataset/mg_build_examples.py` | Cập nhật (partition mới, isp_augmented, sửa import) |
| `Dataset/tri_model/utils.py` | Cập nhật (sample_weight, sparse gather trong `pad()`) |
| `Dataset/tri_model/ETH_GBert.py` | Cập nhật (`VocabGraphConvolution` sparse gather) |
| `Dataset/tri_model/train1.py` | Cập nhật (flags, checkpoint/wandb, F1(pos) selection, Adam+Cosine, imbalance handling) |
| `dynamic_fusion_leakage_audit/preprocessing_and_eval_report.md` | Cập nhật liên tục xuyên suốt phiên |

## 11. Việc còn mở (chưa làm trong phiên này)

1. Kết quả thật của lần chạy 20,000-tài khoản đã sửa lỗi (đang chạy, §9 mục 2 ở trên).
2. Full-scale thật (toàn bộ 1,945,607 tài khoản train) — khả thi về bộ nhớ sau fix §8, nhưng ước tính
   ~1-2 ngày/epoch, chưa chạy.
3. `select_add_features.py` chưa chạy lại trên toàn bộ 2,973,489 node (hiện chỉ có 5,655 node có
   feature).
4. Đặt tên checkpoint (`model_file_4save`) nên bao gồm vocab size/phiên bản code — hiện đã va chạm 3
   lần trong phiên này do trùng tên file giữa các cấu hình khác nhau.
5. `Dataset/bi_model/` (biến thể model khác) vẫn dùng tên/kiến trúc cũ, ngoài phạm vi phiên này.
