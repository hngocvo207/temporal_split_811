# YÊU CẦU: Refactor Pipeline Phân Chia Dữ Liệu Theo Chuẩn Temporal Split (80% Cutoff)
# + Mở Rộng Nhãn Phishing Qua Lan Truyền 1-Hop (Có Kiểm Soát)

## 1. BỐI CẢNH & MỤC TIÊU
Pipeline hiện tại (`mg_*.py`) đã xử lý tốt việc dựng ma trận kề dạng thưa, chống rò rỉ nhãn bằng Focal Loss, hiệu chỉnh ngưỡng trên tập Val, và bóc tách báo cáo kết quả.

Dataset gốc `MulDiGraph.pkl` (Chen et al. 2020, XBlock): 2,973,489 node / 13,551,303 cạnh / **1,165 tài khoản phishing xác thực** (`isp=1`).

Cần thực hiện 2 việc:
1. Refactor logic phân chia tập dữ liệu theo chuẩn **Temporal Split dựa trên Mốc Cắt 80% Giao Dịch Toàn Cục**.
2. Bổ sung bước **mở rộng nhãn phishing bằng lan truyền 1-hop có kiểm soát**, tạo ra nhãn "suspected fraud" tách biệt khỏi ground-truth gốc, phục vụ tăng cường tập train mà không phá vỡ khả năng đánh giá/so sánh với benchmark gốc.

Các file cần refactor:
- `mg_temporal_pipeline.py`
- `mg_build_adjacency.py`
- `mg_refit_features.py`
- `mg_build_examples.py`
- `train1.py`
- **(Mới)** `mg_propagate_labels.py` — script riêng cho bước lan truyền nhãn

---

## 2. CÁC ĐIỂM BẮT BUỘC CHỈNH SỬA

### A. Phân Chia Tập Dữ Liệu (`mg_temporal_pipeline.py`)

**1. Mốc cắt thời gian (T_cutoff):**
- Đọc toàn bộ giao dịch từ `MulDiGraph.pkl`.
- Lấy T_cutoff tại **bách phân vị thứ 80** của TOÀN BỘ timestamp giao dịch (thay logic sắp xếp `T_first` cũ).
- **Pre-cutoff graph:** `timestamp <= T_cutoff` (~80% giao dịch).
- **Post-cutoff graph:** `timestamp > T_cutoff`.

**2. Phân loại tài khoản:**
- `pure-train`: chỉ có giao dịch trước T_cutoff.
- `overlap`: có giao dịch cả hai phía.
- `pure-test`: chỉ bắt đầu giao dịch sau T_cutoff.

**3. Validation & Test:**
- **Validation**: 10% tài khoản muộn nhất trong `pure-train` (sắp xếp theo giao dịch gần nhất trước T_cutoff). Phần còn lại là **train**.
- **Test** = `overlap` ∪ `pure-test`.

### B. Chuẩn Hóa Đặc Trưng (`mg_refit_features.py`)
- `train_mask` chỉ fit StandardScaler trên tập `train` (`df["mg_partition"] == "train"`).
- KHÔNG lấy thống kê từ `overlap`, `val`, `pure_test`.
- Dùng scaler đã fit để `transform` các tập còn lại.

---

### C. MỞ RỘNG NHÃN PHISHING QUA LAN TRUYỀN 1-HOP (`mg_propagate_labels.py`) — MỚI

**Mục tiêu:** từ 1,165 seed phishing account (`isp=1`, ground-truth), sinh thêm tập nhãn mở rộng `is_suspected=1` cho các tài khoản 1-hop liên quan, dùng làm tín hiệu tăng cường khi train — **không thay thế, không trộn lẫn** với ground-truth gốc.

**Bắt buộc tuân thủ các ràng buộc an toàn sau:**

1. **Phân biệt chiều cạnh (directionality):**
   - Chỉ lan truyền theo **out-edge từ seed phishing** (tiền chảy RA từ phishing account → các tài khoản nhận, khả năng là mule/laundering).
   - **KHÔNG** lan truyền theo in-edge (tài khoản gửi tiền TỚI phishing account là nạn nhân, không phải nghi phạm) — loại các node này khỏi tập mở rộng, hoặc gắn nhãn riêng `is_victim=1` để phân tích (không dùng làm positive).

2. **Chống hub node / sàn giao dịch:**
   - Trước khi lan truyền, tính degree (in+out) của toàn bộ node.
   - Loại bỏ khỏi seed-expansion bất kỳ node trung gian nào có degree vượt ngưỡng percentile 99 (coi là hot wallet/sàn giao dịch) — không lan truyền qua các node này.
   - Log lại số lượng node bị loại vì lý do này.

3. **Tách biệt loại nhãn (không gộp class):**
   - Thêm cột riêng `label_source` với 3 giá trị: `"ground_truth"` (1,165 seed gốc), `"propagated_1hop"` (mở rộng), `"benign"`.
   - Cột `isp` (nhãn gốc dùng để so sánh benchmark) **giữ nguyên không đổi**.
   - Thêm cột mới `isp_augmented` = `isp` gốc HOẶC (`propagated_1hop` nếu được bật augmentation) — dùng cột này thay `isp` chỉ khi train, KHÔNG dùng khi report kết quả benchmark.

4. **Không rò rỉ vào Validation/Test:**
   - Nhãn `propagated_1hop` chỉ được áp dụng cho account thuộc partition **`train`**.
   - **Tuyệt đối cấm** áp dụng `propagated_1hop` lên `val`, `overlap`, `pure-test` — các tập này giữ nguyên nhãn gốc `isp` để đảm bảo đánh giá công bằng, so sánh được với benchmark gốc.
   - Thêm assertion kiểm tra: `set(propagated_1hop_accounts) ∩ (val ∪ test) == ∅`.

5. **Trọng số nhãn lan truyền (soft label, không hard label):**
   - Không gán `propagated_1hop` = 1.0 ngang bằng ground-truth.
   - Gán trọng số soft-label mặc định (tham số hoá được, ví dụ `PROPAGATION_WEIGHT = 0.5`) để dùng trong loss (nếu Focal Loss hỗ trợ sample-weight) — hoặc expose thành flag để dễ tắt/bật/tune (`--enable-label-propagation`, `--propagation-weight`).

6. **Logging & thống kê bắt buộc in ra:**
   - Số seed phishing gốc (1,165), số node ứng viên trước lọc, số node bị loại do hub-degree, số node cuối cùng được gán `propagated_1hop`, tỷ lệ tăng positive class trước/sau.

---

## 3. GIỮ NGUYÊN & CẬP NHẬT TÊN TẬP MỚI (`mg_build_adjacency.py`, `mg_build_examples.py`, `train1.py`)

- `adj_train.npz`: cạnh `timestamp <= T_cutoff` (bao gồm `overlap` làm ngữ cảnh message-passing).
- `adj_inference.npz`: toàn bộ đồ thị.
- **Label masking:** nhãn `overlap` bị che hoàn toàn khi train — loss chỉ tính trên `train`.
- `train1.py`:
  - Giữ nguyên Focal Loss, threshold calibration trên Val.
  - Thêm flag để chọn train bằng `isp` (gốc) hoặc `isp_augmented` (có propagation).
  - **Báo cáo kết quả LUÔN dùng `isp` gốc làm ground-truth** (kể cả khi train bằng `isp_augmented`) — không được đánh giá trên nhãn tự sinh.
  - In bảng bóc tách kết quả cho `pure-test`, `overlap`, `Full test` — như cũ.

---

## 4. RÀNG BUỘC KIỂM TRA (ASSERTIONS)
- T_cutoff đúng bách phân vị thứ 80 (sai số ±0.1%).
- `pure-train`, `overlap`, `pure-test` không giao nhau, hợp đủ toàn bộ account.
- `train` ∩ `val` = ∅, `train` ∩ `test` = ∅.
- Scaler chỉ fit trên đúng số dòng `train` (log số lượng).
- **(Mới)** `propagated_1hop` accounts ∩ (`val` ∪ `test`) = ∅.
- **(Mới)** Không có account nào vừa mang `label_source="propagated_1hop"` vừa mang `label_source="ground_truth"` (không ghi đè nhãn gốc).
- **(Mới)** Số lượng seed phishing gốc trong dữ liệu sau xử lý vẫn đúng bằng 1,165 (không bị mất/trùng do lỗi merge).

---

## 5. SAU KHI SỬA: BẮT BUỘC RÀ SOÁT TOÀN BỘ PIPELINE

Không dừng ở việc từng file chạy được — rà soát end-to-end:

1. **Grep toàn repo** tìm các chỗ còn dùng tên partition cũ (`pure_train`, `T_first`, logic split cũ) hoặc còn tham chiếu trực tiếp `isp` ở nơi lẽ ra phải dùng `isp_augmented` (hoặc ngược lại) — liệt kê cụ thể file/dòng.
2. **Kiểm tra tính nhất quán tên cột/tên tập** (`mg_partition`, `label_source`, `isp`, `isp_augmented`) xuyên suốt `mg_build_adjacency.py`, `mg_build_examples.py`, `train1.py`, `mg_propagate_labels.py`.
3. **Kiểm tra thứ tự phụ thuộc dữ liệu:** file sinh sau đọc đúng schema/tên file/tên cột của file sinh trước, không mismatch shape/index/account ID.
4. **Kiểm tra lại toàn bộ logic chống rò rỉ nhãn**, đặc biệt:
   - Xác nhận `propagated_1hop` không lọt vào `val`/`test`.
   - Xác nhận việc gộp `overlap` vào `adj_train.npz` không làm lộ nhãn `overlap`/`test` vào train hoặc fit scaler.
   - Xác nhận báo cáo kết quả cuối cùng luôn dùng `isp` gốc, không vô tình dùng `isp_augmented`.
5. **Chạy thử toàn bộ pipeline từ đầu đến cuối:**
   `mg_temporal_pipeline.py` → `mg_propagate_labels.py` → `mg_build_adjacency.py` → `mg_refit_features.py` → `mg_build_examples.py` → `train1.py`
   Báo cáo lại: lỗi runtime, cảnh báo assertion, và danh sách xung đột còn tồn đọng kèm đề xuất sửa.

**Chỉ báo "hoàn thành" khi đã chạy rà soát mục 5 và không còn xung đột nào tồn đọng.**, chỉnh sửa lại báo cáo preprocessing_and_eval_report.md