# MulDiGraph Temporal Split — Kết quả thực hiện Task Checklist

Trả lời từng mục trong `MulDiGraph_Temporal_Split_Task_Checklist.md`, ưu tiên
Priority 1 → 2 → 3. Mọi số liệu tính trực tiếp từ dữ liệu trên đĩa
(`data/preprocessed/Dataset_MG_v3_no_overlap/`), không ước lượng.

Script: [`analysis_connectivity.py`](analysis_connectivity.py) — chạy 4.2s.
Output thô: [`output/connectivity_summary.json`](output/connectivity_summary.json),
[`output/test_node_connectivity.csv`](output/test_node_connectivity.csv) (487,855 dòng,
1 dòng/test node).

---

## ⚠️ Phát hiện quan trọng nhất: mục 6.2 của checklist bị NGƯỢC CHIỀU

Checklist cũ ghi (mục 6.2): *"Test → Train khoảng 173k, Train → Test khoảng 257k"*.

Số **exact** vừa trích xuất trực tiếp từ `adj_inference.npz`:

| Cặp | Giá trị exact | Checklist cũ ghi | Đúng/Sai |
|---|---:|---|---|
| **Test → Train** | **257,037** | "≈173k" | **Checklist cũ SAI chiều** — đây là con số của Train→Test |
| **Train → Test** | **172,633** | "≈257k" | **Checklist cũ SAI chiều** — đây là con số của Test→Train |
| Test → Val | 77,572 | "≈78k" | Đúng |
| Val → Test | 44,282 | "≈44k" | Đúng |

Cặp Test↔Val trong checklist cũ ghi đúng chiều; chỉ cặp Test↔Train bị hoán đổi.
Độ lớn 2 số (173k/257k) không sai, chỉ gán nhầm chiều mũi tên. Tỷ lệ
`Train→Test / Test→Train = 172,633 / 257,037 = 0.6716` (nghịch đảo của "1.49"
checklist cũ tính — khớp về độ lớn, chỉ ngược chiều diễn giải).

**Đây chính xác là việc mục 6.2/Priority 3 yêu cầu** ("cần extract lại trực
tiếp từ adjacency matrix để tránh sai số") — đã hoàn thành, và phát hiện ra
đây không chỉ là sai số làm tròn mà là **nhầm chiều**.

---

## 1-5. Định nghĩa tập, temporal split, global edge distribution — không đổi

Đã đúng và đã verify độc lập ở các bước trước (`PHASE1_graph_definitions.md`).
Không phát hiện sai lệch nào ở mục 1-5. **Status: DONE — không cần sửa.**

---

## 6. Directed connectivity — exact numbers (mục 6.2, Priority 3 item 15)

| Hướng | Exact count |
|---|---:|
| Train → Train | 4,150,672 |
| Val → Val | 146,993 |
| Test → Test | 136,115 |
| Train → Val | 191,498 |
| Val → Train | 178,353 |
| **Train → Test** | **172,633** |
| **Test → Train** | **257,037** |
| Val → Test | 44,282 |
| Test → Val | 77,572 |

**Status: DONE — exact, thay thế hoàn toàn số ước lượng cũ.**

---

## 7. Test node có kết nối với Test node khác không? (node-level)

### 7.1 Global (đã có từ trước)
Test→Test = 136,115 cạnh. **DONE.**

### 7.2 Node-level (TODO cũ → nay DONE)

Tính `test_to_test(v)` cho toàn bộ 487,855 test node, tách riêng chiều ra
(out) và chiều vào (in) vì đây là đồ thị có hướng — không gộp làm một:

| Thống kê | Test→Test (ra, out) | Test←Test (vào, in) |
|---|---:|---:|
| Mean | 0.279 | 0.279 |
| Median | 0 | 0 |
| P25 | 0 | 0 |
| P75 | 0 | 0 |
| P90 | 1 | 0 |
| Max | 4,432 | 9,315 |
| % node > 0 | **20.83%** | **4.01%** |

**Trả lời trực tiếp câu hỏi của mentor:**
- **20.83%** test node có ≥1 test-neighbor ở chiều RA (gửi tiền cho 1 test node khác).
- **4.01%** test node có ≥1 test-neighbor ở chiều VÀO (nhận tiền từ 1 test node khác).
- **23.83%** test node có ≥1 test-neighbor ở **1 trong 2 chiều bất kỳ** (out HOẶC in).
- Trung vị (median) = 0 ở cả 2 chiều — nghĩa là **hơn một nửa** test node hoàn toàn
  không kết nối trực tiếp với bất kỳ test node nào khác; số 20.83%/4.01% đến từ
  phần đuôi phân bố (một số ít node "hub" có tới 4,432-9,315 kết nối).
- Mean (0.279) mean≈median trùng nhau ở in nhưng khác biệt lớn với max → phân bố
  **rất lệch phải** (long-tail), điển hình cho đồ thị giao dịch on-chain.

**Status: TODO cũ → DONE.**

---

## 8. Phân bố transaction đầy đủ của từng Test node

DataFrame `test_node_connectivity.csv` (487,855 dòng × 9 cột: `address`,
`test_to_train`, `test_to_val`, `test_to_test_out`, `total_outgoing`,
`train_to_test`, `val_to_test`, `test_to_test_in`, `total_incoming`).

(Lưu ý: checklist mẫu ghi cột `test_to_test` xuất hiện 2 lần — 1 lần ở nhóm
outgoing, 1 lần ở nhóm incoming. Đây là 2 đại lượng KHÁC NHAU về mặt hướng
[xem mục 7.2], nên trong file CSV thực tế đặt tên `test_to_test_out` và
`test_to_test_in` để tránh trùng tên cột.)

| Metric | Test→Train | Test→Val | Test→Test (out) | Total outgoing | Train→Test | Val→Test | Test→Test (in) | Total incoming |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean | 0.527 | 0.159 | 0.279 | 0.965 | 0.354 | 0.091 | 0.279 | 0.724 |
| Median | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| P25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P75 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| P90 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| Max | 3,250 | 196 | 4,432 | 4,978 | 2,550 | 2,270 | 9,315 | 10,260 |
| % node > 0 | 44.28% | 14.71% | 20.83% | 74.24% | 22.13% | 5.70% | 4.01% | 30.82% |

**Đọc kết quả**: 74.24% test node có ít nhất 1 cạnh ra (outgoing) nhưng chỉ
30.82% có ít nhất 1 cạnh vào (incoming) — bất đối xứng directed, nhất quán
với đặc điểm "tài khoản mới thường chủ động gửi tiền trước khi có ai gửi lại
cho nó" trên Ethereum. Max rất cao (tới 10,260 cạnh vào cho 1 node) xác nhận
có các "hub" (có thể là sàn giao dịch/dịch vụ) trong tập test.

**Status: TODO cũ → DONE.**

---

## 9. Mức độ phụ thuộc vào Train/historical neighbors

| Thống kê | Test→Train (out, node-level) |
|---|---:|
| Mean | 0.527 |
| Median | 0 |
| P90 | 1 |
| % test node có ≥1 train-neighbor (chiều ra) | **44.28%** |
| % test node có ≥1 train-neighbor (2 chiều bất kỳ) | **63.57%** |
| % tổng test outgoing edges đi tới Train | **54.60%** |
| % cạnh "chạm" test (mỗi cạnh tính 1 lần) mà nối với Train (2 chiều) | **62.48%** |

Số cuối (**62.48%**) chính là con số re-verify chính xác cho observation cũ
*"khoảng 62% edges touching test connect tới train"* — **khớp gần như tuyệt
đối** với ước lượng ban đầu (62% → 62.48% thật). Đây là một trong số ít quan
sát cũ được xác nhận đúng nguyên vẹn, không cần sửa.

**Status: PARTIAL cũ → DONE, xác nhận đúng.**

---

## 10. Train_small vs Train_large — connectivity với Test

| Connectivity | Train-small | Train-large | Δ | % Increase |
|---|---:|---:|---:|---:|
| **Test → Train region** | 257,037 | 334,609 | +77,572 | **+30.18%** |
| **Train region → Test** | 172,633 | 216,915 | +44,282 | **+25.65%** |

(`Train_large = Train ∪ Val`, nên `E(Test→Train_large) = E(Test→Train) +
E(Test→Val)`, và tương tự chiều ngược lại — cộng dồn trực tiếp từ bảng mục 6,
không cần tính lại từ đồ thị.)

**Đọc kết quả**: mở rộng Train_small → Train_large giúp **~30% cạnh Test→Train
region mới xuất hiện** (tăng đáng kể connectivity quan sát được giữa Test và
"vùng train"), nhưng đây là hệ quả tự nhiên của việc Val vốn có
connectivity với Test cao hơn tỷ trọng của nó (Val chỉ 9.03% tổng số node
nhưng đóng góp 30.18%/25.65% mức tăng connectivity với Test) — nhất quán với
observation ở `PHASE1_graph_definitions.md` rằng Val nằm sát biên thời gian
với Test nên tự nhiên "gần" Test hơn về mặt cấu trúc đồ thị.

**Status: TODO cũ → DONE.**

---

## 11. Giữ Test→Train / Test→Val / Test→Test riêng biệt

Đã thực hiện xuyên suốt — không gộp thành 1 số ở bất kỳ bảng nào trên. File
CSV giữ đủ 3 cột outgoing + 3 cột incoming riêng biệt cho từng node.

**Status: TODO cũ → DONE.**

---

## 12. Feature của Test node lấy từ đâu? (audit leakage — quan trọng nhất)

Đã đọc trực tiếp mã nguồn pipeline sinh `features_output_all23_MG_fullscale.csv`
(nguồn của `node_features_all23.pt`, dùng cho toàn bộ 23 đặc trưng node):
`Dataset/fullscale_features/00_graph_to_arrays.py` → `02_groups12_exact.py`
(18 đặc trưng đầu) → `05_group3_centrality.py` (5 đặc trưng cuối) →
`08_assemble_csv.py`.

### Kết luận audit (Case C — Full graph, xác nhận bằng code, không suy đoán)

**Cả 23/23 đặc trưng đều được tính trên đồ thị ĐẦY ĐỦ (Train+Val+Test, mọi
mốc thời gian) — không có bất kỳ cắt thời gian nào được áp dụng cho file
đang dùng thật.**

| Nhóm đặc trưng | Cột | Nguồn dữ liệu | Dùng info Test? | Dùng info tương lai? | Risk |
|---|---|---|---|---|---|
| **Group 1+2** (18 cột) | `out_degree, in_degree, direction_ratio, max/min/avg_out_amount, max/min/avg_in_amount, account_balance, lifetime_days, active_days, freq_out/in_short, freq_out/in_long, short_long_out/in_ratio` | Chỉ cạnh trực tiếp (1-hop) của CHÍNH node, nhưng đọc từ `edges.npz` — **KHÔNG cắt theo t1/t2** (`02_groups12_exact.py` chạy sản phẩm thật KHÔNG truyền `--pre-cutoff`, xác nhận bằng tên file output `groups12_full.npz`, không phải `groups12_precut*.npz`) | Gián tiếp (chỉ nếu chính node đó có giao dịch cả trước/sau mốc) | **Có, với Val** (val node hoạt động thật đến 2018-06-18 nhưng feature phản ánh CẢ giao dịch của nó sau đó nếu có); Train hầu như không bị ảnh hưởng THỰC TẾ (train node theo định nghĩa `t_first<t1`, và dữ liệu quan sát cho thấy accounts này hiếm khi còn hoạt động rất lâu sau đó — nhưng pipeline **không chủ động đảm bảo** điều này, chỉ là hệ quả ngẫu nhiên của dữ liệu) | **Trung bình** |
| **Group 3** (5 cột) | `betweenness_centrality, degree_centrality, clustering_coefficient, in_degree_centrality, out_degree_centrality` | BFS-depth-2 ball + `networkx` centrality trên **TOÀN BỘ đồ thị đầy đủ** (2,973,489 node / 13,551,303 cạnh) — `05_group3_centrality.py` **không có option cắt thời gian nào cả** (chỉ có `--workers`/`--chunk`/`--bench`) | **Có, chắc chắn** | **Có, chắc chắn** | **Cao** |

**Bằng chứng bằng văn bản đã có sẵn trong code** (không phải suy luận của báo
cáo này) — comment tại `02_groups12_exact.py` dòng 25-41, do một phiên làm
việc trước tự ghi nhận:

> "the production artifact actually consumed by train1.py/eval_full_test.py
> ... was generated by running this script WITHOUT `--pre-cutoff` ... That
> means every account's group-1/2 feature values ... reflect the FULL edge
> timeline ... any feature that depends on graph-wide structure (group 3
> centrality...) can still be shaped by post-cutoff activity elsewhere in the
> network. Passing `--pre-cutoff <T_cutoff>` ... would produce a strictly
> train-only-timeline alternative for comparison; **this has not been run**
> (~22h on the full graph per timing estimate)."

Nói cách khác: **giải pháp sửa đã được thiết kế sẵn (cờ `--pre-cutoff`) nhưng
chưa từng chạy** vì chi phí ước tính ~22h cho riêng bước centrality ở full
scale.

**Status: TODO cũ → DONE (audit); FIX (chạy `--pre-cutoff`) vẫn TODO, ước tính ~22h.**

---

## 13. Temporal leakage vs Graph information — formalize

### 13.1 Graph information leakage (Test→Test, Test→Val trong đồ thị suy luận)

**Không phải leakage** theo đúng khung đã dùng xuyên suốt dự án: lúc **suy
luận** (không phải lúc train), dùng `adj_inference.npz` (đồ thị đầy đủ tại
thời điểm hiện tại) là hợp lý — một hệ thống triển khai thật tại thời điểm
"bây giờ" đương nhiên có quyền truy cập toàn bộ đồ thị giao dịch tính đến
"bây giờ", bao gồm cả giao dịch giữa các tài khoản mới với nhau. Cái bị cấm
là **NHÃN** (label) của Val/Test đi vào loss lúc train — không phải cấu trúc
đồ thị đi vào lúc suy luận. Điều này đã verify: `adj_train.npz` /
`adj_train_large.npz` đều cho Test bậc=0 tuyệt đối (mục 5 checklist, đã PASS).

### 13.2 Temporal leakage (feature của node bị ảnh hưởng bởi tương lai TẠI THỜI ĐIỂM TRAIN)

**Đây MỚI là leakage thật, đã xác nhận tồn tại** — qua audit mục 12: 5 cột
centrality (Group 3) của một **TRAIN node** bị định hình bởi cấu trúc đồ thị
hình thành **sau** t1/t2 (node/cạnh mới của Val/Test làm thay đổi
betweenness/clustering/degree-centrality của các node Train cũ). Khi những
đặc trưng này được dùng làm input lúc **train** mô hình, mô hình đang học từ
thông tin đáng lẽ chưa quan sát được tại thời điểm cắt.

### 13.3 Chuỗi quan sát cần formalize (theo đúng khung checklist yêu cầu)

```
Prediction time                = t2 (biên Val/Test) hoặc t_cutoff bất kỳ
        ↓
Observable transaction window  = mọi giao dịch có timestamp <= prediction time
        ↓
Feature construction (HIỆN TẠI)= toàn bộ 23 cột tính trên FULL graph (mọi thời gian)
                                  → VI PHẠM "observable window" ở TRAIN time cho Group 3
Feature construction (ĐÚNG)    = Group 1+2 + Group 3 đều cần bản `--pre-cutoff <t_cutoff>`
                                  tương ứng đúng partition đang xét (Train_small: cutoff=t1;
                                  Train_large: cutoff=t2; Test: không cutoff, dùng full)
        ↓
Label availability             = chỉ nhãn Train (Train_small hoặc Train_large tuỳ thí
                                  nghiệm) được dùng trong loss; Val/Test chỉ dùng để
                                  đánh giá — điều này ĐÃ đúng (đã verify PASS nhiều lần)
```

**Kết luận**: rò rỉ nhãn (label leakage) đã được kiểm soát tốt và verify nhiều
lần trong dự án (partition tách bạch, `adj_train` cắt đúng mốc). Rò rỉ
**feature-level** (temporal leakage qua Group 3 centrality) là lỗ hổng thật,
đã có sẵn thiết kế sửa (`--pre-cutoff`) nhưng chưa chạy — cần quyết định có
đầu tư ~22h compute để chạy bản train-only-timeline hay chấp nhận rủi ro này
làm giới hạn đã biết của thí nghiệm hiện tại.

**Status: TODO cũ → DONE (formalize + xác định rõ root cause); fix vẫn TODO.**

---

## 14-19. Visualization, bảng tổng hợp, 7 câu hỏi mentor

Xem file riêng [`FINAL_REPORT_FOR_MENTOR.md`](FINAL_REPORT_FOR_MENTOR.md) —
gộp bảng tổng hợp cuối (mục 15), trả lời đầy đủ 7 câu hỏi (mục 19), và link
tới visualization đã bổ sung (mục 14).
