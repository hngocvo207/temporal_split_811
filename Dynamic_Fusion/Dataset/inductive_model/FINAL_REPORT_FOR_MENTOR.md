# MulDiGraph Temporal Split — Báo cáo cuối cho mentor

Nguồn số liệu: [`TASK_CHECKLIST_RESULTS.md`](TASK_CHECKLIST_RESULTS.md) (chi tiết từng mục),
[`PHASE1_graph_definitions.md`](PHASE1_graph_definitions.md) (định nghĩa 3 tập),
script [`analysis_connectivity.py`](analysis_connectivity.py),
trực quan hoá: [Cạnh Train/Val/Test](https://claude.ai/artifact/QRYVemftijVdJR5tMBX6aB).

---

## Mục 15 — Bảng kết quả cuối cùng

| Metric | Result | Status |
|---|---:|---|
| Train nodes | 2,217,282 | DONE |
| Val nodes | 268,352 | DONE |
| Test nodes | 487,855 | DONE |
| Total phishing | 1,165 | DONE |
| Train phishing | 757 | DONE |
| Val phishing | 175 | DONE |
| Test phishing | 233 | DONE |
| Full graph edges | 5,355,155 | DONE |
| Train graph edges (adj_train, ≤t1) | 3,859,882 | DONE |
| Train_large graph edges (adj_train_large, ≤t2) | 4,430,051 | DONE |
| Full vs Train edge increase | +38.7% | DONE |
| Train→Train increase after t1 | +7.5% | DONE |
| **Test→Train** (exact) | **257,037** | **DONE — sửa chiều so với checklist cũ (từng ghi ≈173k)** |
| **Train→Test** (exact) | **172,633** | **DONE — sửa chiều so với checklist cũ (từng ghi ≈257k)** |
| Test→Val (exact) | 77,572 | DONE — khớp ước lượng cũ (≈78k) |
| Val→Test (exact) | 44,282 | DONE — khớp ước lượng cũ (≈44k) |
| Test→Test | 136,115 | DONE — global |
| % Test nodes có ≥1 Test-neighbor (chiều ra) | 20.83% | DONE |
| % Test nodes có ≥1 Test-neighbor (chiều vào) | 4.01% | DONE |
| % Test nodes có ≥1 Test-neighbor (2 chiều bất kỳ) | 23.83% | DONE |
| Mean Test→Train per test node | 0.527 | DONE |
| Median Test→Train per test node | 0 | DONE |
| P90 Test→Train per test node | 1 | DONE |
| % edges chạm Test mà nối với Train | 62.48% | DONE — xác nhận đúng observation cũ (~62%) |
| Test→Train_large | 334,609 | DONE |
| Train_large→Test | 216,915 | DONE |
| Increase Test→Train_small→Train_large | +30.18% | DONE |
| Increase Train_small→Train_large→Test | +25.65% | DONE |
| Test feature source | **Full graph (Case C)**, cả 23/23 cột | DONE — audit bằng code, không suy đoán |
| Temporal leakage audit | Group-3 centrality (5 cột) leak thật; Group-1/2 (18 cột) rủi ro thấp hơn nhưng không được đảm bảo chủ động | DONE — formalize, fix (`--pre-cutoff`) còn TODO (~22h compute) |

---

## Mục 19 — 7 câu hỏi mentor cần trả lời được

### Q1. Train nhỏ là gì?

`Train_small = Train`: **2,217,282 node** (757 phishing), **3,859,882 cạnh**
(`adj_train.npz`, cắt tại t1 = 2018-05-18 11:57:10 UTC — 28.5% tổng giao dịch
bị loại).

### Q2. Train lớn là gì?

`Train_large = Train ∪ Val`: **2,485,634 node** (+268,352 so với Train_small,
+932 phishing), **4,430,051 cạnh** (`adj_train_large.npz`, cắt tại t2 =
2018-06-18 23:35:30 UTC — chỉ 15.8% tổng giao dịch bị loại, so với 28.5% ở
Train_small). Thêm **570,169 cạnh** (+14.8%) so với Train_small.

### Q3. Một test node kết nối tới đâu?

| | Test→Train | Test→Val | Test→Test |
|---|---:|---:|---:|
| Global | 257,037 | 77,572 | 136,115 |
| % test node có ≥1 (chiều ra) | 44.28% | 14.71% | 20.83% |

Chiều ngược lại:

| | Train→Test | Val→Test | Test→Test |
|---|---:|---:|---:|
| Global | 172,633 | 44,282 | 136,115 |
| % test node có ≥1 (chiều vào) | 22.13% | 5.70% | 4.01% |

**Kết luận**: test node chủ yếu kết nối RA tới Train (44.3% node, 54.6% tổng
cạnh ra) hơn là nhận VÀO từ đâu đó (chỉ 30.8% test node có ≥1 cạnh vào) —
phản ánh hành vi "tài khoản mới chủ động gửi tiền trước" trên Ethereum.

### Q4. Test node có kết nối với test node khác không?

**Có, nhưng không phổ biến như số global 136,115 nghe có vẻ.**

```
20.83% test node có ≥1 test-neighbor (chiều ra)
4.01%  test node có ≥1 test-neighbor (chiều vào)
23.83% test node có ≥1 test-neighbor (2 chiều bất kỳ)
Median = 0 (cả 2 chiều)
P90    = 1 (ra) / 0 (vào)
Max    = 4,432 (ra) / 9,315 (vào)
```

Tức là **hơn 3/4 test node (76.2%)** hoàn toàn không kết nối trực tiếp với
bất kỳ test node nào khác; con số global 136,115 chủ yếu đến từ một nhóm nhỏ
node có bậc rất cao (hub/sàn giao dịch).

### Q5. Khi dùng Train-large thay Train-small thì test connectivity thay đổi thế nào?

| | Train-small | Train-large | Δ | % increase |
|---|---:|---:|---:|---:|
| Test → Train region | 257,037 | 334,609 | +77,572 | **+30.18%** |
| Train region → Test | 172,633 | 216,915 | +44,282 | **+25.65%** |

Val (9.03% tổng số node) đóng góp gần 1/3 mức tăng — bất tương xứng, vì Val
nằm sát biên thời gian với Test nên tự nhiên "gần" Test hơn về cấu trúc.

### Q6. Feature của test node lấy từ đâu?

**Case C — Full graph (Train + Val + Test), cho cả 23/23 cột.** Xác nhận bằng
đọc trực tiếp mã nguồn (`02_groups12_exact.py`, `05_group3_centrality.py`),
không suy đoán:

- 18 cột Group 1+2 (degree/amount/temporal): đọc từ `edges.npz` không cắt thời
  gian. Rủi ro thực tế thấp hơn cho Train (do đặc điểm dữ liệu, không do thiết
  kế chủ động), nhưng **Val bị ảnh hưởng thật** (feature phản ánh cả hoạt động
  của val account sau t2 nếu có).
- 5 cột Group 3 (centrality): BFS-depth-2 + networkx trên **toàn đồ thị đầy
  đủ**, không có option cắt thời gian. **Leak thật, xác nhận** — centrality
  của 1 train node bị định hình bởi cạnh/node xuất hiện sau t1/t2.

Giải pháp đã thiết kế sẵn trong code (`--pre-cutoff`) nhưng **chưa từng chạy**
(chi phí ước tính ~22h cho group-3 ở full scale).

### Q7. Mô hình thực sự đang sử dụng information nào khi dự đoán test?

| Giai đoạn | Thông tin dùng | Đã kiểm soát đúng chưa? |
|---|---|---|
| Graph structure lúc **train** (cập nhật trọng số) | Chỉ `adj_train`/`adj_train_large` — Test bậc=0 tuyệt đối | ✅ Đã verify PASS nhiều lần |
| Graph structure lúc **suy luận** (eval) | `adj_inference` — toàn bộ đồ thị, kể cả cạnh Test↔Test | ✅ Chủ đích, không phải leakage (label không rò rỉ, chỉ cấu trúc) |
| **Node feature** (cả lúc train lẫn eval) | **Full graph, mọi mốc thời gian** — cho MỌI account, kể cả Train | ❌ **Leak thật ở Group 3 centrality**, chưa fix |
| Nhãn (label) | Chỉ Train/Train_large vào loss; Val/Test chỉ dùng để đánh giá | ✅ Đã verify PASS nhiều lần |

**Kết luận cuối**: kiểm soát rò rỉ ở tầng **đồ thị + nhãn** đã chắc chắn
(nhiều vòng verify độc lập trong dự án). Lỗ hổng còn lại nằm ở tầng
**feature engineering** — cụ thể là 5 cột centrality dùng chung cho toàn bộ
23 cột — đây là hạng mục rủi ro cao nhất cần quyết định trước khi chốt
protocol cuối: chấp nhận làm giới hạn đã biết, hay đầu tư ~22h build lại bản
`--pre-cutoff`.
