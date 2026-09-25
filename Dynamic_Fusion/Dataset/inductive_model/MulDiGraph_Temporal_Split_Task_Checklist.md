# MulDiGraph Temporal Split — Task Checklist & Verification

> **[2026-09-25] Đã thực hiện toàn bộ Priority 1-3.** Kết quả chi tiết từng
> mục (số exact, node-level stats, feature audit, 7 câu hỏi mentor) ở:
> - [`TASK_CHECKLIST_RESULTS.md`](TASK_CHECKLIST_RESULTS.md) — đối chiếu từng
>   mục của checklist này với số liệu thật, có 1 phát hiện quan trọng ở mục
>   6.2 (Test↔Train bị ghi ngược chiều trong bản checklist gốc).
> - [`FINAL_REPORT_FOR_MENTOR.md`](FINAL_REPORT_FOR_MENTOR.md) — bảng tổng
>   hợp mục 15 + trả lời đầy đủ 7 câu hỏi mục 19.
> - Script: [`analysis_connectivity.py`](analysis_connectivity.py) →
>   [`output/connectivity_summary.json`](output/connectivity_summary.json),
>   [`output/test_node_connectivity.csv`](output/test_node_connectivity.csv)
>   (487,855 dòng, node-level).
> - Visualization bổ sung (mục 14): [Cạnh Train/Val/Test](https://claude.ai/artifact/QRYVemftijVdJR5tMBX6aB)
>   (đã thêm mục 5 "node-level distribution" và mục 6 "Train_small vs
>   Train_large").
> - Còn lại 1 hạng mục TODO thật sự (không phải đo/verify mà là build lại dữ
>   liệu): chạy `--pre-cutoff` cho `fullscale_features/02_groups12_exact.py` +
>   `05_group3_centrality.py` để loại temporal leakage ở 5 cột centrality —
>   ước tính ~22h compute, chưa chạy (xem `TASK_CHECKLIST_RESULTS.md` mục 12).

## 1. Mục tiêu

Tài liệu này tổng hợp toàn bộ các nhiệm vụ cần thực hiện để trả lời đầy đủ các câu hỏi của mentor về:

- Temporal split của MulDiGraph.
- `Train_small`, `Train_large = Train ∪ Val`, và `Test`.
- Connectivity giữa Train / Val / Test.
- Phân bố số transaction của từng test node.
- Việc test node có kết nối với test node khác hay không.
- Nguồn feature của test node.
- Khả năng xảy ra temporal leakage / information leakage.
- Visualization phục vụ phân tích và thuyết trình.

> **Quy ước quan trọng:** Những kết quả đã có trong báo cáo được đánh dấu `DONE — đã có`. Những mục chỉ mới được suy luận từ aggregate statistics nhưng chưa có node-level verification được đánh dấu `PARTIAL — cần kiểm chứng`.

---

# 2. Định nghĩa các tập dữ liệu

## 2.1. Train

- Tổng số node: **2,217,282**
- Phishing: **757**
- Normal: **2,216,525**
- Tỷ lệ phishing: **0.0341%**
- Chiếm **64.98%** tổng số phishing nodes.
- Chiếm **74.57%** tổng số nodes.

**Status: DONE — đã có và đã sử dụng trong báo cáo.**

---

## 2.2. Validation

- Tổng số node: **268,352**
- Phishing: **175**
- Normal: **268,177**
- Tỷ lệ phishing: **0.0652%**
- Chiếm **15.02%** tổng số phishing nodes.
- Chiếm **9.03%** tổng số nodes.

**Status: DONE — đã có và đã sử dụng trong báo cáo.**

---

## 2.3. Test

- Tổng số node: **487,855**
- Phishing: **233**
- Normal: **487,622**
- Tỷ lệ phishing: **0.0478%**
- Chiếm **20.00%** tổng số phishing nodes.
- Chiếm **16.41%** tổng số nodes.

**Status: DONE — đã có và đã sử dụng trong báo cáo.**

---

## 2.4. Tổng dataset

- Tổng số node: **2,973,489**
- Phishing: **1,165**
- Normal: **2,972,324**
- Tỷ lệ phishing: **0.0392%**

**Status: DONE.**

---

# 3. Kiểm chứng cách chia temporal split

## 3.1. Quy tắc split

Split `65/15/20` được áp dụng trên **phân bố `t_first` của phishing nodes**, không phải chia 65/15/20 theo thời lượng lịch.

- Train: 65% phishing nodes.
- Validation: 15% phishing nodes.
- Test: 20% phishing nodes.

Tổng:

```text
757 + 175 + 233 = 1,165 phishing nodes
```

**Status: DONE — đã xác định rõ.**

---

## 3.2. Kiểm tra temporal non-stationarity

Quan sát đã có:

- 11/2016 → khoảng 06/2017: gần như không có phishing.
- Từ khoảng 07/2017: số phishing bắt đầu tăng.
- Khoảng 09/2017: khoảng 68.
- Peak khoảng 05/2018: khoảng 190.
- Sau peak: giảm dần.
- Khoảng 09/2018: gần như về 0.

Train/validation boundary nằm gần vùng peak.

Mật độ phishing theo 100k nodes:

| Split | Phishing / 100k nodes |
|---|---:|
| Train | ~34 |
| Validation | ~65 |
| Test | ~48 |

Điều này cho thấy phân bố phishing thay đổi theo thời gian.

**Status: DONE — đã có visualization và discussion.**

### Caveat cần giữ trong báo cáo

Việc số phishing gần 0 ở cuối timeline **không nên kết luận ngay rằng phishing thực sự biến mất**. Có thể tồn tại `right-censoring` hoặc delayed labeling.

**Status: DONE — caveat đã được nêu.**

---

# 4. Global edge distribution

Tổng số edge trong full inference graph:

**5,355,155**

| Edge type | Count | Percentage |
|---|---:|---:|
| Train → Train | 4,150,672 | 77.53% |
| Val → Val | 146,993 | 2.75% |
| Test → Test | 136,115 | 2.54% |
| Train ↔ Val | 369,851 | 6.91% |
| Train ↔ Test | 429,670 | 8.03% |
| Val ↔ Test | 121,854 | 2.28% |
| **Total** | **5,355,155** | **100%** |

> Với các cặp split, `↔` là tổng của hai hướng, không phải một edge vô hướng duy nhất.

**Status: DONE — đã có bảng và global edge matrix visualization.**

---

# 5. Training graph vs Full inference graph

## 5.1. Training graph

Training chỉ sử dụng:

```text
Train → Train
```

Số edge:

**3,859,882**

Không sử dụng:

```text
Train → Val
Train → Test
Val → Train
Val → Test
Test → Train
Test → Val
Test → Test
```

**Status: DONE — đã xác định.**

---

## 5.2. Full inference graph

Full graph có:

**5,355,155 edges**

So với training graph:

```text
5,355,155 - 3,859,882
= 1,495,273 additional edges
```

Mức tăng:

```text
≈ 38.7%
```

**Status: DONE.**

---

## 5.3. Train → Train thay đổi như thế nào?

Train → Train trong training graph:

**3,859,882**

Train → Train trong full graph:

**4,150,672**

Tăng:

```text
4,150,672 - 3,859,882
= 290,790 edges
```

Tương đương khoảng:

```text
+7.5%
```

Điều này chứng minh:

> Full inference graph không đơn giản chỉ là `training graph + new nodes`; một phần connectivity giữa các train nodes cũng thay đổi khi mở rộng temporal window.

**Status: DONE.**

---

# 6. Phân tích directed connectivity

## 6.1. Vì sao phải giữ hướng edge?

Không nên gộp:

```text
A → B
B → A
```

thành một quan hệ duy nhất.

Trong Ethereum:

```text
A → B
```

và

```text
B → A
```

có thể biểu diễn hai hành vi transaction khác nhau.

**Status: DONE — đã giải thích trong báo cáo.**

---

## 6.2. Directed asymmetry

Các quan sát hiện có:

- Test → Train khoảng **173k**
- Train → Test khoảng **257k**
- Test → Val khoảng **78k**
- Val → Test khoảng **44k**

Do đó:

```text
Test → Train
≠
Train → Test
```

và:

```text
Test → Val
≠
Val → Test
```

Ví dụ:

```text
Test → Train ≈ 173k
Train → Test ≈ 257k
```

Tỷ lệ khoảng:

```text
257 / 173 ≈ 1.49
```

Tương tự:

```text
Test → Val ≈ 78k
Val → Test ≈ 44k
```

Tỷ lệ khoảng:

```text
78 / 44 ≈ 1.77
```

> **Lưu ý:** Các con số trên là số liệu quan sát/ước lượng đang có. Trước khi đưa vào bảng kết quả chính thức, cần extract lại trực tiếp từ adjacency matrix để tránh sai số do làm tròn.

**Status: PARTIAL — kết luận đã có, nhưng cần re-extract exact counts từ raw adjacency matrix.**

---

# 7. Câu hỏi quan trọng: Test node có kết nối với Test node khác không?

## 7.1. Global answer

Đã biết:

```text
Test → Test = 136,115 edges
```

Vì vậy:

> Có tồn tại transaction giữa các test nodes.

**Status: DONE ở mức global.**

---

## 7.2. Nhưng mentor cần node-level answer

Global count chưa trả lời được:

> "Một sample test cụ thể có bao nhiêu cạnh đến các test node khác?"

Cần tính cho **từng test node**:

```text
test_to_test(v)
```

với mọi:

```text
v ∈ Test
```

Sau đó thống kê:

- Mean
- Median
- P25
- P75
- P90
- Maximum
- Số test nodes có ≥ 1 test neighbor.
- Tỷ lệ `% test nodes có ≥ 1 test neighbor`.

**Status: TODO — bắt buộc thực hiện.**

---

# 8. Phân bố transaction của từng Test node

Đây là phần còn thiếu quan trọng để trả lời trực tiếp câu hỏi của mentor.

Với mỗi test node `v`, cần tính:

## 8.1. Outgoing transactions

```text
v → Train
v → Val
v → Test
```

Tạo các cột:

```text
test_to_train
test_to_val
test_to_test
total_outgoing
```

---

## 8.2. Incoming transactions

Đồng thời tính:

```text
Train → v
Val → v
Test → v
```

Tạo các cột:

```text
train_to_test
val_to_test
test_to_test
total_incoming
```

---

## 8.3. DataFrame mong muốn

```text
test_node
test_to_train
test_to_val
test_to_test
total_outgoing
train_to_test
val_to_test
test_to_test
total_incoming
```

Sau đó thống kê:

| Metric | Test→Train | Test→Val | Test→Test |
|---|---:|---:|---:|
| Mean | TODO | TODO | TODO |
| Median | TODO | TODO | TODO |
| P25 | TODO | TODO | TODO |
| P75 | TODO | TODO | TODO |
| P90 | TODO | TODO | TODO |
| Max | TODO | TODO | TODO |
| % nodes > 0 | TODO | TODO | TODO |

**Status: TODO — bắt buộc thực hiện.**

---

# 9. Kiểm tra mức độ phụ thuộc vào historical/train neighbors

Đã có observation:

> Khoảng **62% edges touching test** kết nối tới train.

Điều này gợi ý test nodes có mức độ kết nối đáng kể với historical/train region.

Tuy nhiên, để chứng minh rõ hơn ở node-level cần tính:

```text
test_to_train(v)
```

cho từng test node.

Sau đó báo cáo:

- Mean
- Median
- P90
- `% test nodes có ít nhất 1 train neighbor`
- `% tổng test outgoing edges đi tới Train`

**Status: PARTIAL — global observation đã có, node-level verification chưa có.**

---

# 10. Train_small vs Train_large

Đây là một trong những nhiệm vụ mentor yêu cầu rõ nhất.

## 10.1. Định nghĩa

### Train-small

```text
Train_small = Train
```

### Train-large

```text
Train_large = Train ∪ Val
```

### Test

```text
Test = Test
```

---

## 10.2. Cần so sánh Test → Train-small

Tính:

```text
E(Test → Train)
```

---

## 10.3. Cần so sánh Test → Train-large

Tính:

```text
E(Test → (Train ∪ Val))
```

Tức là:

```text
E(Test → Train) + E(Test → Val)
```

---

## 10.4. Tính số edge tăng thêm

```text
ΔE =
E(Test → Train_large)
-
E(Test → Train_small)
```

và:

```text
Increase (%) =
ΔE / E(Test → Train_small) × 100%
```

---

## 10.5. Phải làm cả chiều ngược lại

Ngoài:

```text
Test → Train
```

cần tính:

```text
Train → Test
```

và tương ứng:

```text
Train_large → Test
```

Tạo bảng:

| Connectivity | Train-small | Train-large | Δ | % Increase |
|---|---:|---:|---:|---:|
| Test → Train region | TODO | TODO | TODO | TODO |
| Train region → Test | TODO | TODO | TODO | TODO |

**Status: TODO — bắt buộc thực hiện.**

---

# 11. Test → Test cần được giữ riêng

Không được gộp:

```text
Test → Train
Test → Val
Test → Test
```

thành một số duy nhất nếu mục tiêu là phân tích khả năng inference.

Nên giữ:

```text
Test → Train
Test → Val
Test → Test
```

vì chúng đại diện cho ba loại neighbor khác nhau:

```text
Historical neighbors
        ↓
      Train

Later/validation neighbors
        ↓
       Val

Same-period test neighbors
        ↓
      Test
```

**Status: TODO — cần đưa vào node-level analysis và visualization.**

---

# 12. Feature của Test node

Đây là phần **rất quan trọng về leakage** và hiện chưa được kiểm chứng đầy đủ.

Cần xác định rõ:

> Feature của một test node được xây dựng từ information nào?

Có ba khả năng:

### Case A — Train-only

```text
Feature(test)
    ← Train neighbors only
```

### Case B — Train + Val

```text
Feature(test)
    ← Train + Val neighbors
```

### Case C — Full graph

```text
Feature(test)
    ← Train + Val + Test neighbors
```

---

## 12.1. Audit từng feature

Cần kiểm tra từng feature hiện đang sử dụng, ví dụ:

- In-degree
- Out-degree
- Degree
- Amount statistics
- Transaction count
- Lifetime
- Activity statistics
- Katz centrality
- Betweenness
- Closeness
- Eigenvector centrality
- Clustering coefficient
- Các graph-derived features khác.

Với mỗi feature cần ghi:

| Feature | Source nodes/edges | Có dùng Test info? | Có dùng future info? | Risk |
|---|---|---|---|---|
| Degree | TODO | TODO | TODO | TODO |
| In-degree | TODO | TODO | TODO | TODO |
| Out-degree | TODO | TODO | TODO | TODO |
| Amount stats | TODO | TODO | TODO | TODO |
| Lifetime | TODO | TODO | TODO | TODO |
| Centrality | TODO | TODO | TODO | TODO |

**Status: TODO — bắt buộc audit trước khi chốt experimental protocol.**

---

# 13. Temporal leakage vs Graph information

Cần phân biệt hai vấn đề:

## 13.1. Graph information leakage

Ví dụ test node sử dụng:

```text
Test → Test
```

hoặc:

```text
Test → Val
```

Không tự động có nghĩa là leakage.

Điều quan trọng là phải xác định:

> Những transaction nào được giả định là đã observable tại thời điểm prediction?

---

## 13.2. Temporal leakage

Ví dụ nếu prediction tại thời điểm `t` nhưng feature lại được tính từ:

```text
transaction at t + 6 months
```

thì đây có thể là temporal leakage.

Do đó cần xác định rõ:

```text
Prediction time
↓
Observable transaction window
↓
Feature construction
↓
Label availability
```

**Status: TODO — cần formalize trong methodology.**

---

# 14. Visualization cần bổ sung

## 14.1. Đã có

### Global edge matrix

```text
             Train      Val       Test
Train       Train-T   Train-Val  Train-Test
Val         Val-Train Val-Val    Val-Test
Test        Test-Train Test-Val  Test-Test
```

**Status: DONE.**

---

## 14.2. Cần bổ sung: Test outgoing distribution

Bar chart:

```text
Test → Train
Test → Val
Test → Test
```

Mục tiêu:

> Cho mentor thấy test node chủ yếu giao dịch với region nào.

**Status: TODO.**

---

## 14.3. Cần bổ sung: Incoming to Test

Bar chart:

```text
Train → Test
Val → Test
Test → Test
```

**Status: TODO.**

---

## 14.4. Cần bổ sung: Node-level distribution

Có thể dùng histogram/boxplot cho:

```text
test_to_train
test_to_val
test_to_test
```

Ví dụ:

```text
                 Test nodes
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
 Test→Train      Test→Val      Test→Test
       │             │             │
    histogram     histogram     histogram
```

**Status: TODO.**

---

## 14.5. Cần bổ sung: Train-small vs Train-large

Visualization đề xuất:

```text
                Test connectivity

Train-small  ███████████████
Train-large  ████████████████████
```

Hoặc grouped bar chart:

- Test → Train-small
- Test → Train-large
- Train-small → Test
- Train-large → Test

**Status: TODO.**

---

# 15. Bảng kết quả cuối cùng cần có

Sau khi hoàn thành code analysis, nên có một bảng tổng hợp:

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
| Train graph edges | 3,859,882 | DONE |
| Full vs Train edge increase | +38.7% | DONE |
| Train→Train increase after t1 | +7.5% | DONE |
| Test→Train | ~173k* | PARTIAL |
| Train→Test | ~257k* | PARTIAL |
| Test→Val | ~78k* | PARTIAL |
| Val→Test | ~44k* | PARTIAL |
| Test→Test | 136,115 | DONE — global |
| % Test nodes with Test neighbor | TODO | TODO |
| Mean Test→Train per test node | TODO | TODO |
| Median Test→Train | TODO | TODO |
| P90 Test→Train | TODO | TODO |
| Test→Train-large | TODO | TODO |
| Train-large→Test | TODO | TODO |
| Increase from Train-small→Train-large | TODO | TODO |
| Test feature source | TODO | TODO |
| Temporal leakage audit | TODO | TODO |

`*` Cần re-extract exact counts từ adjacency matrix trước khi đưa vào kết quả chính thức.

---

# 16. Những gì đã hoàn thành

## DONE

- [x] Xác định Train / Val / Test.
- [x] Xác định 65/15/20 là split theo phishing `t_first`.
- [x] Kiểm tra temporal non-stationarity.
- [x] Visualization phishing distribution over time.
- [x] Global edge matrix.
- [x] Tổng hợp 6 loại connectivity.
- [x] Training graph = Train→Train.
- [x] Full inference graph.
- [x] So sánh số edge training vs full graph.
- [x] Phát hiện Train→Train tăng sau temporal boundary.
- [x] Giải thích directed asymmetry.
- [x] Xác định Test→Test tồn tại ở mức global.
- [x] Xác định test graph khá sparse.
- [x] Observation rằng khoảng 62% edges touching test connect tới Train.
- [x] Xác định inductive setting về mặt kiến trúc.

---

# 17. Những gì cần thực hiện tiếp

## 🔴 Priority 1 — Bắt buộc

1. [ ] Tạo `Train_small = Train`.
2. [ ] Tạo `Train_large = Train ∪ Val`.
3. [ ] Extract tất cả test node IDs.
4. [ ] Tính:
   - [ ] Test → Train.
   - [ ] Test → Val.
   - [ ] Test → Test.
   - [ ] Train → Test.
   - [ ] Val → Test.
5. [ ] Tính distribution cho từng test node.
6. [ ] Tính `% test nodes có Test neighbor`.
7. [ ] So sánh Train-small vs Train-large.
8. [ ] Audit source của test features.
9. [ ] Kiểm tra temporal leakage.

---

## 🟡 Priority 2 — Visualization

10. [ ] Bar chart Test → Train / Val / Test.
11. [ ] Bar chart Train / Val / Test → Test.
12. [ ] Histogram/boxplot node-level distribution.
13. [ ] Train-small vs Train-large comparison plot.

---

## 🟢 Priority 3 — Hoàn thiện report

14. [ ] Re-extract exact directed edge counts.
15. [ ] Thay các số "~173k", "~257k", "~78k", "~44k" bằng số exact.
16. [ ] Thêm bảng Test-node transaction distribution.
17. [ ] Thêm bảng Train-small vs Train-large.
18. [ ] Thêm methodology về feature observability.
19. [ ] Thêm temporal leakage protocol.
20. [ ] Viết conclusion gắn trực tiếp với câu hỏi của mentor.

---

# 18. Pipeline phân tích cuối cùng

```text
Raw MulDiGraph
      │
      ▼
Temporal Split
      │
      ├───────────────┐
      ▼               ▼
 Train-small         Val
      │               │
      └──────┬────────┘
             ▼
        Train-large
             │
             ▼
            Test
             │
             ▼
   ┌───────────────────────┐
   │ Connectivity Analysis  │
   ├───────────────────────┤
   │ Test → Train           │
   │ Test → Val             │
   │ Test → Test            │
   │ Train → Test           │
   │ Val → Test             │
   └───────────────────────┘
             │
             ▼
   Node-level Distribution
             │
             ▼
 Train-small vs Train-large
             │
             ▼
 Feature Source Audit
             │
             ▼
 Temporal Leakage Check
             │
             ▼
 Visualization
             │
             ▼
      Final Report
```

---

# 19. Câu hỏi cuối cùng cần trả lời được trước khi gặp mentor

Sau khi hoàn thành các TODO, báo cáo phải trả lời rõ ràng 7 câu hỏi:

### Q1. Train nhỏ là gì?

```text
Train_small = Train
```

Có bao nhiêu node, phishing node, edge?

---

### Q2. Train lớn là gì?

```text
Train_large = Train ∪ Val
```

Nó thêm bao nhiêu node và bao nhiêu connectivity?

---

### Q3. Một test node kết nối tới đâu?

Cần có:

```text
Test → Train
Test → Val
Test → Test
```

và chiều ngược lại.

---

### Q4. Test node có kết nối với test node khác không?

Không chỉ trả lời:

```text
Yes, 136,115 edges
```

mà phải trả lời thêm:

```text
X% test nodes có ít nhất một test neighbor.
Median = ...
P90 = ...
Max = ...
```

---

### Q5. Khi dùng Train-large thay Train-small thì test connectivity thay đổi thế nào?

Phải có:

```text
Test → Train-small
Test → Train-large
Δ
% increase
```

và chiều ngược lại.

---

### Q6. Feature của test node lấy từ đâu?

Phải xác định rõ:

```text
Train only?
Train + Val?
Train + Val + Test?
```

và feature nào có nguy cơ temporal leakage.

---

### Q7. Mô hình thực sự đang sử dụng information nào khi dự đoán test?

Đây là câu hỏi quan trọng nhất về experimental protocol:

```text
What information is observable at prediction time?
```

Sau khi trả lời được câu này, mới có thể chốt chính xác:

- graph construction,
- feature construction,
- RGCN message passing,
- inductive inference,
- và leakage control.
