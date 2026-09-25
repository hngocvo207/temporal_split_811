# 3 kiểm tra ưu tiên — kết quả thật (không phải kế hoạch)

Thực hiện theo đúng thứ tự ưu tiên đã yêu cầu: **kiểm tra trùng địa chỉ
trước tiên** (rẻ nhất, quyết định split còn dùng được không), sau đó đọc code
scaler, sau đó ablation Group 1+2 rẻ trước khi cân nhắc 22h cho Group 3.

---

## 1. Địa chỉ trùng Train/Val/Test — ✅ KHÔNG TRÙNG, split dùng được

Giao 3 tập hợp địa chỉ (`partition.pkl` của `Dataset_MG_v3_no_overlap`), sau
khi chuẩn hoá `str.lower()`:

```
n_train = 2,217,282   n_val = 268,352   n_test = 487,855

Train ∩ Val   = 0
Train ∩ Test  = 0
Val ∩ Test    = 0
Train ∩ Val ∩ Test = 0

Địa chỉ KHÔNG phải lowercase sẵn: 0 / 2,973,489
```

**Kết luận: split hiện tại (`Dataset_MG_v3_no_overlap`) không cần dựng lại.**
Không có rủi ro "cùng 1 account vừa ở train vừa ở test" do khác biệt viết
hoa/thường hay do lỗi logic phân vùng.

---

## 2. Chuẩn hoá toàn cục (scaler) — ⚠️ CÓ VẤN ĐỀ, không phải leakage nhưng ảnh hưởng thật tới nhóm hub/exchange

Đọc trực tiếp code (không chạy gì, đúng như đề xuất):
`Dataset/fullscale_features/08_assemble_csv.py` dòng 161 + `Dataset/inductive_model/data_prep/compute_feature_scaler.py` dòng 32.

```python
train_mask = (df["mg_partition"] == "train").to_numpy()
```

Vấn đề nằm ở **`mg_partition` lấy từ đâu**, không phải công thức scale:

```python
# 08_assemble_csv.py dòng 34, 122
SPLIT_DIR = BASE / "data/preprocessed/Dataset_MG"       # ← split CŨ (T_cutoff percentile 80%)
...
df["mg_partition"] = pd.Series(nodes).map(partition)     # partition.pkl của split CŨ, KHÔNG PHẢI Dataset_MG_v3_no_overlap
```

**Nghĩa là**: mean/std dùng để z-score hoá toàn bộ 23 cột feature
(`features_output_all23_MG_fullscale.csv` → `node_features_all23.pt`, file
DUY NHẤT mọi thí nghiệm `inductive_model/` đang dùng) được fit trên tập
**"train" của split CŨ (1,945,607 account)**, không phải tập "train" của split
v3 đang dùng thật (2,217,282 account).

**Đã đo chính xác mức lệch** (không suy đoán):

```
old_train (co so fit scaler)         = 1,945,607
new_train (v3, dang dung that)       = 2,217,282
old_train ⊆ new_train                = ĐÚNG 100% (old_train la tap con that su cua new_train)
new_train KHONG duoc scaler "thay"   = 271,675 account   (12.25% cua new_train)

v3-val  giao voi old_train (scaler)  = 0 / 268,352
v3-test giao voi old_train (scaler)  = 0 / 487,855
```

**Đọc đúng mức độ nghiêm trọng**:
- **KHÔNG phải leakage** — 0 tài khoản val/test của split v3 lọt vào tập dùng
  để fit scaler. Thống kê mean/std không hề "nhìn thấy" val/test.
- **LÀ một lỗi về tính đại diện (coverage)** — 12.25% tài khoản hiện được gọi
  là "train" (dưới split v3) chưa từng đóng góp vào mean/std đang dùng để
  chuẩn hoá chính feature của chúng. Về lý thuyết, mean/std tính trên
  1,945,607/2,217,282 (87.75%) tài khoản train vẫn là ước lượng hợp lý cho
  toàn bộ phân bố train (không có lý do tin 271,675 tài khoản thêm có phân bố
  khác biệt hệ thống) — nhưng đây là một **sự không nhất quán giữa 2 định
  nghĩa "train"** đang tồn tại song song trong pipeline, cần biết rõ trước khi
  công bố kết quả.

**Việc cần làm (chưa làm)**: refit scaler với `mg_partition` trỏ đúng
`Dataset_MG_v3_no_overlap/partition.pkl` thay vì `Dataset_MG/partition.pkl` —
chi phí thấp (đọc lại CSV + tính mean/std, không cần build lại feature thô),
nhưng **mọi checkpoint đã train** (`graph_only_hypothesis_*`,
`e2_v2_best_checkpoint.pt`) đều dùng feature theo scaler CŨ, nên refit xong
cần train lại mới áp dụng được.

### 2b. Đã refit thử scaler trên split v3 thật — phát hiện thêm, quan trọng hơn dự đoán ban đầu

Tính lại mean/std trực tiếp từ `features_output_fullscale.csv` (chưa scale),
mask theo `mg_partition_v3 == "train"` (2,217,282 dòng, remap qua địa chỉ,
**0 dòng thiếu partition** — coverage đầy đủ, không có account nào bị rơi).

**271,675 account chỉ có trong v3-train (không có trong old-train)**: 0 NaN,
0 Inf trong feature thô của nhóm này — không phải lỗi dữ liệu.

**Nhưng mean/std của scaler CŨ vs MỚI lệch nhau đáng kể ở một số cột**, đặc
biệt liên quan đến degree/lifetime (accounts "bắc cầu" — hoạt động dài hơn,
degree cao hơn nhóm train cũ theo đúng bản chất của chúng: `t_first<t1` nhưng
tiếp tục hoạt động lâu hơn old T_cutoff):

| Cột | old_mean → new_mean | old_std → new_std | % lệch mean | % lệch std |
|---|---|---|---:|---:|
| **in_degree** | 0.99 → 5.49 | 41.67 → 252.45 | **+453.9%** | **+505.8%** |
| **out_degree** | 2.94 → 5.39 | 79.62 → 173.11 | +83.5% | +117.4% |
| **lifetime_days** | 13.50 → 29.53 | 42.38 → 73.71 | +118.7% | +73.9% |
| **active_days** | 1.79 → 2.64 | 3.10 → 9.26 | +47.9% | **+198.7%** |
| 18 cột còn lại (amount/temporal/centrality) | — | — | 0.6%–41% | 0.1%–62% |

**Đo tác động thực tế lên z-score của TEST rows** (nếu đổi từ scaler cũ sang
scaler mới, giá trị chuẩn hoá đưa vào model thay đổi bao nhiêu):

| | Mean |z_old − z_new| (toàn bộ test) | % test row lệch >1 std | % test row lệch >5 std |
|---|---:|---:|---:|
| in_degree | 0.028 | 0.121% | 0.027% |
| out_degree | 0.010 | 0.054% | 0.017% |
| lifetime_days | 0.149 | 2.047% | 0.000% |
| active_days | 0.146 | 1.204% | 0.174% |

**Trường hợp cực đoan nhất tìm thấy**: 1 tài khoản test (`in_degree=10,548`,
khả năng là sàn giao dịch/hub) — `z_old=253.1` vs `z_new=41.8` — lệch **211
độ lệch chuẩn** giữa 2 cách scale. Dưới scaler cũ, giá trị z=253 là một outlier
phi thực tế (do std cũ bị đánh giá thấp vì thiếu các account bắc cầu có degree
cao trong tập fit); scaler mới hợp lý hơn nhưng z=42 vẫn còn rất lớn.

**Đọc đúng mức độ nghiêm trọng (cập nhật so với kết luận trước)**:
- Với **đa số** tài khoản test (>97.9%), chênh lệch chuẩn hoá giữa 2 scaler
  là nhỏ (<1 std) — không đáng lo.
- Nhưng với nhóm **hub/exchange-like** (degree rất cao) — chính xác là nhóm
  hay liên quan tới rửa tiền/trung gian trong bài toán phishing — chênh lệch
  chuẩn hoá có thể lớn tới hàng chục/hàng trăm độ lệch chuẩn. Đây **không còn
  là vấn đề "coverage nhỏ, chắc không sao"** như đánh giá ban đầu — với các
  đặc trưng model dựa nhiều vào degree/centrality của hub để phát hiện gian
  lận, sai lệch scaler ở đúng nhóm node quan trọng nhất này đáng được refit
  và train lại trước khi tin tưởng kết quả hiện có.

Script + số liệu đầy đủ: [`scaler_v3_refit_comparison.json`](scaler_v3_refit_comparison.json).

---

## 3. Ablation Group 1+2 với `--pre-cutoff` — ✅ ĐÃ CHẠY, KẾT QUẢ: 0% leakage đo được

Chạy thật (không phải suy luận): `02_groups12_exact.py --pre-cutoff
1527988904` (đúng T_cutoff của split CŨ — cùng cơ sở `mg_partition` mà
`08_assemble_csv.py`/scaler đang dùng, xem mục 2) — **5 giây**, xác nhận đúng
là phần "rẻ" như dự đoán.

So sánh `groups12_full.npz` (không cắt) vs `groups12_precut1527988904.npz`
(chỉ cạnh ≤ T_cutoff), **trên toàn bộ 1,945,607 tài khoản "train"**, cho cả
18/18 cột Group 1+2:

| Cột | % train account bị đổi giá trị | Mean abs diff | Max abs diff |
|---|---:|---:|---:|
| out_degree, in_degree, direction_ratio | 0.00% | 0.0 | 0.0 |
| max/min/avg_out_amount, max/min/avg_in_amount | 0.00% | 0.0 | 0.0 |
| account_balance, lifetime_days, active_days | 0.00% | 0.0 | 0.0 |
| freq_out/in_short, freq_out/in_long | 0.00% | 0.0 | 0.0 |
| short_long_out/in_ratio | 0.00% | 0.0 | 0.0 |

**KẾT QUẢ: cả 18/18 cột — 0% tài khoản train bị thay đổi giá trị, sai lệch
tuyệt đối = 0 tuyệt đối (không phải "rất nhỏ", mà đúng bằng 0).**

**Đây không phải trùng hợp ngẫu nhiên** — giải thích được bằng chính logic
split (`mg_temporal_pipeline.py::classify_partitions`): tài khoản được gán
`train` khi và chỉ khi `t_last(account) <= T_cutoff`, tức **theo định nghĩa**,
một tài khoản train không có bất kỳ giao dịch nào sau T_cutoff — nên cắt hay
không cắt ở T_cutoff, feature 1-hop (Group 1+2) của nó là **giống hệt nhau về
mặt toán học**. Phép đo này xác nhận **logic split và logic tính feature nhất
quán với nhau** (không có bug ẩn nào phá vỡ giả định đó), chứ không chỉ là
suy luận lý thuyết.

**→ Trả lời câu hỏi gốc**: "Group 1+2 cũng bị ảnh hưởng" — **giả thuyết này
SAI, đã đo và bác bỏ**, khác với Group 3 (đã xác nhận leak thật ở lượt
trước, vì centrality phụ thuộc cấu trúc TOÀN ĐỒ THỊ chứ không chỉ cạnh riêng
của account, nên vẫn bị ảnh hưởng dù account đó không có cạnh sau T_cutoff).

Script + kết quả thô: [`group12_precutoff_ablation_result.json`](group12_precutoff_ablation_result.json).

---

## Quyết định cho khoản đầu tư 22h (Group 3 `--pre-cutoff`)

Với kết quả mục 3, **22h compute chỉ cần dành cho Group 3** (5 cột
centrality) — không cần mở rộng sang Group 1+2 (đã chứng minh sạch 100%,
không phải giả định). Đây đúng là thứ mục 3 được yêu cầu để trả lời: "Group 3
có đáng bỏ 22 giờ để sửa không" — câu trả lời gián tiếp nhưng quan trọng:
**vấn đề bị cô lập gọn trong đúng 5 cột centrality**, không lan sang 18 cột
còn lại, nên chi phí sửa (22h) chỉ áp cho đúng phần bị ảnh hưởng, không phải
build lại toàn bộ 23 cột.

---

## Việc dự định làm trong tương lai (chưa làm ngay)

Biến 3 kiểm tra trên thành `assert` tự động trong pipeline (chạy mỗi lần
build lại split/feature, chặn ngay nếu phát hiện lại) — **để sau khi cả 3 vấn
đề đã được xác nhận và xử lý xong** (scaler refit theo mục 2, quyết định về
Group 3 theo mục 3), không viết ngay bây giờ. Lý do (đúng như đã lưu ý):
viết `assert` khi chưa chốt rõ đang kiểm tra chính xác điều kiện gì (vd
"assert 0 overlap" thì rõ, nhưng "assert scaler coverage" cần chốt trước
ngưỡng % chấp nhận được là bao nhiêu) dễ phải viết lại từ đầu.
