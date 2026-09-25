# Định dạng file gốc — kiểm tra trực tiếp trên dữ liệu thật

Mọi thông tin dưới đây lấy bằng cách `pickle.load()` / đọc trực tiếp file
thật trên đĩa (`raw_data/MulDiGraph/`), không suy đoán từ tên biến trong code.

## 1. `MulDiGraph.pkl`

- **Kiểu đối tượng**: `networkx.MultiDiGraph` — đồ thị có hướng, **cho phép
  cạnh song song** (2 node có thể có nhiều cạnh giữa chúng, mỗi cạnh là 1
  giao dịch riêng biệt).
- **2,973,489 node, 13,551,303 cạnh** (giao dịch).

### Thuộc tính node

```python
G.nodes['0x1f1e784a61a8ca0a90250bcd2170696655b28a21']
# -> {'isp': 0}      # 0 = chưa xác nhận phishing, 1 = phishing xác nhận (Chen et al./XBlock)
```

Chỉ có **1 thuộc tính duy nhất**: `isp` (int, 0 hoặc 1). Không có thuộc tính
nào khác ở cấp node (không có balance, không có thời gian tạo tài khoản, v.v.
— mọi feature khác đều phải **suy ra** từ cạnh).

### Thuộc tính cạnh (1 giao dịch)

```python
G['0x1f1e...']['0x806ceb...']
# -> {0: {'amount': 0.07, 'timestamp': 1504461965.0},
#     1: {'amount': 0.052111, 'timestamp': 1504473420.0}}   # 2 giao dịch song song, cùng cặp địa chỉ
```

| Field | Kiểu | Đơn vị / ý nghĩa |
|---|---|---|
| `amount` | float | Số lượng ETH chuyển trong giao dịch (đơn vị Ether, không phải Wei — giá trị quan sát dao động 0.0 → hàng nghìn, khớp thang ETH thực tế) |
| `timestamp` | float | **Unix epoch giây** (KHÔNG phải mili-giây) — vd `1526454086` = 2018-05-16 07:41:26 UTC, khớp đúng khung thời gian dữ liệu (2015-2019) |

**Lưu ý về tên field không nhất quán**: `00_graph_to_arrays.py` (script đọc
graph này) có fallback chain `d.get("value", d.get("amount", d.get("weight",
0.0)))` và `d.get("timestamp", d.get("time", d.get("ts")))` — dự phòng cho
khả năng tên field khác nhau. Kiểm tra thật trên dữ liệu xác nhận: **file
đang dùng chỉ có `amount` và `timestamp`**, 2 field kia (`value`/`weight`,
`time`/`ts`) không xuất hiện trong file thật — fallback chain là phòng hờ,
không phải mô tả dữ liệu thật.

## 2. Địa chỉ được mã hoá như thế nào

Node ID = **chuỗi string**, định dạng Ethereum address chuẩn:

```
0x1f1e784a61a8ca0a90250bcd2170696655b28a21
└┬┘└──────────────────40 ký tự hex─────────────────┘
 │
 "0x" prefix
```

- Tổng độ dài: **42 ký tự** (`0x` + 40 hex, đã đếm trực tiếp: `len(addr)==42`).
- **Chữ thường (lowercase)** toàn bộ — không có ký tự hoa (đã verify ở lượt
  trước: mọi khoá trong `address_to_index.pkl` đều lowercase, dùng an toàn
  cho tra cứu case-insensitive).
- Không phải hash/anonymize gì thêm — đây **chính là địa chỉ ví Ethereum thật**
  trên mainnet (20 byte, biểu diễn hex). Việc ẩn danh hoá (map sang `addr0,
  addr1, ...` cục bộ trong 1 document) chỉ xảy ra ở bước dựng câu văn bản cho
  BERT (`text_rendering.py`, xem `BERT_debug.md`), không áp dụng cho chính
  `MulDiGraph.pkl`/`address_to_index.pkl`.

## 3. `phisher_account_muldi.txt`

- Plain text, **1 địa chỉ / dòng**, không header, không cột phụ.
- **1,165 dòng** (khớp đúng `total_phishing_original` dùng xuyên suốt dự án).
- Cùng định dạng địa chỉ: `0x` + 40 hex, lowercase.
- Đây là **danh sách con** của các node có `isp==1` trong `MulDiGraph.pkl` —
  dùng làm nguồn `phishing_idx`/quantile cutoff cho các script split
  (`mg_temporal_pipeline_v3_no_overlap.py`), tương đương lọc
  `{addr: G.nodes[addr]['isp']==1}` nhưng đã xuất sẵn ra file text để không
  phải load lại `MulDiGraph.pkl` (1.25GB, tốn ~1-2 phút mỗi lần).

## 4. Vì sao 13,551,303 cạnh → 5,355,155 cạnh trong `adj_inference.npz`

**Đã verify bằng cách đếm trực tiếp trên `MulDiGraph.pkl` thật** (không suy
luận từ code):

```
raw transaction-edges (có lặp, MultiDiGraph):     13,551,303
unique (from_addr, to_addr) address pairs:         5,355,155   ← khớp CHÍNH XÁC nnz của adj_inference.npz
số cạnh bị "gộp" vào 1 cặp đã tồn tại:              8,196,148
cặp địa chỉ có > 1 giao dịch song song:             1,286,990   (24.03% tổng số cặp)
số giao dịch song song nhiều nhất giữa 1 cặp:      10,000       (rất có thể là hub/sàn giao dịch)
```

**Cơ chế gộp** (`mg_graph_weight_formula.py::build_edge_weights()`):
mỗi giao dịch riêng lẻ được tính trọng số theo công thức Eq.1-3 (n-gram
ΔTₙ × αₙ × amount), sau đó **`groupby(["from_addr","to_addr"])["weight"].sum()`**
— nghĩa là **N giao dịch song song giữa cùng 1 cặp địa chỉ được cộng dồn
thành đúng 1 cạnh trong ma trận kề**, giá trị cạnh đó = tổng trọng số của mọi
giao dịch giữa 2 địa chỉ đó. Đây là lý do `adj_train.npz`/`adj_train_large.npz`/
`adj_inference.npz` đều có **nnz < số giao dịch gốc** — nnz đếm **số cặp địa
chỉ phân biệt có ít nhất 1 giao dịch**, không đếm số giao dịch.

**Hệ quả cần nhớ khi đọc số liệu**: mọi "edge count" nhắc tới trong các báo
cáo trước (vd "Test→Train = 257,037") là **số cặp địa chỉ phân biệt**, không
phải số giao dịch — 2 tài khoản giao dịch với nhau 10,000 lần vẫn chỉ tính là
**1 cạnh**.
