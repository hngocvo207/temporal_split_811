# dataset_check — toàn bộ code sinh ra split + feature + train/eval dùng cho model

Thư mục này gom **bản sao** (không phải bản gốc — xem "Lưu ý quan trọng" cuối
file) của mọi script chịu trách nhiệm sinh ra:
`partition.pkl`, `t_first.pkl`, `labels.pkl`, `adj_train.npz`,
`adj_train_large.npz`, `adj_inference.npz`, 23 cột feature node
(`features_output_all23_MG_fullscale.csv`), **và toàn bộ pipeline
train/eval** (data loader, scaler, class weight, threshold, early stopping,
feature selection, pre-training) — để review/audit tập trung một chỗ.

Tài liệu liên quan trong thư mục này:
- [`THREE_CHECKS_RESULT.md`](THREE_CHECKS_RESULT.md) — **đọc trước tiên**:
  kết quả 3 kiểm tra ưu tiên cao nhất — (1) trùng địa chỉ Train/Val/Test:
  KHÔNG trùng, split dùng được; (2) scaler fit trên tập nào: fit trên split
  CŨ (1,945,607), lệch 12.25% so với train v3 đang dùng thật (2,217,282) —
  không phải leakage, nhưng **đã refit thử và đo được**: lệch mean/std lớn ở
  in_degree (+454%/+506%), out_degree, lifetime_days, active_days — ảnh
  hưởng rõ nhất tới nhóm tài khoản degree cao (hub/exchange); (3) ablation
  Group 1+2 `--pre-cutoff`: **0% leakage đo được** trên cả 18/18 cột (khác
  Group 3 đã xác nhận leak).
- [`RAW_DATA_FORMAT.md`](RAW_DATA_FORMAT.md) — định dạng `MulDiGraph.pkl`
  (cột/đơn vị timestamp của cạnh), `phisher_account_muldi.txt`, cách địa chỉ
  được biểu diễn, và cách 13,551,303 cạnh gốc gộp còn 5,355,155 cạnh trong
  `adj_inference.npz` — mọi số liệu đã verify trực tiếp trên dữ liệu thật.
- [`EXISTING_METRICS.md`](EXISTING_METRICS.md) — bảng tổng hợp mọi kết quả đã
  chạy thật (bi_model/tri_model, graph-only, E2 fusion, GBM baseline), so
  sánh mức lệch giữa các slice (`pure_test`/`overlap`/`val`) và giữa split cũ
  vs split v3.
- [`group12_precutoff_ablation_result.json`](group12_precutoff_ablation_result.json) —
  kết quả thô của ablation mục (3) ở trên.
- [`scaler_v3_refit_comparison.json`](scaler_v3_refit_comparison.json) —
  kết quả thô refit scaler trên v3-train + so sánh z-score với scaler cũ,
  mục (2b) ở trên.

## 0. Code train/eval (data loader, scaler, class weight, threshold, early stopping, feature selection, pre-training)

| File | Vai trò |
|---|---|
| `attempt3_corpus.py` | **Data loader** — nạp corpus BERT đã build sẵn, resolve địa chỉ sang global index, ghép với node feature |
| `compute_feature_scaler.py` | **Scaler** — StandardScaler fit CHỈ trên partition `train` (không train+overlap như bản cũ), áp dụng cho toàn bộ 23 cột |
| `label_aware_sampler.py` | **Class weight / sampling mất cân bằng** — label-aware neighbor sampler kiểu PC-GNN (budget lớn hơn + oversample có hoàn lại cho node dương) |
| `graph_sampling.py` | Lõi lấy mẫu subgraph dùng chung cho `label_aware_sampler.py` |
| `metrics.py` | **Chọn threshold** — `find_best_f1_threshold()` quét trên val, không bao giờ nhìn test lúc chọn; `compute_metrics()` (F1/AUPRC/P@K/Recall@K) |
| `train_e2_v2.py` | **Vòng lặp train/eval chính** (BERT+GraphSAGE fusion) — `WeightedRandomSampler`, `CrossEntropyLoss` (tuỳ chọn `--pos-weight`), **early stopping theo patience** (mặc định 4 epoch không cải thiện `--selection-metric`, mặc định P@100 trên val), threshold calibration sau khi chọn checkpoint |
| `train_graph_only_hypothesis.py` | Vòng lặp train/eval cho nhánh graph-only (GraphSAGE thuần) — cùng cơ chế early-stop + threshold, hỗ trợ `--split-dir` để chạy trên split v3 |
| `eval_full_test.py` | Script **evaluation-only** trên toàn bộ 811,704 test thật (không train lại), dùng checkpoint đã có |
| `feature_count_ablation_multiseed.py` | **Feature selection** — xếp hạng 23 đặc trưng theo \|correlation\| với nhãn, multi-seed để tránh chọn nhầm do may rủi |
| `graph_mae.py` | **Pre-training** — masked graph autoencoder (GraphMAE-style, phỏng theo LMAE4Eth) cho `GraphSAGEEncoder`, không cần nhãn |
| `pretrain_graph_encoder.py` | Driver chạy pre-training ở mục trên trên toàn bộ 2,973,489 node (không chỉ node có nhãn) |
| `gbm_baseline.py` | Model baseline tabular thuần (Gradient Boosting, không graph/BERT) — train/eval + threshold riêng, dùng làm mốc so sánh |

**Lưu ý**: các file này thuộc `inductive_model/` (kiến trúc GraphSAGE +
BERT fusion, đang target split v3) — KHÔNG phải `bi_model/train_origin.py`
hay `tri_model/train1.py` (kiến trúc GCN transductive cũ, vẫn dùng split cũ).
Nếu cần bản train/eval của `bi_model`/`tri_model`, báo lại để bổ sung riêng.

## 1. Thứ tự chạy thật (provenance) — script nào sinh ra file nào

```
raw_data/MulDiGraph/MulDiGraph.pkl  (2,973,489 node, 13,551,303 cạnh, nguồn duy nhất)
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                            ▼
mg_temporal_pipeline.py                                    mg_build_adjacency.py
(split GỐC, T_cutoff = percentile                          (Step 3 — dùng LẠI T_cutoff
 80% trên TOÀN BỘ giao dịch)                                 của mg_temporal_pipeline.py)
        │                                                            │
        ├─ labels.pkl        {addr: 0/1}  ◄── QUY TẮC GÁN NORMAL     ├─ address_to_index.pkl
        ├─ t_first.pkl       {addr: ts}       xem mục 2 bên dưới     ├─ adj_train.npz   (CŨ, cắt T_cutoff)
        ├─ t_last.pkl        {addr: ts}                              └─ adj_inference.npz (FULL, không cắt)
        └─ partition.pkl (CŨ, 4 nhóm: train/val/overlap/pure_test)
        │
        │  (labels.pkl / t_first.pkl / address_to_index.pkl / adj_inference.npz
        │   được TÁI SỬ DỤNG nguyên vẹn, không tính lại)
        ▼
mg_temporal_pipeline_v3_no_overlap.py   ◄── "NO_OVERLAP" — xem mục 3 bên dưới
        │
        ├─ partition.pkl (MỚI, 3 nhóm: train/val/test, KHÔNG có "overlap")
        ├─ adj_train.npz   (MỚI, cắt tại t1 — khác T_cutoff của bản CŨ)
        └─ adj_inference.npz  (copy nguyên bản từ mg_build_adjacency.py, không đổi)
        │
        ▼
mg_build_adj_train_large.py
        └─ adj_train_large.npz  (cắt tại t2, dùng cho Train_large = Train ∪ Val)


── Song song, độc lập với nhánh split ──────────────────────────────────────

00_graph_to_arrays.py  (đọc MulDiGraph.pkl 1 lần, dump nodes.npy + edges.npz)
        ▼
02_groups12_exact.py   (18 cột: degree/amount/temporal — đọc edges.npz KHÔNG cắt thời gian)
        │
05_group3_centrality.py (5 cột: centrality — BFS-depth-2 + networkx trên edges.npz KHÔNG cắt thời gian)
        │
        ▼
08_assemble_csv.py  →  features_output_all23_MG_fullscale.csv  (23 cột, 2,973,489 dòng)
```

## 2. Quy tắc gán "normal node" — chính xác từng dòng code

File: `mg_temporal_pipeline.py`, hàm `compute_labels()`:

```python
labels = {addr: int(d.get("isp", 0) == 1) for addr, d in G.nodes(data=True)}
```

**Đọc đúng nghĩa**: một node được gán `label = 1` (phishing) **chỉ khi** thuộc
tính `isp` của nó trong `MulDiGraph.pkl` (do Chen et al. 2020 / XBlock gắn sẵn)
bằng đúng `1`. **Mọi trường hợp còn lại — kể cả node hoàn toàn không có
thuộc tính `isp`** (do `.get("isp", 0)` trả về mặc định `0` khi thiếu key) —
đều bị gán `label = 0` (normal).

**Điểm cần lưu ý (không phải lỗi, nhưng là giới hạn epistemic của nhãn)**:
`label = 0` ở đây có nghĩa là **"chưa được xác nhận là phishing"**, **KHÔNG
phải** "đã được xác minh chủ động là tài khoản sạch". Đây là kiểu nhãn
**residual / mặc định**, đúng với quy ước `confirmed_phishing_only` đã ghi
xuyên suốt dự án (`label_scheme` trong mọi `split_config.json`). Hệ quả thực
tế: trong 2,972,324 node "normal", chắc chắn có một tỷ lệ nào đó là phishing
thật nhưng chưa bị XBlock/Chen et al. gắn nhãn (false negative ẩn trong tập
âm) — đây là giới hạn cố hữu của MỌI benchmark phishing-detection dùng
ground-truth dạng blacklist, không phải vấn đề riêng của pipeline này.

## 3. Ý nghĩa "no_overlap"

**Bản split GỐC** (`mg_temporal_pipeline.py`, hàm `classify_partitions()`)
phân loại từng account theo **TOÀN BỘ khoảng hoạt động** `(t_first, t_last)`
so với 1 mốc cắt `T_cutoff` duy nhất:

```python
if tmax <= T_cutoff:                    partition[addr] = "pure_train_candidates"
elif tmin <= T_cutoff < tmax:           partition[addr] = "overlap"   # ← đây
else:                                    partition[addr] = "pure_test"
```

Một account rơi vào nhóm **"overlap"** nếu nó bắt đầu hoạt động **trước**
`T_cutoff` nhưng **còn tiếp tục hoạt động sau** `T_cutoff` — tức là account đó
**"bắc cầu"** qua ranh giới train/test, không thuộc gọn về bên nào. Nhóm này
được dùng làm ngữ cảnh graph (message-passing) lúc train nhưng **không** được
gán InputExample vào loss (xem `mg_build_examples.py` ở các bước trước), và
lúc đánh giá thì tách riêng khỏi `pure_test` để so sánh (`pure_test` vs
`overlap` breakdown).

**Bản `..._v3_no_overlap.py`** loại bỏ hẳn khái niệm "bắc cầu" này: phân loại
**chỉ dựa vào `t_first`** (bỏ qua `t_last` hoàn toàn), nên **mọi account đều
rơi gọn vào đúng 1 trong 3 nhóm train/val/test**, không có nhóm trung gian:

```python
if tf < t1:        partition[addr] = "train"
elif tf < t2:       partition[addr] = "val"
else:                partition[addr] = "test"
```

→ **"no_overlap" = không còn account nào được xếp vào một nhóm "vừa train vừa
test"** — đổi lại, hệ quả (đã verify ở các bước trước): vì `adj_train.npz`
cắt tại `t1`, **cả val LẪN test đều có bậc = 0 tuyệt đối** trong đồ thị
train-time (không chỉ test như bản gốc) — đây là bài kiểm tra inductive
nghiêm ngặt hơn bản gốc, đúng lý do split này được chọn để dùng cho
`inductive_model/`.

## 4. Lưu ý quan trọng — đây là BẢN SAO, không phải bản chạy được độc lập

Các file trong thư mục này được **copy nguyên văn** từ vị trí gốc để tiện
review tập trung — **không di chuyển (move)** bản gốc, vì nhiều script khác
trong dự án (`run_pipeline_end_to_end.sh`, các script trong
`inductive_model/`) vẫn import/gọi chúng theo đúng đường dẫn cũ.

**Không chạy trực tiếp các file trong `dataset_check/`** — chúng dùng đường
dẫn tương đối theo vị trí gốc (vd `02_groups12_exact.py` đọc
`HERE/"arrays"` — ở vị trí gốc là `fullscale_features/arrays/`, không tồn tại
trong `dataset_check/`). Muốn chạy lại, dùng đúng bản gốc:

| File ở đây | Vị trí gốc (chạy thật tại đây) |
|---|---|
| `00_graph_to_arrays.py`, `02_groups12_exact.py`, `05_group3_centrality.py`, `08_assemble_csv.py` | `Dataset/fullscale_features/` |
| `mg_temporal_pipeline.py`, `mg_temporal_pipeline_v3_no_overlap.py`, `mg_build_adjacency.py`, `mg_build_adj_train_large.py`, `mg_graph_weight_formula.py` | `Dataset/` |
| `attempt3_corpus.py`, `compute_feature_scaler.py` | `Dataset/inductive_model/data_prep/` |
| `label_aware_sampler.py`, `graph_sampling.py`, `graph_mae.py` | `Dataset/inductive_model/model/` |
| `metrics.py`, `train_e2_v2.py`, `train_graph_only_hypothesis.py`, `eval_full_test.py`, `feature_count_ablation_multiseed.py`, `pretrain_graph_encoder.py`, `gbm_baseline.py` | `Dataset/inductive_model/train_eval/` |

Các script train/eval còn phụ thuộc import nội bộ package (`from
data_prep.labels_io import ...`, `from model.gnn_encoder import ...`) — chạy
đúng bằng `python3 -m train_eval.<ten_file>` từ `Dataset/inductive_model/`,
không chạy trực tiếp file trong `dataset_check/`.
