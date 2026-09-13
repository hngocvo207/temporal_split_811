# inductive_model — tiến độ so với new_propose.md

Theo dõi song song với danh sách nhiệm vụ ở `../../dynamic_fusion_leakage_audit/new_propose.md` mục 3.
Không sửa đè `tri_model/` — pipeline này độc lập hoàn toàn (đúng quyết định ở đầu tài liệu).

## Giai đoạn A — Chuẩn bị dữ liệu

- [x] **A1** `data_prep/build_node_features.py` — CSV 23 cột → `data/node_features_all23.pt`
      `[2973489, 23]` float32, index hoá theo `address_to_index.pkl` (đã verify permutation đầy
      đủ, không NaN/Inf). Metadata: `data/node_features_all23.meta.json`.
- [x] **A2** `data_prep/build_graph_data.py` — `adj_train.npz`/`adj_inference.npz` →
      `data/graph_train.pt` / `data/graph_inference.pt` (`torch_geometric.data.Data`). Đã verify
      train ⊆ inference theo cạnh (giữ đúng phân biệt train-time/inference-time). Metadata:
      `data/graph_data.meta.json`.
- [x] **A3** Chọn **PyTorch Geometric**, không dùng DGL — lý do & bằng chứng thực nghiệm ở
      `A3_library_decision.md` (DGL không có wheel PyPI khớp torch 2.9.1+cu128 hiện tại; PyG cài
      sạch, đã cài thật vào môi trường).

Lưu ý kỹ thuật phát sinh trong lúc làm A (ghi lại để B1 không giẫm lại):
- `edge_weight` trong graph_*.pt là trọng số thô, dải giá trị rất rộng (0 → ~6.8e12) — **chưa**
  chuẩn hoá/log-transform, B1 phải xử lý trước khi đưa vào GNN.
- `torch.load` mặc định `weights_only=True` (torch ≥2.6) sẽ **fail** khi load các file
  `graph_*.pt` (là `torch_geometric.data.Data`, không phải tensor thuần) — dùng
  `data_prep/io_utils.py::load_graph()`/`load_node_features()` thay vì gọi `torch.load` trực
  tiếp trong code sau này.

Addendum phát hiện khi làm B4 (không có trong danh sách A gốc, nhưng thuộc phạm vi A1):
- [x] **A1b** `data_prep/compute_feature_scaler.py` — 23 cột trong CSV đã qua StandardScaler
      (train-only fit, `fullscale_features/08_assemble_csv.py` dòng 160-185) nhưng mean/std KHÔNG
      được lưu lại ở đâu cả — recompute từ `features_output_fullscale.csv` (bản chưa scale) và
      **verify khớp tuyệt đối** (`max abs diff = 5.7e-14`) với `features_output_all23_MG_fullscale.csv`
      thật. Lưu ở `data/node_features_all23.scaler.json`. Cần cho B4 (đặt feature của account mới vào
      đúng không gian đã chuẩn hoá).

## Giai đoạn B — Inductive GNN (`h_graph(x0)`)
- [x] **B1** `model/gnn_encoder.py` — `GraphSAGEEncoder` (WeightedSAGEConv, mean có trọng số theo
      `edge_weight` đã log1p-normalize) + `GATEncoder` (thay thế, dùng `edge_dim=1`). Đã đo thật:
      `full_graph_forward` trên `graph_train.pt` (2.97M node, 4.16M cạnh) chạy 0.3s, peak 8.0GB VRAM
      (torch.no_grad() thuần suy luận, KHÔNG dùng khi backward — train luôn qua B2's minibatch nhỏ).
- [x] **B2** `model/label_aware_sampler.py` (+ lõi dùng chung `model/graph_sampling.py`) —
      label-aware neighbor sampling kiểu PC-GNN (đơn giản hoá: budget theo nhãn CHÍNH node đang mở
      rộng, không có similarity-aware selector học được như bản gốc). Node dương: budget lớn hơn +
      oversample có hoàn lại; node âm/chưa nhãn: budget nhỏ hơn, không hoàn lại. Verify: subgraph 8
      seed (4 dương + 4 âm) → 103 node/342 cạnh, forward qua cả 2 encoder không lỗi.
- [x] **B3** `model/ego_subgraph.py` + `model/predict_account.py` — `predict_account(addr)` cho
      account ĐÃ có trong graph (Case A), dùng `adj_inference.npz`, sampler uniform (không label-aware
      — nhãn thật của seed là ẩn số cần đoán). Verify bằng account thật trong dataset.
- [x] **B4** `model/new_account_features.py` (công thức G1/G2, **verify khớp tuyệt đối với dữ liệu
      giao dịch thật** tái dựng từ `fullscale_features/arrays/edges.npz`, sau khi sửa 1 lỗi làm tròn
      sớm ở tỉ lệ short/long) + `model/case_b_subgraph.py` (ghép node mới vào slice đồ thị, không
      sửa `adj_inference.npz` gốc) + `model/etherscan_client.py` (gọi Etherscan REST trực tiếp, không
      dùng package `etherscan` mà LMAE4Eth phụ thuộc — package đó không có trên PyPI dưới dạng dùng
      được). Cờ `low_confidence` dựa trên tỉ lệ node trong receptive field có bậc=0 ở `adj_train`
      (chưa có lịch sử train-time) + luôn `True` cho Case B, đúng yêu cầu "độ tin cậy GCN thấp nếu
      hàng xóm cũng là node mới".
      **CHƯA test được end-to-end thật**: không có `ETHERSCAN_API_KEY`/mạng trong sandbox — đã verify
      phần graph-splicing bằng giao dịch giả lập nối vào node thật, và `predict_account` raise lỗi rõ
      ràng (không silent-fail) khi thiếu API key. Xem TODO cuối `model/etherscan_client.py`.

Smoke test: `smoke_test_stage_b.py` (B1/B2), `smoke_test_stage_b4.py` (B4 feature formulas vs dữ
liệu thật), `smoke_test_stage_b_predict.py` (B3+B4 end-to-end, Case B dùng dữ liệu giả lập).

## Giai đoạn C — Tx-BERT (`h_text(x0)`)
- [x] **C1** `model/txbert.py` — `TxBertEncoder` dùng thẳng `BertModel` chuẩn của
      `pytorch_pretrained_bert`, không còn subclass `ETH_GBertEmbeddings` riêng (không còn tensor phụ
      nào phải luồn qua embedding layer sau khi bỏ GCN+FeatureProjector+DynamicFusionLayer). Trả về
      `pooled_output` (h_text) thay vì logits — phân loại chuyển sang Giai đoạn D. Verify:
      `forward()` chỉ còn `input_ids`/`token_type_ids`/`attention_mask`, không còn
      `gcn_vocab_ids`/`graph_features`/`vocab_adj_list`. **Lưu ý**: checkpoint cũ của
      `tri_model`/`bi_model` không load thẳng được vào đây (khác state_dict keys) — chưa viết
      partial-load, để E1 quyết định có cần warm-start hay không.
      Smoke test: `smoke_test_stage_c1.py`.
- [ ] **C2** (tùy chọn, P0 không cần) — chưa làm, đúng kế hoạch (hoãn tới sau khi kiến trúc cơ bản
      chạy được).

## Giai đoạn D — Fusion & Classifier
- [x] **D1** `model/fusion.py` (`GatedFusion`, `diff_softmax` viết lại cục bộ — không import từ
      `tri_model/` để giữ 2 pipeline cô lập) + `model/fraud_model.py` (`FraudClassifier`, ghép
      C1+D1; nhận `h_graph` làm tham số thay vì tự tính, vì cách tính h_graph khác nhau giữa
      train/eval/predict_account — xem docstring). Verify **forward + backward thật**
      (`smoke_test_stage_d1.py`): subgraph thật từ B2's sampler → `GraphSAGEEncoder` → `h_graph`
      (không detach) → `FraudClassifier` (input_ids giả lập, vì dữ liệu tokenized giao dịch thật
      thuộc phạm vi E1) → loss → `backward()` → gradient xác nhận chảy vào cả 3 nhánh (text encoder,
      graph encoder, fusion+classifier), không chỉ shape khớp.
      **Chưa train thật** (theo yêu cầu, dừng trước Giai đoạn E).
- [ ] D2 — chưa làm, đúng kế hoạch (chỉ làm nếu D1 không đủ mạnh, sau khi có kết quả E).

## Rà soát: chỗ nào chưa chắc / dùng dữ liệu giả lập (2026-08-26)

Tự rà lại theo yêu cầu — liệt kê trung thực, không chỉ những gì đã verify tốt:

1. **G3 (5 cột centrality) cho account mới ở B4 là placeholder, KHÔNG phải tính toán thật**
   — gán `train_mean` (→ z=0) vì betweenness/clustering cần toàn bộ đồ thị, không thể tính đúng cho
   1 node cô lập. Đây là compromise nằm trong CODE PRODUCTION (`new_account_features.py`), không
   chỉ ở test — luôn kèm `low_confidence=True` để không lộ ra như 1 con số đáng tin.
2. **`smoke_test_stage_b_predict.py` Case B dùng địa chỉ và giao dịch HOÀN TOÀN BỊA** (`0xfff...`,
   giá trị 1.5 ETH, timestamp tự chọn) nối vào 5 neighbor CÓ THẬT trong graph — vì không có account
   mới thật nào để test (đúng bản chất Case B) và không có mạng/API key. Test này chỉ chứng minh
   luồng ghép graph không crash và ra shape đúng, KHÔNG chứng minh giá trị dự đoán có ý nghĩa.
3. **`model/etherscan_client.py` chưa test được END-TO-END** vì thiếu `ETHERSCAN_API_KEY` — nhưng
   **SỬA LẠI 1 GIẢ ĐỊNH SAI**: lúc đầu tưởng sandbox không có mạng ra ngoài, kiểm tra lại bằng
   `requests.get` thật thì mạng DÙNG ĐƯỢC. Nhân đó phát hiện thêm 1 bug thật: endpoint V1
   (`api.etherscan.io/api`) đã bị Etherscan deprecate (response thật: "deprecated V1 endpoint"),
   đã sửa sang V2 (`api.etherscan.io/v2/api` + `chainid`) và verify V2 nhận đúng format (trả về
   "Invalid API Key" thay vì lỗi deprecated). Vẫn thiếu API key thật để test toàn bộ luồng
   (parse JSON response thật, phân trang >10k tx — xem TODO cuối file).
4. **Phát hiện lúc rà soát, ĐÃ SỬA**: `GATEncoder.full_graph_forward` ở config mặc định
   (hidden=64, out=128, heads=4) **OOM thật** trên GPU 12GB khi test lại ở scale thật (trước đó chỉ
   test GAT ở subgraph nhỏ 103 node, chưa test full-scale nên tuyên bố ngầm "GAT cũng full-scale
   được như SAGE" là SAI). Config nhỏ hơn (hidden=16, out=32, heads=2) chạy được, 3.5GB. Đã cập nhật
   docstring `gnn_encoder.py` với số đo thật thay vì để trống.
5. **C1/D1 ban đầu chỉ test với BertConfig đồ chơi** (vocab=1000, hidden=32) và `input_ids` random
   — không phải kích thước BERT thật của dự án. Rà soát lại thì đã kiểm tra thêm: checkpoint thật
   (`tri_model/output/*_all23_fullcov.pt`) dùng `bert-base-uncased` (vocab=30522, hidden=768, 12
   lớp, 12 head) — vừa test `FraudClassifier` ở đúng kích thước này (`peak 1.1GB`, forward+backward
   OK). Vẫn CHƯA test với dữ liệu tokenized giao dịch thật (thuộc phạm vi E1, chưa có sẵn để load).
6. **B2 (label-aware sampling) là 1 lựa chọn thiết kế chưa được validate bằng kết quả thật** —
   budget theo nhãn CHÍNH node đang mở rộng là simplification tự đề xuất (đã ghi rõ trong
   `label_aware_sampler.py`), không phải PC-GNN nguyên bản, và hiệu quả thật của nó chỉ biết được
   sau khi train so sánh ở Giai đoạn E — chưa có bằng chứng nào ngoài "chạy không lỗi".
7. **Mọi trọng số model hiện đều random-init (chưa train)** — mọi con số cụ thể đã in ra (giá trị
   `h_graph`, `logits`, `loss=0.7474` ở smoke test D1) chỉ chứng minh pipeline nối đúng
   (shape/gradient), KHÔNG có ý nghĩa dự đoán gì. Đừng đọc chúng như kết quả model.
8. Đã verify (không phải giả định): tất cả 2,973,489 khoá trong `address_to_index.pkl` đều
   lowercase — giả định case-insensitive lookup ở B4 (`resolve_counterparties`, `predict_account`)
   là đúng, không phải chỗ cần lo.

## Giai đoạn E — Hạ tầng train/eval
- [x] **E1 (hạ tầng, CHƯA train thật)** `data_prep/attempt3_corpus.py` (nạp corpus Attempt-3 có sẵn
      — 35,000 doc thật, resolve địa chỉ sang global address_to_index, verify 0 miss — nhưng graph
      branch dùng đồ thị FULL-SCALE thật qua B2, không dùng `gcn_adj_*.npz` local-vocab 35k của
      corpus, xem lý do trong docstring), `train_eval/train_e2.py` (train loop: B2 sampler → B1
      GraphSAGEEncoder → C1 TxBertEncoder.from_pretrained("bert-base-uncased") → D1 FraudClassifier
      → AdamW; `WeightedRandomSampler` cấp-seed theo tần suất lớp, bổ sung B2's cấp-hàng-xóm, không
      thay thế — đúng phân biệt đã nói ở lượt trước), `train_eval/metrics.py` (F1(pos)/AUPRC).
      Checkpoint/log riêng trong `output/`, không đụng `tri_model/output/`.

      **Phát hiện + sửa 1 lỗi thật lúc smoke-test** (không phải suy đoán): `evaluate()` gọi
      `full_graph_forward` (GraphSAGE, ~8GB đo ở B1) trong khi BERT-base + optimizer state đồng trú
      trên GPU 12GB đã chiếm ~7.7GB thật (không phải cache nhàn rỗi — đã thử `torch.cuda.empty_cache()`
      trước, KHÔNG hết OOM) → 7.7+8=15.7GB, OOM thật. Fix: chạy `full_graph_forward` trên CPU lúc eval
      (đo thật: 3s cho toàn bộ 2.97M node/5.35M cạnh — chấp nhận được vì chỉ chạy 1 lần/epoch, không
      phải hot loop), chuyển encoder về GPU lại sau. Đây là bằng chứng cụ thể xác nhận lo ngại về
      ngân sách 12GB ở Giai đoạn F là có thật, không phải lý thuyết.

      Smoke test (`smoke_test_stage_e1.py`, KHÔNG phải training thật): 3 bước train (0.9s) + eval đầy
      đủ trên val split thật (5000 ví dụ, 84.9s — chủ yếu BERT-base forward theo batch) chạy được,
      không lỗi. Loss/metric in ra là của model **random-init/vừa warm-start vài bước**, không có ý
      nghĩa dự đoán gì — chỉ xác nhận wiring đúng.
- [ ] **E2 (chạy training thật + so sánh Attempt-3 baseline)** — CHƯA CHẠY, theo yêu cầu dừng lại sau
      khi tạo xong hạ tầng. Baseline cần đánh bại/so sánh: Attempt-3 F1(pos)=87.63%, AUPRC overall=
      0.9155 (pure_test F1=0.9038/AUPRC=0.9474, overlap F1=0.8605/AUPRC=0.8846).
- [ ] **E3 (test inductive thật trên account ngoài vocab train)** — chưa làm, phụ thuộc E2.

## Thí nghiệm tối giản: prove inductive hypothesis trước E2 (2026-08-26)

Theo đề xuất tái sắp xếp thứ tự "fix experimental definition -> prove inductive hypothesis -> then
build full model" thay vì chạy thẳng B1->F3. Đã fix định nghĩa thực nghiệm bằng dữ liệu có sẵn (không
tạo mới): `pure_test` (609,773 acc, 312 dương) verify **100% bậc=0 tuyệt đối trong adj_train** — bài
test inductive nghiêm ngặt nhất; `overlap` (201,931 acc, 217 dương, 197,351 CÓ cạnh train-time) chỉ để
đối chiếu, không phải bài test inductive thật.

Đã VIẾT VÀ CHẠY THẬT `train_eval/train_graph_only_hypothesis.py` — chỉ GraphSAGE (B1) + label-aware
sampler (B2) + 1 Linear classifier, KHÔNG BERT/fusion, train 30 epoch (519 dương + neg_ratio=3, ~9s
tổng), eval bằng `full_graph_forward` trên `adj_inference` thật:

| split | F1(pos)@0.5 | AUPRC | best-F1 (best threshold) | top-100 precision | prevalence |
|---|---|---|---|---|---|
| val (in-distribution) | 0.021 | 0.0198 | 0.056 | 0.0% | 0.054% |
| overlap (thấy 1 phần cấu trúc) | 0.014 | 0.0078 | 0.019 | 4% | 0.107% |
| **pure_test (inductive thật)** | 0.012 | 0.0133 | 0.041 | 4% | 0.051% |
| baseline ngẫu nhiên (sàn) | ~0.001 | ~0.0005-0.001 | — | — | — |

**Đọc kết quả trung thực (không tô hồng)**:
- ✅ Tín hiệu để tiếp tục: **không sập kiểu tri_model** (0.62→0.055, giảm ~11 lần) — pure_test AUPRC
  (0.0133) chỉ thấp hơn val (0.0198) ~1.5 lần, không phải sập theo cấp số nhân. Mọi split đều vượt
  baseline ngẫu nhiên 10-80 lần (có tín hiệu thật, không phải nhiễu) — ủng hộ giả thuyết là kiến trúc
  inductive (bỏ vocab cố định) tránh được đúng lớp lỗi cũ.
- ❌ Chưa đủ mạnh để dùng được: F1(pos) tuyệt đối 0.01-0.06 là quá thấp cho 1 hệ thống chống gian lận
  thật — đây là model **chưa tune gì cả** (30 epoch, hidden=128 tuỳ chọn, không warmup/scheduler, chỉ
  519 mẫu dương, 1 Linear classifier). Không kết luận được là "kiến trúc thất bại" hay "chỉ chưa đủ
  công sức tune" từ 1 lần chạy này.

**Kết luận**: đèn vàng, không phải đỏ hay xanh — đáng để tune tiếp graph-only (nhiều epoch hơn, tìm
hyperparameter, có thể sampler B2 cần chỉnh) TRƯỚC KHI đầu tư vào E2 (BERT, tốn hơn nhiều), vì việc
tái sắp xếp thứ tự đã trả giá trị ngay: phát hiện "chưa đủ mạnh" này chỉ tốn ~9s train, nếu đi thẳng
B1->F3 như cũ thì phải build xong cả BERT+fusion (đã tốn nhiều giờ ở E1) mới nhìn thấy cùng vấn đề.
Kết quả lưu ở `output/graph_only_hypothesis_result.json`, checkpoint ở `output/graph_only_hypothesis_model.pt`.

### Tuning phase (2026-08-27)

- Sửa lỗi thật thứ 2 (cùng dạng OOM ở E1): `evaluate()` trong
  `train_graph_only_hypothesis.py` OOM khi `hidden=256` vì `full_graph_forward` cần bộ nhớ liền mạch
  lớn hơn — fix bằng cách chuyển TOÀN BỘ model (không chỉ encoder) sang CPU lúc eval, chuyển lại GPU
  sau. Cũng cache `graph_inference.pt` (364MB) qua các lần eval thay vì đọc lại từ đĩa mỗi lần.
- `train()` giờ chọn checkpoint theo **val AUPRC tốt nhất** (early stopping, patience=30 epoch) —
  không chọn theo pure_test, để không "nhìn trộm" tập test lúc tune (kỷ luật train/val/test).
- Log trực tiếp lên wandb, project riêng `fraud_detection_inductive` (tách khỏi `fraud_detection` của
  tri_model) — mỗi cấu hình trong sweep là 1 run riêng, group `graph_only_sweep`.
- `train_eval/sweep_graph_only.py`: quét 12 cấu hình (lr×{1e-3,3e-4}, hidden/out×{64,128,256/128},
  mlp_classifier×{có,không}, 150 epoch + early stop). Đang chạy — xem
  `output/graph_only_sweep_results.json` và wandb project khi xong.
- **Hạ tầng GPU remote**: đã cấu hình SSH key-based (`ssh remote-gpu`, host `100.106.237.81`,
  máy `bailab-worker-64`, cũng RTX 3060 12GB, gần như trống) để dùng khi cần thêm GPU (vd chạy song
  song 2 sweep, hoặc E2 sau này) — không dùng mật khẩu nữa sau lần cài key ban đầu.

**Kết quả sweep (12 cấu hình, đã chạy xong thật)** — chọn theo val AUPRC, đọc pure_test SAU khi chọn:

| config (lr, hidden/out, classifier) | val AUPRC | pure_test AUPRC | pure_test F1 | overlap AUPRC |
|---|---|---|---|---|
| **1e-3, 64/64, Linear (BEST theo val)** | **0.0313** | **0.0198** | **0.0166** | 0.0121 |
| 1e-3, 256/128, Linear | 0.0310 | 0.0215 | 0.0138 | 0.0105 |
| 3e-4, 128/128, Linear | 0.0282 | 0.0164 | 0.0233 | 0.0128 |
| 1e-3, 256/128, MLP | 0.0241 | 0.0228 | 0.0194 | 0.0109 |
| ...8 config còn lại | 0.0135-0.0230 | 0.0067-0.0161 | 0.0088-0.0199 | 0.0056-0.0096 |

**Phát hiện quan trọng từ sweep**: hidden dim lớn hơn (256) và classifier phức tạp hơn (MLP) **không**
nhất quán cho kết quả tốt hơn — so 4 cặp Linear/MLP cùng lr+hidden, Linear thắng ở cả 4/4 cặp. Đây là
dấu hiệu **bottleneck không phải model capacity** (thêm tham số chỉ overfit trên 519 mẫu dương) — tune
hyperparameter đã chạm trần thật của riêng nhánh graph (AUPRC pure_test dao động hẹp quanh 0.01-0.02
suốt cả sweep), không phải do chưa tune đủ.

**Phát hiện phụ, QUAN TRỌNG — tự chặn 1 bẫy rò rỉ nhãn trước khi dùng**: lúc tìm cách tăng cỡ tập
dương (chỉ 519 mẫu, đã nghi là nguyên nhân chính khiến AUPRC thấp), thấy `isp_expanded.pkl` +
`propagation_expanded_stats.json` cho tập dương lớn hơn NHIỀU (train 519→3,541, pure_test 312→1,581)
— đúng khoảng "519–3,541" nhắc trong new_propose.md. NHƯNG đọc `mg_propagate_labels_expanded_test.py`
thì nhãn "expanded" này là **lan truyền 1-hop từ 1,165 seed đã xác nhận** (mọi out-neighbor của 1 tài
khoản phisher xác nhận bị gán nhãn "fraud lan truyền", không phải xác nhận độc lập). Dùng nhãn này để
train/eval CHO NHÁNH GRAPH cụ thể sẽ **circular**: model sẽ học "cách 1 bước từ seed đã biết" — gần
như tầm thường khi chính input là cấu trúc đồ thị — không phải học tín hiệu gian lận thật, sẽ làm AUPRC
"tăng" giả tạo mà không chứng minh được gì. **Không dùng isp_expanded cho thí nghiệm này** — đúng
tinh thần "fix experimental definition" mà new_propose.md mục G2 cũng để ngỏ câu hỏi provenance nhãn
này, chưa giải quyết.

**Kết luận sau tuning**: giả thuyết inductive được ủng hộ thêm (không sập pure_test vs val, tune capacity
không đổi được trần ~0.02 AUPRC — nhất quán, không nhiễu), nhưng **519 mẫu dương xác nhận sạch là quá
ít cho riêng nhánh graph tự học đủ mạnh**. Đây chính là lý do kiến trúc gốc đề xuất fusion với text
(BERT) — nhánh graph không cần tự nó đủ mạnh, chỉ cần đủ tín hiệu bổ sung cho nhánh text. Khuyến nghị
tiếp theo: thử fusion đầy đủ (E2) xem có nâng được performance lên mức dùng được không, vì tune riêng
graph đã chạm trần và không phải hướng đáng đầu tư thêm nữa.

Kết quả đầy đủ: `output/graph_only_sweep_results.json`, wandb project `fraud_detection_inductive`
(group `graph_only_sweep`, 12 run).

## E2 — Kết quả training thật (2026-08-27, chạy trên remote-gpu)

Đã chạy training thật (không phải smoke test): `train_eval/train_e2.py --epochs 15 --patience 3
--batch-size 8 --lr 2e-5`, corpus Attempt-3 đầy đủ (20k train), chạy trong tmux session trên
`remote-gpu` (bailab-worker-64, RTX 3060 12GB) để không tranh chấp GPU local, log trực tiếp wandb
(project `fraud_detection_inductive`, group `e2_full_fusion`). Early stop ở epoch 9 (không cải thiện
val AUPRC 3 epoch liên tiếp kể từ epoch 6) — ~1097s/epoch (~18 phút), tổng ~3 giờ.

**Chọn checkpoint theo val AUPRC tốt nhất (epoch 6)**, đọc pure_test/overlap SAU khi chọn xong (đúng
kỷ luật không nhìn trộm test set):

| split | F1(pos) | AUPRC | so với baseline Attempt-3 |
|---|---|---|---|
| val | 0.817 | 0.870 | (baseline không tách val riêng) |
| **pure_test (inductive thật, 0 cạnh train-time)** | **0.919** | **0.973** | baseline: F1=0.904, AUPRC=0.947 → **tốt hơn** |
| overlap | 0.916 | 0.957 | baseline: F1=0.861, AUPRC=0.885 → **tốt hơn** |

**Đọc kết quả**: kiến trúc mới (inductive GNN + BERT fusion, GraphSAGE thay GCN vocab cố định) **bằng
hoặc vượt** checkpoint Attempt-3 gốc (GCN vocab cố định) ở CẢ HAI chỉ số, trên CẢ HAI tập test — kể cả
`pure_test`, bài test inductive nghiêm ngặt nhất (0 cạnh train-time, xem mục "fix experimental
definition" ở trên). Đây là bằng chứng thực nghiệm mạnh cho toàn bộ giả thuyết kiến trúc: nhánh graph
yếu tự nó (AUPRC ~0.02, xem sweep) + nhánh BERT mạnh, fusion cho kết quả tốt hơn kiến trúc cũ mà KHÔNG
cần vocab cố định (giải quyết đúng gốc rễ vấn đề "F1 0.62→0.055" nêu ở đầu new_propose.md).

**Thận trọng, chưa nên tuyên bố thắng tuyệt đối**:
1. So sánh F1 với baseline có thể không hoàn toàn tương đương — log gốc của Attempt-3 ghi "Test_set
   ... calibrated" (có thể đã tune threshold), còn ở đây tôi dùng threshold mặc định 0.5, không
   calibrate gì — nếu vậy kết quả của model mới còn "thiệt" hơn baseline một chút trong so sánh này,
   càng củng cố kết luận. AUPRC (không phụ thuộc threshold) là so sánh công bằng nhất và cũng thắng.
2. **Vẫn ở scale bounded (20k train)**, chưa phải full-scale (1.9M) — đây đúng là cổng cần vượt qua
   trước khi quyết định F1-F3 (scale lên), không phải kết luận cuối cùng.
3. Chưa kiểm tra "generalization theo token/vocab văn bản" tách biệt khỏi "generalization theo cấu
   trúc đồ thị" — pure_test đảm bảo cấu trúc đồ thị chưa từng thấy, nhưng KHÔNG đảm bảo mọi
   token/địa chỉ xuất hiện trong text của pure_test cũng hoàn toàn mới với BERT (BERT có thể đã thấy
   cùng địa chỉ đó xuất hiện trong docs của account KHÁC ở tập train). Chưa loại trừ được khả năng 1
   phần hiệu năng đến từ "nhớ mặt địa chỉ" thay vì học hành vi gian lận tổng quát — cần thêm 1 test
   riêng nếu muốn khẳng định chắc chắn hơn.

Kết quả đầy đủ: `output/e2_final_metrics.json`, `output/e2_best_checkpoint.pt` (trên remote-gpu),
wandb run `e2_bert_graphsage_fusion`.

## CẢNH BÁO QUAN TRỌNG: E2 dùng nhầm nhãn lan truyền, không phải nhãn xác nhận (2026-08-28)

Theo yêu cầu "eval full pure_test", đã chạy eval CHECKPOINT E2 (epoch 6) trên TOÀN BỘ pure_test
(609,773 account thật) + overlap (201,931 account thật) -- không phải mẫu 5,000 như E2 -- dùng lại
corpus 811,704 account đã tokenized sẵn (`data_prep/full_test_corpus.py`).

**Kết quả sập nghiêm trọng, gần giống kiểu tri_model cũ**:

| split | F1(pos) | AUPRC | n | n_pos | thời gian |
|---|---|---|---|---|---|
| pure_test (609,773 thật) | 0.031 | 0.060 | 609,773 | 312 | 10,124s (~2.8h), 60.2 acc/s |
| overlap (201,931 thật) | 0.011 | 0.016 | 201,931 | 217 | 3,358s (~0.9h), 60.1 acc/s |

So với E2's mẫu 5,000 (F1=0.919/0.916, AUPRC=0.973/0.957) -- **sập gần 20-90 lần**.

**NGUYÊN NHÂN ĐÃ XÁC NHẬN (không suy đoán, đối chiếu số liệu trực tiếp)**: corpus Attempt-3
(`runs/expanded_test_attempt3/corpus/`) mà E2 dùng để train/eval được build bằng
`--labels_source isp_expanded.pkl` (nhãn LAN TRUYỀN 1-hop từ seed đã xác nhận), KHÔNG PHẢI
`labels.pkl` (nhãn xác nhận độc lập) -- verify bằng cách khớp số chính xác:
`train_y n_pos=3541` == `propagation_expanded_stats.json`'s "train fraud_after: 3541";
`test_y n_pos=4168` == "test_combined fraud_after: 4168". Đây CHÍNH XÁC là loại nhãn tôi đã tự phát
hiện và **chủ động từ chối dùng** cho thí nghiệm graph-only (xem mục "Phát hiện phụ" ở trên, vì rủi
ro circular) -- nhưng khi làm E2, tôi tái sử dụng corpus có sẵn của tri_model **mà không kiểm tra lại
nguồn gốc nhãn của chính nó**. Đây là lỗi nhất quán của tôi, không áp dụng cùng kỷ luật đã tự đặt ra.

`full_test_corpus.py` (dùng cho eval full-scale vừa chạy) dùng `test_y_strict` (đã verify khớp đúng
312 pure_test + 217 overlap, canonical) -- nên bản thân kết quả full-scale này là ĐÚNG và SẠCH, chỉ
là nó đang so sánh với 1 baseline (E2 mẫu 5k) bị đo trên bài toán DỄ HƠN (đoán "cách 1 bước từ fraud
đã biết" thay vì hành vi gian lận độc lập).

**Đang chạy chẩn đoán để tách 2 nguyên nhân gộp lại** (đổi nhãn propagated→strict VÀ đổi tỉ lệ dương
31.6%→0.05% cùng lúc): `train_eval/eval_diagnostic.py` -- 1 lần forward trên mẫu 100,000 (subset ngẫu
nhiên của pure_test, vì docs đã "shuffled" từ trước), tính metric cho CẢ HAI loại nhãn cùng lúc để so
sánh trực tiếp. Kết quả sẽ cập nhật khi xong (~30 phút, đang chạy trong tmux `diag` trên remote-gpu).

**KẾT QUẢ CHẨN ĐOÁN (đã chạy xong, 100,000 mẫu ngẫu nhiên của pure_test)**:

| cấu hình | F1(pos) | AUPRC |
|---|---|---|
| E2 mẫu 5,000 (propagated, curated) | 0.919 | 0.973 |
| 100,000 mẫu, **cùng nhãn propagated** | 0.180 | 0.293 |
| 100,000 mẫu, nhãn strict (sạch) | 0.054 | 0.078 |
| Full 609,773, nhãn strict | 0.031 | 0.060 |

**Kết luận dứt điểm**: giữ NGUYÊN loại nhãn (propagated, giống hệt E2 train) mà chỉ đổi từ mẫu 5,000
sang 100,000 đã sập từ F1=0.92→0.18 — chứng tỏ **nguyên nhân chính KHÔNG PHẢI do đổi nhãn propagated→
strict, mà do mẫu 5,000 của corpus Attempt-3 không đại diện cho quần thể thật** (rất có thể đã được
chọn/cân bằng theo cách nào đó ở `mg_build_examples.py`, chưa xác định chính xác cơ chế). Đổi nhãn chỉ
làm tệ thêm 1 phần (0.180→0.054), không phải nguyên nhân chính.

**Hệ quả quan trọng nhất**: kết luận "E2 vượt baseline Attempt-3" ở mục trên **KHÔNG ĐÁNG TIN** —
không phải vì kiến trúc sai, mà vì **checkpoint E2 được train/chọn model trên đúng cùng loại mẫu nhỏ
không đại diện đó** (train 20k cũng từ cùng corpus). Đây là train-test mismatch, không phải bằng
chứng kiến trúc thất bại — nhưng cũng không phải bằng chứng kiến trúc thành công. Cần coi ĐÂY là
điểm quan trọng: thí nghiệm graph-only (mục "Thí nghiệm tối giản" ở trên) KHÔNG bị lỗi này vì luôn
eval trên toàn bộ quần thể thật (609,773/201,931) ngay từ đầu — kết luận "graph-only ổn định, không
sập kiểu tri_model, nhưng yếu" của thí nghiệm đó VẪN ĐÁNG TIN. Chỉ có E2 (BERT fusion) là chưa có
con số full-scale nào đáng tin, vì model đó chưa từng được train/chọn bằng mẫu đại diện thật.

**Việc cần làm tiếp (chưa làm)**: train lại E2 với (1) nhãn strict (519 dương thật, không phải 3,541
propagated) và (2) tập validation/model-selection dùng mẫu ĐẠI DIỆN quần thể thật (không dùng nguyên
val 5k cân bằng sẵn của Attempt-3) — chỉ khi đó mới có con số full-scale đáng tin cho toàn bộ kiến
trúc fusion.

## E2 v2 — sửa 2 lỗi trên, train thật xong (2026-09-02)

Sửa đúng 2 lỗi đã phát hiện ở mục trên, train THẬT (không phải smoke test), chạy
qua tmux (`e2_v2_train`), log wandb group `e2_v2_full_fusion` (run
`e2_v2_bert_graphsage_fusion`, id `esxgswwc`, state=`finished`).

**Sửa lỗi #1 (nhãn lan truyền)**: corpus train MỚI, 100,000 account, build bằng
`mg_build_examples.py --cap_train 100000 --labels_source labels.pkl` (nhãn
STRICT, không phải `isp_expanded.pkl`) — giữ toàn bộ 519 dương thật + 99,481 âm
thật (`data_prep/e2_train_corpus.py`, log build ở
`runs/inductive_e2_corpus_100k/build.log`).

**Sửa lỗi #2 (val không đại diện)**: val là mẫu con CỐ ĐỊNH (seed cố định) rút từ
`overlap` population thật (`full_test_corpus.py`, 201,931 account, 217 dương
thật, nhãn `label_strict` đã verify khớp canonical) — giữ toàn bộ 217 dương +
lấy mẫu ngẫu nhiên âm tới 20,000 (`data_prep/e2_val_subsample.py`). Prevalence
mẫu val 1.08% vẫn thiên lệch so với thật 0.11% (vì giữ 100% dương) nhưng KHÔNG
cực đoan như slice 5k cũ (từng oversample dương ~43x).

**Điều chỉnh khác theo yêu cầu**: `make_train_loader()` dùng tỷ lệ sample mục
tiêu 1 dương : 4 âm (`--pos-neg-ratio 4`, xem `train_e2.py`) thay vì cân bằng
đầy đủ 1:1 (sẽ lặp mỗi dương ~96 lần/epoch, dễ overfit đúng 519 example) — loss
vẫn `CrossEntropyLoss` không trọng số (tránh bù imbalance 2 lần). `metrics.py`
bổ sung precision/recall/roc_auc; `find_best_f1_threshold()` quét threshold F1
tối ưu CHỈ trên val (không đụng test). Patience tăng 3→4. `train_e2_v2.py` CHỈ
train+val+chọn checkpoint, không eval pure_test/overlap thật trong lúc train.

**Kết quả train thật** (11 epoch, dừng sớm ở epoch 10, patience=4 không cải
thiện kể từ epoch 6; ~91.5 phút/epoch, tổng ~14h wall-clock kể cả thời gian
tokenize corpus lúc khởi động):

| epoch | val AUPRC | val F1(pos) | precision | recall | ROC-AUC |
|---|---|---|---|---|---|
| 0 | 0.073 | 0.126 | 0.068 | 0.783 | 0.911 |
| 1 | 0.102 | 0.140 | 0.080 | 0.576 | 0.783 |
| 2 | 0.113 | 0.141 | 0.081 | 0.558 | 0.784 |
| 3 | 0.142 | 0.156 | 0.090 | 0.590 | 0.843 |
| 4 | 0.124 | 0.144 | 0.082 | 0.571 | 0.854 |
| 5 | 0.112 | 0.155 | 0.090 | 0.562 | 0.884 |
| **6 (best)** | **0.167** | 0.191 | 0.115 | 0.571 | 0.898 |
| 7 | 0.131 | 0.150 | 0.087 | 0.558 | 0.862 |
| 8 | 0.126 | 0.149 | 0.085 | 0.585 | 0.883 |
| 9 | 0.097 | 0.123 | 0.070 | 0.535 | 0.858 |
| 10 | 0.139 | 0.173 | 0.104 | 0.525 | 0.880 |

Checkpoint chọn theo val AUPRC tốt nhất = **epoch 6** (val AUPRC=0.1669).
Threshold F1 tối ưu tìm trên val (không đụng test) = **0.9328**; ở threshold
này: F1=0.314, precision=0.294, recall=0.336, n=20,000 (217 dương).

**Checkpoint lưu tại**: `Dataset/inductive_model/output/e2_v2_best_checkpoint.pt`
(~419MB, chứa `graph_encoder`, `classifier`, `threshold`, `best_epoch`,
`best_val_auprc`). Lịch sử train đầy đủ:
`Dataset/inductive_model/output/e2_v2_train_history.json`.

**Đánh giá trung thực**: val AUPRC ~0.11–0.17 thấp hơn NHIỀU so với con số giả
0.973 của E2 v1 (do lỗi mẫu val 43x lệch) — đây là con số đáng tin trên val đại
diện thật, không nhìn trộm test. (Nhận định "cải thiện ~6–8 lần so với graph-only"
ghi ở đây lúc đầu, dựa trên val AUPRC, **đã bị bác bỏ bởi eval full-scale ngay
dưới đây — xem lại**.)

### Eval full pure_test/overlap thật (2026-09-03)

Đã chạy `eval_full_test.py --checkpoint output/e2_v2_best_checkpoint.pt` MỘT LẦN
DUY NHẤT (sau khi checkpoint+threshold đã cố định từ val, không nhìn trộm test
lúc chọn model), trên TOÀN BỘ pure_test (609,773) + overlap (201,931) thật:

| split | F1(pos) | precision | recall | AUPRC | ROC-AUC | n | n_pos | thời gian |
|---|---|---|---|---|---|---|---|---|
| **pure_test (inductive thật)** | 0.047 | 0.025 | 0.397 | **0.0189** | 0.911 | 609,773 | 312 | 11,275s (~3.1h), 54.1 acc/s |
| overlap | 0.072 | 0.041 | 0.336 | **0.0231** | 0.900 | 201,931 | 217 | 3,344s (~0.9h) |

**So với val đã dùng để chọn checkpoint (AUPRC=0.167)**: sập ~7–9 lần — val KHÔNG
dự đoán đúng hiệu năng full-scale, dù đã sửa lỗi oversample dương 43x của E2 v1.
**So với graph-only** (trần pure_test AUPRC ~0.013–0.02, xem mục "Thí nghiệm tối
giản"/sweep ở trên): E2 v2 full-scale (0.0189) **nằm đúng trong khoảng đó** —
nhận định "fusion cải thiện 6–8 lần" ở trên KHÔNG đứng vững ở full-scale, phải
rút lại.

**Chẩn đoán nguyên nhân (không suy đoán, có cơ sở định lượng)**: khác với lỗi
oversample DƯƠNG 43x của E2 v1 (đã sửa ở v2), val subsample của v2 vẫn
**undersample ÂM ~30 lần** so với thật (19,783 âm trong val vs 609,773 âm thật ở
pure_test). Recall giữ khá ổn định (val 0.336 → pure_test 0.397, overlap 0.336 —
gần như không đổi), nhưng precision sập mạnh (val 0.294 → pure_test 0.025, overlap
0.041) — đúng như dự đoán lý thuyết: ở threshold cố định, số lượng false positive
tuyệt đối tỉ lệ thuận với cỡ pool âm, nên pool âm nhỏ hơn ~30 lần ở val làm
precision/F1/AUPRC bị thổi phồng một cách hệ thống, không phải do model thay đổi
hành vi. ROC-AUC (0.90–0.91, ít nhạy với imbalance hơn AUPRC/F1) khá nhất quán
giữa val và full-scale — xác nhận model vẫn xếp hạng dương đúng hướng, chỉ là
ngưỡng/precision không transfer được từ 1 tập val nhỏ sang quần thể 1:1955 thật.

**Bài học phương pháp luận cho các lần sau**: "val đại diện" cần giữ đúng tỉ lệ cả
2 lớp (không chỉ giữ đủ dương) mới dự đoán đúng precision/F1/AUPRC ở full-scale —
giữ 100% dương nhưng subsample âm (như đã làm ở v2) vẫn đủ để chọn đúng hướng
tương đối giữa các checkpoint (ROC-AUC ổn định), nhưng KHÔNG đủ để đọc trị số
tuyệt đối của F1/AUPRC như thể đó là số full-scale. Chỉ eval full-scale 1 lần
cuối (đúng kỷ luật đã theo) mới cho con số đáng báo cáo.

**Lưu ý phát hiện khi cập nhật mục này**: repo hiện có thêm các file của hướng
"E3 — Pretrain GraphMAE" (`model/graph_mae.py`, `train_eval/pretrain_graph_encoder.py`,
`train_eval/train_e2_v3.py`, `data_prep/e2_v3_train_corpus.py`,
`smoke_test_stage_pretrain.py`, `smoke_test_stage_e3.py`, và cả
`output/graph_encoder_pretrained.pt` đã có timestamp thật) — các file này được
tạo/chạy song song, KHÔNG phải bởi phiên làm việc train E2 v2 này (không đụng
tới, xem mục E3 ngay dưới). Ghi chú lại để tránh nhầm là cùng 1 luồng công việc.

## E3 — Pretrain không-nhãn cho graph_encoder + mở rộng corpus (2026-09-01, CHỈ MỚI VIẾT CODE, CHƯA CHẠY THẬT)

Bổ sung theo yêu cầu so sánh cách training của LMAE4Eth (paper "LMAE4Eth", MAGAE
— masked graph autoencoder pretrain trên *toàn bộ* node đồ thị, không cần
nhãn) với `train_e2_v2.py` (GraphSAGEEncoder khởi tạo ngẫu nhiên, chỉ học từ
519 mẫu dương xác thực). Lý do chọn hướng này *dựa trên phát hiện thật đã có
sẵn trong chính file này* (mục "Thí nghiệm tối giản" ở trên): tune capacity
nhánh graph riêng lẻ (hidden 64→256, Linear/MLP) **không** nâng được trần
AUPRC ~0.01–0.02 — kết luận lúc đó là "519 mẫu dương xác thực là quá ít cho
riêng nhánh graph tự học đủ mạnh TỪ ĐẦU", không phải vấn đề model capacity.
Pretrain masked-feature-reconstruction không cần nhãn nên dùng được cả
2,973,489 node của `graph_train.pt` — tăng ~2500 lần lượng tín hiệu không-nhãn
so với 519 mẫu dương hiện có, trước khi warm-start vào bước finetune có nhãn.

**Code mới (không đụng bất kỳ file nào của E2/E2 v2 — `train_e2.py`,
`train_e2_v2.py`, `data_prep/e2_train_corpus.py`, `output/e2_best_checkpoint.pt`,
`output/e2_v2_best_checkpoint.pt` giữ nguyên, vẫn đối chiếu lại được)**:
- `model/graph_mae.py` — `GraphMAE`: bọc `GraphSAGEEncoder` (B1) thành masked
  autoencoder kiểu GraphMAE (Hou et al. 2022, cùng ý tưởng MAGAE của
  LMAE4Eth): mask ngẫu nhiên 1 phần node của subgraph, encode, remask `h` tại
  vị trí masked bằng token học được rồi mới decode (bắt buộc, tránh decoder
  "gian lận" nhờ đường residual `lin_self` của `WeightedSAGEConv`), loss SCE
  y hệt công thức `loss_func.py:sce_loss` của LMAE4Eth. CÓ CHỦ Ý đơn giản hơn
  `PreModel` gốc của LMAE4Eth (bỏ nhánh BYOL/EMA-teacher + multi-remask) — xem
  docstring module.
- `train_eval/pretrain_graph_encoder.py` — vòng lặp pretrain: seed mỗi bước
  lấy từ TOÀN BỘ 2,973,489 node (không lọc nhãn/partition, khác hẳn
  `train_e2*.py` chỉ lấy seed có sẵn doc BERT+nhãn), sample subgraph bằng
  `sample_union_subgraph` có sẵn (B2/B3) với budget hàng xóm **cố định, không
  label-aware** (khác `LabelAwareNeighborSampler`, vì đây là pretrain không
  nhãn). Lưu checkpoint `output/graph_encoder_pretrained.pt`
  (`encoder_state_dict` + `encoder_config`).
- `data_prep/e2_v3_train_corpus.py` — bản tổng quát hoá của
  `e2_train_corpus.py`: bỏ `assert n_train==100000`, nhận `corpus_dir` tuỳ ý —
  để dùng được corpus lớn hơn 100k nếu chạy lại
  `python -m Dataset.mg_build_examples --cap_train <N> --labels_source labels.pkl`
  (N tối đa 1,945,607 — toàn bộ partition 'train' thật, xem `split_stats.json`).
  Mặc định vẫn trỏ về đúng corpus 100k có sẵn nên `train_e2_v3.py` chạy được
  ngay cả khi chưa rebuild corpus lớn hơn.
- `train_eval/train_e2_v3.py` — giống hệt kỷ luật của E2 v2 (nhãn strict, val
  subsample cố định từ overlap thật, không nhìn trộm pure_test/overlap lúc
  train), thêm `--pretrained-graph-encoder <path>` để warm-start
  `graph_encoder` từ checkpoint pretrain (assert `in_channels`/`out_channels`
  khớp `build_models()` trước khi `load_state_dict`, tránh lỗi khó hiểu) và
  `--corpus-dir` để trỏ corpus mở rộng. Checkpoint/log riêng
  (`output/e2_v3_best_checkpoint.pt`, wandb group `e2_v3_full_fusion`).
- `smoke_test_stage_pretrain.py`, `smoke_test_stage_e3.py` — đã chạy thật (vài
  bước, không phải training thật, cùng tinh thần `smoke_test_stage_e1.py`):
  xác nhận `GraphMAE.forward` không crash + loss hữu hạn trên subgraph thật
  từ `graph_train.pt`, checkpoint pretrain nạp đúng vào `load_state_dict` của
  `GraphSAGEEncoder` (state_dict khớp tuyệt đối sau save/load), và
  `train_one_epoch` chạy được vài bước với `graph_encoder` đã warm-start —
  KHÔNG có lỗi. Không gọi `build_fixed_val_subsample()` trong smoke test (sẽ
  tokenize 811,704 doc — tốn nhiều phút, để dành cho chạy thật).

**Rủi ro khả thi đã biết, CHƯA giải quyết (không giấu)**: `sample_union_subgraph`
là vòng lặp Python thuần (không vector hoá) — đây là lý do `train_e2*.py` chỉ
dùng `batch_size=8` seed/bước cho nhánh fusion. Duyệt hết 2,973,489 node/epoch
cho pretrain sẽ CHẬM (chưa đo thời gian thật ở quy mô đầy đủ) — dùng
`--max-steps-per-epoch` để đo thử trước khi cam kết chạy nhiều epoch/qua đêm,
khuyến nghị chạy trên remote-gpu qua tmux giống cách đã làm với E2.

**Việc cần làm tiếp (chưa làm)**:
1. Đo thời gian/step thật của `pretrain_graph_encoder.py` ở batch-size dự
   định (chưa đo) — quyết định epoch/steps khả thi trước khi chạy full.
2. Chạy pretrain thật, rồi chạy `train_e2_v3.py --pretrained-graph-encoder
   output/graph_encoder_pretrained.pt`, so `val AUPRC` với E2 v2 (cùng corpus
   100k, chỉ khác warm-start hay không) để cô lập đúng 1 biến.
3. (Tuỳ chọn, tách biệt với #2) rebuild corpus lớn hơn 100k rồi so lần nữa.
4. Chỉ sau khi đã chọn checkpoint theo val AUPRC mới chạy `eval_full_test.py`
   MỘT LẦN DUY NHẤT trên pure_test/overlap thật — đúng kỷ luật đã đặt ra từ
   E2 v2.

## E2 v3 — kết quả train thật với warm-start pretrain (2026-09-03)

Chạy `train_e2_v3.py --pretrained-graph-encoder output/graph_encoder_pretrained.pt`
thật (tmux `e2_v3_train`, wandb group `e2_v3_full_fusion`, run
`e2_v3_bert_graphsage_fusion`, id `wes47zyy`) — thực hiện đúng việc #2 còn
thiếu ở mục E3 trên. Cùng corpus 100k + val subsample cố định + kỷ luật như
E2 v2, CHỈ khác đúng 1 biến: `graph_encoder` warm-start từ pretrain GraphMAE
thay vì khởi tạo ngẫu nhiên.

| epoch | val AUPRC | val F1(pos) | precision | recall | ROC-AUC |
|---|---|---|---|---|---|
| **0 (best)** | **0.173** | 0.230 | 0.140 | 0.650 | 0.927 |
| 1 | 0.129 | 0.164 | 0.093 | 0.677 | 0.925 |
| 2 | 0.125 | 0.201 | 0.127 | 0.493 | 0.917 |
| 3 | 0.147 | 0.210 | 0.131 | 0.530 | 0.926 |
| 4 | 0.133 | 0.150 | 0.084 | 0.682 | 0.919 |

Dừng sớm ở epoch 4 (patience=4, không cải thiện kể từ epoch 0). Checkpoint
chọn: **epoch 0** — val AUPRC=0.1734, threshold F1 tối ưu (tìm trên val)
=0.9903, tại threshold đó: F1=0.303, precision=0.283, recall=0.327, ROC-AUC=0.927.

**So sánh trực tiếp với E2 v2** (cùng corpus/val/kỷ luật, chỉ khác warm-start):

| | val AUPRC (best) | val F1 @ best threshold | ROC-AUC |
|---|---|---|---|
| E2 v2 (encoder ngẫu nhiên, best=epoch 6) | 0.1669 | 0.314 | 0.898 |
| E2 v3 (encoder pretrain warm-start, best=epoch 0) | **0.1734** | 0.303 | **0.927** |

**Đọc kết quả trung thực**: warm-start cho cải thiện AUPRC nhẹ (+3.9%
tương đối) và ROC-AUC tốt hơn rõ (0.898→0.927), nhưng KHÔNG phải bước nhảy
vọt — nằm trong khoảng dao động tự nhiên giữa các epoch của chính E2 v2
(v2 dao động 0.097–0.167 giữa các epoch). Điểm đáng chú ý nhất: checkpoint
tốt nhất của v3 rơi ngay ở **epoch 0** (trước khi finetune có nhãn bắt đầu
thay đổi gì nhiều — train_loss epoch 0 đã giảm rất nhanh 0.11→0.02 ở epoch
1-4 mà val KHÔNG cải thiện thêm, có dấu hiệu overfit nhanh lên đúng 519 mẫu
dương) — nghĩa là phần lớn giá trị đến từ chính pretrain (biểu diễn graph
tốt hơn ngay cả trước finetune), còn finetune có nhãn ở scale 519 dương này
nhanh chóng phản tác dụng (val giảm) thay vì tiếp tục cải thiện.

**Checkpoint lưu tại**: `output/e2_v3_best_checkpoint.pt` (~419MB, có field
`pretrained_graph_encoder`/`corpus_dir` để truy vết). History đầy đủ:
`output/e2_v3_train_history.json`.

**Việc cần làm tiếp (chưa làm)**: chưa chạy `eval_full_test.py` cho v3 (giống
v2, đang chờ xác nhận trước khi chạy 1 lần duy nhất trên pure_test/overlap
thật). Việc #3 (rebuild corpus >100k) của mục E3 vẫn chưa làm, độc lập với
kết quả này.

## PHÁT HIỆN QUAN TRỌNG NHẤT: baseline GBM (không BERT, không graph) vượt fusion ~8 lần (2026-09-05)

Theo đề xuất kiểm tra "signal-limited vs data-limited" (train-AUPRC + test-AUPRC theo
%dương, cố định pool test) + chạy song song 1 baseline không-BERT để biết nhánh BERT
(30h/epoch nếu train full) có đóng góp gì không. Trước khi learning-curve chạy xong,
baseline đã cho câu trả lời dứt khoát hơn dự kiến.

`train_eval`/script tạm `gbm_baseline.py`: `sklearn.HistGradientBoostingClassifier`
(native, không cần cài thêm), CHỈ dùng 23 đặc trưng tabular có sẵn
(`node_features_all23.pt` — degree, volume, gas, thời gian... KHÔNG BERT, KHÔNG
GraphSAGE/message-passing), train trên TOÀN BỘ train partition thật (519 dương +
1,945,088 âm, không subsample) — **22.5 giây tổng cộng** (load data + train + eval).
Eval trên ĐÚNG 1 pool cố định đã dùng cho matched-pool của E2 v2 (150,000 mẫu đầu
`full_test_corpus`, deterministic → 234 dương pure_test chưa thấy + 149,766 âm thật)
để so sánh thẳng hàng, không bị nhiễu bởi cỡ pool khác nhau.

| | AUPRC | ROC-AUC | n_pos |
|---|---|---|---|
| **GBM — test (234 dương pure_test CHƯA thấy, cùng pool)** | **0.381** | 0.990 | 234 |
| GBM — train (519 dương ĐÃ thấy, cùng pool) | 0.813 | 0.999 | 519 |
| *E2 v2 (BERT+GraphSAGE) — test (cùng pool, đã đo trước đó)* | *0.046* | *0.905* | *234* |
| *E2 v2 (BERT+GraphSAGE) — train (cùng pool)* | *0.105* | *0.926* | *519* |

**GBM vượt kiến trúc fusion ~8.3 lần trên test (0.381 vs 0.046) và ~7.7 lần trên
train (0.813 vs 0.105), trong 22.5 giây so với ~14h/checkpoint của fusion.** Đây là
bằng chứng độc lập, dứt khoát: nhánh BERT+GraphSAGE (random-init, học từ đầu chỉ với
519 gradient dương) đang **không khai thác được** ngay cả lượng thông tin đã có sẵn
trong 23 đặc trưng tabular — bài toán tối ưu hoá biểu diễn từ đầu (representation
learning from scratch) với quá ít giám sát dương là nút thắt thật, KHÔNG phải "cần
train BERT full-scale" hay "cần graph message-passing tinh vi hơn". Chi phí 30h/epoch
của nhánh BERT ở quy mô đầy đủ vì vậy **không đáng đầu tư tiếp** cho tới khi có lý do
khác để tin nó sẽ vượt được baseline tabular đơn giản này.

**Khuyến nghị thay đổi hướng đi**: ưu tiên (1) dùng GBM/tabular làm baseline chính
thức để so sánh mọi kiến trúc phức tạp hơn từ nay về sau (rẻ, nhanh, đã có con số
thật); (2) nếu vẫn muốn theo hướng deep learning, warm-start/pretrain (hướng E3) có
lý hơn train-from-scratch, nhưng cần vượt qua được 0.381 của GBM mới coi là có giá
trị tăng thêm; (3) việc thu thập thêm nhãn dương (đã đề xuất ở mục trước) vẫn có thể
có ích nhưng KHÔNG còn là ưu tiên số 1 — ưu tiên số 1 bây giờ là hiểu vì sao
GraphSAGE+BERT thua xa GBM trên cùng dữ liệu đầu vào cơ bản, trước khi đầu tư thêm
compute vào đúng kiến trúc đó.

Kết quả đầy đủ: `output/gbm_baseline_result.json`.

## Learning curve theo %dương (2 đường, pool cố định) — xác nhận thêm phát hiện GBM (2026-09-06)

Train lại `train_e2_ablation.py` từ đầu (encoder ngẫu nhiên) với 25/50/75/100% của
519 dương thật (130/260/389/519 dương), âm huấn luyện giữ CỐ ĐỊNH 20,000 ở cả 4 lần
để cô lập đúng 1 biến (`data_prep/e2_ablation_corpus.py`, nested subset theo seed cố
định). Sau đó `eval_ablation_matched_pool.py` đo LẠI cả train-AUPRC và test-AUPRC
trên **đúng 1 pool cố định** (234 dương pure_test chưa thấy + 149,766 âm thật — cùng
pool dùng cho GBM) ở mọi điểm, để không lặp lại lỗi nhiễu-do-cỡ-pool.

| %dương | n_pos train | Train AUPRC (matched-pool) | Test AUPRC (matched-pool, pool cố định) |
|---|---|---|---|
| 25% | 130 | 0.240 | 0.063 |
| 50% | 260 | 0.118 | 0.065 |
| 75% | 389 | 0.199 | 0.133 |
| 100% | 519 | 0.094 | 0.039 |

**Đọc kết quả trung thực**: cả 2 đường **không đơn điệu** — không khớp gọn với kịch
bản lý tưởng "hội tụ phẳng ở mức thấp" (signal-limited) hay "còn dốc lên rõ ràng ở
100%" (data-limited). Nhiễu giữa các lần train đơn-seed (mỗi fraction chỉ 1 lần chạy,
early-stop ở epoch khác nhau: 5/1/2/2) đủ lớn để che mất xu hướng hệ thống, nếu có.
Đáng chú ý: train-AUPRC giảm dần khi thêm dương (0.240→0.094) dù có NHIỀU giám sát
hơn — khả năng do cách sample: tổng số lượt sample dương/epoch gần như cố định
(~1/5 tổng draw), nên càng nhiều dương duy nhất thì mỗi dương càng ĐƯỢC LẶP LẠI ÍT
LẦN hơn/epoch — làm loãng hiệu ứng "ghi nhớ" từng ví dụ, một biến nhiễu khác cần tính
đến khi đọc train-AUPRC như tín hiệu năng lực thuần tuý.

**Kết luận vững chắc bất kể nhiễu**: ở **MỌI** fraction (kể cả 100%, dùng hết 519
dương sẵn có), fusion đều thua GBM (AUPRC=0.381, cùng pool) từ **3–10 lần**
(0.039–0.133 vs 0.381). Nếu nút thắt thuần tuý là "thiếu nhãn dương", ta sẽ kỳ vọng
đường cong tiệm cận hoặc có xu hướng tiến gần 0.381 khi dùng hết dữ liệu — điều này
KHÔNG xảy ra. Đây là bằng chứng bổ sung (dù đường cong tự nó nhiễu) củng cố kết luận
chính từ mục GBM ở trên: nút thắt chủ đạo là **kiến trúc/tối ưu hoá** (GraphSAGE+BERT
học từ đầu với quá ít giám sát), không phải đơn thuần "cần thêm nhãn dương" — muốn có
kết luận dứt khoát hơn về hình dạng đường cong cần chạy nhiều seed/fraction để lọc
nhiễu, nhưng không cần thiết nữa vì kết luận chính (ưu tiên hiểu vì sao thua GBM
trước khi scale) đã đủ vững từ cả 2 thí nghiệm.

Kết quả đầy đủ: `output/e2_ablation_matched_pool_curve.json`,
`output/e2_ablation_pos{25,50,75,100}_checkpoint.pt`, wandb group
`e2_ablation_pos_fraction`.

## Kiểm tra tính hợp lệ của so sánh GBM vs fusion (2026-09-06)

Theo yêu cầu kiểm tra 2 điều trước khi tin số 8.3×: (1) pool âm eval (149,766) có
lọt vào tập train của GBM không (leakage), (2) GBM và fusion có đang được chấm
trên cùng điều kiện dữ liệu train không.

**(1) Giao tập ID — 0 ở mọi phép kiểm**: `train (partition=='train', n=1,945,607)
∩ fixed_test_neg (149,766) = 0`; `∩ fixed_test_pos (234) = 0`. Đối chiếu trực tiếp
`partition[global_idx]` của toàn bộ 150,000 mẫu eval: 100% là `'pure_test'`, không
lẫn `'train'`. **Không có leakage — AUPRC=0.381 giữ nguyên**, không cần chạy lại.

**(2) Xác nhận bất đối xứng CÓ THẬT**: GBM (chạy ở mục trên) train trên **TOÀN BỘ
1,945,088 âm** (100% train partition, không subsample). Fusion thì KHÔNG:
- E2 v2: 99,481 âm (~5.11% của GBM) — subsample cố định trong corpus 100k.
- Ablation (25/50/75/100%): 20,000 âm (~1.03% của GBM).

Đúng như nghi ngờ: 2 mô hình đã bị chấm trên 2 điều kiện train khác nhau — so sánh
8.3× ban đầu có thể một phần đến từ việc GBM đơn giản là THẤY NHIỀU DỮ LIỆU HƠN,
không hẳn "kiến trúc tốt hơn".

**Thí nghiệm cô lập biến**: train lại GBM trên **ĐÚNG 99,481 âm + 519 dương** mà
E2 v2 đã dùng — không phải mẫu ngẫu nhiên mới, mà chính xác cùng tập ID (giảm
95% dữ liệu train so với bản gốc). Eval trên cùng fixed pool:

| GBM config | n_neg train | Test AUPRC (cùng pool) |
|---|---|---|
| Full train partition (bản gốc) | 1,945,088 | 0.381 |
| **Khớp đúng tập E2 v2 đã dùng** | **99,481** | **0.398** |

**Kết luận dứt khoát**: giảm 95% dữ liệu train của GBM (khớp chính xác điều kiện
fusion) **không làm GBM yếu đi** — nếu có thì nhỉnh hơn (0.398 > 0.381, trong biên
độ nhiễu). Khoảng cách với fusion (0.046, cùng đúng 99,481 âm này) vẫn ~8.6×, thậm
chí lớn hơn phép so sánh ban đầu. **Loại trừ hẳn giả thuyết "GBM thắng chỉ vì thấy
nhiều âm hơn"** — khoảng cách là do kiến trúc/cách tối ưu, không phải bất đối xứng
dữ liệu train. Kết luận "signal-limited" ở 2 mục trên được củng cố thêm, không bị
lung lay bởi câu hỏi hợp lệ này.

Kết quả đầy đủ: `output/gbm_matched_negset_result.json`.

## Tổng kết chuỗi chẩn đoán E2 v2 → GBM (2026-09-06) và khuyến nghị hiện tại

Toàn bộ chuỗi (E2 v2 full eval → matched-pool → GBM baseline → learning curve 2 đường
→ kiểm tra leakage/bất đối xứng dữ liệu) đã đóng vòng, không còn câu hỏi mở nào về
tính hợp lệ của kết luận chính. Tóm tắt 1 chỗ để không phải lần theo từng mục:

1. E2 v2 (BERT+GraphSAGE, encoder ngẫu nhiên) full-scale: pure_test AUPRC=**0.019**.
2. Matched-pool (loại nhiễu cỡ pool): dương đã thấy 0.105 vs dương chưa thấy **0.046**
   — có generalization gap thật nhưng vừa phải (2.3×), không cực đoan.
3. GBM (23 đặc trưng tabular, KHÔNG BERT/graph, 22.5s): **0.381** trên cùng pool —
   vượt fusion **8.3×**.
4. Learning curve TRÊN FUSION, 25/50/75/100% dương (pool cố định, 1 seed/fraction):
   nhiễu, không đơn điệu, nhưng test AUPRC luôn thấp hơn GBM 3–10× ở MỌI fraction.
5. Kiểm tra hợp lệ: không leakage (giao tập ID=0); train GBM trên đúng 99,481 âm
   (khớp E2 v2, giảm 95% dữ liệu) vẫn cho **0.398** — khoảng cách không phải do
   GBM thấy nhiều dữ liệu hơn.
6. **Learning curve TRÊN GBM** (không phải fusion), 25/50/75/100% dương × 20
   seed/fraction (95% CI, âm train CỐ ĐỊNH 99,481 mọi điểm, chỉ đổi số dương) —
   xem mục ngay dưới. Kết quả: đường **CÒN DỐC LÊN rõ** ở 519 dương (0.296→0.361,
   CI 25%/100% không chồng lấn), độ dốc giảm dần (bão hoà kiểu logarit) nhưng
   CHƯA phẳng.

**Verdict (đã tinh chỉnh sau bước 6 — 2 kết luận cùng đúng, không loại trừ nhau)**:
- **Nút thắt CHÍNH hiện tại vẫn là kiến trúc/tối ưu hoá của fusion**: GraphSAGE+BERT
  học biểu diễn từ đầu với quá ít giám sát dương, tối ưu kém hơn hẳn boosted trees
  trên đúng cùng đặc trưng/dữ liệu (mục 3, 5) — đây là khoảng cách LỚN NHẤT (8×) và
  cấp bách nhất cần giải quyết trước.
- **Nhưng thêm nhãn dương KHÔNG VÔ ÍCH như kết luận ban đầu (mục 4, dựa trên fusion
  nhiễu) từng gợi ý** — trên model đang hoạt động tốt (GBM), đường cong CÒN DỐC LÊN
  thật ở quy mô 519, nghĩa là mở rộng seed dương xác thực vẫn có lợi ích đo được,
  chỉ là lợi ích nhỏ dần (diminishing returns), không phải đòn bẩy khẩn cấp nhất.

Khuyến nghị:
1. Dùng GBM (AUPRC 0.361–0.398, giây thay vì giờ) làm **baseline chính thức** cho
   mọi so sánh từ nay, thay Attempt-3 cũ.
2. **Ưu tiên số 1**: **không** đầu tư compute vào scale fusion hiện tại (full 1.9M
   account, ~30h/epoch) cho tới khi có 1 biến thể vượt qua được mốc GBM ở quy mô
   nhỏ trước.
3. Hướng còn đáng thử cho fusion (chưa loại trừ): pretrain sâu hơn (E3, mới chỉ
   warm-start nhẹ 1 lần), hoặc **feed thẳng 23 đặc trưng tabular vào làm 1 nhánh
   input bổ sung** cho fusion thay vì chỉ dựa vào GraphSAGE tự học biểu diễn — cả
   2 đều rẻ hơn nhiều so với train full-scale và có thể kiểm chứng ở quy mô nhỏ
   trước.
4. **Ưu tiên số 2 (không còn "bỏ qua" như bản trước, nhưng vẫn sau #2-3)**: thu
   thập thêm nhãn dương xác thực vẫn có giá trị đo được thật (mục 6) — đáng làm
   song song, không cần chờ giải quyết xong vấn đề kiến trúc mới bắt đầu.

## Learning curve trên GBM theo %dương (20 seed/fraction, 95% CI) — (2026-09-06)

Theo yêu cầu: chạy learning curve trên chính GBM (mô hình mạnh nhất hiện có) thay vì
fusion (yếu, nhiễu) để trả lời "thêm nhãn có giá trị không" một cách đáng tin hơn.
`gbm_learning_curve.py`: 4 fraction (25/50/75/100% của 519 dương) × 20 seed/fraction,
**âm train CỐ ĐỊNH 99,481** ở mọi seed/fraction (chỉ đổi seed nào được chọn trong số
99,481, và số dương) — loại nhiễu cỡ pool đã gặp trước đó. Eval trên đúng 1 fixed
pool (234 dương pure_test chưa thấy + 149,766 âm, đã xác nhận 0 giao với train).
Tổng thời gian: 43.7 giây cho cả 80 lần train+eval (nhanh hơn nhiều so với ước tính
~30 phút, vì data chỉ load 1 lần, mỗi GBM fit <1s).

| %dương | n_pos | AUPRC trung bình (20 seed) | 95% CI |
|---|---|---|---|
| 25% | 130 | 0.296 | [0.272, 0.321] |
| 50% | 260 | 0.332 | [0.307, 0.357] |
| 75% | 389 | 0.345 | [0.325, 0.364] |
| **100%** | **519** | **0.361** | **[0.346, 0.376]** |

**Đọc kết quả**: đường tăng đều đặn qua cả 4 điểm, CI của 25% và 100% KHÔNG chồng
lấn (0.321 < 0.346) → tăng có ý nghĩa thống kê thật, không phải nhiễu ngẫu nhiên
(khác hẳn learning curve trên fusion ở mục 4, vốn quá nhiễu để đọc xu hướng). Độ dốc
giảm dần qua mỗi lần tăng gấp đôi dương (~+0.036 → +0.013 → +0.016) — dạng bão hoà
kiểu logarit, NHƯNG đoạn cuối (75%→100%) vẫn tăng gần bằng đoạn giữa (50%→75%), tức
**chưa phẳng hẳn ở 519 dương**. Ngoại suy thô: mỗi lần tăng gấp đôi số dương gần đây
cho thêm ~0.013–0.016 AUPRC — muốn vượt mốc ví dụ 0.42–0.43 nhiều khả năng cần cỡ
1,000–1,500 dương xác thực (gấp ~2–3 lần hiện tại), không phải một bước nhảy nhỏ.

Kết quả đầy đủ: `output/gbm_learning_curve_result.json` (kèm toàn bộ 20 AUPRC/fraction
để tự tính lại CI hoặc vẽ phân phối nếu cần).

## P@K/Recall@K full-scale, 5-fold CV, và bootstrap generalization gap cho GBM (2026-09-06)

Theo yêu cầu bổ sung 3 phân tích trên GBM gốc (train toàn bộ 1,945,088 âm train
thật — mô hình "0.381"): (1) precision/recall@K trên **FULL pure_test thật**
(609,773 account, không phải mẫu 150,000 như trước), (2) 5-fold CV nội bộ, (3)
bootstrap kiểm tra generalization gap có kiểm soát đúng N. Script
`gbm_full_eval.py`, tổng thời gian 55.4 giây.

**(1) P@K / Recall@K trên FULL pure_test (609,773 account thật, 312 dương)**:

| Metric | Giá trị | Ý nghĩa thực tế |
|---|---|---|
| P@100 | **0.660** | Duyệt 100 tài khoản rank cao nhất → 66 là phisher thật |
| P@1000 | **0.188** | Duyệt 1,000 tài khoản rank cao nhất → 188 là phisher thật |
| Recall@1000 | **0.603** | Duyệt 1,000 tài khoản bắt được 60.3% TOÀN BỘ 312 phisher thật trong 609,773 |
| AUPRC (full 609,461 âm) | 0.343 | Thấp hơn ước tính mẫu 150k trước (0.381) — đúng cơ chế đã biết (pool âm lớn hơn ~4x) |
| ROC-AUC (full) | 0.981 | |

Đây là con số **có giá trị triển khai thực tế**: một quy trình review thủ công chỉ
cần duyệt 1,000/609,773 tài khoản (0.16%) đã bắt được quá nửa số phisher xác nhận
trong toàn bộ population — hữu ích hơn nhiều so với AUPRC trừu tượng khi trình bày
cho người ra quyết định không chuyên ML.

**(2) 5-fold StratifiedKFold CV** (trên chính train partition, độc lập với pure_test):

| Fold | n_val | n_pos_val | AUPRC |
|---|---|---|---|
| 0 | 389,122 | 104 | 0.214 |
| 1 | 389,122 | 104 | 0.287 |
| 2 | 389,121 | 103 | 0.223 |
| 3 | 389,121 | 104 | 0.303 |
| 4 | 389,121 | 104 | 0.237 |
| **Mean ± std** | | | **0.253 ± 0.040** |

CV nội bộ (0.253) **thấp hơn** cả AUPRC full-scale trên pure_test (0.343) — nghĩa là
con số test KHÔNG bị thổi phồng bởi may mắn chọn threshold/model; nếu có thiên lệch
thì test thật còn tốt hơn CV nội bộ, ngược hướng "overfit". Khả năng train partition
tự nó là quần thể khó hơn pure_test (đặc điểm cấu trúc khác nhau giữa 2 partition).

**(3) Bootstrap generalization gap — kiểm soát đúng N (câu hỏi chính)**: lấy ngẫu
nhiên 234/519 dương TRAIN (khớp đúng N=234 với dương pure_test thật), tính AUPRC
trên **CÙNG pool âm FULL** (609,461 âm thật), lặp 200 lần:

| | AUPRC | Ghi chú |
|---|---|---|
| Bootstrap dương TRAIN (234/519, 200 lần) | mean=**0.648**, std=0.016 | 95% CI [0.618, 0.681] |
| Dương PURE_TEST thật (234, chưa thấy) | **0.343** | Percentile trong phân phối bootstrap: **0%** |

**Trả lời câu hỏi chính — generalization gap có thật không, ở mức nào**: **CÓ, và ở
mức LỚN**. AUPRC test thật (0.343) thấp hơn TOÀN BỘ 200/200 lần bootstrap của dương
train ở cùng N (thấp nhất trong 200 lần là 0.610) — không phải hiệu ứng "so sánh N
khác nhau" (519 vs 234) như nghi ngờ ban đầu, vì đã kiểm soát đúng N=234 ở cả hai
phía. Khoảng cách ~1.9× (0.648→0.343) là generalization gap THẬT của chính GBM,
không phải giả tạo do thống kê. Đồng thời GBM vẫn vượt fusion xa (0.343 vs
0.019–0.046 tuỳ điều kiện) — 2 kết luận cùng tồn tại: (a) ngay cả model tốt nhất
hiện có cũng chưa khái quát hoá hoàn hảo tới dương chưa từng thấy (ủng hộ thêm nhãn
có ích, khớp với learning curve ở mục trên), (b) nhưng khoảng cách kiến trúc GBM vs
fusion vẫn là vấn đề lớn hơn nhiều, cấp bách hơn để giải quyết trước.

Kết quả đầy đủ: `output/gbm_full_scale_analysis.json` (kèm toàn bộ 200 giá trị
bootstrap AUPRC để vẽ phân phối nếu cần).

## Embedding GraphMAE (đóng băng) nạp vào GBM — tách graph representation khỏi bài toán tối ưu end-to-end (2026-09-06)

Theo yêu cầu: lấy `h_graph` từ encoder GraphMAE đã pretrain (`output/graph_encoder_pretrained.pt`,
mục E3), **đóng băng hoàn toàn** (`eval()`, `requires_grad_(False)`, không finetune),
chạy `full_graph_forward` trên `graph_inference.pt` MỘT LẦN (1.0 giây, đúng số đo B1)
để lấy embedding 128-chiều cho toàn bộ 2,973,489 node. Mục đích: tách "graph
structure có tín hiệu thật không" ra khỏi "vấn đề tối ưu hoá end-to-end của fusion"
(đã xác định là nút thắt chính) — nạp thẳng embedding vào GBM (không cần train GNN
lại) để kiểm tra representation TỰ NÓ có hữu ích không.

3 cấu hình GBM, cùng train (toàn bộ 1,945,088 âm + 519 dương train that), cùng eval
(FULL pure_test 609,773 thật):

| Cấu hình | dim | AUPRC | ROC-AUC | P@100 | P@1000 | Recall@1000 |
|---|---|---|---|---|---|---|
| A. 23 đặc trưng tabular gốc | 23 | **0.343** | 0.981 | **0.660** | 0.188 | 0.603 |
| B. Embedding GraphMAE đóng băng | 128 | 0.252 | 0.980 | 0.500 | 0.161 | 0.516 |
| C. Ghép A+B | 151 | 0.228 | 0.975 | 0.280 | **0.218** | **0.699** |

**Đọc kết quả — không đơn giản, cần nêu đúng cả 2 chiều**:
- Embedding đóng băng ĐƠN LẺ (B) kém hơn 23 đặc trưng gốc (A) trên AUPRC/P@100/
  Recall@1000 — pretrain tự giám sát (masked-feature-reconstruction, không dùng
  nhãn) không "biết" khái niệm fraud nên không cô đọng đúng tín hiệu phân loại tốt
  bằng đặc trưng đã thiết kế thủ công (degree, volume, thời gian...). Không có gì
  bất ngờ về mặt lý thuyết — pretrain tự giám sát tối ưu cho reconstruction, không
  phải cho downstream task.
- NHƯNG ghép cả hai (C) cho **Recall@1000 tốt nhất** (0.699, vượt cả A) dù AUPRC/
  P@100 tệ hơn — embedding KHÔNG hoàn toàn vô nghĩa, nó thay đổi cách GBM xếp hạng
  theo hướng khác (bắt được nhiều phisher hơn trong top 1000 nhưng xếp hạng top-100
  kém chính xác hơn). Đây là tín hiệu bổ sung thật, chỉ là chưa "cộng dồn" có lợi
  theo nghĩa đơn giản với GBM mặc định (có thể do tăng chiều 23→151 trên chỉ 519
  mẫu dương dễ khiến GBM học nhiễu ở embedding, chưa tune hyperparameter riêng cho
  cấu hình này).

**Giới hạn cần nêu rõ (không giấu)**: encoder pretrain này mới chạy RẤT NHẸ
(`best_epoch=0` — chỉ 1 epoch, xem mục E3/E2 v3) — chưa phải 1 lần pretrain đã tối
ưu kỹ. Kết quả ở đây KHÔNG đủ để kết luận dứt khoát "graph representation vô dụng
cho bài toán này" — chỉ đủ để kết luận "ở trạng thái pretrain hiện tại, embedding
đóng băng không vượt được đặc trưng tabular đơn giản khi dùng với GBM". Muốn kết
luận chắc hơn cần pretrain lâu hơn/kỹ hơn trước khi lặp lại đúng thí nghiệm này.

**Ý nghĩa với hướng đi tổng thể**: củng cố thêm (không phải bằng chứng mới hoàn
toàn, nhưng nhất quán) rằng vấn đề của fusion không chỉ ở "graph message-passing
không có tín hiệu" — ngay cả graph representation tốt nhất hiện có (qua pretrain)
cũng không tự động vượt qua đặc trưng tabular đơn giản. Điều này làm giảm bớt kỳ
vọng rằng "pretrain kỹ hơn" sẽ tự nó giải quyết được khoảng cách 8× với GBM — có
thể cần trực tiếp đưa 23 đặc trưng tabular vào làm input bổ sung cho classifier
(khuyến nghị #3 ở mục Tổng kết) quan trọng hơn là chỉ cải thiện riêng nhánh graph.

Kết quả đầy đủ: `output/gbm_graphmae_embedding_result.json`.

## Sweep hyperparameter GBM + 5-fold CV — mốc baseline "thật" là bao nhiêu (2026-09-06)

Mọi con số GBM báo cáo trước giờ (0.343–0.398 tuỳ điều kiện eval) đều dùng
hyperparameter MẶC ĐỊNH của `HistGradientBoostingClassifier` (chỉ chỉnh
`max_iter=200`) — chưa tune gì. Theo yêu cầu: sweep 18 config
(`learning_rate`×{0.03,0.1,0.3} × `max_leaf_nodes`×{15,31,63} ×
`l2_regularization`×{0,1.0}) × 5-fold StratifiedKFold CV trên train partition
thật (519 dương + 1,945,088 âm) — **chọn config theo CV, KHÔNG nhìn pure_test**
(đúng kỷ luật train/val/test đã theo suốt dự án) — rồi mới retrain config tốt
nhất trên toàn bộ train, eval MỘT LẦN DUY NHẤT trên FULL pure_test.

**Xu hướng rõ trong sweep**: `learning_rate` cao (0.3) + `max_leaf_nodes` lớn
(63) + `l2_regularization`>0 nhất quán cho CV AUPRC cao nhất (0.30–0.34), trong
khi `learning_rate` thấp (0.03) cho kết quả tệ hơn (0.22–0.25) — hợp lý với quy
mô cực nhỏ của lớp dương (519): learning_rate cao hội tụ nhanh trước khi early
stopping cắt, tránh underfitting trên tín hiệu dương hiếm.

| Config | CV AUPRC (5-fold) | Full pure_test AUPRC | P@100 | P@1000 | Recall@1000 |
|---|---|---|---|---|---|
| Default (`max_iter=200`, còn lại mặc định — đã dùng xuyên suốt) | 0.253 | 0.343 | 0.660 | 0.188 | 0.603 |
| **Best theo CV** (`lr=0.3, max_leaf_nodes=63, l2=1.0`) | **0.340** | **0.347** | **0.720** | 0.186 | 0.596 |

**Trả lời câu hỏi chính**: mốc baseline GBM "thật" (đã tune đúng kỷ luật CV,
không nhìn trộm test) là **AUPRC ≈ 0.347** trên full pure_test — chỉ nhỉnh hơn
con số default (0.343) **+1.2%**, không phải một cú nhảy lớn. Điểm cải thiện rõ
nhất là **P@100: 0.66→0.72** (thực chiến hơn), còn Recall@1000 gần như không đổi.

**Ý nghĩa quan trọng nhất**: việc tune hyperparameter GBM **không phải đòn bẩy
lớn** ở bài toán này — GBM mặc định đã gần sát mức tối ưu có thể đạt được với
đúng 519 mẫu dương và 23 đặc trưng hiện có. Điều này củng cố thêm độ tin cậy của
mọi so sánh GBM-vs-fusion đã làm trước đó trong toàn bộ chuỗi chẩn đoán: khoảng
cách 8× với fusion KHÔNG phải do "GBM chưa tune đủ" — ngay cả GBM đã tune kỹ
cũng chỉ nhích lên chút ít, xác nhận `0.34–0.40` là mức trần thực sự của cách
tiếp cận tabular hiện tại (không phải GBM báo cáo trước đó bị đánh giá thấp).

Kết quả đầy đủ (toàn bộ 18 config × 5 fold): `output/gbm_hparam_sweep_result.json`.

## Đặc trưng đồ thị thủ công (thay embedding học được) → GBM — dạng tín hiệu nào hữu ích (2026-09-06)

Theo yêu cầu: xây 9 đặc trưng đồ thị THỦ CÔNG, DIỄN GIẢI ĐƯỢC (khác `[[embedding
GraphMAE đóng băng]]` — không học, không cần train GNN) từ `adj_train.npz`/
`adj_inference.npz`, nạp vào GBM. **Bắt buộc chống leakage đã tuân thủ nghiêm**:
seed set cho PPR/khoảng cách CHỈ lấy từ 519 dương TRAIN — không đụng nhãn
pure_test. Đồ thị dùng để LAN TRUYỀN là `adj_inference` (có cạnh tới pure_test,
cần thiết vì pure_test có 0 bậc trong `adj_train` — giống hệt cách
`full_graph_forward` đã dùng `adj_inference` cho mọi node kể cả pure_test ở B1/
E2, không phải leakage mới). Đơn giản hoá có chủ đích: đối xứng hoá đồ thị
(undirected) cho mọi đặc trưng lan truyền/khoảng cách/clustering/component.

**9 đặc trưng**: `ppr_seed` (Personalized PageRank từ seed), `n_seed_neighbor`
(số hàng xóm 1-hop là seed), `dist_to_seed` (BFS đa nguồn, virtual super-source
— 1 lần duy nhất thay vì 519 lần riêng lẻ), `degree_1hop`, `degree_2hop` (proxy:
tổng bậc hàng xóm), `clustering_coef` (giới hạn bậc ở 300 để tránh nổ tính toán
ở ~622 hub — xấp xỉ cho các hub đó, ghi rõ), `component_size_log1p`,
`logw_mean_1hop`/`logw_mean_2hop` (log1p(edge_weight) tự tính, vì `.npz` gốc là
trọng số THÔ, chưa log1p). Tính cho toàn bộ 2,973,489 node trong **20 giây**
(đồ thị `adj_inference` thưa hơn tưởng: 5,355,155 cạnh có hướng, bậc trung bình
đối xứng chỉ 1.34 — hơn 50% node nằm trong thành phần liên thông nhỏ/cô lập).

**Phát hiện phụ quan trọng (giải thích tại sao PPR yếu ở đồ thị NÀY)**: 98.7%
dương pure_test (308/312) reach được ít nhất 1 seed trong ≤6 hop (trung vị 2
hop), nhưng chỉ 36.9% âm reach được (63.1% hoàn toàn không tới được seed nào)
— khác biệt reachability RÕ RỆT, PPR trung bình dương cao gấp ~9.3 lần âm. Tín
hiệu THẬT có tồn tại, nhưng đồ thị quá thưa/phân mảnh (>50% node cô lập) khiến
"gần một fraud đã biết trong vài hop" vẫn bắt luôn rất nhiều node vô tội trong
cùng thành phần liên thông khổng lồ (1,477,413 node) — làm giảm mạnh precision
so với các đồ thị dày đặc hơn (vd Elliptic) nơi PPR nổi tiếng là đặc trưng mạnh.

**Kết quả GBM (hyperparameter mặc định — bộ đã tune riêng cho 23 đặc trưng ở
mục sweep trên gây suy biến hoàn toàn khi dùng với thang đo khác hẳn của 9 đặc
trưng này, đã tự phát hiện và sửa, xem dưới)**:

| Cấu hình | AUPRC | ROC-AUC | P@100 | P@1000 | Recall@1000 |
|---|---|---|---|---|---|
| D. 9 đặc trưng đồ thị thủ công (riêng) | **0.0142** | 0.635 | 0.070 | 0.008 | 0.026 |
| *So sánh: A raw23 (đã có)* | *0.343* | *0.981* | *0.660* | *0.188* | *0.603* |
| *So sánh: B embedding GraphMAE (đã có)* | *0.252* | *0.980* | *0.500* | *0.161* | *0.516* |

**9 đặc trưng thủ công yếu hơn NHIỀU so với cả raw23 lẫn embedding học được** —
dù reachability khác biệt rõ (đoạn trên), GBM chỉ khai thác được rất ít trong
số đó thành AUPRC thật.

**Permutation importance (trên model D không suy biến, scoring=average_precision,
50k mẫu âm + toàn bộ dương)** — trả lời trực tiếp "tín hiệu đồ thị dạng nào hữu ích":

| Đặc trưng | Importance |
|---|---|
| **clustering_coef** | **+0.094** |
| **degree_2hop** | **+0.092** |
| degree_1hop | +0.017 |
| logw_mean_1hop | +0.001 |
| logw_mean_2hop | +0.0002 |
| n_seed_neighbor, dist_to_seed, component_size_log1p | 0.000 (không được dùng) |
| ppr_seed | **−0.032** (xáo trộn ngẫu nhiên lại CẢI THIỆN kết quả!) |

**Trả lời câu hỏi chính — tín hiệu đồ thị dạng nào hữu ích, diễn giải được**:
đặc trưng **mật độ cấu trúc cục bộ** (`clustering_coef`, `degree_2hop`) hữu ích
nhất — ngược hẳn kỳ vọng ban đầu rằng PPR/khoảng cách-tới-seed sẽ mạnh nhất.
`ppr_seed` không những vô dụng mà còn **phản tác dụng** (importance âm) — hệ
quả trực tiếp của phát hiện phụ ở trên: trên đồ thị thưa/phân mảnh này, PPR từ
519 seed lan truyền tới rất nhiều node vô tội trong cùng thành phần khổng lồ,
biến nó thành nhiễu nhiều hơn tín hiệu đối với GBM. `dist_to_seed`/
`n_seed_neighbor` có tương quan thật với nhãn (đoạn phát hiện phụ) nhưng
importance=0 — bị `degree_2hop`/`clustering_coef` (tương quan, dùng thay thế)
che khuất trong thuật toán tham lam của GBM, không phải bằng chứng chúng vô
nghĩa tuyệt đối.

**Sự cố kỹ thuật cần nêu trung thực (không giấu)**: ghép raw23+9 đặc trưng
(cấu hình E) qua GBM mặc định cho kết quả SUY BIẾN (AUPRC≈0.0005≈nền ngẫu
nhiên, ROC-AUC=0.500 chính xác) — TỆ HƠN cả 2 cấu hình riêng lẻ. Đã loại trừ
nguyên nhân "hyperparameter tune sai" (thử cả early_stopping=True/False, vẫn
suy biến khi tắt hẳn early stopping, huấn luyện đủ 200 vòng).

**Sanity check (theo yêu cầu) — bác bỏ giả thuyết sentinel/thang đo**: ghép
raw23 với (F) 9 cột nhiễu Gaussian chuẩn thuần tuý, và (G) 9 cột xáo trộn
NGẪU NHIÊN từ chính 9 đặc trưng thủ công (giữ nguyên phân phối/sentinel=99
thật, chỉ phá tương quan với nhãn) — cả hai đều KHÔNG suy biến:

| Cấu hình | dim | unique_probs | AUPRC | ROC-AUC |
|---|---|---|---|---|
| A. raw23 (đối chiếu) | 23 | 21,850 | 0.343 | 0.981 |
| F. raw23 + 9 nhiễu Gaussian | 32 | 111,173 | 0.296 | 0.972 |
| G. raw23 + 9 cột xáo trộn (giữ sentinel/skew thật) | 32 | 58,466 | 0.366 | 0.978 |
| E. raw23 + 9 đặc trưng THẬT (chưa xáo trộn) | 32 | 2 | 0.0005 | 0.500 |

**Kết luận đã sửa (giả thuyết trước SAI)**: không phải do sentinel=99 hay
thang đo lệch — G giữ nguyên đúng phân phối/giá trị cực đoan đó và vẫn chạy
bình thường (thậm chí AUPRC nhỉnh hơn A). Suy biến CHỈ xảy ra khi ghép đúng
giá trị THẬT có tương quan thật (với nhãn và với nhau) — nhiều khả năng do đa
cộng tuyến giữa các đặc trưng gần-trùng-khái-niệm (`degree_2hop`,
`clustering_coef`, `ppr_seed`, `dist_to_seed` đều là proxy của "gần seed")
kết hợp với `sample_weight` cực đoan (~3748:1) gây bất ổn cho thuật toán
tìm-split tham lam của `HistGradientBoostingClassifier` — **vẫn là giả thuyết,
chưa xác nhận dứt điểm, không nên diễn giải đây là bằng chứng "ghép 2 loại
đặc trưng không có lợi"**. Cần kỹ thuật khác (loại bớt đặc trưng trùng lặp,
hoặc stacking 2 model riêng thay vì ghép cột thẳng) nếu muốn kết luận chắc hơn
về việc ghép có thực sự cải thiện so với raw23 riêng lẻ hay không.

Kết quả đầy đủ: `output/gbm_handcrafted_graph_result_v2.json`,
`output/handcrafted_graph_features.npy` (9 cột, N=2,973,489).

## BUG QUAN TRỌNG: lệch hướng cạnh giữa train và eval của toàn bộ nhánh graph (2026-09-06)

**Phát hiện** (theo yêu cầu rà soát): `model/graph_sampling.py::sample_union_subgraph`
(dùng bởi B2 `LabelAwareNeighborSampler` — train, B3 `ego_subgraph` — inference
1 account, và `pretrain_graph_encoder.py` — GraphMAE) và `full_graph_forward()`
(dùng bởi **MỌI** `evaluate()`/eval hàng loạt trong E2/E2v2/E2v3/ablation/
GraphMAE-embedding-extraction đã làm) dùng **HAI QUY ƯỚC HƯỚNG CẠNH NGƯỢC NHAU**:

- `adj_csr[i,j] != 0` nghĩa là **i gửi tiền cho j** (row=from_addr, col=to_addr
  — xác nhận từ `mg_build_adjacency.py`).
- `sample_union_subgraph`: center aggregate từ `adj_csr[center,:]` — tức từ
  **người center ĐÃ GỬI TIỀN CHO** (payee/out-neighbor).
- `full_graph_forward` (trước fix): dùng thẳng `edge_index=[coo.row, coo.col]`
  từ `graph_train.pt`/`graph_inference.pt` → node v aggregate từ
  `edge_index[0]` tại vị trí `edge_index[1]==v`, tức từ **người ĐÃ GỬI TIỀN
  CHO v** (payer/in-neighbor) — **NGƯỢC HẲN** quy ước train.

**Hệ quả**: mọi checkpoint (GraphMAE pretrain, GraphSAGE trong E2/v2/v3, graph-only
hypothesis, sweep) đều TRAIN đúng theo quy ước payee (không bị ảnh hưởng, vì train
luôn qua `sample_union_subgraph`) — nhưng **MỌI con số eval/suy luận hàng loạt đã
báo cáo trong toàn bộ investigate này** (E2 v2 full pure_test AUPRC=0.019,
matched-pool 0.105/0.046, learning curve trên fusion, embedding GraphMAE
đóng băng→GBM...) đều tính `h_graph` với hướng cạnh SAI (payer thay vì payee) —
KHÔNG khớp với những gì encoder thực sự học được lúc train.

**Quyết định sửa**: chọn quy ước của `sample_union_subgraph` (payee) làm chuẩn —
**không phải vì nó đúng hơn về domain** (cả 2 hướng đều có tín hiệu fraud thật:
nạn nhân gửi tiền CHO phisher = payer-signal; phisher gửi tiền ĐI để rửa tiền =
payee-signal — không có câu trả lời domain rõ ràng) — mà vì **mọi checkpoint đã
train từ đầu dự án đều học theo quy ước này**, sửa phía eval để khớp tránh phải
train lại toàn bộ (GraphMAE pretrain 2.97M node, BERT+GraphSAGE E2/v2/v3...).
Đã ghi đầy đủ lý do vào `data/graph_data.meta.json` field
`edge_direction_convention` (kèm cảnh báo đây là lựa chọn thực dụng, không phải
khẳng định "payee là hướng đúng duy nhất").

**Fix**: `data_prep/build_graph_data.py::csr_to_pyg_data` — đổi
`edge_index=[coo.row, coo.col]` thành `[coo.col, coo.row]`. Build lại
`graph_train.pt`/`graph_inference.pt` từ `adj_train.npz`/`adj_inference.npz`
gốc (deterministic, không mất dữ liệu). **Assert chống tái diễn**:
`data_prep/verify_edge_direction_consistency.py` — đối chiếu tập "nguồn
aggregate" của `sample_union_subgraph` vs `full_graph_forward` trên 100 node
ngẫu nhiên, cả `train` và `inference` graph. **Đã chạy, PASS 100/100 cả hai.**

**Việc cần làm tiếp (đang chạy)**: chạy lại `eval_full_test.py` với
`e2_v2_best_checkpoint.pt` (checkpoint CŨ, không cần train lại — chỉ eval lại
với `graph_inference.pt` đã sửa) trên FULL pure_test/overlap để biết con số
AUPRC=0.019 có đổi không — đang chạy trong tmux `e2_v2_eval_fixed`, kết quả
lưu `output/e2_v2_full_eval_result_FIXED_EDGEDIR.json`. **CẢNH BÁO: kết luận
"fusion thua GBM 8×, signal-limited không phải data-limited" ở các mục trên
được rút ra TỪ SỐ LIỆU BỊ ẢNH HƯỞNG BỞI BUG NÀY — cần xác nhận lại sau khi có
kết quả eval đã sửa, TRƯỚC KHI coi các kết luận đó là chốt.**

### SỬA LẠI LẦN 2 — đảo ngược quyết định trên, theo chuẩn HGT (cùng ngày 2026-09-06)

Sau khi rà soát lại theo đúng quy trình chuẩn GNN (đối chiếu paper **Heterogeneous
Graph Transformer**, Hu et al. 2020, WWW'20 — `heterogeneous graph transformer.pdf`):
**Definition 2/Eq.1**: `H[t] ← Aggregate_{s∈N(t)}(...)`, `N(t)` = các **source node**
có cạnh `s→t` — tức target LUÔN aggregate từ nguồn TRỎ VÀO nó (incoming). Đây là
chuẩn GNN thông thường (GCN/GraphSAGE/GAT/HGT đều vậy), không phải lựa chọn tuỳ ý.

**Quyết định LẦN 1 ở trên (chọn payee/outgoing làm chuẩn) đã SAI theo chuẩn này —
đảo ngược lại**: `full_graph_forward`/`graph_train.pt`/`graph_inference.pt` giữ
**NGUYÊN BẢN GỐC** `edge_index=[coo.row, coo.col]` (không đổi gì, revert lại
đúng như trước khi có bug) — target aggregate từ payer (đúng chuẩn incoming).
Thay vào đó, **sửa phía `sample_union_subgraph`'s CALLERS** (không sửa chính
`sample_union_subgraph`/`_neighbors` — đổi tên biến `edge_src`/`edge_dst`
KHÔNG đủ, vì `_neighbors()` luôn đọc HÀNG của center trong CSR bất kể gán vai
trò nào; phải TRANSPOSE ma trận đầu vào để hàng của center trở thành payer
thật): `model/label_aware_sampler.py` (`LabelAwareNeighborSampler.__init__`),
`model/ego_subgraph.py` (`build_ego_subgraph`), `model/case_b_subgraph.py`
(`build_case_b_subgraph`), `train_eval/pretrain_graph_encoder.py` — cả 4 nơi
đều thêm `.T.tocsr()` khi nạp `adj_train`/`adj_inference`.

`data_prep/verify_edge_direction_consistency.py` viết lại để gọi THẲNG hàm
`_neighbors()` thật trên ma trận đã transpose, đối chiếu với `full_graph_forward`
— **PASS 100/100 cả train và inference graph** với quy ước đã sửa đúng.
`data/graph_data.meta.json` cập nhật lại lý do đầy đủ (field
`edge_direction_convention`, có `revision_history` ngầm qua `why_this_direction`).

**HỆ QUẢ NGHIÊM TRỌNG HƠN LẦN SỬA TRƯỚC**: lần sửa này thay đổi **CHÍNH VIỆC
SAMPLE LÚC TRAIN** (không chỉ eval) — `sample_union_subgraph` giờ lấy đúng
payer thay vì payee cho MỌI lần gọi, kể cả trong vòng lặp train thật. Nghĩa là
**MỌI checkpoint đã train trước ngày 2026-09-06** (GraphMAE pretrain,
GraphSAGE trong E2/E2v2/E2v3, graph-only hypothesis, sweep, ablation) đều học
theo quy ước payee CŨ — **không còn khớp với quy ước payer hiện tại, cần TRAIN
LẠI TỪ ĐẦU mới có kết quả đáng tin** dưới quy ước đã sửa đúng. Việc chạy lại
`eval_full_test.py` (chỉ eval, không train lại) đang chạy ở mục trên vì vậy
**không còn cần thiết nữa** — đã dừng — vì checkpoint đó vẫn học theo quy ước
payee, eval với graph payer-convention (dù đã đúng chuẩn) vẫn cho kết quả
KHÔNG Ý NGHĨA (model chưa từng thấy payer-relationship lúc train).

**Trạng thái hiện tại — DỪNG lại chờ xác nhận (theo đúng yêu cầu "chỉ sau khi
kiểm chứng thành công mới tiến hành các bước sửa tiếp theo")**: code đã sửa
đúng + verify PASS. CHƯA train lại bất kỳ checkpoint nào. Toàn bộ kết luận
"fusion thua GBM 8×, signal-limited" từ các mục trước đó trong file này giờ
**không còn cơ sở đáng tin** (dựa trên checkpoint train sai quy ước) — cần
train lại ít nhất 1 biến thể (khuyến nghị: GraphMAE pretrain trước — rẻ, không
cần nhãn — rồi warm-start E2 v3 kiểu cũ, hoặc train thẳng E2 v2 lại từ đầu) để
có con số đáng tin dưới quy ước đã sửa, trước khi quyết định bước tiếp theo.

## Graph-only baseline train lại SAU KHI sửa Direction — xác nhận bug có tác động thật (2026-09-06)

Theo yêu cầu: train lại `train_graph_only_hypothesis.py` (GraphSAGE tự học, KHÔNG
BERT/fusion/GraphMAE) từ đầu dưới quy ước hướng cạnh đã sửa, để biết trần năng
lực thật của riêng nhánh graph trước khi đưa BERT/fusion vào lại. Bổ sung
`p_at_1000`/`recall_at_1000` vào `train_eval/metrics.py::compute_metrics`
(tham số `k_list`, mặc định `None` — không đổi hành vi các nơi gọi cũ).

**Kết quả sơ bộ (LR mặc định 1e-3, hidden=out=128, Linear, 30 epoch, 54.9 giây,
CHƯA sweep)** — so với thí nghiệm gốc (hướng cạnh SAI, trước fix):

| | pure_test AUPRC | pure_test F1 | ROC-AUC |
|---|---|---|---|
| Trước fix (hướng cạnh sai, mục "Thí nghiệm tối giản") | 0.013–0.020 | 0.01–0.06 | ~0.5–0.6 |
| **Sau fix (LR mặc định, chưa tune)** | **0.099** | 0.075 | **0.979** |

**Tăng ~5–7 lần chỉ với việc sửa đúng hướng cạnh, LR còn chưa tune** — xác nhận
mạnh mẽ bug hướng cạnh là nguyên nhân thật, đáng kể, không chỉ là chi tiết kỹ
thuật. Đang chạy sweep đầy đủ để tìm LR/cấu hình tốt nhất dưới quy ước đã sửa
(theo yêu cầu "sửa Direction + LR" — không giả định LR cũ 1e-3 vẫn tối ưu vì
hướng aggregate đã đổi hoàn toàn từ payee sang payer).

**Sweep v2** (`train_eval/sweep_graph_only_v2.py`): mở rộng lưới so với sweep
gốc (12 config, đã lỗi thời do bug hướng cạnh) — `learning_rate`×{3e-4, 1e-3,
3e-3, 1e-2} × `(hidden,out)`×{(64,64),(128,128),(256,128)} × `mlp_classifier`
×{có,không} = **24 config**, cùng kỷ luật chọn theo val AUPRC (không nhìn
pure_test lúc chọn). Kết quả lưu
`output/graph_only_sweep_v2_fixed_direction_results.json` (giữ nguyên
`output/graph_only_sweep_results.json` cũ để đối chiếu, không ghi đè).

### KẾT QUẢ SWEEP V2 — ĐẢO NGƯỢC HOÀN TOÀN KẾT LUẬN TRƯỚC (đã chạy xong, 2026-09-06)

**Cấu hình tốt nhất theo val AUPRC**: `lr=0.003, hidden=64, out=64,
mlp_classifier=True` (dừng ở epoch 144/150, 125.5 giây train, val AUPRC=0.3118).

**GraphSAGE THUẦN (không BERT/fusion/GraphMAE) trên `pure_test`** (609,773
account thật, 312 dương, bài test inductive nghiêm ngặt nhất — 0% cạnh
train-time):

| Metric | Giá trị |
|---|---|
| AUPRC | **0.3025** |
| ROC-AUC | **0.9836** |
| F1(pos) @ threshold 0.5 | 0.0861 |
| Precision @ 0.5 | 0.0453 |
| Recall @ 0.5 | 0.8846 |
| P@1000 | **0.193** |
| Recall@1000 | **0.619** |

(val: AUPRC=0.312, ROC-AUC=0.995, Recall@1000=0.821 | overlap: AUPRC=0.170,
ROC-AUC=0.973, Recall@1000=0.608 — đầy đủ 3 split trong
`output/graph_only_sweep_v2_fixed_direction_results.json`)

**So với thí nghiệm gốc (hướng cạnh sai)**: AUPRC 0.013–0.020 → **0.303**,
tăng **~15–23 lần**. Xu hướng learning_rate rõ trong sweep: LR quá nhỏ (3e-4)
hoặc quá lớn (1e-2) đều kém hẳn — 3e-3 là điểm rơi tốt nhất (khác hẳn LR=1e-3
từng "tốt nhất" ở sweep gốc bị bug — xác nhận đúng lo ngại "không thể giả định
LR cũ còn tối ưu khi hướng aggregate đổi hoàn toàn").

**So sánh trực tiếp với GBM (baseline tốt nhất trước đó)**:

| Metric | GraphSAGE thuần (đã sửa hướng) | GBM (đã tune) |
|---|---|---|
| AUPRC | 0.303 | **0.347** |
| ROC-AUC | **0.984** | 0.959 |
| P@1000 | **0.193** | 0.186 |
| Recall@1000 | **0.619** | 0.596 |

**GraphSAGE thuần giờ NGANG NGỬA hoặc VƯỢT GBM** trên ROC-AUC/P@1000/Recall@1000,
chỉ thua nhẹ về AUPRC (0.303 vs 0.347, khoảng cách ~13%, không còn là 8× như
trước).

**Kết luận phải đảo ngược hoàn toàn**: mục "Tổng kết chuỗi chẩn đoán E2 v2 →
GBM" và mọi mục liên quan ở TRÊN (kết luận "signal-limited, kiến trúc graph
kém hơn hẳn tabular") **dựa trên checkpoint bị bug hướng cạnh — KHÔNG còn
đúng**. Bug hướng cạnh (aggregate từ payee thay vì payer) đã che giấu tín hiệu
graph THẬT SỰ MẠNH suốt toàn bộ investigate trước đó — không phải "kiến trúc
GraphSAGE+BERT tối ưu hoá kém hơn boosted trees trên đặc trưng có sẵn" như kết
luận cũ, mà đơn giản là graph encoder đã học sai chiều thông tin từ đầu.

**Việc cần làm tiếp**: với tín hiệu graph THUẦN đã mạnh gần bằng GBM, BERT/fusion
(giờ cũng cần train lại dưới quy ước đã sửa) có cơ hội thật sự vượt qua GBM —
khác hẳn nhận định "không đáng đầu tư" trước đó. Ưu tiên: (1) train lại E2 v2
(BERT+GraphSAGE) từ đầu dưới quy ước đã sửa, (2) train lại GraphMAE pretrain
rồi warm-start E2 v3, (3) so sánh cả hai với GBM và với GraphSAGE-thuần-đã-sửa
(0.303) làm mốc tham chiếu mới — TOÀN BỘ cần chạy lại, không checkpoint nào từ
trước 2026-09-06 còn dùng được trực tiếp.

## Đề xuất split mới cho MulDiGraph — `Dataset/temporal_pyg.py` (2026-09-07)

Theo yêu cầu: đọc lại cách split CŨ (`tổng_hợp_markdown/preprocessing_and_eval_report.md`)
rồi đổi sang cách split MỚI trong `Dataset/temporal_pyg.py` — đã implement
`load_raw_data()` (trước đó chỉ là khung `NotImplementedError`) với dữ liệu
THẬT, dùng ĐÚNG index space với `address_to_index.pkl`/`node_features_all23.pt`
của pipeline `inductive_model` hiện có (để 2 cách split đối chiếu trực tiếp
được). Đã chạy thật, không phải mô phỏng.

**Split CŨ (đang dùng cho toàn bộ `data/preprocessed/Dataset_MG/`)**: **1 mốc
cắt toàn cục** `T_cutoff` = phân vị 80% trên TOÀN BỘ 13,551,303 timestamp giao
dịch (không riêng phishing) = 2018-06-03. Phân loại theo `t_first`/`t_last` mỗi
account so với mốc này → train/val/overlap/pure_test = **519/117/217/312**
(44.5%/10.0%/18.6%/26.8% của 1,165 phisher) — tỉ lệ này là HỆ QUẢ của phân bố
thời gian, không phải thiết kế chủ đích.

**Split MỚI (`temporal_pyg.py`)**: **2 mốc cắt theo QUANTILE trên chính t_first
của 1,165 phisher** (q=0.65, q=0.80) — do STATUS.md/artifact histogram trước đó
đã phát hiện phân bố phisher lệch cực mạnh (67.5% rơi vào H1/2018, đỉnh T5/2018)
nên cắt theo % lịch thông thường sẽ làm méo tỉ lệ dương giữa các split. Đồ thị
theo kiểu **"growing graph" tích lũy** (val chứa toàn bộ train + phần mới; test
chứa toàn bộ trước đó + mới nhất) — 3 subgraph RIÊNG BIỆT, không phải 1 đồ thị
dùng chung như split cũ. Nhãn chỉ gán cho node MỚI xuất hiện đúng cửa sổ của
split đó; âm lấy mẫu (tỉ lệ 1:10) cũng CHỈ trong đúng cửa sổ thời gian đó
(tránh âm từ tương lai/quá khứ xa).

**Kết quả chạy thật** (`python3 Dataset/temporal_pyg.py`, 19.2 giây cho
`load_raw_data()`, tổng cộng dưới 30 giây):

| Split | Mốc cắt | Dương mới | Âm mới (1:10) | Node subgraph (tích lũy) | Cạnh (tích lũy) |
|---|---|---|---|---|---|
| train | ≤ 2018-05-18 11:57 UTC | 757 (**65.0%**) | 7,570 | 2,217,282 | 10,238,978 |
| val | → 2018-06-18 23:35 UTC | 175 (**15.0%**) | 1,750 | 2,485,634 | 11,413,288 |
| test | → hết dữ liệu | 233 (**20.0%**) | 2,330 | 2,973,489 (toàn bộ) | 13,551,296* |

(*13,551,296 vs tổng thật 13,551,303 cạnh — lệch 7 cạnh, có thể do vài cạnh có
timestamp nhỉnh hơn `max(node_first_time)` một chút; chưa điều tra sâu, sai số
0.00005%, không đáng kể.)

Tổng dương = 757+175+233 = **1,165** ✓ (không mất account nào), đúng CHÍNH XÁC
tỉ lệ 65/15/20 thiết kế — khác hẳn split cũ vốn không nhắm tỉ lệ cụ thể nào.

**Khác biệt cốt lõi so với split cũ, cần cân nhắc trước khi thay thế**:
1. Split cũ dùng **1 đồ thị suy luận chung** (`adj_inference.npz`) cho mọi split,
   chỉ khác nhãn/partition gán trên cùng tập node — kiến trúc `inductive_model`
   hiện tại (`LabelAwareNeighborSampler`, `full_graph_forward`...) được thiết kế
   xung quanh giả định NÀY. Split mới dùng **3 đồ thị con tích lũy KHÁC NHAU**
   (số node/cạnh khác nhau ở mỗi split) — cần viết lại đáng kể phần data-loading
   của `inductive_model` (không chỉ đổi file `labels.pkl`/`partition.pkl`) nếu
   muốn áp dụng đầy đủ tinh thần "growing graph" thay vì chỉ lấy lại ngưỡng cắt.
2. Âm lấy mẫu THEO CỬA SỔ THỜI GIAN (mới) khác hẳn lấy mẫu TOÀN BỘ partition
   (cũ) — ảnh hưởng trực tiếp đến những gì `train_e2*.py`/`train_graph_only_hypothesis.py`
   coi là "âm train" (hiện đang dùng toàn bộ ~1.9M âm của partition `train` cũ).
3. Đây là **artifact riêng, độc lập**, CHƯA ghi đè `data/preprocessed/Dataset_MG/`
   — mọi pipeline/checkpoint hiện có (kể cả GraphSAGE thuần vừa sửa hướng cạnh,
   AUPRC=0.303) vẫn dùng split cũ, không bị ảnh hưởng bởi thay đổi này.

**Việc cần làm tiếp (chưa làm, chờ xác nhận hướng)**: quyết định xem có muốn
(a) chỉ lấy 2 mốc cắt quantile mới áp vào ĐÚNG kiến trúc dữ liệu cũ (1 đồ thị
suy luận chung, chỉ đổi ngưỡng phân loại) — thay đổi nhỏ, tương thích ngược, hay
(b) chuyển hẳn sang kiến trúc "growing graph" 3-subgraph của `temporal_pyg.py`
— thay đổi lớn, cần viết lại data-loading, nhưng đúng tinh thần thiết kế gốc
của file này hơn. Chưa train lại gì với split mới này.

### Đã chọn (a): áp 2 mốc cắt quantile vào kiến trúc dữ liệu CŨ — hoàn tất, verify PASS (2026-09-07)

Viết `Dataset/mg_temporal_pipeline_v2_quantile.py` — **giữ nguyên logic phân
loại của `mg_temporal_pipeline.py` gốc** (train/val/overlap/pure_test theo
`(t_first, t_last)` của từng account so với 2 mốc cắt, 1 đồ thị suy luận
chung `adj_inference.npz`) — **chỉ thay cách tính mốc cắt**:

| | Bản gốc | Bản v2 (mới) |
|---|---|---|
| Mốc `T_cutoff` (biên pure_train_candidates/overlap/pure_test) | percentile 80% trên TOÀN BỘ 13,551,303 giao dịch | = **t2** = quantile 0.80 trên t_first của 1,165 phisher = **2018-06-18** |
| Mốc train/val | "latest 10% của pure_train_candidates theo t_last" | = **t1** = quantile 0.65 trên t_first của 1,165 phisher = **2018-05-18** |

Tái sử dụng `labels.pkl`/`t_first.pkl`/`t_last.pkl`/`address_to_index.pkl` đã
có sẵn (không đọc lại `MulDiGraph.pkl` cho bước này) + build lại
`adj_train.npz` với `T_cutoff=t2` mới (51.5 giây) — `adj_inference.npz` không
đổi (toàn bộ giao dịch, không phụ thuộc T_cutoff) nên copy thẳng từ bản gốc.

**Kết quả thật, đầy đủ 7 kiểm định PASS** (`data/preprocessed/Dataset_MG_v2_quantile/`):

| Split | Bản gốc (n, %/1165) | Bản v2 (n, %/1165) |
|---|---|---|
| train | 519 (44.5%) | **558 (47.9%)** |
| val | 117 (10.0%) | **180 (15.5%)** |
| overlap | 217 (18.6%) | **194 (16.7%)** |
| pure_test | 312 (26.8%) | **233 (20.0%)** — khớp CHÍNH XÁC mục tiêu thiết kế 20% |
| `adj_train.npz` nnz | 4,162,599 | **4,430,051** (T_cutoff muộn hơn 15 ngày) |

**Verify invariant nghiêm ngặt nhất** (đã làm lại đúng kỷ luật xuyên suốt dự
án): `pure_test` (487,855 account) có bậc = 0 TUYỆT ĐỐI trong `adj_train.npz`
mới — **PASS 487,855/487,855**. 7/7 kiểm định khác (boundary consistency,
disjoint, phishing coverage, phân bố không bị cân bằng nhân tạo...) đều PASS,
xem `data/preprocessed/Dataset_MG_v2_quantile/split_config.json`/`split_stats.json`.

**Lưu ý về khác biệt val/overlap so với thiết kế 3-way của `temporal_pyg.py`**:
vì giữ logic `(t_first, t_last)` đầy đủ của kiến trúc cũ (rigorous hơn thiết
kế đơn giản chỉ dùng `t_first` của `temporal_pyg.py`), val/overlap ở đây
KHÔNG khớp tỉ lệ 15%/0% mà `temporal_pyg.py` báo cáo riêng lẻ — một phần
account "sinh ra" trong cửa sổ [t1,t2) nhưng còn hoạt động QUA t2 bị xếp vào
`overlap` (194) thay vì `val`, đúng tinh thần chống leakage nghiêm ngặt hơn
của kiến trúc cũ. Chỉ số quan trọng nhất — `pure_test` = 20.0% — khớp tuyệt
đối với mục tiêu.

**Trạng thái**: artifact ĐỘC LẬP tại `data/preprocessed/Dataset_MG_v2_quantile/`,
**CHƯA thay thế** `data/preprocessed/Dataset_MG/` gốc — mọi checkpoint/kết quả
hiện có (GraphSAGE thuần đã sửa hướng cạnh AUPRC=0.303, GBM baselines...) vẫn
dùng split gốc, không bị ảnh hưởng. Muốn dùng split v2 để train, các script
`train_e2*.py`/`train_graph_only_hypothesis.py`/`data_prep/labels_io.py` cần
trỏ `PREPROC_DIR` sang thư mục mới này (chưa làm — chờ xác nhận có muốn
chính thức chuyển sang split này hay chỉ dùng để đối chiếu).

### SỬA LẠI (theo yêu cầu): bỏ hẳn "overlap" — 3-way split thuần theo t_first, đúng `temporal_pyg.py` (2026-09-07)

v2 ở trên (giữ `(t_first, t_last)` straddle-check của kiến trúc cũ) vẫn còn
category `overlap` — **không đúng ý muốn**: chỉ cần 3 nhóm train/val/test,
phân loại THUẦN theo `t_first` (bỏ hẳn kiểm tra `t_last`), khớp CHÍNH XÁC logic
`build_splits()` trong `Dataset/temporal_pyg.py`. Viết
`Dataset/mg_temporal_pipeline_v3_no_overlap.py` (thay thế v2, giữ file v2 lại
để đối chiếu lịch sử, không xoá).

**Quy tắc**: `train`: `t_first < t1` | `val`: `t1 <= t_first < t2` | `test`:
`t_first >= t2` (t1/t2 giống hệt v2: quantile 0.65/0.80 trên t_first của 1,165
phisher = 2018-05-18 / 2018-06-18). **`adj_train.npz` build lại với
`T_cutoff = t1`** (khác v2 dùng t2) — khớp đúng định nghĩa "train window" của
`temporal_pyg.py` (train chỉ thấy cạnh đến t1, không đến t2).

**Kết quả thật, 8/8 kiểm định PASS** (`data/preprocessed/Dataset_MG_v3_no_overlap/`):

| Split | n account | Dương | %/1,165 |
|---|---|---|---|
| train | 2,217,282 | 757 | **65.0%** |
| val | 268,352 | 175 | **15.0%** |
| test | 487,855 | 233 | **20.0%** |

Khớp **CHÍNH XÁC** tỉ lệ thiết kế 65/15/20 (không còn lệch do overlap như v2).
`adj_train.npz` (v3): nnz=3,859,882 (T_cutoff=t1, sớm hơn t2 nên ít cạnh hơn
v2's 4,430,051). **Invariant nghiêm ngặt nhất verify lại PASS**: `test`
(487,855 account) có bậc = 0 TUYỆT ĐỐI trong `adj_train.npz` mới — 0/487,855.
Các boundary/disjoint/coverage/phân-bố-không-nhân-tạo khác đều PASS.

**Lưu ý khác biệt cần nhớ khi dùng split này**: `val` (268,352 account, 175
dương) giờ **cũng có bậc = 0 trong adj_train** giống `test` (vì `t_first>=t1`
đồng nghĩa chưa có cạnh nào trước t1) — tức **val giờ mang tính "inductive"
giống test**, không còn là "in-distribution, đã thấy 1 phần cấu trúc" như
`val` của kiến trúc cũ. Đây là hệ quả trực tiếp, có chủ đích, của việc bỏ
`t_last`/overlap — cần nhớ khi diễn giải kết quả val lúc train (val giờ đóng
vai trò "test sớm hơn", không phải "validation nhẹ nhàng" nữa).

**Trạng thái**: artifact tại `data/preprocessed/Dataset_MG_v3_no_overlap/`,
**thay thế v2 làm phiên bản split-mới chính thức** (v2 giữ lại chỉ để đối
chiếu lịch sử) — vẫn **CHƯA thay thế** `data/preprocessed/Dataset_MG/` gốc,
chưa train lại gì với split này. Muốn dùng cần trỏ `PREPROC_DIR` trong
`data_prep/labels_io.py` sang thư mục v3 này, và các nơi có check cứng
`partition=='overlap'`/`partition=='pure_test'` (vd `eval_full_test.py`,
`train_graph_only_hypothesis.py`) cần đổi tên thành `'test'` cho khớp.

### Đã wire split v3 vào `train_graph_only_hypothesis.py` — KHÔNG đổi `PREPROC_DIR` toàn cục (2026-09-07)

**Quyết định quan trọng, khác với đề xuất ban đầu**: KHÔNG sửa
`data_prep/labels_io.py::PREPROC_DIR` trực tiếp — vì hằng số này được import
TOÀN CỤC bởi mọi script fusion (`train_e2*.py`, `pretrain_graph_encoder.py`,
mọi `smoke_test_*.py`). Đổi thẳng sẽ khiến nhánh **graph** của các script đó
âm thầm chuyển sang split v3 trong khi nhánh **text** (`e2_train_corpus.py`,
`full_test_corpus.py`) vẫn dùng corpus BERT đã build sẵn cho split CŨ — 2
nhánh học 2 tập account khác nhau mà không có cảnh báo gì, một lỗi ngầm mới.
Việc phát hiện ra rủi ro này khi triển khai, không phải đoán trước.

**Giải pháp an toàn**: thêm tham số `--split-dir` **CHỈ RIÊNG**
`train_eval/train_graph_only_hypothesis.py` (mặc định `None` = hành vi CŨ,
100% tương thích ngược — đã smoke-test xác nhận không đổi) — vì nhánh
graph-only KHÔNG phụ thuộc corpus BERT nào, tự nó luôn nhất quán bất kể trỏ
tới split nào. Thêm `_load_labels_partition_from_dir()` (đọc trực tiếp
`labels.pkl`/`partition.pkl`/`address_to_index.pkl` từ 1 thư mục bất kỳ) và
tổng quát hoá `build_partition_indices()`/`evaluate()`/`naive_baselines()` để
**tự phát hiện tên các nhóm eval** từ chính `partition.pkl` (không hard-code
`"overlap"`/`"pure_test"` nữa) — dùng được với CẢ split cũ (4 nhóm) lẫn v3 (3
nhóm, không overlap) bằng cùng 1 file code.

**Đã smoke-test cả 2 chế độ, đều chạy đúng**:
- Mặc định (không `--split-dir`): vẫn ra đúng `val`/`overlap`/`pure_test` như
  trước — xác nhận không phá vỡ hành vi cũ.
- `--split-dir .../Dataset_MG_v3_no_overlap`: ra đúng `val`/`test` (không còn
  `overlap`), `test` n=487,855 — khớp đúng số liệu đã build.

**`eval_full_test.py` — KHÔNG đổi, cần nói rõ lý do (tránh hiểu lầm)**: khác
`train_graph_only_hypothesis.py`, script này KHÔNG đọc `partition.pkl` trực
tiếp — nó đọc từ `data_prep/full_test_corpus.py`, vốn nạp 1 corpus BERT ĐÃ
TOKENIZE SẴN (`runs/expanded_test_attempt3/full_test_corpus/`) được build một
lần cho ĐÚNG tập 811,704 account của split CŨ (609,773 pure_test + 201,931
overlap), với nhãn `.split` gán CỨNG lúc build corpus đó, không lấy từ
`partition.pkl` hiện tại. **Nếu chỉ đổi tên chuỗi filter `'pure_test'`/`'overlap'`
thành `'test'` mà không build lại corpus, script sẽ chạy nhưng tìm thấy 0 mẫu
khớp `split=='test'` — lỗi câm, không phải cảnh báo rõ ràng** — vì vậy đã
CHỦ ĐỘNG KHÔNG sửa file này. Muốn `eval_full_test.py` dùng được split v3 cần
**build lại corpus BERT cho đúng 487,855 account `test` mới** (dựng câu giao
dịch + tokenize — việc tốn thời gian hơn nhiều, tương tự quy mô đã làm cho
`runs/expanded_test_attempt3/full_test_corpus/` ban đầu) — đây là việc RIÊNG,
lớn hơn, chưa làm, cần xác nhận trước khi bắt tay vào.

## Giai đoạn F — Quyết định scale

## Giai đoạn F — Quyết định scale
- [ ] F1, F2, F3 — TẠM DỪNG, chờ train lại E2 v2/GraphMAE dưới quy ước hướng cạnh đã sửa trước khi quyết định (xem mục "KẾT QUẢ SWEEP V2" ở trên — kết luận "không scale fusion" trước đó không còn cơ sở)

## Giai đoạn G — Dọn dẹp & tài liệu
- [ ] G1, G2, G3 — có thể làm song song
