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

## Giai đoạn F — Quyết định scale
- [ ] F1, F2, F3 — chờ E xong

## Giai đoạn G — Dọn dẹp & tài liệu
- [ ] G1, G2, G3 — có thể làm song song
