# Rà soát pipeline Dynamic_Fusion vs. pipeline mong muốn (code_check.md)

> Review dựa trên mã nguồn thực tế trong `Dynamic_Fusion/Dataset` (các script `mg_*.py`, `tri_model/`, `bi_model/`),
> đối chiếu với lịch sử audit trong `tong_hop_quy_trinh.md`, so với pipeline mong muốn mô tả trong `code_check.md`.
> Format theo đúng yêu cầu A–H của `code_check.md`.

---

## A. PIPELINE HIỆN TẠI CỦA CODE

Có **hai lineage** trong repo, đừng nhầm:

- **Lineage cũ (B4E, đã bị thay thế)**: mô tả trong `Dynamic_Fusion/CONTEXT_SUMMARY.md` và `Dynamic_Fusion/README.md` —
  `dataset0.py…dataset11.py`, `adjust_matrix.py`, `shared_sampling.py`. Các file `dataset*.py`/`shared_sampling.py`
  **không còn tồn tại** trong thư mục, chỉ còn `adjust_matrix.py` làm chứng tích — dựng adjacency **dense N×N trên
  toàn bộ dataset trước khi split** (bất khả thi + rò rỉ, đã bị `tong_hop_quy_trinh.md` mục 1 xác nhận và khai tử).
  Hai tài liệu này **chưa được cập nhật** để phản ánh quyết định "Path B" — dễ gây hiểu nhầm cho người đọc sau.
- **Lineage hiện hành (MulDiGraph, dùng cho mọi số liệu đã công bố)**: `mg_temporal_pipeline.py` →
  `mg_propagate_labels.py` → `mg_build_adjacency.py` (+ `mg_graph_weight_formula.py`) → `mg_refit_features.py` /
  `fullscale_features/*.py` → `mg_build_examples.py` → `tri_model/train1.py` (hoặc `bi_model/train_origin.py`) →
  `mg_build_full_test_eval.py` → `tri_model/eval_full_test.py`.

Kiến trúc model thực tế (`tri_model/ETH_GBert.py`):

```
input_ids (text: "from: 0x.. to: 0x.. amount.. in_out.. n-gram..")
        │
        ├─► BERT word_embeddings ──────────────────► words_embeddings [B,L,H]
        │
        ├─► VocabGraphConvolution: H_vh = sparse_mm(adj_norm, W0_vh) [vocab,128]
        │        (MỘT hop toàn-đồ-thị trên TOÀN BỘ vocab, mỗi forward() gọi lại)
        │        gather theo gcn_vocab_ids của từng token → ghi đè vào
        │        `gcn_embedding_dim` vị trí cuối chuỗi token (ETH_GBert.py:366-371)
        │
        ├─► FeatureProjector(23 đặc trưng thủ công) → broadcast ra mọi vị trí token
        │
        └─► DynamicFusionLayer (4 gate) fuse 3 luồng NGAY TRONG embedding layer
                        │
                        ▼
                 BertEncoder (self-attention) → pooler → Linear classifier
                        │
                        ▼
                 P(fraud | account)
```

GCN không phải "graph G → GNN 2 lớp → h_GCN(x)" độc lập rồi fuse muộn — nó là **một bảng embedding transductive
per-account** (`W0_vh` [vocab_size,128], `ETH_GBert.py:55`) được làm mượt 1-hop qua sparse adjacency, tiêm trực tiếp
vào chuỗi token BERT trước self-attention.

---

## B. PIPELINE MONG MUỐN (từ code_check.md)

```
transactions → graph branch (G) → GNN/GCN → h_GCN(x)  ┐
            └→ account behavior branch → BERT → h_BERT(x) ┘→ Dynamic Fusion → Classifier → P(fraud|x0)
```

Hai nhánh độc lập, fuse sau khi đã có 2 embedding hoàn chỉnh; GNN dùng k-hop neighborhood cục bộ, không giả định
nhìn toàn đồ thị.

---

## C. DIFFERENCES

| # | Mong muốn | Code hiện tại |
|---|---|---|
| 1 | h_GCN(x) và h_BERT(x) tính độc lập rồi fuse | GCN được tiêm **vào bên trong** embedding layer của BERT trước attention (`ETH_GBert.py:361-382`) — hai luồng không độc lập, fusion xảy ra ở mức token, không ở mức account-embedding |
| 2 | GNN dùng k-hop neighborhood cục bộ | Một sparse `H_vh = adj_norm @ W0_vh` tính trên **toàn bộ vocab** mỗi lần forward (`ETH_GBert.py:98`), không sampling, không giới hạn k-hop tường minh (thực chất là 1-hop toàn cục do không stack layer) |
| 3 | Inductive/incremental cho x0 mới | Không có — `W0_vh` là bảng tra cứu cố định theo account cụ thể, transductive by construction |
| 4 | Inference cho 1 account, không rebuild toàn graph | Mọi pipeline eval (`mg_build_full_test_eval.py`, `eval_full_test.py`) build **toàn bộ tập test** cùng lúc, không có API `predict_one(x0)` |
| 5 | Feature account chỉ dùng info ≤ T | Đặc trưng đồ thị (23 cột) tính trên **toàn bộ timeline**, dùng chung cho mọi partition kể cả train (chi tiết ở D) |

---

## D. LEAKAGE

| Hạng mục (theo checklist §5 code_check.md) | Đánh giá | Vị trí |
|---|---|---|
| Test account vào training graph gây label leak | **[OK]** | `mg_build_examples.py:380-389` assert overlap/test không có trong `train_examples`; `mg_propagate_labels.py:177-179,210` assert `propagated_1hop ∩ (val∪test) = ∅` |
| Neighbor label bị dùng làm feature | **[OK]** | Không tìm thấy feature nào derive từ nhãn hàng xóm — 23 đặc trưng đều là thống kê giao dịch/centrality thuần |
| Graph dựng từ toàn bộ dataset trước split | **[OK]** (đã fix) / lịch sử **[CRITICAL]** ở `adjust_matrix.py` (dead code, vẫn còn trong repo) | `mg_build_adjacency.py:93-102` filter đúng `timestamp <= T_cutoff` cho train adjacency |
| Transaction của x0 sau prediction time vẫn dùng | **[HIGH]** | `mg_build_examples.py:284-286` và `mg_build_full_test_eval.py:183-184`: `build_account_transactions(..., ts_max=None)` cho val/overlap/pure_test — comment tự nhận "others use full history — transductive eval". Với account `overlap` (có giao dịch cả trước lẫn sau T_cutoff), văn bản BERT feed vào model chứa **toàn bộ lịch sử tương lai** của chính account đó, không chỉ phần ≤T |
| Aggregated features tính trên toàn timeline thay vì đến cutoff | **[HIGH]** | `fullscale_features/02_groups12_exact.py:72-91` có flag `--pre-cutoff` nhưng **optional, mặc định None**; artifact thực tế là `groups12_full.npz` (không có hậu tố `precut`) → 23 đặc trưng dùng chung một file `features_output_all23_MG_fullscale.csv` cho **mọi partition kể cả train** (`train1.py:367-370`). Khác với adjacency (có tách rõ `adj_train` vs `adj_inference` và được tài liệu hoá là "transductive có chủ đích"), việc feature tính trên toàn đồ thị **không được ghi nhận rõ** là một lựa chọn tương tự trong `tong_hop_quy_trinh.md` — đây là điểm mới, chưa từng bị flag trong 5 báo cáo trước |
| Normalization fit trên cả train+test | **[OK]** | `mg_refit_features.py:62-77`, `fullscale_features/08_assemble_csv.py:160-168` fit `mean/std` chỉ trên `mg_partition=='train'` |
| Node ID test tra cứu embedding đã train sẵn (nhầm lẫn) | **[OK]** cho case đã implement | `address_to_index`/`doc_accounts` được remap tường minh mỗi lần build corpus, không dựa vào vị trí ngầm định |
| Full-graph GCN inference cho test node access info từ future/test nodes | **[HIGH, có chủ đích]** | `mg_build_adjacency.py:104-116` + `train1.py:326`: `gcn_adj_list_eval` dùng **full graph (mọi edge, mọi thời điểm)** cho cả **validation lẫn test** — nghĩa là ngay cả khi hiệu chỉnh ngưỡng trên val, message-passing đã "nhìn thấy" cấu trúc đồ thị sau T_cutoff |

---

## E. INFERENCE x0

**Case 1 — x0 đã có trong `MulDiGraph.pkl` là 1 node**, chia làm 2 nhánh nhỏ:

- Nếu x0 nằm trong vocab lúc train (hiếm, vì corpus train chỉ subsample 20k/2.97M): tra cứu trực tiếp, có hàng
  `W0_vh` đã học.
- Nếu x0 là 1 trong 811,704 tài khoản test thật (chưa từng ở trong vocab train): phải chạy lại quy trình như
  `mg_build_full_test_eval.py` — mở rộng `address_to_index`, cắt lại `adj_inference.npz` theo CSR (rẻ, O(nnz)),
  rồi **rebuild toàn bộ model** ở vocab_size mới và partial-load checkpoint (`eval_full_test.py:255-272`, hàng
  `W0_vh` mới là random-init, không có self-loop nên tín hiệu GCN của x0 hoàn toàn phụ thuộc hàng xóm 1-hop đã
  học). Hiện **không có wrapper cho 1 account đơn lẻ** — script chỉ chạy batch cho toàn bộ 811,704 account cùng
  lúc.

**Case 2 — x0 hoàn toàn mới, địa chỉ chưa từng là node trong `MulDiGraph.pkl`**: **không có code path nào xử lý**.
`mg_temporal_pipeline.py`, `mg_build_adjacency.py`, `mg_build_examples.py`, `mg_build_full_test_eval.py` đều giả
định account đã tồn tại sẵn trong `G.nodes()` của pickle tĩnh. Không có logic nào build
`G' = G_historical + node(x0) + edges(x0, neighbors)` từ transaction thô của một account mới hoàn toàn — đây là
khoảng trống lớn nhất so với pipeline mong muốn.

---

## F. GRAPH CONSTRUCTION

- **Có thể reuse**: `adj_inference.npz` (sparse, tĩnh, build 1 lần) — cắt hàng/cột theo CSR cho 1 account thêm
  vào là rẻ, không cần rebuild toàn ma trận.
- **Không thể reuse / chưa tồn tại**: (1) không có cơ chế thêm 1 node + cạnh mới vào `adj_inference` on-the-fly
  cho account thực sự mới (Case B); (2) `W0_vh` là `nn.Parameter` kích thước cố định theo vocab — mỗi lần vocab
  đổi (dù chỉ +1 account) phải **tái tạo toàn bộ model object** và partial-load lại checkpoint, không có API
  "thêm 1 hàng embedding" nhẹ nhàng.

---

## G. SCALE

- Không có dense N×N nào ở pipeline hiện hành (đã xác nhận qua `tong_hop_quy_trinh.md` mục 4 và code: toàn bộ
  dùng `scipy.sparse`/`torch.sparse`). Tốt.
- GCN full-batch **khả thi về mặt sparse-matmul** (nnz≈13.5M cạnh) nhưng **triển khai lãng phí**:
  `H_vh = torch.sparse.mm(adj, W_i)` nằm trong `VocabGraphConvolution.forward()` (`ETH_GBert.py:98`) nên bị
  **tính lại từ đầu ở MỌI batch**, kể cả khi eval (`model.eval()` + `torch.no_grad()`, trọng số và adjacency
  không đổi trong suốt pass). Với batch_size=8 trên 811,704 account (~101,463 batch), đây là ~101,463 lần lặp
  lại đúng một phép tính bất biến — khớp với con số thực đo "213.4 phút" ở `full_scale_test_eval_report.md`. Đây
  là bottleneck tối ưu hoá cụ thể nhất tìm được, dễ sửa, tác động lớn.
- Không có neighbor sampling/GraphSAGE — đúng như tài liệu ghi nhận, đây là lựa chọn kiến trúc có chủ đích
  (1-hop transductive), nhưng cũng là lý do không thể inference "local subgraph" nhẹ cho 1 x0 như pipeline mong
  muốn.

---

## H. CODE CHANGES (patch tối thiểu, ưu tiên)

1. **[ĐÃ SỬA] Cache `H_vh` trong eval** — `VocabGraphConvolution` (`ETH_GBert.py`) có thêm `compute_H_vh()` +
   `forward(..., precomputed_H_vh=None)`; `ETH_GBertEmbeddings.forward` và `ETH_GBertModel.forward` thread tham
   số này xuống, và `ETH_GBertModel.compute_gcn_H_vh()` là tiện ích gọi từ ngoài. `train1.py`'s `evaluate()` và
   `eval_full_test.py`'s `evaluate()` giờ tính `precomputed_H_vh` một lần ngay sau `model.eval()` / trước vòng
   lặp batch, thay vì để mỗi batch tự gọi `sparse.mm(adj, W)` lại. Đã verify bằng test số học: output với cache
   khớp bit-for-bit với output không cache (`max abs diff = 0.0`, `dropout_rate=0.0`). Lưu ý: vòng lặp TRAIN
   (không phải `evaluate()`) vẫn tính lại mỗi batch vì `W0_vh` đổi mỗi optimizer step — cache chỉ áp dụng đúng
   ở nhánh eval/no_grad.
2. **Viết API inference 1 account**: hàm `predict_account(addr, ...)` tái dùng logic của
   `mg_build_full_test_eval.py` nhưng tham số hoá cho N=1 thay vì cả tập test — extend vocab +1, cắt CSR, rebuild
   model 1 lần (không phải rebuild-batch-cho-cả-811k mỗi lần muốn xem 1 account).
3. **Xử lý Case B tường minh**: cần một bước tiền xử lý nhận transaction thô của x0 chưa có trong
   `MulDiGraph.pkl`, build node+edges tạm thời nối vào bản sao/slice của `adj_inference`, gắn nhãn "no learned
   W0_vh row" rõ ràng trong output (thay vì im lặng dùng random-init) để người dùng model biết độ tin cậy GCN
   cho case này gần như bằng 0 nếu hàng xóm cũng đều là node mới.
4. **[ĐÃ SỬA — hướng (b)] Đối xứng hoá "transductive" giữa adjacency và feature**: không chạy lại
   `--pre-cutoff` (option a) vì tốn ~22h trên toàn bộ 2.97M node và sẽ ghi đè file feature mà các checkpoint đã
   train phụ thuộc vào — quá rủi ro để tự ý chạy. Thay vào đó đã thêm cảnh báo tường minh, cùng style với
   `mg_build_adjacency.py:111-115`, ở 3 điểm: docstring `fullscale_features/02_groups12_exact.py` (xác nhận
   artifact production là `groups12_full.npz`, không có biến thể `precut`), comment tại `train1.py:367-370`
   (nơi load feature CSV cho train), và comment tại `eval_full_test.py`'s `FEATURE_SETS` (nơi load cùng CSV cho
   val/test). Option (a) — chạy lại `--pre-cutoff <T_cutoff>` để có bộ feature train-only-timeline thật sự —
   vẫn là việc còn mở, cần ~22h compute khi có nhu cầu.
5. **Dọn/đánh dấu dead code**: `adjust_matrix.py`, `CONTEXT_SUMMARY.md`, `README.md` phản ánh lineage B4E đã bị
   khai tử — nên gắn header "DEPRECATED, xem tong_hop_quy_trinh.md" để tránh người review sau nhầm đây là
   pipeline đang chạy.

**Phần chưa thể kết luận chắc chắn**: liệu file `features_output_all23_MG_fullscale.csv` thực tế có từng được
chạy lại với `--pre-cutoff` ở một lần build khác ngoài lần tạo ra `groups12_full.npz` hay không — tôi chỉ thấy
một bộ artifact không mang hậu tố cutoff trong `fullscale_features/arrays/` và `fullscale_features/groups12_full.npz`;
nếu có một bản build khác đã áp `--pre-cutoff`, cần chỉ rõ đường dẫn để xác nhận lại.
