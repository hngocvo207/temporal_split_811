# Dynamic_Fusion — Tổng hợp toàn bộ quy trình (từ 5 báo cáo)

> Tài liệu này gộp nội dung của 5 báo cáo gốc thành một quy trình duy nhất, theo đúng trình tự thời gian/logic:
> `preprocessing_and_eval_report.md` (báo cáo trung tâm, nhiều lần cập nhật) · `fullscale_feature_extraction_report.md` ·
> `bi_modal_run_report.md` · `expanded_test_propagation_report.md` · `full_scale_test_eval_report.md`.

---

## 0. Bối cảnh chung

Dự án **Dynamic_Fusion** phát hiện gian lận phishing trên đồ thị giao dịch Ethereum **MulDiGraph**
(Chen et al. 2020, XBlock): **2,973,489 node / 13,551,303 cạnh / 1,165 tài khoản phishing xác thực
(`isp=1`)**. Mô hình chính (`tri_model`) là **ETH_GBert**: BERT (chuỗi giao dịch dạng văn bản) + GCN
(cấu trúc đồ thị) + top‑10 đặc trưng đồ thị thủ công + tầng "Dynamic Fusion". `bi_model` là biến thể
không có nhánh đặc trưng đồ thị (chỉ BERT+GCN).

Toàn bộ công việc bắt nguồn từ một **audit tính hợp lệ của pipeline đánh giá** (rò rỉ dữ liệu giữa
train/val/test, ngưỡng quyết định không hiệu chỉnh, hạ tầng không mở rộng được tới quy mô thật). Từ đó
quy trình được làm lại theo từng bước, ghi nhận đầy đủ trong 5 tài liệu trên.

---

## 1. Audit ban đầu & các lỗi được xác nhận (nền tảng cho mọi việc sau)

| Hạng mục | Vị trí | Kết luận |
|---|---|---|
| Resampling chạm cả val/test | `shared_sampling.py:62-68` (toàn bộ phisher + 2× normal, áp dụng **trước** khi chia tách) | Xác nhận rò rỉ, kế thừa sang val/test |
| Adjacency dựng trên toàn bộ dataset | `adjust_matrix.py:154-172` (`np.zeros((N,N))` đặc, không biên giới) | Rò rỉ + **bất khả thi ở quy mô MulDiGraph** (N×N đặc ở N=2.97M ≈ 35 TB) |
| Ngưỡng quyết định | hardcode argmax, không hiệu chỉnh | Sửa: hiệu chỉnh trên **validation only** |
| Artifact MulDiGraph trên đĩa | `data/preprocessed/Multigraph/*.pkl` | Phần lớn là con trỏ Git‑LFS/rỗng — không dùng, xây lại từ `MulDiGraph.pkl` gốc |

Kết luận: chọn **Path B — khôi phục & xây lại toàn bộ trên MulDiGraph thật**, không phải vá dữ liệu cũ.

---

## 2. Chia tách dữ liệu theo thời gian — T_cutoff 80th percentile (thay thế split 60/20/20 cũ)

Script: `mg_temporal_pipeline.py`, chạy trên **toàn đồ thị thật** (không mẫu).

- `T_cutoff` = **percentile thứ 80 của mọi timestamp cạnh** (ở mức giao dịch, không phải `T_first` từng tài khoản) → `T_cutoff = 1527988904` (2018‑06‑03 01:21:44 UTC), xác nhận đúng 80.0000%.
- Phân loại tài khoản theo `(t_first, t_last)`: `pure_train_candidates` (`t_last ≤ T_cutoff`), `overlap` (`t_first ≤ T_cutoff < t_last`), `pure_test` (`t_first > T_cutoff`).
- **Validation** = 10% của `pure_train_candidates` có `t_last` **gần cutoff nhất**; 90% còn lại → **train**.
- **Test** = `overlap ∪ pure_test` (giữ tách riêng để báo cáo theo từng lát cắt).

| Partition | Số tài khoản | Phishing | Tỉ lệ |
|---|---:|---:|---:|
| train | 1,945,607 | 519 | 0.0267% |
| val | 216,178 | 117 | 0.0541% |
| overlap | 201,931 | 217 | 0.1075% |
| pure_test | 609,773 | 312 | 0.0512% |
| **test (overlap∪pure_test)** | **811,704** | **529** | **0.0652%** |
| **Tổng** | **2,973,489** | **1,165** | — |

Mất cân bằng thật của test: **529/811,704 ≈ 1:1534**. Toàn bộ ràng buộc disjoint train/val/test được assert và **PASS**.

---

## 3. Lan truyền nhãn 1‑hop có kiểm soát (`mg_propagate_labels.py`) — chỉ dành cho train

Từ 1,165 seed xác thực, lan truyền **chỉ theo OUT‑edge** (tiền chảy ra khỏi kẻ lừa đảo đã xác nhận) tới hàng xóm 1‑hop, coi là "nghi ngờ mule/rửa tiền" — **soft label**, không phải ground truth.

Cơ chế an toàn:
- **Có hướng**: chỉ OUT‑edge lan truyền; IN‑edge (người gửi TỚI seed) gắn `is_victim=1`, không bao giờ dùng làm nhãn dương.
- **Chặn hub/sàn giao dịch**: loại node trên percentile‑99 độ (in+out) khỏi cả hai vai trò (nguồn & ứng viên).
- **Tách nhãn**: `label_source ∈ {ground_truth, propagated_1hop, benign}`; `isp` gốc **không bao giờ bị sửa**; `isp_augmented = isp OR propagated_1hop` chỉ dùng để train.
- **Không rò rỉ val/test**: `propagated_1hop` chỉ được gán trong `train`, assert `propagated_1hop ∩ (val∪test) = ∅` — **PASS**, kiểm tra lại độc lập ở bước build corpus.
- **Soft label**: trọng số mẫu `sample_weight=0.5` (tùy chỉnh), dùng làm hệ số nhân loss.

| Chỉ số | Giá trị |
|---|---:|
| Seed ground-truth | 1,165 |
| Ngưỡng hub (percentile 99) | độ 40.0 |
| Seed bị loại vì là hub | 236 |
| Ứng viên trước lọc hub | 1,743 |
| Ứng viên bị loại vì là hub | 424 |
| Ứng viên cuối (`propagated_1hop`, toàn đồ thị) | **1,319** |
| Bị loại vì không thuộc `train` | 800 |
| **`propagated_1hop` cuối cùng trong train** | **519** |
| Train dương: trước/sau tăng cường | 519 → 1,038 (2.0×) |

---

## 4. Sửa lỗi kiến trúc để chạy được ở quy mô đầy đủ — sparse gather/embedding‑lookup (§10)

**Vấn đề:** `CorpusDataset.pad()` dựng one‑hot đặc `gcn_swop_eye` (`[batch, vocab_size, seq_len]`), `VocabGraphConvolution.forward` nhân ma trận `X_dv.matmul(H_vh)` — về bản chất là **scatter**, lãng phí vì mỗi batch chỉ dùng ≤~416 dòng trong số 2,973,489 dòng vocab.

**Sửa:** viết lại thành **gather** toán học tương đương — với mỗi token, chỉ lấy đúng 1 dòng của `H_vh` theo `gcn_vocab_ids`, rồi rút gọn qua einsum nhỏ. Không đổi bất kỳ tham số học nào (`W0_vh`, `fc_hc`, `DynamicFusionLayer`, `FeatureProjector`, BERT, classifier).

**Kiểm chứng:**
- Sai số số học tối đa so với công thức one‑hot cũ: **2.4e‑7** (chỉ do làm tròn float32) — PASS.
- Loss bước đầu của smoke test giống hệt trước khi sửa — PASS.
- Benchmark bộ nhớ GPU thật (V=2,973,489, batch=8, seq=416): **7,362 MB** — so với one‑hot cũ cần **~39.6 GB** chỉ riêng tensor đầu (bản gốc của báo cáo ghi nhầm là "~39.6 TB", đã đính chính đơn vị).

→ **Rào cản bộ nhớ bị xóa bỏ**; rào cản còn lại là **thời gian tính toán** (không phụ thuộc vocab_size, phụ thuộc batch/seq).

---

## 5. Cải tổ quy trình huấn luyện — xử lý mất cân bằng, chọn F1(pos), optimizer/warmup (§11)

**Vấn đề phát hiện:** hai lần chạy corpus 20,000 lớn (dưới split T_cutoff mới) đều chọn **epoch 0** là "tốt nhất" và không cải thiện thêm. Nguyên nhân gốc: tiêu chí chọn mô hình (`perform_metrics`) là **weighted F1** trên tập val có tới ~97%+ là lớp đa số (benign) → thưởng cho việc phân loại đúng lớp dễ, không phản ánh chất lượng phát hiện phishing.

**4 thay đổi** (chỉ ở `train1.py`, không đổi kiến trúc mô hình):
1. `train_dataloader` chuyển sang **`WeightedRandomSampler`** (cân bằng trong từng batch).
2. Tiêu chí chọn checkpoint/early-stopping đổi sang **F1(pos)** (F1 riêng lớp dương), weighted F1 vẫn được báo cáo song song.
3. `evaluate()` in F1(pos) như số liệu đầu dòng, không còn ẩn trong `classification_report`.
4. LR mặc định nâng lên `2e‑5` (bỏ hardcode `8e‑6`); optimizer đổi từ `BertAdam` sang `Adam`+`CosineAnnealingLR`.

**Lỗi thứ hai bị bắt giữa chừng:** kết hợp `WeightedRandomSampler` + focal‑loss `alpha` theo tần suất lớp gốc (`[0.026, 0.974]`) chồng lấn hai cơ chế cân bằng → mô hình **sụp đổ, luôn dự đoán lớp dương**. **Sửa:** đặt `alpha` trung lập `[0.5, 0.5]` — chỉ giữ một cơ chế cân bằng (sampler).

### Bảng so sánh 4 lần thử (cùng corpus 20,000/5,000×3, cùng split T_cutoff)

| Cấu hình | Optimizer | LR | Warmup | Sampler | Focal α | Tiêu chí chọn | Best epoch | Test F1(pos) tổng |
|---|---|---:|---|---|---|---|---:|---:|
| Before (§9-scale) | BertAdam | 8e-6 | 0.1 (danh nghĩa) | shuffle thường | theo tần suất | weighted F1 | 0 | 0.5035 |
| Attempt 1 (lỗi) | Adam+Cosine | 2e-5 | không | weighted | theo tần suất | F1(pos) | 0 (kẹt, sụp đổ) | 0.1005 |
| Attempt 2 (đã sửa) | Adam+Cosine | 2e-5 | không | weighted | trung lập 0.5/0.5 | F1(pos) | 0 (vẫn kẹt) | 0.5104 |
| **Attempt 3 (thêm warmup)** | **BertAdam** | 2e-5 | **0.1** | weighted | trung lập | F1(pos) | **7** | **0.6227** |

**Kết luận then chốt:** nguyên nhân thật của việc "kẹt ở epoch 0" là **thiếu LR warmup** cho các tầng khởi tạo ngẫu nhiên (GCN/feature‑projector/fusion/classifier — chỉ BERT được pretrain), **không phải** do split, mất cân bằng, hay sampler/alpha. Khôi phục `BertAdam` (có warmup+decay tích hợp) là thay đổi **duy nhất** giữa Attempt 2 và 3, và phá vỡ mẫu hình kẹt lần đầu tiên.

**Attempt 3 — kết quả đầy đủ** (wandb `h3gxwf4k`, wall‑clock ~6.65h, ngưỡng hiệu chỉnh **0.8134**):

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean |
|---|---:|---:|---:|---:|---:|---:|
| pure_test | 5,000 | 312 | 0.7406 | 0.9701 | 0.8194 | 0.7927 |
| overlap | 5,000 | 217 | 0.5107 | 0.9510 | 0.6285 | 0.7946 |
| **overall** | 10,000 | 529 | **0.6227** | 0.9594 | 0.6904 | 0.7922 |

→ Đây là **số tốt nhất tại quy mô train=20,000** trong toàn bộ tài liệu, và là checkpoint chuẩn ("Attempt 3") dùng làm mốc so sánh cho mọi thí nghiệm sau này. `overlap` vẫn yếu hơn `pure_test` rõ rệt — nhất quán với giả thuyết mô hình gặp khó với tài khoản "biên giới" (không bao giờ có `InputExample` trong loss huấn luyện, theo thiết kế split).

---

## 6. Kiến trúc thay thế — bi_model (BERT+GCN, không có nhánh đặc trưng đồ thị)

Toàn bộ fix từ tri_model (sparse gather §10, protocol Attempt‑3 §11) được port sang `bi_model/{train_origin.py, ETH_GBert_origin.py, utils_origin.py}`. Trước đó phát hiện thêm 1 lỗi riêng: `ETH_GBertModel.forward()` **nhân toàn bộ adjacency với 0** (`vocab_adj_list = [adj*0 ...]`) — vô hiệu hóa hoàn toàn nhánh GCN mà không báo lỗi; đã sửa.

### So sánh bi_model vs tri_model, cùng split mới, cùng protocol Attempt‑3 (wandb `nq5fvb0y`)

| Slice | F1(pos) tri_model (Attempt 3) | F1(pos) bi_model |
|---|---:|---:|
| pure_test | 0.7406 | **0.8209** |
| overlap | **0.5107** | 0.5007 |
| overall | 0.6227 | **0.6476** |
| Best epoch | 7 | **1** |
| Wall clock | ~6.65h | **~3.59h** |

**Kết luận:** ở quy mô bounded (20,000 train), **bi_model (không có đặc trưng đồ thị) bằng hoặc tốt hơn tri_model trên mọi chỉ số chính**, hội tụ nhanh hơn (1 epoch so với 7) và chạy nhanh hơn ~gấp đôi. Đây là **2 điểm dữ liệu độc lập** (split cũ ở §7 của báo cáo bi_modal, và split mới ở đây) cho cùng kết luận — nhưng lý do được làm rõ ở mục 8 dưới: đặc trưng đồ thị lúc đó gần như **trơ** (chỉ phủ 0.2% dữ liệu), nên đây không phải bằng chứng "đặc trưng đồ thị vô ích" một cách tổng quát.

---

## 7. Trích xuất đặc trưng đồ thị full‑scale (2,973,489 node) & phát hiện đặc trưng đã "trơ" từ đầu

### 7.1 Tại sao cần làm lại

`select_add_features.py` gốc chỉ trích xuất được **5,655/2,973,489 node (0.19%)** — bị chặn bởi bộ nhớ (đồ thị networkx ~15 GB, không song song hóa được), chứ không phải chặn bởi khối lượng tính toán (~18 core‑ngày là khả thi).

### 7.2 Giải pháp kỹ thuật

Chuyển việc duyệt đồ thị sang `scipy.sparse` CSR (~200 MB, chia sẻ read‑only giữa các worker song song), chỉ đưa networkx phần đồ thị con nhỏ cần cho tính centrality:
- Nhóm đặc trưng 1+2 (18 đặc trưng cơ bản/thời gian): vector hóa hoàn toàn → **7 giây** cho toàn bộ 2.97M node.
- Nhóm 3 (8 centrality): giữ nguyên thuật toán networkx gốc trên đồ thị con, chạy 16 worker song song, theo *strided chunk* để cân bằng tải (vì node id gần như xếp theo bậc — decile đầu có bậc trung bình 20.7, decile cuối chỉ 1.02).
- Bỏ `eigenvector`/`katz`/`closeness` (46.8% chi phí tính toán, không nằm trong top‑10 hiện dùng) → còn **~23h** thay vì ~44h.

**2 lỗi thật được tìm và sửa trong lúc xây dựng:**
1. Tràn số `int8` khi cộng dồn CSR (cạnh song song lên tới 10,000 trên một cặp node) → xóa âm thầm một số cạnh thật. Sửa bằng `float64` + assertion.
2. Bản gốc dùng 2 quy ước "0" khác nhau (`0.0 or 1.0` → `1.0` cho centrality có trọng số, nhưng giữ `0.0` cho đặc trưng amount) — tái tạo trung thành, không "dọn dẹp" để không đổi kết quả.

**Độ trung thực:** nhóm 1+2 khớp chính xác (hoặc sai số 1e‑6 do làm tròn) với 5,655 dòng gốc; nhóm 3 khớp chính xác trừ `betweenness_centrality` (bản gốc dùng sampling không seed, không tự tái lập được chính nó — giá trị mới luôn nằm trong khoảng dao động của bản gốc).

### 7.3 Kết quả trích xuất

- Coverage: **0.19% → 100%** (2,973,489/2,973,489), sau **22.13h** trên 16 CPU worker.
- Scaler train‑only fit lại trên thật **1,945,607** dòng train (thay vì mẫu 0.2% cũ) — mẫu cũ đã **thổi phồng** `betweenness_centrality` khoảng **7×** (std 0.023 vs 0.0033 thật).

### 7.4 Ba khiếm khuyết nghiêm trọng được phát hiện

1. **Nhánh đặc trưng đồ thị đã trơ trong MỌI thí nghiệm ghi nhận trước đó.** `train1.py` đọc file CSV 5,655 dòng, fallback về vector‑0 cho phần còn thiếu → **99.81% tài khoản nhận vector toàn‑0**. Tệ hơn: vì CSV đó lấy từ mẫu 5:5 phisher/normal, chỉ riêng việc "có vector khác 0" đã là tín hiệu **202×** cho lớp dương (P(có đặc trưng|phishing)=35.5% vs P(có đặc trưng|benign)=0.18%). → **Mọi F1 ghi nhận ở §9/§11/§12/§17 (báo cáo chính) và trong `bi_modal_run_report.md` đều đo dưới điều kiện này**, không phản ánh một mô hình 3‑modal thực sự hoạt động.
2. **Top‑10 đặc trưng hiện dùng dựa trên một lựa chọn bị lỗi**: tính |Spearman| trên split cũ (38% dữ liệu "TRAIN" của nó thực ra thuộc val/overlap/pure_test theo split mới), ở tỉ lệ dương 16% (thật là 0.0267%, phóng đại ~600×), trên 0.23% dữ liệu train, và theo nhãn `phisher_accounts.txt` chứ không phải ground truth.
3. **Hai bộ nhãn mâu thuẫn nhau**: `phisher_accounts.txt` (5,480 trong đồ thị) vs `labels.pkl`/`isp` (1,165) chỉ trùng **963** địa chỉ (4,517 chỉ có trong .txt, 202 chỉ có trong isp) — mâu thuẫn hai chiều, không phải một bộ là tập con của bộ kia. `.txt` **không hề** ảnh hưởng tới nhãn huấn luyện mô hình (mô hình dùng `isp`), nhưng đã **định hướng sai** bước chọn đặc trưng ở mục (2). `.txt` bị **khai tử**, thay bằng `phisher_account_muldi.txt` xuất trực tiếp từ `isp`.

### 7.5 Baseline chỉ‑đặc‑trưng — đặc trưng có đáng giữ không?

`HistGradientBoosting` trên 23 đặc trưng, không BERT/GCN/fusion, **4 giây** để fit:

| | HistGBM (all‑23), full test 811,704 | Attempt 3 (BERT+GCN+fusion), full test |
|---|---:|---:|
| F1(pos) overall | **0.3479** | **0.0549** |
| AUPRC overall | 0.2558 | 0.1241 (§ mục 8 dưới) |

→ Trên **phân bố thật** (0.065% dương), một mô hình gradient‑boosting đơn giản trên đặc trưng đồ thị **vượt hẳn** (6.3×) mô hình Dynamic‑Fusion đã huấn luyện. Nhưng trên **test bounded 5.29% dương** thứ tự đảo ngược (HistGBM 0.4937 vs Attempt 3 0.6227) — đây chính là cảnh báo rằng **test bounded làm mô hình có vẻ tốt hơn thực tế**. (Không phải so sánh công bằng: baseline train trên toàn bộ 1,945,607 dòng, Attempt 3 chỉ trên 20,000 — 97× ít hơn.)

Tất cả 23 đặc trưng vượt top‑10 cũ trên cả hai mô hình baseline → xác nhận thực nghiệm cho quyết định **bỏ bước chọn lọc top‑10**, dùng đủ cả 23 đặc trưng.

### 7.6 Huấn luyện lại với đủ 23 đặc trưng, full coverage — Attempt 3 "all‑23"

Cùng cấu hình Attempt 3 (BertAdam, lr=2e‑5, warmup=0.1, sampler, alpha trung lập, chọn theo F1(pos)), chỉ đổi nhánh đặc trưng:

| | Attempt 3 (top‑10, 0.2% coverage) | all‑23 (100% coverage) | Δ |
|---|---:|---:|---:|
| overall F1(pos) | 0.6227 | **0.6852** | **+0.0625** |
| pure_test F1(pos) | 0.7406 | **0.7582** | +0.0176 |
| overlap F1(pos) | 0.5107 | **0.5987** | **+0.0880** |
| overall AUPRC | **0.6904** | 0.5822 | **−0.1082** |
| Best epoch | 7 | 2 | |
| Wall clock | 6.65h | 4.00h | |

**Đọc kết quả một cách trung thực:** F1(pos) tăng ở mọi lát cắt, đặc biệt `overlap` (điểm yếu cố hữu của mô hình) — nhưng **AUPRC (không phụ thuộc ngưỡng) lại giảm**, nghĩa là mô hình mới xếp hạng kém hơn một chút nhưng rơi đúng vào điểm vận hành tốt hơn khi hiệu chỉnh ngưỡng. Với chỉ 529 dương ở test, chênh lệch 0.06 F1 cũng nằm trong biên độ ngẫu nhiên của seed khác nhau. Kết luận: **"không tệ hơn, có thể tốt hơn tại điểm vận hành, chất lượng xếp hạng chưa ngã ngũ"** — không phải một cải thiện rõ ràng. Đây vẫn là **test bounded 5.29% dương**, nên **không kết luận được gì về năng lực thật ở quy mô đầy đủ** — phép đo quyết định là đánh giá full‑scale (chưa chạy tính đến thời điểm viết báo cáo đặc trưng).

---

## 8. Đánh giá full‑scale trên toàn bộ 811,704 tài khoản test thật (không train lại)

### 8.1 Hai rào cản kỹ thuật gặp phải trước khi chạy được

**(a) Va chạm tên checkpoint.** `tri_model/train1.py` và `bi_model/train_origin.py` đều tính `output_dir` tương đối theo cwd khi chạy (không theo thư mục script) và dùng công thức đặt tên checkpoint giống hệt nhau → **checkpoint "Attempt 3" thật của tri_model đã bị ghi đè âm thầm** bởi một lần chạy bi_model sau đó (kiểm tra bằng `state_dict`: file trên đĩa không có `feature_projector`, không thể là tri_model). **Sửa:** neo `output_dir` theo thư mục script, không theo cwd; các checkpoint bi_model bị đặt nhầm chỗ được **di chuyển** (không xóa) về đúng thư mục.

**(b) Nhánh GCN bị khóa từ vựng (transductive).** `W0_vh` là `nn.Parameter` cố định `[35000, 128]` — một dòng embedding học được cho từng tài khoản cụ thể trong vocab huấn luyện. Muốn chấm điểm 811,704 tài khoản test cần vocab chứa cả chúng, không load trực tiếp vào tensor shape cố định được. **Sửa:** vocab mới = 35,000 vocab cũ (giữ nguyên thứ tự, để mọi trọng số học được load nguyên vẹn) + 801,704 tài khoản test thật chưa từng nằm trong vocab, nối thêm vào cuối; `W0_vh` cũ được copy vào 35,000 dòng đầu (xác minh `torch.equal()` bit‑identical), các dòng mới giữ khởi tạo ngẫu nhiên.

### 8.2 Huấn luyện lại checkpoint Attempt‑3 để có bản gốc đúng vị trí (`attempt3v2`)

Chạy lại nguyên xi lệnh Attempt 3 → **tái tạo chính xác** từng epoch F1(pos) tới 4 chữ số thập phân, best epoch 7 (F1(pos)=0.6667), kết quả bounded‑test khớp hệt bản gốc. Checkpoint được lưu đúng chỗ, không va chạm.

### 8.3 Xây corpus test full‑scale & chạy đánh giá — tri_model

- Vocab mở rộng: 35,000 → **836,704** (thêm 801,704 tài khoản test mới).
- Ngưỡng hiệu chỉnh (trên val, không đổi): **0.8134** — giống hệt lần hiệu chỉnh gốc, xác nhận việc mở rộng vocab không ảnh hưởng tài khoản đã có trong vocab huấn luyện.

**Kết quả — toàn bộ 811,704 tài khoản test thật:**

| Slice | n | pos | F1(pos) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 609,773 | 312 | 0.0840 | 0.1176 | 0.7919 | 0.1058 | 0.3173 |
| overlap | 201,931 | 217 | 0.0371 | 0.1793 | 0.7969 | 0.2028 | 0.3963 |
| **overall** | **811,704** | **529** | **0.0549** | 0.1241 | 0.7960 | 0.0643 | 0.2590 |

Runtime: **~3.6h** (213.4 phút test + 1.6 phút hiệu chỉnh val), nhanh hơn ước tính ban đầu (~5h).

### 8.4 Vì sao F1(pos) rơi từ 0.6227 (bounded) xuống 0.0549 (full‑scale) — không phải bug

1. **98.8% tài khoản test (801,704/811,704) chưa từng xuất hiện lúc huấn luyện** — dòng `W0_vh` của chúng là khởi tạo ngẫu nhiên, không có tín hiệu GCN riêng theo tài khoản (chỉ có tín hiệu gián tiếp qua hàng xóm đã học, nếu có).
2. **Độ phủ đặc trưng đồ thị chỉ 0.2% toàn dự án** (tại thời điểm chạy báo cáo này — trước khi mục 7 ở trên hoàn tất) — phần lớn 801,704 tài khoản mới rơi vào fallback vector‑0.

→ Số 0.6227 (bounded) đo trên một tập test mà 100% tài khoản **đã có trong vocab huấn luyện** — một điều kiện dễ hơn, không phải "thực tế hơn". **AUPRC (0.12) và Recall@500 (0.26)** là chỉ số đáng tin hơn F1(pos) tại ngưỡng cố định, vì ngưỡng được hiệu chỉnh trên tập val 100%-in-vocab, không phù hợp với một quần thể test 98.8% out‑of‑vocab.

### 8.5 Cùng đánh giá cho bi_model — không cần train lại

Checkpoint bi_model tốt nhất (từ `bi_modal_run_report.md` §9, protocol Attempt‑3‑parity) **còn nguyên vẹn**, không bị ghi đè — chính là file mà mục 8.1(a) xác định là bi‑modal. Chỉ cần di chuyển về đúng thư mục, tái sử dụng corpus full‑test đã xây ở mục 8.3.

**Kết quả — toàn bộ 811,704 tài khoản test:**

| Slice | n | pos | F1(pos) | AUPRC | G-Mean |
|---|---:|---:|---:|---:|---:|
| pure_test | 609,773 | 312 | 0.0858 | 0.1159 | 0.8788 |
| overlap | 201,931 | 217 | 0.0293 | 0.0566 | 0.8719 |
| **overall** | **811,704** | **529** | **0.0474** | 0.0801 | 0.8797 |

### 8.6 bi_model vs tri_model ở full‑scale (cùng test set, cùng phương pháp)

| Slice | F1(pos) tri_model | F1(pos) bi_model | AUPRC tri_model | AUPRC bi_model |
|---|---:|---:|---:|---:|
| pure_test | 0.0840 | 0.0858 | 0.1176 | 0.1159 |
| overlap | **0.0371** | 0.0293 | **0.1793** | 0.0566 |
| overall | **0.0549** | 0.0474 | **0.1241** | 0.0801 |

Cả hai đều "sụp" về F1(pos) thấp vì cùng một lý do (98.8% out‑of‑vocab); tri_model nhỉnh hơn ở `overlap`/AUPRC tổng, bi_model nhỉnh hơn nhẹ ở `pure_test` F1(pos). Cả hai vẫn nằm trong cùng chế độ "chủ yếu out‑of‑vocab", nên **so sánh kiến trúc bi‑vs‑tri ở mục 6 chỉ có ý nghĩa ở quy mô bounded**, không phải kết luận cuối cùng.

---

## 9. Thí nghiệm bổ sung — mở rộng nhãn test bằng lan truyền không giới hạn (2 phiên bản)

Đây là thí nghiệm **tách biệt**, không sửa/ghi đè lên `mg_propagate_labels.py`, split, hay checkpoint Attempt‑3 gốc. Mục tiêu: tập test thật chỉ có 529 dương/811,704 (≈1:1534) — quá thưa để đánh giá tin cậy → thử làm giàu bằng lan truyền 1‑hop **cho phép rơi vào cả val/test** (ngược thiết kế train‑only ở mục 3).

### 9.1 Phiên bản 1 — có hub‑guard, giữ toàn bộ cơ chế an toàn (`expanded_test_propagation_report.md`)

Dùng lại y hệt cơ chế của `mg_propagate_labels.py` (có hướng, chặn hub, tách nhãn, soft label 0.5) — chỉ **bỏ duy nhất** ràng buộc "không chạm val/test".

| Partition | Phishing trước | Propagated thêm | Phishing sau | Tỉ lệ sau |
|---|---:|---:|---:|---:|
| train | 519 | 519 | 1,038 | 0.0534% |
| val | 117 | 137 | 254 | 0.1175% |
| overlap | 217 | 228 | 445 | 0.2204% |
| pure_test | 312 | 435 | 747 | 0.1225% |
| **test** | **529** | **663** | **1,192** | **0.1469%** |

→ Test tăng từ 529 lên 1,192 dương (**2.25×**). Toàn bộ assertion an toàn (không trùng ground_truth/propagated, seed count bảo toàn 1,165) **PASS**. Tại thời điểm viết, **chưa có mô hình nào được huấn luyện/đánh giá** trên nhãn `isp_expanded` này — chỉ là bước xây dựng nhãn.

### 9.2 Phiên bản 2 — bỏ toàn bộ guard, chạy huấn luyện thật (báo cáo chính §17)

Theo yêu cầu đơn giản hóa: mọi hàng xóm OUT 1‑hop của 1,165 seed đều gắn nhãn gian lận, **không lọc hub, không soft‑weight, không theo dõi nạn nhân**.

| Partition | Fraud trước (`isp`) | Fraud sau (`isp_expanded`) |
|---|---:|---:|
| train | 519 | 3,541 |
| val | 117 | 463 |
| overlap | 217 | 2,587 |
| pure_test | 312 | 1,581 |
| **test** | **529** | **4,168** |
| **Tổng** | **1,165** | **8,172** |

**Huấn luyện lại Attempt‑3 trên nhãn mở rộng** (corpus riêng, không va chạm checkpoint gốc): 15 epoch đầy đủ, best epoch 12, ngưỡng hiệu chỉnh 0.6976.

**Test bounded (10,000 mẫu, 41.68% dương — do nhãn lan truyền làm giàu test):**

| Slice | F1(pos) |
|---|---:|
| pure_test | 0.9038 |
| overlap | 0.8605 |
| **overall** | **0.8763** |

**Vì sao con số 0.8763 KHÔNG được đọc là "cải thiện so với Attempt 3 (0.6227)":** hai lần chạy chấm trên hai tập test có độ khó khác nhau hoàn toàn — 41.68% dương so với 5.29% dương là hai bài toán phân loại khác hẳn về độ khó; hơn nữa ~87% số dương ở đây là **suy diễn (propagated)**, không phải xác thực, nên phần lớn "cải thiện" chỉ là mô hình học khớp với chính họ hàng nhãn suy diễn của nó.

### 9.3 Đánh giá full‑scale checkpoint "expanded" — kiểm chứng quyết định

Chạy đúng quy trình full‑811,704 như mục 8, cho checkpoint huấn luyện trên `isp_expanded`, chấm điểm theo **cả hai** bộ nhãn (đồng thời, không cần forward pass thứ hai):

**Chấm theo strict ground‑truth `isp` — so sánh trực tiếp với `attempt3v2` (mục 8.3):**

| Metric (overall, strict isp) | attempt3v2 (checkpoint gốc) | expanded_test_attempt3 |
|---|---:|---:|
| F1(pos) | 0.0549 | **0.0383** |
| AUPRC | 0.1241 | **0.0408** |
| Recall@500 | 0.2590 | **0.0605** |

**Kết luận quan trọng nhất của toàn bộ thí nghiệm mở rộng:** checkpoint huấn luyện trên nhãn mở rộng **không hề tốt hơn** ở phát hiện phishing thật quy mô đầy đủ — mà **tệ hơn** trên mọi chỉ số, rõ nhất ở AUPRC (giảm 3×) và Recall@500 (giảm 4.3×). Con số "F1(pos)=0.88" ở bước bounded chỉ đo mức độ mô hình khớp với chính nhãn suy diễn của nó, không phải chất lượng phát hiện phishing thật — đúng như cảnh báo đã nêu trước khi kết quả full‑scale này tồn tại để xác nhận.

**Lưu ý phạm vi:** đây là kết luận cho **phiên bản không guard, không soft‑weight** của lan truyền mở rộng test — cơ chế train‑only/hub‑guard/soft‑weight gốc ở mục 3 là một cơ chế thận trọng hơn, **chưa được đánh giá ở quy mô full‑scale** trong tài liệu này.

---

## 10. Dòng thời gian quyết định chính (rút gọn)

```
Audit rò rỉ dữ liệu (mục 1)
        │
        ▼
Split T_cutoff 80th-percentile mới (mục 2) ──► Lan truyền nhãn 1-hop train-only, hub-guard (mục 3)
        │
        ▼
Fix bộ nhớ: dense one-hot → sparse gather (mục 4) ──► gỡ rào cản bộ nhớ, không đổi kết quả số học
        │
        ▼
Fix quy trình train: sampler + F1(pos) selection + BertAdam warmup (mục 5)
        │      "Before → Attempt1(sập) → Attempt2(vẫn kẹt epoch 0) → Attempt3(hội tụ thật, F1=0.6227)"
        ▼
So sánh kiến trúc bi_model vs tri_model, cùng protocol (mục 6) ──► bi_model thắng ở scale bounded
        │
        ▼
Trích xuất đủ 23 đặc trưng ở 100% coverage (mục 7) ──► phát hiện nhánh đặc trưng cũ đã TRƠ (99.8% zero-fallback)
        │      + baseline chỉ-đặc-trưng cho thấy full-scale đảo ngược thứ hạng model vs baseline
        ▼
Đánh giá full-scale thật (811,704 tài khoản), không train lại (mục 8)
        │      F1(pos) 0.62 (bounded) → 0.05 (full-scale) — không phải bug, do 98.8% out-of-vocab
        ▼
Thí nghiệm mở rộng nhãn test bằng lan truyền không giới hạn (mục 9)
        └──► xác nhận full-scale: KHÔNG cải thiện, thậm chí TỆ HƠN so với checkpoint gốc trên isp thật
```

---

## 11. Những điểm mấu chốt cần nhớ khi đọc bất kỳ con số F1 nào trong dự án này

1. **F1(pos) bounded‑scale (test 5.29% dương, 100% in‑vocab) luôn lạc quan hơn thực tế** — chỉ dùng để so sánh nội bộ giữa các cấu hình cùng corpus, không phải ước lượng năng lực thật.
2. **Chỉ số full‑scale đáng tin nhất hiện có**: `attempt3v2` (tri_model, top‑10 feature, 0.2% coverage) F1(pos)=0.0549, AUPRC=0.1241 trên toàn bộ 811,704 tài khoản test thật — đây là mốc so sánh chuẩn.
3. **Baseline chỉ‑đặc‑trưng (HistGBM, 23 đặc trưng, full coverage) đạt F1(pos)=0.3479 full‑scale**, vượt xa mô hình BERT+GCN+fusion hiện có — nhưng không phải so sánh công bằng (baseline train trên 1.9M dòng, mô hình chỉ trên 20,000). Đây là gợi ý rằng **nhánh đặc trưng đồ thị đáng được sửa và huấn luyện lại ở quy mô đầy đủ**, chưa phải bằng chứng BERT+GCN vô dụng.
4. **Chưa có lần chạy nào huấn luyện thật trên toàn bộ 1,945,607 tài khoản train** — mọi con số F1 hiện có đều từ corpus bounded (tối đa 20,000 train). Ước tính wall‑clock cho huấn luyện full‑scale: ~1–2 ngày/epoch trên 1 GPU RTX 3060.
5. **Hai bộ nhãn phishing khác nhau tồn tại** (`phisher_accounts.txt` 5,480 vs `labels.pkl`/`isp` 1,165, trùng 963) — mọi kết quả trong tài liệu dùng `isp`; `.txt` đã bị khai tử nhưng câu hỏi "bộ nào đúng cho dataset này" vẫn để ngỏ.
6. **Việc "làm giàu" nhãn test bằng lan truyền không giới hạn KHÔNG được khuyến nghị** dựa trên bằng chứng full‑scale ở mục 9.3 — số liệu bounded từ cách làm này (F1(pos)=0.88) hấp dẫn nhưng gây hiểu lầm.

---

## 12. Việc còn mở (chưa làm, ghi nhận xuyên suốt các báo cáo)

1. Huấn luyện thật (không chỉ đánh giá) ở quy mô đầy đủ 1,945,607 tài khoản train, cho một trong hai kiến trúc — ước tính nhiều ngày/epoch.
2. Chạy `eval_full_test.py --feature_set all23` (checkpoint all‑23‑fullcov, mục 7.6) trên toàn bộ 811,704 tài khoản test — đây là phép đo quyết định còn thiếu để biết đặc trưng đồ thị đầy đủ có thật sự giúp ở quy mô thật hay không.
3. Hiệu chỉnh lại ngưỡng quyết định trên một tập validation cũng chứa tài khoản out‑of‑vocab (thay vì tập val 100% in‑vocab hiện dùng) — có thể phù hợp hơn với quần thể test 98.8% out‑of‑vocab.
4. Khôi phục `eigenvector`/`katz`/`closeness` centrality (hiện bị bỏ để tiết kiệm ~44h → ~23h) nếu cần dùng sau này — cần chạy lại không `--reduced` trên toàn bộ 2.97M node (~44h).
5. Giải quyết câu hỏi provenance: bộ nhãn `phisher_accounts.txt` vs `isp` — bộ nào đúng cho dataset MulDiGraph này.