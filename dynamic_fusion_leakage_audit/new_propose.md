
**Quyết định**: xây một pipeline song song `Dataset/inductive_model/`, không sửa đè lên `tri_model/`
hiện có, theo đúng quy ước cô lập thí nghiệm của dự án.

---

## 1. Kiến trúc pipeline v2

```
                          x0
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Transaction history        Account network
             │                         │
             ▼                         ▼
       Tx-BERT/TxLM              Inductive GNN
        (tái dùng phần lớn         GraphSAGE/GAT
         BERT hiện có,          + neighbor sampling
         bỏ mọi tensor GCN)     (node feature = 23-cột
             │                   full-coverage đã có)
             │                         │
             ▼                         ▼
         h_text(x0)              h_graph(x0)
             │                         │
             └────────────┬────────────┘
                          ▼
                Cross/Dynamic Fusion
           (gated-fusion trước, cross-attention
              kiểu LMAE4Eth sau nếu cần)
                          │
                          ▼
                    P(fraud | x0)
```

### 1.1 Khác biệt so với kiến trúc hiện có (`tri_model/ETH_GBert.py`)

| | Kiến trúc cũ (`tri_model`) | Kiến trúc mới (`inductive_model`) |
|---|---|---|
| GCN | `W0_vh` bảng embedding cố định theo vocab, tiêm trực tiếp vào token embedding của BERT | Encoder GraphSAGE/GAT độc lập, input là **23 đặc trưng đã trích xuất full-coverage**, không học embedding per-account |
| Tổng quát hóa account mới | Không — phải mở rộng vocab + partial-load checkpoint | Có — chỉ cần tính 23 đặc trưng + lấy mẫu hàng xóm |
| Fusion | Token-level, bên trong embedding layer, trước self-attention | Account-level, sau khi đã có `h_text`, `h_graph` hoàn chỉnh |
| FeatureProjector (23 đặc trưng) | Broadcast riêng vào token, là nhánh thứ 3 | Trở thành **input trực tiếp** của GNN, không còn là nhánh riêng |
| API 1 account | Không có | Có, hệ quả tự nhiên của neighbor sampling (mục 4.5 bên dưới) |

### 1.2 Thành phần tái sử dụng được từ LMAE4Eth (đã kiểm chứng bằng cách clone code thật)

| Thành phần | Trạng thái | Dùng cho |
|---|---|---|
| `model/sage/model.py` (`SAGE`, `inference()`) | ✅ sạch, chạy được | Khung GraphSAGE + suy luận full-graph theo tầng, theo batch, không tính lặp lại |
| `model/gcn.py`, `model/gat.py` | ✅ sạch (DGL, block-based) | Khung encoder inductive cho `h_graph(x0)` |
| `model/contrastive.py` | ✅ sạch, độc lập | Tham khảo nếu sau này muốn pretrain Tx-BERT bằng contrastive loss |
| `loss_func.py` (`sce_loss`) | ✅ sạch | Nếu làm masked-autoencoder pretraining cho node feature |
| `sampler analyze/ladies_sampler.py` | ✅ sạch (port từ ví dụ chính thức DGL) | Sampler LADIES thay thế neighbor sampling đơn giản nếu cần |
| `etherscan.py` + `Data/actions_enum.py` | ✅ sạch | Lấy giao dịch thô cho account hoàn toàn mới (Case B) |
| `model/fusion_blocks.py` + `lgfusion.py` | ⚠️ có bug (import sai, logic loop đáng ngờ) | Tham khảo ý tưởng cross-attention, **không copy nguyên** |
| `model/edcoder.py` (MAGAE) | ⚠️ đúng hướng nhưng nặng (EMA teacher, BYOL-style) | Cân nhắc sau, không phải ưu tiên hiện tại |
| `model/google_bert.py`, `finetune.py`, `test.py`, `inference.py`, `sampler analyze/train_lightning*.py` | ❌ không dùng | Trùng lặp, thiếu dependency, hoặc hoàn toàn không liên quan (rác project khác) |


## 3. Danh sách nhiệm vụ đầy đủ — theo giai đoạn

### Giai đoạn A — Chuẩn bị dữ liệu cho kiến trúc mới

- [ ] **A1. Chuyển `features_output_all23_MG_fullscale.csv` → tensor `[N, 23]` + index map**, tái sử
      dụng code đã tối ưu ở `fullscale_features` (đã tránh lỗi "1 tensor/account" cũ) — làm node
      feature input cho GraphSAGE/GAT.
- [ ] **A2. Chuyển `adj_train.npz`/`adj_inference.npz` (scipy CSR) sang định dạng đồ thị của thư viện
      đã chọn** (PyTorch Geometric `Data`/`HeteroData` hoặc DGL `graph`) — giữ nguyên phân biệt
      train-time graph (≤T_cutoff) vs inference-time graph (đầy đủ), chỉ đổi định dạng lưu trữ.
- [ ] **A3. Chọn thư viện graph**: PyTorch Geometric (`NeighborLoader`) hoặc DGL (`LadiesSampler`/
      `MultiLayerNeighborSampler`, có thể tái dùng trực tiếp `ladies_sampler.py` của LMAE4Eth nếu chọn
      DGL) — quyết định trước khi viết code A2/B1.

### Giai đoạn B — Nhánh Inductive GNN (`h_graph(x0)`)

- [ ] **B1. Cài đặt encoder GraphSAGE hoặc GAT 2 lớp**, dùng `NeighborLoader`/sampler đã chọn ở A3,
      budget hàng xóm mỗi lớp (ví dụ [15, 10]) — tham khảo `model/gcn.py`/`model/gat.py` của LMAE4Eth
      làm khung, không copy nguyên.
- [ ] **B2. Kết hợp label-aware sampling ở cấp đồ thị** (PC-GNN-style: oversample hàng xóm quanh node
      dương, undersample quanh node âm) vào sampler ở B1 — thay thế `WeightedRandomSampler` cấp-mẫu
      hiện tại của `tri_model`, vốn không cân bằng được cấp-hàng-xóm.
- [ ] **B3. Viết `predict_account(addr)`**: given 1 địa chỉ (có thể chưa từng ở trong graph), build
      ego-subgraph on-the-fly, forward qua encoder B1 → `h_graph`. Đây là hệ quả tự nhiên của A2+B1,
      không cần rebuild toàn model như cách làm cũ.
- [ ] **B4. Xử lý Case B tường minh (account hoàn toàn mới, chưa từng là node)**: dùng `etherscan.py` +
      `Data/actions_enum.py` (từ LMAE4Eth) để lấy giao dịch thô, build node+feature+cạnh tạm thời nối
      vào bản sao/slice của đồ thị suy luận, gắn nhãn rõ ràng "độ tin cậy GCN thấp nếu hàng xóm cũng là
      node mới" trong output.

### Giai đoạn C — Nhánh Tx-BERT (`h_text(x0)`)

- [ ] **C1. Tách nhánh BERT khỏi mọi tensor liên quan tới GCN**: bỏ `gcn_vocab_ids` khỏi
      `ETH_GBertEmbeddings.forward`/`ETH_GBertModel.forward`, bỏ `FeatureProjector` cũ (23-đặc-trưng đã
      chuyển sang input của GNN ở A1). Nhánh BERT trở về gần giống BERT chuẩn, chỉ nhận `input_ids`.
- [ ] **C2 (tùy chọn, không ưu tiên P0)**: cân nhắc contrastive pretraining cho Tx-BERT theo
      `model/contrastive.py` của LMAE4Eth nếu muốn giảm tính đồng nhất (anisotropy) của embedding giao
      dịch — hoãn tới sau khi kiến trúc cơ bản chạy được.

### Giai đoạn D — Fusion & Classifier

- [ ] **D1. Viết module fusion account-level, bắt đầu bằng gated-fusion** (tái dùng ý tưởng
      `DynamicFusionLayer` hiện có nhưng đổi input từ token-level 3 luồng → account-level 2 vector
      `h_text`, `h_graph`) — rẻ hơn, ít tham số hơn cross-attention, phù hợp với tập dương nhỏ
      (519–3,541 mẫu).
- [ ] **D2 (nâng cấp sau, nếu D1 chưa đủ mạnh)**: cross-attention fusion kiểu LMAE4Eth — tham khảo
      `model/fusion_blocks.py`/`lgfusion.py` **nhưng tự viết lại**, không copy nguyên vì có bug import
      (`from models import fusion_blocks`, đúng phải là `model`) và logic loop đáng ngờ
      (`zip(self.fusion_blocks, self.fusion_blocks, self.fusion_blocks)` dùng chung 1 list 3 lần).

### Giai đoạn E — Hạ tầng train/eval & kiểm chứng ở scale nhỏ

- [ ] **E1. Xây pipeline train/eval mới trong `Dataset/inductive_model/`** (thư mục riêng, checkpoint
      riêng, không đụng `tri_model/output/`), theo đúng quy ước cô lập thí nghiệm đã dùng xuyên suốt dự
      án (`runs/expanded_test_attempt3/` là tiền lệ).
- [ ] **E2. Kiểm chứng ở scale bounded (20k, tái dùng corpus Attempt‑3 hiện có)** — so sánh trực tiếp
      F1(pos)/AUPRC với checkpoint Attempt‑3 gốc.
- [ ] **E3. Test inductive thật sự**: chấm điểm trên vài trăm account **cố tình giữ ngoài vocab train**
      — đây là bài test mà kiến trúc cũ (`tri_model`) không vượt qua được (chính là nguyên nhân gây sập
      F1 0.62→0.055 ở full-scale). Nếu kiến trúc mới vượt qua bài test này ở scale nhỏ, đó là tín hiệu
      mạnh để đầu tư tiếp; nếu không, dừng lại xem xét trước khi scale lên.

### Giai đoạn F — Quyết định scale (chỉ sau khi Giai đoạn E cho kết quả tốt)

- [ ] **F1.** Tối ưu hạ tầng trước khi tăng scale: tăng batch size (đã đo dư ~4.6GB bộ nhớ ở batch=8
      trên card 12GB), thêm mixed-precision (fp16/bf16), cân nhắc freeze vài lớp dưới của BERT.
- [ ] **F2.** Chạy mốc trung gian (200,000–500,000 account, giữ toàn bộ 519 dương thật) trước khi cam
      kết train full 1,945,607 — rẻ hơn nhiều so với train full ngay, đủ để thấy xu hướng scale có giúp
      hay không.
- [ ] **F3.** Chỉ sau F1+F2 và nếu kết quả tích cực: cân nhắc train full 1,945,607 tài khoản (ước tính
      1–2 tuần trên 1 GPU nếu không tối ưu, ít hơn nhiều sau F1).

### Giai đoạn G — Dọn dẹp & tài liệu (có thể làm song song, không phụ thuộc kiến trúc)

- [ ] **G1.** Xác nhận lại `--pre-cutoff` đã áp dụng đúng cho toàn bộ artifact đang dùng (kiểm tra lại
      sau khi A1 hoàn tất, vì node feature giờ là input trực tiếp của GNN, sai sót ở đây ảnh hưởng lớn
      hơn trước).
- [ ] **G2.** Giải quyết câu hỏi provenance nhãn: `phisher_accounts.txt` (5,480) vs `isp` (1,165) — vẫn
      để ngỏ, không phụ thuộc kiến trúc.
- [ ] **G3.** Cập nhật tài liệu (`tong_hop_quy_trinh.md`, `pipeline_review_vs_code_check.md`) để phản
      ánh kiến trúc v2 một khi Giai đoạn E hoàn tất — tránh tình trạng tài liệu lệch code như đã xảy ra
      với `CONTEXT_SUMMARY.md`/`README.md` ở lineage B4E.

---

## 4. Việc đã loại bỏ khỏi danh sách (không còn là nhiệm vụ riêng lẻ)

Ghi lại để tránh làm trùng lặp — các mục này của bản kế hoạch trước đã được **gộp vào nhiệm vụ ở mục 3**
vì kiến trúc mới giải quyết tận gốc, không cần patch riêng:

- ~~Cache `H_vh` trong eval~~ → không còn `VocabGraphConvolution` toàn-vocab trong kiến trúc mới; thay
  bằng B1 (encoder inductive) + cách suy luận theo tầng của `SAGE.inference()`.
- ~~Viết API `predict_account` bằng cách tham số hoá `mg_build_full_test_eval.py` cho N=1~~ → thay bằng
  B3, là hệ quả tự nhiên của kiến trúc inductive.
- ~~Xử lý Case B bằng cách sửa `mg_build_full_test_eval.py`~~ → thay bằng B4, dùng hạ tầng Etherscan mới
  tìm được.
- ~~Thay `W0_vh` bằng encoder inductive~~ → đây chính là toàn bộ Giai đoạn B, không còn là 1 mục đơn lẻ.

---

## 5. Tóm tắt thứ tự thực hiện đề xuất

```
A (chuẩn bị dữ liệu) → B + C (song song, 2 nhánh độc lập) → D1 (gated-fusion) → E1-E3 (kiểm chứng nhỏ)
        │
        ▼
   Kết quả E3 tốt? ──No──► dừng, xem lại B/C/D trước khi đi tiếp
        │Yes
        ▼
   F1 (tối ưu hạ tầng) → F2 (mốc trung gian 200k-500k) → F3 (train full, nếu vẫn đáng)
        │
        ▼
   D2 (cross-attention, nếu cần) + C2 (contrastive pretrain, nếu cần) — nâng cấp tùy chọn
```

G (dọn dẹp/tài liệu) chạy song song, không chặn đường trên.