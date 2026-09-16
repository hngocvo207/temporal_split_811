# BERT 

## 0. Vai trò

Bạn là kỹ sư ML được giao sửa một lỗi thiết kế dữ liệu (không phải bug logic) trong pipeline
fraud detection Ethereum có 2 nhánh: graph (đang hoạt động đúng, AUPRC 0.303 trên pure_test) và
text/BERT (đang overfit nặng: ROC-AUC 0.979 nhưng P@100 = 0.07). Nhiệm vụ của bạn là sửa root
cause của nhánh text, KHÔNG được đụng vào nhánh graph.

## 1. Bối cảnh / Root cause (đọc kỹ trước khi code)

- Corpus text tại `runs/inductive_e2_corpus_100k/corpus/` build câu theo template
  `from: {addr} to: {addr} amount: ... in_out: ...` ở `Dataset/mg_build_examples.py:172-180`.
- Vì account đang xét luôn đứng ở một vế `from`/`to` của MỌI giao dịch của chính nó, địa chỉ hex
  40 ký tự của chính nó lặp lại ở mọi dòng trong document của nó (~45% token sau front-truncate ở
  `MAX_SEQ_LEN=400`, xem `attempt3_corpus.py:36,58-60`).
- Kết hợp `WeightedRandomSampler` (pos:neg = 1:4, 9 epoch), mỗi document dương được BERT nhìn
  ~346 lần → model học thuộc "chuỗi sub-token địa chỉ X → nhãn 1" thay vì hành vi giao dịch, không
  transfer được sang 609,773 địa chỉ mới ở pure_test.
- Đối chiếu SOTA (BERT4ETH, Hu et al., WWW'23):
  - Họ không lặp self-address: chỉ chèn **một** dummy self-transaction ở đầu chuỗi (address =
    self-address, mọi field khác = "Null"), không xuất hiện lại ở các dòng sau.
  - Repetitiveness Reduction gồm **2 phần**, không phải 1: (RR#1) dedup giao dịch liên tiếp, và
    (RR#2) masking ratio/dropout ratio cao (80%) khi pretrain — vì bản thân paper gốc xác nhận
    dedup liên tiếp một mình KHÔNG đủ, phần "discontinuous repetitiveness" vẫn còn nhiều.
  - Skew Alleviation là frequency-aware negative sampling trong contrastive loss khi pretrain,
    vận hành trên nền một bảng embedding TOÀN CỤC cho từng address cụ thể — khác về bản chất với
    việc ẩn danh hoá địa chỉ, KHÔNG dùng làm căn cứ tương đương cho Task 2 bên dưới.
  - Có thêm cơ chế thứ 3 (Heterogeneity Modeling: tách in/out + encoder riêng cho log ERC-20) —
    ngoài phạm vi sửa lỗi lần này, chỉ ghi nhận làm backlog.
  - Các paper text-based khác cùng dòng (TLMG4Eth — Sun et al.; LMAE4Eth) textualize giao dịch
    thành attribute-value words (`amount: ..., direction: ...`) và **không đưa chuỗi hex address
    thật vào câu văn bản dưới bất kỳ hình thức nào**.

## 2. Nguyên tắc bắt buộc khi sửa

1. Không bao giờ để một chuỗi hex address thật (của chính account hoặc của đối tác) xuất hiện dưới
   dạng token WordPiece trong text đưa vào BERT.
2. Ưu tiên sửa data design (Task 1–4) trước, chỉ động đến kiến trúc (Task 5) nếu 1–4 không đủ.
3. Mọi thay đổi phải có bước validation cụ thể, chạy được, có output đo lường được (không chỉ
   "có vẻ đúng").
4. Không copy structure/code cũ khi không cần thiết — nếu một hàm cũ (`clean_str()`, v.v.) không
   còn phù hợp với thiết kế mới thì sửa hoặc thay thế, đừng vá chồng lên.

---

## Task 1 — Loại bỏ self-address khỏi template câu (ưu tiên cao nhất)

**File:** `Dataset/mg_build_examples.py`, hàm build sentence, dòng 172-180.

- Bỏ field địa chỉ chính chủ khỏi câu. Giữ: `counterparty`, `amount`, `in_out`.
- Đổi template: `from: {addr} to: {addr} amount: ... in_out: ...` →
  `counterparty: {addr} amount: ... in_out: ...`
- `in_out` đã đủ để suy vai trò from/to của account chính, không cần address chính chủ.

**Validation:** grep corpus mới theo từng file, đảm bảo địa chỉ của chính account (biết trước) —
không còn xuất hiện trong nội dung document của chính nó. Report số document còn vi phạm (kỳ vọng: 0).

---

## Task 2 — Anonymize địa chỉ đối tác theo local ID trong từng document

**File:** `Dataset/tri_model/utils.py` (hàm mới, gọi trước `clean_str()`).

- Viết `anonymize_addresses(transactions) -> transactions_với_id_cục_bộ`: map mỗi địa chỉ đối tác
  xuất hiện trong document sang `addr0, addr1, addr2, ...` theo thứ tự xuất hiện đầu tiên. Ánh xạ
  chỉ có hiệu lực **trong phạm vi 1 document** (không global).
- Mục đích thật sự của bước này (ghi rõ trong code comment, tránh nhầm lẫn khi review sau này):
  **ngăn model học thuộc chuỗi hex cụ thể** của bất kỳ address nào (chính chủ lẫn đối tác), đồng
  thời vẫn giữ tín hiệu hành vi hợp lệ (giao dịch lặp lại nhiều lần với cùng 1 đối tác). Đây KHÔNG
  phải bản rút gọn của "Skew Alleviation" (cơ chế đó của BERT4ETH vận hành trên embedding toàn cục
  cho từng address, khác bản chất) — mà là áp dụng nguyên tắc chung của các pipeline text-based
  SOTA: không đưa raw address string vào text.
- **BẮT BUỘC, không phải fallback:** random hoá lại thứ tự gán ID mỗi lần document được sampler
  draw ra (không cố định 1 lần lúc build corpus), để cùng 1 document dương không luôn sinh ra đúng
  một chuỗi token giống hệt qua các epoch. Lý do bắt buộc: paper gốc BERT4ETH xác nhận dedup liên
  tiếp một mình (Task 3) không đủ để chặn overfit do repetitiveness — cần thêm một cơ chế biến
  thiên ở mức token qua các epoch; đây chính là cơ chế đó cho pipeline này.
- **Validation:** lấy ngẫu nhiên ≥20 document dương sau xử lý, xác nhận không còn địa chỉ hex thật
  nào trong text (chỉ còn `addrN`); xác nhận qua 2 epoch liên tiếp, cùng 1 document sinh ra 2 phép
  gán ID khác nhau (log trước/sau để đối chiếu).

---

## Task 3 — Dedup giao dịch liên tiếp trước khi truncate

**File:** `Dataset/inductive_model/data_prep/attempt3_corpus.py`, trước `_encode()` (dòng ~58).

- Viết `dedup_consecutive(transactions, window_hours=72)`: gộp giao dịch liên tiếp cùng
  `counterparty_anon` + cùng `in_out` + timestamp cách nhau ≤ window, cộng dồn `amount`, giữ
  timestamp đầu tiên, thêm field `count`.
- Áp dụng dedup **trước** khi gọi `_encode()`/front-truncate.
- **Lưu ý bắt buộc:** dedup liên tiếp chỉ xử lý "continuous repetitiveness". Theo đúng phát hiện
  của BERT4ETH, "discontinuous repetitiveness" (đối tác lặp lại nhưng không liền kề nhau trong
  chuỗi) vẫn còn tồn tại sau bước này — đây là lý do Task 2 có bước random-hoá ID theo epoch ở
  trên là **bắt buộc đi kèm**, không phải optional.
- **Đo lường:** tính tỷ lệ repetitiveness trước/sau dedup trên corpus 100k (cách đo của BERT4ETH:
  % giao dịch có cùng counterparty với giao dịch liền trước). Dùng số này làm metric so sánh.
- Nếu số giao dịch/account vẫn > sức chứa 398 token sau dedup: đánh giá sampling (đầu+cuối chuỗi,
  hoặc random có trọng số theo amount) thay vì front-truncate thuần.

---

## Task 4 — Giảm rủi ro overfit do oversampling

**File:** pipeline training (nơi khởi tạo `WeightedRandomSampler`) + training loop.

- Giảm tỷ lệ oversample pos:neg từ 1:4 xuống thấp hơn (thử 1:2), bù bằng weighted loss
  (`BCEWithLogitsLoss(pos_weight=...)` hoặc focal loss) thay vì oversample thô.
- Đổi tiêu chí early stopping sang theo dõi P@K (ví dụ P@100) trên validation set tách riêng,
  không theo loss/AUC.
- **Validation:** log P@100 mỗi epoch trên val set; dừng training khi P@100 bắt đầu giảm dù
  loss/AUC vẫn cải thiện.

---

## Task 5 (dài hạn, chỉ làm nếu Task 1–4 chưa đủ) — Thay WordPiece bằng address embedding

Đánh giá **cả hai** phương án dưới đây trên tập inductive (pure_test), không chỉ chọn một:

- **(a)** Categorical address embedding kiểu BERT4ETH (vocab theo tần suất, OOV → bucket chung)
  + amount/timestamp binning + embedding, thay vì sinh câu rồi tokenize WordPiece.
- **(b)** Bỏ hẳn address (kể cả dạng anonymized) khỏi luồng text: sentence chỉ gồm
  `amount / direction (in_out) / time-interval`, để toàn bộ tín hiệu "ai giao dịch với ai" do
  nhánh graph đảm nhiệm (theo hướng TLMG4Eth/LMAE4Eth).

Ghi rõ lý do cần so sánh cả 2: (a) vẫn có rủi ro OOV embedding cho địa chỉ hoàn toàn mới ở
pure_test; (b) né rủi ro đó triệt để hơn nhưng mất tín hiệu định danh đối tác trong nhánh text.
Chỉ triển khai Task 5 nếu sau Task 1–4, fusion vẫn không vượt graph-only baseline (AUPRC 0.303).

---

## Thứ tự thực hiện & tiêu chí nghiệm thu

| Bước | Task | Output kỳ vọng |
|---|---|---|
| 1 | Task 1 + 2 | Corpus mới: 0 địa chỉ hex thật trong text; xác nhận ID đổi qua epoch |
| 2 | Task 3 | Repetitiveness ratio giảm, đo bằng script riêng, kèm số liệu trước/sau |
| 3 | Rebuild corpus 100k, train lại BERT nhánh text | So sánh P@100 trước/sau trên pure_test |
| 4 | Task 4 | P@100 ổn định qua epoch, không sập sau khi AUC bão hòa |
| 5 | So sánh fusion vs graph-only (0.303) trên pure_test | Fusion > graph-only → dừng. Nếu không → Task 5 |

## Yêu cầu báo cáo lại (bắt buộc sau mỗi task)

Với mỗi task, trả về:
1. Diff code thực tế đã sửa (đường dẫn file + đoạn thay đổi).
2. Lệnh/script dùng để validate, và kết quả chạy thật (không phải dự đoán).
3. Số liệu đo lường liên quan (repetitiveness ratio, P@100, AUC, AUPRC — tuỳ task).
4. Nếu phát hiện sai lệch so với plan này trong lúc code (ví dụ path không khớp, hàm không tồn
   tại), dừng lại và báo trước khi tự ý đổi hướng.

## Tham khảo

- Hu, S. et al. "BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection." WWW 2023.
  (Repetitiveness Reduction, Skew Alleviation, Heterogeneity Modeling)
- Sun, J. et al. "Ethereum Fraud Detection via Joint Transaction Language Model and Graph
  Representation Learning" (TLMG4Eth), 2024.
- "LMAE4Eth: Generalizable and Robust Ethereum Fraud Detection by Exploring Transaction
  Semantics and Masked Graph Embedding," 2025.