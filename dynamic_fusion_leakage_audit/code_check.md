Hãy review code của tôi theo đúng pipeline fraud detection account-level bằng GNN/GCN + BERT, lấy cảm hứng từ ETH-GBERT, nhưng tập trung vào bài toán inference cho một account test x0.

Bối cảnh:
- Có khoảng 2 triệu account trong historical training data, đã biết label fraud/legitimate.
- Dữ liệu gốc là các blockchain transactions: from_address, to_address, value, timestamp, direction...
- Mỗi account là một sample và có một label ở account level.
- Test sample là một account x0 cần dự đoán fraud probability.
- x0 có thể đã xuất hiện trong historical graph hoặc có thể là một account mới/unseen.
- Mục tiêu là tránh rebuild toàn bộ graph 2 triệu node khi inference.

Pipeline mong muốn:

1. RAW TRANSACTIONS
   Transaction records
   (from, to, value, timestamp, ...)
          |
          +-----------------------------+
          |                             |
          v                             v
   GRAPH BRANCH                  ACCOUNT BEHAVIOR BRANCH
          |                             |
          v                             v
   Build temporal/weighted       Aggregate transactions
   transaction graph             belonging to each account
          |                             |
          v                             v
   Account graph G=(V,E)         Temporal transaction features
          |                      (value, in/out, time deltas...)
          |                             |
          v                             v
       GCN/GNN                         BERT
          |                             |
          v                             v
   Account graph embedding      Transaction semantic embedding
       h_GCN(x)                      h_BERT(x)
          |                             |
          +-------------+---------------+
                        |
                        v
                Dynamic Fusion
                        |
                        v
                  Classifier
                        |
                        v
              P(fraud | account)

2. GRAPH CONSTRUCTION
- Node = account/address.
- Edge = transaction relationship between two accounts.
- Edge có thể có weight dựa trên transaction value/frequency/temporal information.
- Nếu dùng GCN 2 layers thì embedding của x0 nên phụ thuộc vào local k-hop neighborhood tương ứng, thay vì giả định rằng GCN trực tiếp nhìn toàn bộ graph.
- Kiểm tra code có đúng directionality, self-loop, normalization và edge weighting hay không.

3. TRAINING
- Training accounts có label.
- Không được để label của validation/test account đi vào node features, edge features hoặc message passing nếu setting inductive.
- Kiểm tra train/validation/test split có leakage hay không.
- Đặc biệt kiểm tra temporal leakage: nếu prediction time là T, model chỉ được dùng transactions/graph information có timestamp <= T.
- Nếu code đang random split 80/10/10 giống paper, hãy chỉ ra rõ rằng đây là transductive/random-node split và có khác biệt với production temporal inference.
- Kiểm tra class imbalance và loss/evaluation metric.

4. INFERENCE CHO x0
Phân biệt rõ hai trường hợp:

CASE A — x0 đã tồn tại trong historical graph:
- Không cần build một graph độc lập cho x0.
- Lấy graph context/k-hop neighborhood của x0 từ historical graph.
- Chạy GNN/GCN để lấy h_GCN(x0).
- Lấy transaction history hợp lệ của x0 rồi chạy BERT để lấy h_BERT(x0).
- Fuse hai embedding rồi predict fraud probability.

CASE B — x0 là unseen/new account:
- Không được rebuild toàn bộ graph 2 triệu node nếu không cần thiết.
- Cần incremental graph update hoặc local inference subgraph:
    G' = G_historical + node(x0) + edges(x0, neighbors)
- Lấy k-hop neighborhood cần thiết cho GNN.
- Chỉ sử dụng information available up to prediction time T.
- Chạy GNN trên local subgraph hoặc cơ chế inductive GraphSAGE/GNN tương đương.
- Chạy BERT trên transaction history của x0.
- Fuse và predict.
- Kiểm tra code có vô tình yêu cầu x0 phải tồn tại trong training node index hay full adjacency matrix hay không.

5. IMPORTANT LEAKAGE CHECK
Hãy đặc biệt kiểm tra các lỗi sau:
- Test account x0 được đưa vào training graph theo cách làm label/message passing bị leakage.
- Neighbor của x0 có label và code vô tình sử dụng label đó làm feature.
- Graph được xây bằng toàn bộ dataset trước khi split, khiến future transactions xuất hiện trong training/prediction context.
- Transaction của x0 xảy ra sau prediction time nhưng vẫn được dùng.
- Aggregated account features được tính từ toàn bộ timeline thay vì chỉ đến cutoff time.
- Normalization/statistics được fit trên cả train + test.
- Node ID/index của test được dùng để tra cứu embedding đã được train từ trước.
- Full-graph GCN inference vô tình cho test node access tới information từ future/test nodes.

6. SCALE CHECK
Tôi có khoảng 2 triệu nodes, nên hãy kiểm tra:
- Có đang materialize dense NxN adjacency matrix hay không.
- Có nên dùng sparse adjacency/edge list hay không.
- GCN full-batch có khả thi không.
- Có cần neighbor sampling / GraphSAGE / mini-batch inference không.
- Có đang rebuild graph toàn bộ mỗi lần có một x0 hay không.
- Memory complexity và inference complexity có hợp lý không.

7. OUTPUT CỦA REVIEW
Hãy trả lời theo format:

A. PIPELINE HIỆN TẠI CỦA CODE
Vẽ lại pipeline thực tế mà code đang thực hiện.

B. PIPELINE MONG MUỐN
Vẽ pipeline:
transactions
→ temporal features
→ graph
→ GNN/GCN
→ BERT
→ dynamic fusion
→ classifier
→ P(fraud|x0)

C. DIFFERENCES
Liệt kê từng điểm code khác pipeline mong muốn.

D. LEAKAGE
Đánh dấu từng leakage risk:
[CRITICAL], [HIGH], [MEDIUM], [OK]

E. INFERENCE x0
Mô tả chính xác code hiện tại sẽ làm gì nếu x0:
1. đã tồn tại trong graph;
2. là unseen/new account.

F. GRAPH CONSTRUCTION
Kiểm tra có cần build graph mới cho x0 không. Nếu không, chỉ rõ phần graph nào nên reuse và phần nào cần incremental update.

G. SCALE
Đánh giá khả năng chạy với 2 triệu nodes và đề xuất thay đổi nếu cần.

H. CODE CHANGES
Đưa ra các thay đổi code cụ thể, ưu tiên patch/minimal changes thay vì viết lại toàn bộ hệ thống.

Quan trọng:
- Không chỉ review model architecture; hãy trace DATA FLOW từ raw transaction → graph/text features → model → prediction.
- Phân biệt rõ transductive inference và inductive/unseen-node inference.
- Nếu code hiện tại không đủ thông tin để kết luận, hãy nói chính xác phần nào chưa thể kết luận thay vì tự giả định.
- Ưu tiên phát hiện data leakage và sai logic inference trước optimization.
- Với mỗi vấn đề, trích dẫn function/class/file/line code cụ thể nếu có thể.