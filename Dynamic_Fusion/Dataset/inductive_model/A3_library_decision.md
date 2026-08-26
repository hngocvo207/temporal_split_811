# A3 — Chọn thư viện graph: PyTorch Geometric

**Quyết định: PyTorch Geometric (PyG)**, không dùng DGL, dù phần lớn code tham khảo
sạch trong LMAE4Eth (`model/gcn.py`, `model/gat.py`, `model/sage/model.py`,
`sampler analyze/ladies_sampler.py`) được viết bằng DGL.

## Lý do

Môi trường hiện tại: `torch==2.9.1+cu128` (đã cài, dùng cho toàn bộ `tri_model`/`bi_model`).

- `pip index versions dgl` chỉ trả về `0.1.3/0.1.2/0.1.0` trên PyPI — bản DGL thật
  (2.x) không phân phối qua PyPI thường mà cần index riêng (`data.dgl.ai/wheels/...`)
  với wheel build sẵn khớp *chính xác* cặp torch+CUDA. DGL lịch sử luôn chậm theo kịp
  bản torch mới; chưa xác nhận có wheel nào tương thích `torch 2.9.1+cu128` (rất mới).
  Rủi ro: phải build DGL từ source hoặc hạ cấp torch — ảnh hưởng cả `tri_model`/`bi_model`
  đang chạy tốt trên torch hiện tại.
- `pip install torch_geometric` cài sạch, tự resolve về `torch_geometric==2.8.0.post1`
  ngay trên môi trường hiện tại, không cần biên dịch — đã kiểm chứng thực tế (không suy
  đoán), xem log lệnh cài đặt.
- PyG cung cấp đủ mọi thứ mục 1.2 liệt kê cần tái dùng ý tưởng (không phải copy code):
  `SAGEConv`/`GATConv` tương đương `model/gcn.py`/`model/gat.py`, `NeighborLoader`
  tương đương `MultiLayerNeighborSampler` của DGL, và `ladies_sampler.py` chỉ là
  *tham khảo thuật toán* LADIES (sample theo importance theo layer) — thuật toán này
  không phụ thuộc DGL, có thể port sang PyG nếu cần dùng tới (chưa cần ở B1).

## Hệ quả cho A2/B1

- A2 xuất ra `torch_geometric.data.Data` (`x`, `edge_index`, `edge_weight`,
  `num_nodes`), không phải `dgl.graph`.
- B1 dùng `torch_geometric.loader.NeighborLoader` với `num_neighbors=[15, 10]`
  (budget 2 lớp theo đề xuất ở mục 3), encoder viết bằng `SAGEConv`/`GATConv`.
- Nếu sau này cần LADIES sampler thật (ưu tiên thấp, chỉ nếu neighbor sampling
  đơn giản không đủ), port thuật toán từ `sampler analyze/ladies_sampler.py`
  sang thao tác trên `edge_index`/CSR của PyG thay vì kéo cả DGL vào.

## Đã kiểm chứng bằng thực nghiệm (không chỉ theo tài liệu)

```
$ pip index versions dgl
Available versions: 0.1.3, 0.1.2, 0.1.0

$ pip install --dry-run torch_geometric
...
Would install ... torch-geometric-2.8.0.post1 ...

$ pip install torch_geometric   # cài thật, thành công
Successfully installed ... torch_geometric-2.8.0.post1
```
