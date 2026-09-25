# Phase 1 — Định nghĩa 3 tập dữ liệu (Train_small / Train_large / Test)

Nguồn: `data/preprocessed/Dataset_MG_v3_no_overlap/` (`partition.pkl`, `t_first.pkl`,
`adj_train.npz`, `adj_inference.npz`) — split quantile (t1=65%, t2=80% trên t_first của
1,165 tài khoản phishing xác nhận), xem `Dataset/mg_temporal_pipeline_v3_no_overlap.py`.
Mọi con số dưới đây tính trực tiếp từ dữ liệu trên đĩa, không ước lượng.

## 1. Ba tập

```
Train_small = Train
Train_large = Train ∪ Val
Test        = Test
```

## 2. Timeline

```
Train_small (Train)         Val                      Test
2015-08-07 → 2018-05-18     2018-05-18 → 2018-06-18   2018-06-18 → 2019-01-19
        |                        |                          |
        |<──────── Train_large = Train ∪ Val ────────>|      |
        |                                              |      |
        └──────────────── past ───────────────────────┴───── future ─────┘
                          t1                          t2
                  2018-05-18 11:57:10           2018-06-18 23:35:30
                  (quantile 65% t_first          (quantile 80% t_first
                   của 1,165 phishing)             của 1,165 phishing)
```

## 3. Số node mỗi tập

| Tập | Số node | Định nghĩa |
|---|---:|---|
| **Train_small (Train)** | **2,217,282** | `t_first < t1` |
| **Val** | **268,352** | `t1 ≤ t_first < t2` |
| **Train_large (Train ∪ Val)** | **2,485,634** | `t_first < t2` |
| **Test** | **487,855** | `t_first ≥ t2` |
| Tổng | 2,973,489 | = Train_large + Test (khớp tổng đồ thị) |

## 4. Khoảng thời gian tương ứng

t_first thực tế, tính trên **toàn bộ node** của từng tập (không chỉ phishing):

| Tập | t_first min | t_first max |
|---|---|---|
| **Train_small** | 2015-08-07 05:01:09 UTC | 2018-05-18 11:56:54 UTC |
| **Val** | 2018-05-18 11:57:36 UTC | 2018-06-18 23:34:55 UTC |
| **Train_large** | 2015-08-07 05:01:09 UTC | 2018-06-18 23:34:55 UTC |
| **Test** | 2018-06-18 23:35:38 UTC | 2019-01-19 08:32:09 UTC |

Biên giới `train.max < t1 ≤ val.min` và `val.max < t2 ≤ test.min` — không chồng lấn
thời gian giữa 3 tập (tự kiểm chứng trực tiếp trên dữ liệu).

## 5. Đồ thị cạnh tương ứng cho mỗi tập

| Tập | Đồ thị cạnh nên dùng | Trạng thái |
|---|---|---|
| Train_small | Cạnh `timestamp ≤ t1` | **Đã có** — `data/preprocessed/Dataset_MG_v3_no_overlap/adj_train.npz` (3,859,882 cạnh) |
| Train_large | Cạnh `timestamp ≤ t2` | **Đã build** — `data/preprocessed/Dataset_MG_v3_no_overlap/adj_train_large.npz` (4,430,051 cạnh, 84.22% tổng giao dịch) |
| Test | Cạnh toàn bộ (transductive) | **Đã có** — `adj_inference.npz` (5,355,155 cạnh, không cắt) |

Build bằng `Dataset/mg_build_adj_train_large.py` (sao chép logic + công thức Eq.1-3
của `mg_temporal_pipeline_v3_no_overlap.py`, chỉ đổi `T_cutoff` từ t1 sang t2, không
đụng đến `adj_train.npz`/`adj_inference.npz`/split hiện có). Chạy thật: 61.5s.
Kiểm tra đã PASS: test vẫn bậc=0 tuyệt đối trong `adj_train_large` (0/487,855 —
Train_large không hề "nhìn thấy" Test); val nay CÓ bậc > 0 (253,580/268,352 —
94.5%, khác hẳn `adj_train.npz` cũ nơi val bậc=0 tuyệt đối); `adj_train_large.nnz
(4,430,051) ≥ adj_train.nnz (3,859,882)` đúng như kỳ vọng vì t2 > t1.

## 6. Ma trận cạnh giữa 3 tập (tham khảo, từ phần trực quan hoá đã làm)

Hướng: hàng = tài khoản gửi, cột = tài khoản nhận.

**Trong `adj_train.npz` (cắt tại t1 — dùng cho Train_small)**:

| | → Train | → Val | → Test |
|---|---:|---:|---:|
| Train → | 3,859,882 | 0 | 0 |
| Val → | 0 | 0 | 0 |
| Test → | 0 | 0 | 0 |

100% cạnh là Train–Train — xác nhận Val và Test có bậc = 0 tuyệt đối trong đồ thị
train-time (bất biến inductive nghiêm ngặt nhất của split này).

**Trong `adj_inference.npz` (đầy đủ — dùng cho Test/transductive eval)**:

| | → Train | → Val | → Test |
|---|---:|---:|---:|
| Train → | 4,150,672 | 191,498 | 172,633 |
| Val → | 178,353 | 146,993 | 44,282 |
| Test → | 257,037 | 77,572 | 136,115 |

6 loại cạnh gộp 2 chiều (nội bộ / liên tập):

| Loại | Số cạnh | % tổng (5,355,155) |
|---|---:|---:|
| Train–Train (nội bộ) | 4,150,672 | 77.53% |
| Val–Val (nội bộ) | 146,993 | 2.75% |
| Test–Test (nội bộ) | 136,115 | 2.54% |
| Train–Val (liên tập) | 369,851 | 6.91% |
| Train–Test (liên tập) | 429,670 | 8.03% |
| Val–Test (liên tập) | 121,854 | 2.28% |

Trực quan hoá đầy đủ (heatmap, bar chart, histogram t_first): [Cạnh Train/Val/Test](https://claude.ai/artifact/QRYVemftijVdJR5tMBX6aB).
