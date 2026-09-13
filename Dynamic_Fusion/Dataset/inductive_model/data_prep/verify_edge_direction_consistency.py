"""
Doi chieu quy uoc huong canh giua 2 duong (xem data/graph_data.meta.json field
'edge_direction_convention' cho boi canh day du bug da sua 2026-09-06, sua LAN
2 -- dao nguoc quyet dinh lan 1 theo dung chuan "incoming aggregation" cua HGT
(Hu et al. 2020, Definition 2: target t aggregate tu source s co canh s->t)):

  1. full_graph_forward() qua graph_train.pt/graph_inference.pt's edge_index =
     [coo.row, coo.col] (KHONG doi, giu nguyen ban goc tu adj_csr) -- node v
     aggregate tu {u : edge_index[1]==v, source=edge_index[0]=u}, tuc tu
     PAYER (nguoi da gui tien cho v, vi adj_csr[u,v]!=0 nghia la u gui cho v).
  2. model/graph_sampling.py::_neighbors(adj_csr, center) doc HANG cua center
     trong CSR -- luon la vay, khong doi duoc bang cach doi ten bien. De
     sample_union_subgraph cho ra dung PAYER cua center (khop voi (1)), MOI
     noi goi no (LabelAwareNeighborSampler, build_ego_subgraph,
     build_case_b_subgraph, pretrain_graph_encoder.py) deu TRANSPOSE adj_csr
     truoc khi truyen vao -- hang cua center trong adj_csr.T = cot cua center
     trong adj_csr goc = payer that.

Script nay goi THANG _neighbors() that (khong viet lai logic rieng) tren ma
tran DA TRANSPOSE, doi chieu voi full_graph_forward's edge_index tren 100 node
ngau nhien. Chay lai sau bat ky thay doi nao o 1 trong 2 duong de bug nay
khong tai dien.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep.io_utils import load_graph
from data_prep.labels_io import PREPROC_DIR
from model.graph_sampling import _neighbors


def verify(split: str, n_check: int = 100, seed: int = 44):
    adj_csr = sp.load_npz(PREPROC_DIR / f"adj_{split}.npz").tocsr()
    adj_csr_T = adj_csr.T.tocsr()  # dung HET NHU cach LabelAwareNeighborSampler/build_ego_subgraph transpose
    data = load_graph(split)
    edge_index = data.edge_index.numpy()
    n_nodes = adj_csr.shape[0]

    # index hoa "nguon theo dich" 1 lan cho toan bo edge_index (nhanh hon tra cuu O(E) moi node)
    order = np.argsort(edge_index[1])
    dst_sorted = edge_index[1][order]
    src_sorted = edge_index[0][order]
    starts = np.searchsorted(dst_sorted, np.arange(n_nodes))
    ends = np.searchsorted(dst_sorted, np.arange(n_nodes), side="right")

    rng = np.random.default_rng(seed)
    check_nodes = rng.choice(n_nodes, size=n_check, replace=False)

    mismatches = []
    for v in check_nodes:
        nbr_ids, _ = _neighbors(adj_csr_T, int(v))  # HAM THAT, tren ma tran da transpose -- payer cua v
        expected = set(nbr_ids.tolist())
        actual = set(src_sorted[starts[v]:ends[v]].tolist())  # nguon aggregate cua v qua full_graph_forward
        if expected != actual:
            mismatches.append((int(v), sorted(expected)[:10], sorted(actual)[:10]))

    status = "PASS" if not mismatches else "FAIL"
    print(f"[{split}] {status}: {n_check - len(mismatches)}/{n_check} node khop tuyet doi "
          f"giua sample_union_subgraph (tren ma tran transpose) va full_graph_forward")
    if mismatches:
        for v, exp, act in mismatches[:5]:
            print(f"  MISMATCH node={v} expected(payer that, dau 10)={exp} actual(dau 10)={act}")
    assert not mismatches, f"[{split}] {len(mismatches)}/{n_check} node LECH huong canh -- xem lai build_graph_data.py / cac noi goi sample_union_subgraph"
    return True


if __name__ == "__main__":
    for split in ("train", "inference"):
        verify(split)
    print("\nTAT CA PASS -- 2 duong da khop huong canh (incoming/payer, chuan HGT Definition 2).")
