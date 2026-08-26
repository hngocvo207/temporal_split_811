"""
Task E2 (new_propose.md giai đoạn E): nạp corpus "Attempt-3" đã có sẵn
(runs/expanded_test_attempt3/corpus/) để kiểm chứng kiến trúc mới ở scale
bounded (20k train / 5k val / 5k overlap / 5k pure_test = 35k docs), so trực
tiếp F1(pos)/AUPRC với checkpoint Attempt-3 gốc (F1(pos)=87.63%, AUPRC=0.9155
overall -- xem STATUS.md).

QUAN TRỌNG -- khác 1 điểm có chủ đích so với cách tri_model dùng corpus này:
corpus có `gcn_adj_train.npz`/`gcn_adj_eval.npz` riêng, nhưng đó là bản SLICE
CSR chỉ 35,000x35,000 (chỉ các cạnh giữa 35k account này với nhau) -- dùng nó
cho GraphSAGE sẽ cắt cụt receptive field ở biên corpus, mất mọi hàng xóm thật
ngoài 35k account này. Pipeline mới KHÔNG dùng gcn_adj_*.npz của corpus --
`doc_accounts` (địa chỉ 0x... thật) được resolve sang global address_to_index
(đã verify: toàn bộ 35,000 địa chỉ đều có trong global index, 0 miss) rồi lấy
neighbor từ ĐÚNG đồ thị full-scale (adj_train.npz/adj_inference.npz, A2) --
đúng tinh thần kiến trúc inductive (B1), không tái tạo giới hạn vocab cũ.

Text branch: dùng nguyên `shuffled_clean_docs` (đã pre-tokenize WordPiece, xem
mg_build_examples.py) + tokenizer bert-base-uncased giống tri_model, front-
truncate về MAX_SEQ_LEN -- không còn dành chỗ cho gcn_embedding_dim SEP token
như tri_model/utils.py::example2feature (không còn GCN token injection sau C1).
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_prep.labels_io import _load_addr_to_idx

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Dynamic_Fusion
CORPUS_DIR = REPO_ROOT / "runs" / "expanded_test_attempt3" / "corpus"
MAX_SEQ_LEN = 400  # khớp tri_model (400 + gcn_embedding_dim(16) = 416 ở bản cũ; bỏ +16 vì không còn GCN slot)


@dataclass
class Example:
    input_ids: torch.Tensor      # [MAX_SEQ_LEN], padded
    attention_mask: torch.Tensor  # [MAX_SEQ_LEN]
    token_type_ids: torch.Tensor  # [MAX_SEQ_LEN], toàn 0
    label: int
    global_idx: int  # index trong address_to_index.pkl / node_features_all23.pt / graph_*.pt
    address: str
    split: str  # 'train' | 'val' | 'overlap' | 'pure_test'


def _load_pickle(name: str):
    with open(CORPUS_DIR / f"data_Dataset_MG.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")


def _encode(doc: str, tokenizer: BertTokenizer) -> (torch.Tensor, torch.Tensor):
    tokens_a = doc.split()
    if len(tokens_a) > MAX_SEQ_LEN - 2:
        tokens_a = tokens_a[: MAX_SEQ_LEN - 2]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    n = len(ids)
    pad = MAX_SEQ_LEN - n
    input_ids = torch.tensor(ids + [0] * pad, dtype=torch.long)
    attention_mask = torch.tensor([1] * n + [0] * pad, dtype=torch.long)
    return input_ids, attention_mask


def load_attempt3_examples(tokenizer: BertTokenizer = None, max_examples: int = None) -> List[Example]:
    """Trả về list Example cho cả 35,000 doc (train 20k, valid 5k, overlap+pure_test
    5k+5k -- test_partition phân biệt 2 tập con trong test_y)."""
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    shuffled_clean_docs = _load_pickle("shuffled_clean_docs")
    doc_accounts = _load_pickle("doc_accounts")
    train_y = _load_pickle("train_y")
    valid_y = _load_pickle("valid_y")
    test_y = _load_pickle("test_y")
    test_partition = _load_pickle("test_partition")

    n_train, n_valid, n_test = len(train_y), len(valid_y), len(test_y)
    assert len(shuffled_clean_docs) == n_train + n_valid + n_test == len(doc_accounts)

    splits = (
        ["train"] * n_train
        + ["val"] * n_valid
        + list(test_partition)  # 'pure_test' hoặc 'overlap', len == n_test
    )
    labels = np.concatenate([train_y, valid_y, test_y]).astype(int)

    global_a2i = _load_addr_to_idx()

    examples = []
    n = len(shuffled_clean_docs) if max_examples is None else min(max_examples, len(shuffled_clean_docs))
    for i in range(n):
        addr = doc_accounts[i].lower()
        if addr not in global_a2i:
            raise ValueError(f"doc {i} address {addr} not in global address_to_index -- corpus/global mismatch")
        input_ids, attention_mask = _encode(shuffled_clean_docs[i], tokenizer)
        examples.append(Example(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros(MAX_SEQ_LEN, dtype=torch.long),
            label=int(labels[i]),
            global_idx=global_a2i[addr],
            address=addr,
            split=splits[i],
        ))
    return examples
