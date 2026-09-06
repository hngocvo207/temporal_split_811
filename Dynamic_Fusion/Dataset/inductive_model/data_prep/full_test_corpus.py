"""
Eval-full-pure_test (theo yêu cầu sau E2): dùng corpus 811,704 account
(pure_test 609,773 + overlap 201,931) đã có sẵn tokenized text từ trước
(runs/expanded_test_attempt3/full_test_corpus/) -- không cần build gì thêm,
chỉ dùng để EVAL (không train), thay cho mẫu 5,000/609,773 quá nhỏ ở E2.

QUAN TRỌNG: file `test_y` gốc dùng isp_expanded.pkl (nhãn lan truyền 1-hop,
xem STATUS.md mục "phát hiện phụ" -- rủi ro circular cho nhánh graph). Dùng
`test_y_strict` thay thế (nhãn xác nhận độc lập, isp gốc) -- đã verify khớp
tuyệt đối 312 pure_test + 217 overlap dương, đúng canonical labels.pkl/
partition.pkl.
"""
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_prep.attempt3_corpus import MAX_SEQ_LEN, Example, _encode
from data_prep.labels_io import _load_addr_to_idx

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "runs" / "expanded_test_attempt3" / "full_test_corpus"


def _load_pickle(name: str):
    with open(CORPUS_DIR / f"data_Dataset_MG_full_test.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_full_test_examples(tokenizer: BertTokenizer = None, max_examples: int = None) -> List[Example]:
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    shuffled_clean_docs = _load_pickle("shuffled_clean_docs")
    doc_accounts = _load_pickle("doc_accounts")
    test_y = _load_pickle("test_y")  # propagated (isp_expanded) -- khop nhan corpus Attempt-3 dung cho E2
    test_y_strict = _load_pickle("test_y_strict")  # xac nhan doc lap (labels.pkl goc)
    test_partition = _load_pickle("test_partition")

    n = len(shuffled_clean_docs)
    assert n == len(doc_accounts) == len(test_y) == len(test_y_strict) == len(test_partition) == 811704

    global_a2i = _load_addr_to_idx()

    n_use = n if max_examples is None else min(max_examples, n)
    examples = []
    for i in range(n_use):
        addr = doc_accounts[i].lower()
        if addr not in global_a2i:
            raise ValueError(f"doc {i} address {addr} not in global address_to_index")
        input_ids, attention_mask = _encode(shuffled_clean_docs[i], tokenizer)
        examples.append(Example(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros(MAX_SEQ_LEN, dtype=torch.long),
            label=int(test_y[i]),
            label_strict=int(test_y_strict[i]),
            global_idx=global_a2i[addr],
            address=addr,
            split=test_partition[i],  # 'pure_test' | 'overlap'
        ))
    return examples
