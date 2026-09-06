"""
E2 v2 (sua 2 loi da phat hien o E2 v1, xem STATUS.md muc "CANH BAO QUAN
TRONG"): corpus train MOI, 100,000 account (toan bo 519 duong that + ~99,481
am that), build boi mg_build_examples.py --cap_train 100000
--labels_source labels.pkl (nhan strict, KHONG phai isp_expanded.pkl nhu
Attempt-3) -- xem runs/inductive_e2_corpus_100k/build.log cho log that.

Chi dung phan TRAIN cua corpus nay -- val/overlap/pure_test build kem chi de
script mg_build_examples.py chay duoc (cap rat nho, khong dung). Val that su
dung overlap population tu full_test_corpus.py (representative hon nhieu,
xem data_prep/e2_val_subsample.py); test cuoi cung dung pure_test cua
full_test_corpus.py, giu nguyen, chi eval 1 lan.
"""
import pickle
from pathlib import Path
from typing import List

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_prep.attempt3_corpus import MAX_SEQ_LEN, Example, _encode
from data_prep.labels_io import _load_addr_to_idx

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "runs" / "inductive_e2_corpus_100k" / "corpus"


def _load_pickle(name: str):
    with open(CORPUS_DIR / f"data_Dataset_MG.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_e2_train_examples(tokenizer: BertTokenizer = None, max_examples: int = None) -> List[Example]:
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    shuffled_clean_docs = _load_pickle("shuffled_clean_docs")
    doc_accounts = _load_pickle("doc_accounts")
    train_y = _load_pickle("train_y")  # strict (labels_source=labels.pkl khi build)
    n_train = len(train_y)
    assert n_train == 100000

    global_a2i = _load_addr_to_idx()
    n_use = n_train if max_examples is None else min(max_examples, n_train)

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
            label=int(train_y[i]),
            label_strict=int(train_y[i]),
            global_idx=global_a2i[addr],
            address=addr,
            split="train",
        ))
    return examples
