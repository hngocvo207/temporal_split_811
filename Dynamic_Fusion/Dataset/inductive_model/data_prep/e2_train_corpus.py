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

BERT_debug.md Task 1-2 (root-cause fix, xem STATUS.md muc "RA SOAT CODE"):
Example tra ve o day KHONG con input_ids co san -- chi mang theo `records`
(raw per-tx, chua anonymize). input_ids duoc dien vao moi epoch boi
train_eval/train_e2_v2.py (goi data_prep.text_rendering.render_and_encode_corpus
voi 1 seed rieng/epoch) de dia chi doi tac duoc anonymize LAI voi 1 phep gan
ID khac moi epoch, khong dong cung 1 chuoi token co dinh cho model hoc thuoc.
"""
import pickle
from pathlib import Path
from typing import List

import torch

from data_prep.attempt3_corpus import MAX_SEQ_LEN, Example
from data_prep.labels_io import _load_addr_to_idx

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "runs" / "inductive_e2_corpus_100k" / "corpus"


def _load_pickle(name: str):
    with open(CORPUS_DIR / f"data_Dataset_MG.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_e2_train_examples(max_examples: int = None) -> List[Example]:
    raw_records = _load_pickle("raw_records")  # dict address -> list[dict], xem mg_build_examples.py
    doc_accounts = _load_pickle("doc_accounts")
    train_y = _load_pickle("train_y")  # strict (labels_source=labels.pkl khi build)
    n_train = len(train_y)
    assert n_train == 100000

    global_a2i = _load_addr_to_idx()
    n_use = n_train if max_examples is None else min(max_examples, n_train)

    examples = []
    for i in range(n_use):
        raw_addr = doc_accounts[i]  # casing goc -- khoa cua raw_records dict (xem mg_build_examples.py)
        addr = raw_addr.lower()
        if addr not in global_a2i:
            raise ValueError(f"doc {i} address {addr} not in global address_to_index")
        examples.append(Example(
            input_ids=None,
            attention_mask=None,
            token_type_ids=torch.zeros(MAX_SEQ_LEN, dtype=torch.long),
            label=int(train_y[i]),
            label_strict=int(train_y[i]),
            global_idx=global_a2i[addr],
            address=addr,
            split="train",
            records=raw_records[raw_addr],
        ))
    return examples
