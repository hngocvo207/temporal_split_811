"""
Bien the tong quat hoa cua e2_train_corpus.py (E2 v2) cho train_e2_v3.py: cho
phep tro toi corpus voi cap_train BAT KY (khong khoa cung n_train==100000 nhu
ban v2), de mo rong corpus finetune vuot qua 100k khi da chay lai

    python -m Dataset.mg_build_examples --cap_train <N> --labels_source labels.pkl

voi N lon hon (toi da 1,945,607 -- toan bo partition 'train' that, xem
data/preprocessed/Dataset_MG/split_stats.json muc "train.total"). Day la nua
thu 2 cua huong da chon khi so sanh LMAE4Eth vs train_e2_v2.py (nua thu nhat
la pretrain khong nhan, xem model/graph_mae.py + train_eval/pretrain_graph_encoder.py).

KHONG sua e2_train_corpus.py (E2 v2) -- giu nguyen file/checkpoint/ket qua v2
da co, dung quy uoc co lap experiment cua du an (moi attempt/version co
output/corpus rieng, xem STATUS.md). Mac dinh corpus_dir van tro ve dung thu
muc 100k co san (runs/inductive_e2_corpus_100k/corpus) nen train_e2_v3.py van
chay duoc NGAY CA KHI chua rebuild corpus lon hon -- mo rong corpus la buoc
TUY CHON, tach biet voi warm-start tu pretrain.

BERT_debug.md Task 1-2 (root-cause fix, xem STATUS.md muc "RA SOAT CODE"):
corpus da doi tu shuffled_clean_docs (text tokenize san, co dia chi vi lap
lai) sang raw_records (Task 1: khong self-address). Ham nay dong bo theo cung
mau voi data_prep/e2_train_corpus.py -- tra ve Example mang `records` (chua
anonymize), input_ids/attention_mask duoc dien vao moi epoch boi script train
(xem data_prep/text_rendering.py::render_examples_inplace).
"""
import pickle
from pathlib import Path
from typing import List

import torch

from data_prep.attempt3_corpus import MAX_SEQ_LEN, Example
from data_prep.labels_io import _load_addr_to_idx

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Dynamic_Fusion
DEFAULT_CORPUS_DIR = REPO_ROOT / "runs" / "inductive_e2_corpus_100k" / "corpus"


def load_e2_v3_train_examples(corpus_dir: Path = DEFAULT_CORPUS_DIR, max_examples: int = None) -> List[Example]:
    def _load_pickle(name: str):
        with open(corpus_dir / f"data_Dataset_MG.{name}", "rb") as f:
            return pickle.load(f, encoding="latin1")

    raw_records = _load_pickle("raw_records")
    doc_accounts = _load_pickle("doc_accounts")
    train_y = _load_pickle("train_y")  # nhan strict (--labels_source labels.pkl luc build corpus)
    n_train = len(train_y)

    global_a2i = _load_addr_to_idx()
    n_use = n_train if max_examples is None else min(max_examples, n_train)

    examples = []
    for i in range(n_use):
        raw_addr = doc_accounts[i]
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
