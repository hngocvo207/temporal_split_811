"""
Learning-curve ablation (theo yeu cau: "train lai voi 25/50/75/100% cua 519
duong, ve AUPRC" de kiem tra gia thuyet "thieu du lieu duong" so voi "van de
dac trung/kien truc"). Chi bien doi DUY NHAT la SO LUONG duong -- am giu CO
DINH (fixed seed, cung 1 tap 20,000 am cho ca 4 lan chay) de co lap dung 1
bien, va giam quy mo corpus (20k thay vi 100k) de 4 lan train kha thi ve thoi
gian (~18 phut/epoch thay vi ~91 phut/epoch, do that o E2 v2).

Positive subset LONG NHAU (nested): 25% subset la con cua 50%, la con cua 75%,
la con cua 100% -- dung 1 hoan vi co dinh (seed) cua 519 duong, lay N dau tien.
Tai su dung corpus 100k da tokenize san cua E2 v2 (data_prep/e2_train_corpus.py)
-- KHONG tokenize lai gi ca, chi loc lai theo Python.
"""
import random
from typing import List

from data_prep.attempt3_corpus import Example
from data_prep.e2_train_corpus import load_e2_train_examples

N_NEG_FIXED = 20000
ABLATION_SEED = 44
POS_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def build_ablation_train_examples(pos_fraction: float, n_neg: int = N_NEG_FIXED,
                                   seed: int = ABLATION_SEED) -> List[Example]:
    assert pos_fraction in POS_FRACTIONS, f"pos_fraction={pos_fraction} phai la 1 trong {POS_FRACTIONS}"

    all_train = load_e2_train_examples()
    pos = [e for e in all_train if e.label == 1]
    neg = [e for e in all_train if e.label == 0]
    assert len(pos) == 519, f"expected 519 duong, got {len(pos)}"

    rng_pos = random.Random(seed)
    pos_shuffled = pos[:]
    rng_pos.shuffle(pos_shuffled)
    n_pos_keep = round(519 * pos_fraction)
    pos_keep = pos_shuffled[:n_pos_keep]

    rng_neg = random.Random(seed + 1)
    neg_shuffled = neg[:]
    rng_neg.shuffle(neg_shuffled)
    neg_keep = neg_shuffled[:n_neg]

    examples = pos_keep + neg_keep
    rng_shuffle = random.Random(seed + 2)
    rng_shuffle.shuffle(examples)
    print(f"ablation corpus pos_fraction={pos_fraction}: n_pos={len(pos_keep)} n_neg={len(neg_keep)} "
          f"n_total={len(examples)}")
    return examples
