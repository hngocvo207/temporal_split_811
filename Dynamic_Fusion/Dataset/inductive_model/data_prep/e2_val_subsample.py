"""
Val cho E2 v2 (sua loi #2 cua E2 v1, xem STATUS.md): mau con CO DINH (seed
co dinh, khong doi giua cac epoch) rut tu overlap population THAT
(full_test_corpus.py, 201,931 account, 217 duong that, nhan `label_strict`
da verify khop canonical labels.pkl) -- thay cho slice 5k cua Attempt-3 (da
xac nhan chua gan nhu TOAN BO 117 duong that cua val partition, oversample
~43x so voi prevalence that).

Giu toan bo 217 duong that + lay mau ngau nhien am toi tong ~20,000 -- van
thien vi hon quan the that (prevalence mau ~1.09% vs that ~0.11%, vi giu
100% duong that trong khi chi lay ~9.9% so am that) nhung khong cuc doan nhu
slice cu (43x), va du re de eval moi epoch (~20,000/61.6 vi du/s do that =
~5.4 phut, xem STATUS.md).

Full overlap that (201,931) + pure_test (609,773) CHI duoc dung 1 LAN DUY
NHAT, SAU KHI da chon xong checkpoint tu val subsample nay -- xem
eval_full_test.py. Val subsample nay KHONG BAO GIO duoc dung de bao cao ket
qua cuoi cung (chi de chon checkpoint + threshold).
"""
import random
from typing import List

from data_prep.attempt3_corpus import Example
from data_prep.full_test_corpus import load_full_test_examples

VAL_TARGET_N = 20000
VAL_SEED = 44
N_OVERLAP_POS_EXPECTED = 217


def build_fixed_val_subsample(target_n: int = VAL_TARGET_N, seed: int = VAL_SEED) -> List[Example]:
    print("loading full_test_corpus de rut val subsample tu overlap (mot lan, ~811,704 doc)...")
    all_examples = load_full_test_examples()
    overlap = [e for e in all_examples if e.split == "overlap"]
    pos = [e for e in overlap if e.label_strict == 1]
    neg = [e for e in overlap if e.label_strict == 0]
    assert len(pos) == N_OVERLAP_POS_EXPECTED, f"expected {N_OVERLAP_POS_EXPECTED} overlap positives (strict), got {len(pos)}"

    rng = random.Random(seed)
    n_neg = max(0, target_n - len(pos))
    neg_sample = rng.sample(neg, min(n_neg, len(neg)))

    val_subsample = pos + neg_sample
    rng.shuffle(val_subsample)
    # gan split='val' de evaluate()/compute_probs_labels() cua train_e2.py
    # filter dung -- day la mau con dung lam VAL, khong phai overlap that.
    for e in val_subsample:
        e.split = "val"
    prevalence = len(pos) / len(val_subsample) * 100
    print(f"val subsample: n={len(val_subsample)} n_pos={len(pos)} prevalence={prevalence:.2f}% "
          f"(that: overlap 201,931 account, prevalence 0.11%)")
    return val_subsample
