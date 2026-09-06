"""
Ve 2 duong (train AUPRC + test AUPRC) theo % duong, tren CUNG 1 POOL AM
CO DINH qua moi diem -- neu khong se dinh dung lai loi "cỡ pool âm khac
nhau lam sap AUPRC" da chan doan o E2 v2 full eval (xem STATUS.md).

Pool co dinh (giong het train_vs_test_matched_pool.py): 150,000 mau dau cua
full_test_corpus (deterministic, luon la pure_test) -> 234 duong / 149,766
am that. Dung CHO CA train-line LAN test-line, CHO CA 4 checkpoint fraction:
  - train_AUPRC(fraction) = AUPRC(fraction's train positives vs fixed_neg)
  - test_AUPRC(fraction)  = AUPRC(fixed_test_pos vs fixed_neg)  [test_pos KHONG doi]

Chi eval checkpoint da co (bo qua fraction chua train xong). Chay lai duoc
nhieu lan, se tu cap nhat khi co checkpoint moi.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch

from data_prep.e2_ablation_corpus import POS_FRACTIONS, build_ablation_train_examples
from data_prep.full_test_corpus import load_full_test_examples
from train_eval.train_e2 import build_models, compute_probs_labels
from train_eval.metrics import compute_metrics

OUTPUT_DIR = Path("output")
FIXED_POOL_N = 150000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

print(f"loading fixed test pool ({FIXED_POOL_N} mau dau full_test_corpus, co dinh qua moi diem)...")
pt_sample = load_full_test_examples(max_examples=FIXED_POOL_N)
assert all(e.split == "pure_test" for e in pt_sample)
fixed_test_pos = [e for e in pt_sample if e.label_strict == 1]
fixed_test_neg = [e for e in pt_sample if e.label_strict == 0]
print(f"fixed pool: n_pos={len(fixed_test_pos)} n_neg={len(fixed_test_neg)} (CO DINH cho moi fraction)")

results = {}
for frac in POS_FRACTIONS:
    tag = f"pos{int(frac * 100)}"
    ckpt_path = OUTPUT_DIR / f"e2_ablation_{tag}_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"[{tag}] checkpoint chua co, bo qua")
        continue

    print(f"\n[{tag}] loading checkpoint {ckpt_path} ...")
    graph_encoder, classifier = build_models(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("pos_fraction") != frac:
        print(f"[{tag}] CANH BAO: checkpoint['pos_fraction']={ckpt.get('pos_fraction')} != {frac} -- "
              f"co the la file cu (smoke test), BO QUA de tranh doc nham")
        continue
    graph_encoder.load_state_dict(ckpt["graph_encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    print(f"[{tag}] checkpoint loaded (best_epoch={ckpt['best_epoch']}, n_pos_train={ckpt.get('n_pos_train')})")

    train_examples_full = build_ablation_train_examples(frac)
    train_pos = [e for e in train_examples_full if e.label == 1]
    print(f"[{tag}] train_pos (fraction nay) = {len(train_pos)}")

    combined = train_pos + fixed_test_neg + fixed_test_pos
    probs, _ = compute_probs_labels(graph_encoder, classifier, combined, device, split_filter=None, batch_size=16)
    n1, n2, n3 = len(train_pos), len(fixed_test_neg), len(fixed_test_pos)
    probs_train_pos, probs_fixed_neg, probs_test_pos = probs[:n1], probs[n1:n1 + n2], probs[n1 + n2:]

    y_true_train = np.concatenate([np.ones(n1), np.zeros(n2)])
    y_prob_train = np.concatenate([probs_train_pos, probs_fixed_neg])
    m_train = compute_metrics(y_true_train, y_prob_train)

    y_true_test = np.concatenate([np.ones(n3), np.zeros(n2)])
    y_prob_test = np.concatenate([probs_test_pos, probs_fixed_neg])
    m_test = compute_metrics(y_true_test, y_prob_test)

    print(f"[{tag}] train_AUPRC={m_train['auprc']:.4f} (n_pos={n1}) | test_AUPRC={m_test['auprc']:.4f} (n_pos={n3}, CO DINH)")
    results[tag] = {"pos_fraction": frac, "n_train_pos": n1, "train_metrics": m_train, "test_metrics": m_test}

with open(OUTPUT_DIR / "e2_ablation_matched_pool_curve.json", "w") as f:
    json.dump({"fixed_pool_n": FIXED_POOL_N, "n_fixed_test_pos": len(fixed_test_pos),
                "n_fixed_test_neg": len(fixed_test_neg), "results": results}, f, indent=2)
print(f"\nSaved -> {OUTPUT_DIR / 'e2_ablation_matched_pool_curve.json'}")
print(f"Fractions co san: {sorted(results.keys())} / {len(POS_FRACTIONS)}")
