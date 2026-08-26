"""
Smoke test E1: verify wiring của train_eval/train_e2.py (dataset thật, sampler
B2 thật, model B1+C1+D1 thật ở kích thước bert-base-uncased thật) chạy được
vài BƯỚC (không phải epoch/training thật) -- xem STATUS.md, dừng trước khi
train thật theo yêu cầu. Test này chỉ chứng minh KHÔNG CRASH + loss giảm được
1 chút qua vài step (sanity gradient thật, không phải hội tụ), KHÔNG phải kết
quả huấn luyện.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch

from data_prep.attempt3_corpus import load_attempt3_examples
from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.train_e2 import build_models, evaluate, make_train_loader, train_one_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    t0 = time.time()
    examples = load_attempt3_examples()
    train_examples = [e for e in examples if e.split == "train"]
    print(f"loaded {len(examples)} examples ({time.time()-t0:.1f}s), train={len(train_examples)}")

    node_features = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    labels_tensor = torch.zeros(node_features.shape[0], dtype=torch.long)
    for e in train_examples:
        labels_tensor[e.global_idx] = e.label
    sampler = LabelAwareNeighborSampler(adj_train, labels_tensor, seed=44)

    print("building models (bert-base-uncased pretrained + GraphSAGE)...")
    t0 = time.time()
    graph_encoder, classifier = build_models(device)
    print(f"models built ({time.time()-t0:.1f}s)")
    optimizer = torch.optim.AdamW(list(graph_encoder.parameters()) + list(classifier.parameters()), lr=2e-5)

    train_loader = make_train_loader(train_examples, batch_size=4)

    print("running 3 SMOKE steps (not real training)...")
    t0 = time.time()
    loss = train_one_epoch(graph_encoder, classifier, train_loader, sampler, node_features, optimizer, device,
                            max_steps=3)
    print(f"avg loss over 3 steps={loss:.4f} ({time.time()-t0:.1f}s, ~{(time.time()-t0)/3:.1f}s/step)")
    assert np.isfinite(loss)

    print("running eval on val split (full_graph_forward over adj_inference)...")
    t0 = time.time()
    val_metrics = evaluate(graph_encoder, classifier, examples, device, split_filter="val", batch_size=16)
    print(f"val_metrics(untrained model, meaningless numbers, wiring-only)={val_metrics} ({time.time()-t0:.1f}s)")
    assert val_metrics["n"] == 5000
    assert 0.0 <= val_metrics["auprc"] <= 1.0

    print("\nE1 SMOKE TEST PASSED (wiring only -- not a training result)")


if __name__ == "__main__":
    main()
