"""
Smoke test E3: verify wiring rieng cua train_e2_v3.py so voi E1 (da co
smoke_test_stage_e1.py cho train_e2.py) -- 2 diem MOI can kiem tra:
  1. data_prep/e2_v3_train_corpus.py doc dung corpus 100k co san (khong con
     assert n_train==100000 cung nhu v2, tong quat hoa hon).
  2. train_e2_v3.load_pretrained_graph_encoder() nap dung checkpoint tu
     pretrain_graph_encoder.py vao graph_encoder cua build_models() -- dung
     kien truc that (hidden=128, out=D_GRAPH=128) -- roi chay duoc vai buoc
     train_one_epoch KHONG crash (gradient chay xuyen qua trong so da
     warm-start, khong bi dong bang/lech kich thuoc).

KHONG goi build_fixed_val_subsample() (se tokenize 811,704 doc cua
full_test_corpus.py -- ton nhieu phut, danh cho chay that, khong phai smoke
test nhanh) -- chi kiem tra train_one_epoch tren mau train nho (khop tinh
than "vai buoc, khong phai epoch that" cua smoke_test_stage_e1.py).
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from data_prep.e2_v3_train_corpus import load_e2_v3_train_examples
from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.graph_mae import GraphMAE
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.train_e2 import D_GRAPH, build_models, make_train_loader, train_one_epoch
from train_eval.train_e2_v3 import load_pretrained_graph_encoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    t0 = time.time()
    train_examples = load_e2_v3_train_examples(max_examples=200)
    print(f"loaded {len(train_examples)} train examples tu corpus 100k co san ({time.time()-t0:.1f}s)")
    assert len(train_examples) == 200

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

    print("\nchecking --pretrained-graph-encoder warm-start voi checkpoint dung kien truc build_models()...")
    fake_pretrain = GraphMAE(in_channels=23, hidden_channels=128, out_channels=D_GRAPH, num_layers=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke_pretrained.pt"
        torch.save({
            "encoder_state_dict": fake_pretrain.encoder.state_dict(),
            "encoder_config": {"in_channels": 23, "hidden_channels": 128, "out_channels": D_GRAPH,
                                "num_layers": 2, "dropout": 0.2},
            "best_epoch": 0, "best_val_recon_loss": 0.0,
        }, ckpt_path)
        load_pretrained_graph_encoder(graph_encoder, str(ckpt_path))
    graph_encoder = graph_encoder.to(device)

    optimizer = torch.optim.AdamW(list(graph_encoder.parameters()) + list(classifier.parameters()), lr=2e-5)
    train_loader = make_train_loader(train_examples, batch_size=4, pos_neg_ratio=4.0)

    print("\nrunning 3 SMOKE steps voi graph_encoder DA warm-start (not real training)...")
    t0 = time.time()
    loss = train_one_epoch(graph_encoder, classifier, train_loader, sampler, node_features, optimizer, device,
                            max_steps=3)
    print(f"avg loss over 3 steps={loss:.4f} ({time.time()-t0:.1f}s)")
    assert np.isfinite(loss)

    print("\nE3 SMOKE TEST PASSED (corpus loader v3 + warm-start wiring only -- not a training result)")


if __name__ == "__main__":
    main()
