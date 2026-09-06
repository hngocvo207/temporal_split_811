"""
Smoke test cho pretrain khong-nhan (model/graph_mae.py + train_eval/
pretrain_graph_encoder.py) + hop dong checkpoint voi train_e2_v3.py. Khong
phai training that (giong smoke_test_stage_e1.py) -- chi chay VAI BUOC tren
du lieu that (node_features_all23.pt + adj_train.npz that) de xac nhan:
  1. GraphMAE.forward khong crash, loss huu han, tren subgraph sample that
     tu sample_union_subgraph (khong dung full_graph_forward -- xem docstring
     pretrain_graph_encoder.py ve ly do khong du VRAM cho backward).
  2. Checkpoint luu boi pretrain (encoder_state_dict + encoder_config) nap
     duoc dung vao load_pretrained_graph_encoder() cua train_e2_v3.py --
     day la hop dong quan trong nhat giua 2 script (pretrain viet checkpoint,
     v3 doc checkpoint), sai kich thuoc/ten key o day se lam v3 crash khi
     dung --pretrained-graph-encoder.
KHONG goi build_models() (nang, tai BERT-base pretrained that) -- chi can GraphSAGEEncoder
rieng, dung dung kien truc build_models() dung (in_channels=23, out=D_GRAPH)
de test co y nghia.
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.gnn_encoder import GraphSAGEEncoder
from model.graph_mae import GraphMAE
from train_eval.pretrain_graph_encoder import D_GRAPH, eval_recon_loss, pretrain_one_epoch
from train_eval.train_e2_v3 import load_pretrained_graph_encoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    node_features = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    num_nodes = node_features.shape[0]
    print(f"num_nodes={num_nodes} in_channels={node_features.shape[1]}")

    perm = torch.randperm(num_nodes)
    train_seed_pool, val_seed_nodes = perm[:400], perm[400:450]
    fanouts = [15, 10]
    rng = np.random.default_rng(44)

    mae = GraphMAE(
        in_channels=node_features.shape[1], hidden_channels=32, out_channels=D_GRAPH,
        num_layers=2, dropout=0.2, mask_rate=0.5,
    ).to(device)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-3)

    print("running 3 SMOKE pretrain steps (not real training)...")
    t0 = time.time()
    train_loss = pretrain_one_epoch(
        mae, node_features, adj_train, train_seed_pool, batch_size=32, fanouts=fanouts,
        rng=rng, optimizer=optimizer, device=device, max_steps=3,
    )
    print(f"avg recon loss over 3 steps={train_loss:.4f} ({time.time()-t0:.1f}s)")
    assert np.isfinite(train_loss)

    val_loss = eval_recon_loss(mae, node_features, adj_train, val_seed_nodes, fanouts, rng, device)
    print(f"val recon loss (50 held-out seed, wiring-only)={val_loss:.4f}")
    assert np.isfinite(val_loss)

    print("\nchecking checkpoint roundtrip contract voi train_e2_v3.load_pretrained_graph_encoder()...")
    encoder_config = {
        "in_channels": node_features.shape[1], "hidden_channels": 32,
        "out_channels": D_GRAPH, "num_layers": 2, "dropout": 0.2,
    }
    checkpoint = {
        "encoder_state_dict": mae.encoder.state_dict(),
        "encoder_config": encoder_config,
        "best_epoch": 0,
        "best_val_recon_loss": val_loss,
        "args": {},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke_pretrained.pt"
        torch.save(checkpoint, ckpt_path)

        fresh_encoder = GraphSAGEEncoder(in_channels=23, hidden_channels=32, out_channels=D_GRAPH)
        load_pretrained_graph_encoder(fresh_encoder, str(ckpt_path))

        for (name_a, p_a), (name_b, p_b) in zip(
            fresh_encoder.state_dict().items(), mae.encoder.state_dict().items()
        ):
            assert name_a == name_b
            assert torch.equal(p_a.cpu(), p_b.cpu()), f"param {name_a} khong khop sau load_state_dict"
    print("checkpoint roundtrip OK (state_dict khop tuyet doi sau save/load)")

    print("\nALL PRETRAIN SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
