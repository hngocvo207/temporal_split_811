"""
Pretrain khong-nhan cho GraphSAGEEncoder (B1), theo huong da chon khi so sanh
LMAE4Eth (MAGAE: masked graph autoencoder, pretrain tren TOAN BO node, khong
can nhan) voi train_e2_v2.py (GraphSAGEEncoder khoi tao ngau nhien, chi hoc
tu 519 mau duong xac thuc). Ly do cu the (khong phai suy doan): STATUS.md muc
"Thi nghiem toi gian" da do that -- tune capacity nhanh graph rieng le (hidden
64->256, Linear/MLP) KHONG nang duoc tran AUPRC ~0.01-0.02, ket luan la "519
mau duong xac thuc la qua it cho rieng nhanh graph tu hoc du manh TU DAU",
khong phai van de model capacity. Pretrain masked-feature-reconstruction
(model/graph_mae.py) khong dung nhan nen dung duoc CA 2,973,489 node cua
graph_train.pt -- tang ~2500 lan luong tin hieu khong-nhan so voi 519 mau
duong, truoc khi warm-start vao train_e2_v3.py (finetune co nhan, dung lai
LabelAwareNeighborSampler cua B2).

QUAN TRONG ve kha thi (xem model/gnn_encoder.py docstring + smoke_test_stage_b.py
da do that): full_graph_forward (torch.no_grad(), 1 lan duy nhat) dung 8.0GB/
0.3s tren graph_train.pt that -- nhung do la INFERENCE THUAN, khong co
activation cache cho backward. KHONG dung full_graph_forward o day (se OOM khi
co grad) -- van phai di qua sample_union_subgraph (model/graph_sampling.py)
theo minibatch giong B2, chi khac 2 diem:
  1. KHONG label-aware (moi node binh dang, khong phan biet duong/am -- day
     la pretrain khong nhan, budget hang xom co dinh moi tang).
  2. Seed moi buoc lay tu TOAN BO 2,973,489 node (khong loc theo partition/
     nhan/co-doc-BERT), khac han train_e2*.py chi lay seed tu account co san
     doc BERT + nhan.

CHUA CHAY THAT (chi viet code + smoke test, xem STATUS.md muc pretrain graph
encoder o cuoi file) -- pure-Python neighbor loop trong sample_union_subgraph
(khong vector-hoa) la nut that toc do da biet (train_e2.py dung batch_size=8
seed/buoc cho nhanh fusion vi ly do nay), nen duyet het 2,973,489 node/epoch
se CHAM. Dung --max-steps-per-epoch de smoke-test truoc khi chay that nhieu
epoch/qua dem (khuyen nghi chay tren remote-gpu qua tmux giong E2, xem
STATUS.md).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import wandb

from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.graph_mae import GraphMAE
from model.graph_sampling import sample_union_subgraph, subgraph_to_data

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"
GLOBAL_SEED = 44
D_GRAPH = 128  # khop D_GRAPH cua train_e2.py/train_e2_v2.py de checkpoint nap thang vao build_models()


def _uniform_budget_fn(fanouts):
    """Budget hang xom co dinh theo tang, KHONG phan biet nhan (khac
    LabelAwareNeighborSampler cua B2) -- moi node binh dang vi day la pretrain
    khong nhan. allow_oversample=False: khong lap lai hang xom (khac budget
    duong cua B2), giu dung phan phoi bac that cua do thi."""
    def fn(center, hop):
        return fanouts[hop], False
    return fn


def pretrain_one_epoch(mae, node_features, adj_csr, seed_perm, batch_size, fanouts, rng, optimizer, device,
                        max_steps=None):
    mae.train()
    budget_fn = _uniform_budget_fn(fanouts)
    total_loss, n_steps = 0.0, 0
    num_batches = (seed_perm.numel() + batch_size - 1) // batch_size
    for b in range(num_batches):
        if max_steps is not None and b >= max_steps:
            break
        seed_batch = seed_perm[b * batch_size: (b + 1) * batch_size]
        global_ids, edge_index, edge_weight, _ = sample_union_subgraph(
            adj_csr, seed_batch.tolist(), len(fanouts), budget_fn, rng
        )
        data = subgraph_to_data(global_ids, edge_index, edge_weight, node_features).to(device)
        loss = mae(data.x, data.edge_index, data.edge_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1
        if n_steps % 50 == 0:
            print(f"  step={n_steps}/{num_batches} loss={loss.item():.4f} subgraph_nodes={data.num_nodes}")
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def eval_recon_loss(mae, node_features, adj_csr, seed_nodes, fanouts, rng, device):
    """Recon loss tren 1 mau seed CO DINH (khong doi giua cac epoch) -- dong
    vai tro nhu 'val' de chon checkpoint tot nhat. Khong can nhan nen khong co
    rui ro ro ri nhan kieu E2 v1 -- day chi la 1 tap con node giu rieng ra
    khong dua vao train_seed_pool, khong lien quan gi partition train/val/
    test cua nhan (labels.pkl)."""
    mae.eval()
    budget_fn = _uniform_budget_fn(fanouts)
    global_ids, edge_index, edge_weight, _ = sample_union_subgraph(
        adj_csr, seed_nodes.tolist(), len(fanouts), budget_fn, rng
    )
    data = subgraph_to_data(global_ids, edge_index, edge_weight, node_features).to(device)
    loss = mae(data.x, data.edge_index, data.edge_weight)
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512,
                     help="so seed node (do thi) moi buoc -- KHAC batch-size cua train_e2*.py (do la so vi du BERT)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None,
                     help="gioi han so buoc/epoch, dung de smoke-test truoc khi chay full ~2.97M node")
    ap.add_argument("--fan-out", type=str, default="15,10",
                     help="budget hang xom moi tang, khop receptive field mac dinh cua GraphSAGEEncoder")
    ap.add_argument("--hidden-channels", type=int, default=128)
    ap.add_argument("--out-channels", type=int, default=D_GRAPH)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mask-rate", type=float, default=0.5)
    ap.add_argument("--sce-alpha", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-size", type=int, default=2000,
                     help="so seed node giu rieng (khong phai nhan) de theo doi recon loss/chon checkpoint")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group="e2_v3_graph_pretrain", name="graph_mae_pretrain", config=vars(args)
    )

    print("loading full node features + adj_train (toan bo graph_train.pt, khong loc nhan/partition)...")
    node_features = load_node_features()
    # Transpose -- de _neighbors() trong sample_union_subgraph lay dung PAYER
    # (nguoi da gui tien cho center), khop quy uoc "incoming" cua
    # full_graph_forward/graph_train.pt -- xem data/graph_data.meta.json field
    # 'edge_direction_convention' va model/label_aware_sampler.py.
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz").T.tocsr()
    num_nodes = node_features.shape[0]
    print(f"num_nodes={num_nodes} in_channels={node_features.shape[1]}")

    fanouts = [int(_) for _ in args.fan_out.split(",")]
    assert len(fanouts) == args.num_layers, "fan-out phai co dung num-layers gia tri"

    rng_np = np.random.default_rng(GLOBAL_SEED)
    perm_generator = torch.Generator().manual_seed(GLOBAL_SEED)
    full_perm = torch.randperm(num_nodes, generator=perm_generator)
    val_seed_nodes, train_seed_pool = full_perm[:args.val_size], full_perm[args.val_size:]
    print(f"val_seed_nodes (giu rieng, khong phai nhan)={val_seed_nodes.numel()} "
          f"train_seed_pool={train_seed_pool.numel()}")

    mae = GraphMAE(
        in_channels=node_features.shape[1], hidden_channels=args.hidden_channels, out_channels=args.out_channels,
        num_layers=args.num_layers, dropout=args.dropout, mask_rate=args.mask_rate, alpha=args.sce_alpha,
    ).to(device)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_perm = train_seed_pool[torch.randperm(train_seed_pool.numel())]
        train_loss = pretrain_one_epoch(
            mae, node_features, adj_train, epoch_perm, args.batch_size, fanouts, rng_np, optimizer, device,
            max_steps=args.max_steps_per_epoch,
        )
        val_loss = eval_recon_loss(mae, node_features, adj_train, val_seed_nodes, fanouts, rng_np, device)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in mae.encoder.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(f"epoch={epoch} train_recon_loss={train_loss:.4f} val_recon_loss={val_loss:.4f} time={dt:.1f}s"
              f"{' *best*' if improved else ''}")
        history.append({"epoch": epoch, "train_recon_loss": train_loss, "val_recon_loss": val_loss, "time_s": dt})
        if run is not None:
            run.log({"train_recon_loss": train_loss, "val_recon_loss": val_loss, "epoch_time_s": dt,
                      "best_val_recon_loss_so_far": best_val_loss}, step=epoch)

        if args.patience and epochs_since_improve >= args.patience:
            print(f"early stop at epoch={epoch} (khong cai thien val recon loss sau {args.patience} epoch)")
            break

    if best_state is None:
        raise RuntimeError("khong epoch nao chay xong (epochs=0?) -- khong co checkpoint de luu")

    encoder_config = {
        "in_channels": node_features.shape[1], "hidden_channels": args.hidden_channels,
        "out_channels": args.out_channels, "num_layers": args.num_layers, "dropout": args.dropout,
    }
    checkpoint = {
        "encoder_state_dict": best_state,
        "encoder_config": encoder_config,
        "best_epoch": best_epoch,
        "best_val_recon_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, OUTPUT_DIR / "graph_encoder_pretrained.pt")
    print(f"\n=== pretrain checkpoint saved -> output/graph_encoder_pretrained.pt "
          f"(best_epoch={best_epoch}, val_recon_loss={best_val_loss:.4f}) ===")
    print("Dung checkpoint nay bang: train_eval/train_e2_v3.py "
          "--pretrained-graph-encoder output/graph_encoder_pretrained.pt")

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_recon_loss"] = best_val_loss
        run.finish()

    with open(OUTPUT_DIR / "graph_pretrain_history.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_recon_loss": best_val_loss,
                    "args": vars(args)}, f, indent=2)
    print(f"Saved -> {OUTPUT_DIR / 'graph_pretrain_history.json'}")


if __name__ == "__main__":
    main()
