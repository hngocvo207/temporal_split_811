"""Pretrain MAGAE (Eq. 7-16) on the MulDiGraph account-interaction graph
using LABOR sampling mini-batches."""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from model import MAGAE, labor_sample, build_padded_block

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--fanout", type=int, default=10)
    ap.add_argument("--mask_rate", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_steps", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ART / "magae.pt"))
    args = ap.parse_args()

    log("loading artifacts ...")
    feats = np.load(ART / "expert_feats.npy")
    n, in_dim = feats.shape
    adj = sp.load_npz(ART / "adj.npz").tocsr()
    log(f"n_nodes={n} in_dim={in_dim} nnz={adj.nnz}")

    device = args.device
    feats_t = torch.from_numpy(feats).to(device)

    model = MAGAE(in_dim, args.hidden, heads=args.heads, mask_rate=args.mask_rate).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_steps = int(args.epochs * n / args.batch_size)
    if args.limit_steps:
        n_steps = min(n_steps, args.limit_steps)
    log(f"training for {n_steps} steps (batch_size={args.batch_size}, fanout={args.fanout})")

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(n_steps):
        batch_nodes = rng.choice(n, size=args.batch_size, replace=False)
        sampled, _ = labor_sample(adj, batch_nodes, args.fanout, rng)
        block_idx, block_mask = build_padded_block(sampled, batch_nodes, args.fanout)
        block_idx = block_idx.to(device)
        block_mask = block_mask.to(device)

        mask_bool = torch.zeros(n, dtype=torch.bool, device=device)
        universe = torch.unique(block_idx)
        n_mask = int(args.mask_rate * universe.numel())
        mask_sel = universe[torch.randperm(universe.numel(), device=device)[:n_mask]]
        mask_bool[mask_sel] = True

        loss = model(feats_t, block_idx, block_mask, mask_bool)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 50 == 0:
            elapsed = time.time() - t0
            ips = (step + 1) * args.batch_size / max(elapsed, 1e-6)
            log(f"step {step}/{n_steps} loss={loss.item():.4f} ({ips:.0f} nodes/s)")
        if step % 500 == 0 and step > 0:
            torch.save({"encoder": model.encoder.state_dict(), "in_dim": in_dim,
                        "hidden": args.hidden, "heads": args.heads}, args.out)
            log(f"checkpoint saved -> {args.out}")

    torch.save({"encoder": model.encoder.state_dict(), "in_dim": in_dim,
                "hidden": args.hidden, "heads": args.heads}, args.out)
    log(f"DONE. final model saved -> {args.out}")


if __name__ == "__main__":
    main()
