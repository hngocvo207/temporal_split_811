"""Pretrain TxCLM (Eq. 1-6) on all MulDiGraph accounts' transaction sentences."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import TxCLM

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
CLS_ID = 1  # see data_prep.SPECIAL_TOKENS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--limit_steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ART / "txclm.pt"))
    args = ap.parse_args()

    log("loading artifacts ...")
    tx_tokens = np.load(ART / "tx_tokens.npy", mmap_mode="r")
    tx_mask = np.load(ART / "tx_mask.npy", mmap_mode="r")
    with open(ART / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = max(int(k) for k in vocab) + 1
    n, L = tx_tokens.shape
    max_len = L + 1  # + CLS
    log(f"n_accounts={n} seq_len={L} vocab_size={vocab_size}")

    device = args.device
    model = TxCLM(vocab_size, hidden=args.hidden, layers=args.layers, heads=args.heads, max_len=max_len).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    n_steps = int(args.epochs * n / args.batch_size)
    if args.limit_steps:
        n_steps = min(n_steps, args.limit_steps)
    log(f"training for {n_steps} steps (batch_size={args.batch_size})")

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(n_steps):
        idx = rng.integers(0, n, size=args.batch_size)
        idx.sort()  # friendlier for mmap access
        toks = torch.from_numpy(np.ascontiguousarray(tx_tokens[idx])).long()
        mask = torch.from_numpy(np.ascontiguousarray(tx_mask[idx]))

        cls_col = torch.full((toks.size(0), 1), CLS_ID, dtype=torch.long)
        toks = torch.cat([cls_col, toks], dim=1).to(device)
        mask = torch.cat([torch.ones(mask.size(0), 1, dtype=torch.bool), mask], dim=1).to(device)

        loss, mlm_loss, ta_loss = model(toks, mask, mask_id=2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.enhanced.parameters(), 5.0)
        opt.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            ips = (step + 1) * args.batch_size / max(elapsed, 1e-6)
            log(f"step {step}/{n_steps} loss={loss.item():.4f} mlm={mlm_loss.item():.4f} "
                f"ta={ta_loss.item():.4f} ({ips:.0f} seq/s)")
        if step % 2000 == 0 and step > 0:
            torch.save({"enhanced": model.enhanced.state_dict(), "vocab_size": vocab_size,
                        "hidden": args.hidden, "layers": args.layers, "heads": args.heads,
                        "max_len": max_len}, args.out)
            log(f"checkpoint saved -> {args.out}")

    torch.save({"enhanced": model.enhanced.state_dict(), "vocab_size": vocab_size,
                "hidden": args.hidden, "layers": args.layers, "heads": args.heads,
                "max_len": max_len}, args.out)
    log(f"DONE. final model saved -> {args.out}")


if __name__ == "__main__":
    main()
