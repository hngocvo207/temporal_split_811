"""Stage 3: freeze pretrained TxCLM (semantic) + MAGAE (interaction) encoders,
train CAFN (Algorithm 1) for fraud classification on the paper's 7:1:2 split,
evaluate Precision/Recall/F1/Balanced-Accuracy on the held-out test set."""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

from model import TxCLM, TinyBert, MAGAE, SimpleGAT, CAFN, labor_sample, build_padded_block

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
RAW_MG = ROOT.parent / "raw_data" / "MulDiGraph"
CLS_ID = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_semantic_batch(tx_tokens, tx_mask, ids, device):
    toks = torch.from_numpy(np.ascontiguousarray(tx_tokens[ids])).long()
    mask = torch.from_numpy(np.ascontiguousarray(tx_mask[ids]))
    cls_col = torch.full((toks.size(0), 1), CLS_ID, dtype=torch.long)
    toks = torch.cat([cls_col, toks], dim=1).to(device)
    mask = torch.cat([torch.ones(mask.size(0), 1, dtype=torch.bool), mask], dim=1).to(device)
    return toks, mask


def build_interaction_batch(adj, feats_t, ids, fanout, rng, device):
    sampled, _ = labor_sample(adj, ids, fanout, rng)
    block_idx, block_mask = build_padded_block(sampled, ids, fanout)
    return block_idx.to(device), block_mask.to(device)


@torch.no_grad()
def encode_batch(txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj, ids, fanout, rng, device):
    toks, mask = build_semantic_batch(tx_tokens, tx_mask, ids, device)
    semantic_seq = txclm_bert(toks, mask)
    block_idx, block_mask = build_interaction_batch(adj, feats_t, ids, fanout, rng, device)
    interaction_emb = magae_enc(feats_t, block_idx, block_mask)
    return semantic_seq, mask, interaction_emb


@torch.no_grad()
def get_probs(cafn, txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj, node_ids,
              fanout, batch_size, device, rng):
    cafn.eval()
    all_probs = []
    for i in range(0, len(node_ids), batch_size):
        ids = node_ids[i:i + batch_size]
        semantic_seq, mask, interaction_emb = encode_batch(
            txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj, ids, fanout, rng, device)
        logits = cafn(semantic_seq, mask, interaction_emb)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.cpu())
    return torch.cat(all_probs).numpy()


def metrics_at_threshold(probs, y, thr):
    preds = (probs >= thr).astype(np.int64)
    p = precision_score(y, preds, zero_division=0)
    r = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    bacc = balanced_accuracy_score(y, preds)
    return p, r, f1, bacc, int(preds.sum())


def best_threshold(probs, y, grid=np.linspace(0.05, 0.95, 19)):
    """Class-balanced oversampling during training shifts the model's output
    distribution away from the real (very skewed) class prior, so a fixed
    0.5 cutoff is miscalibrated (paper doesn't specify how this is handled).
    We pick the threshold that maximizes F1 on the validation set instead,
    a standard remedy, and apply that same threshold to the test set."""
    best_t, best_f1 = 0.5, -1
    for t in grid:
        _, _, f1, _, _ = metrics_at_threshold(probs, y, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(cafn, txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj, node_ids, labels_t,
             fanout, batch_size, device, rng, thr=0.5):
    probs = get_probs(cafn, txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj, node_ids,
                       fanout, batch_size, device, rng)
    y = labels_t[node_ids].cpu().numpy()
    p, r, f1, bacc, npred = metrics_at_threshold(probs, y, thr)
    return p, r, f1, bacc, npred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--pos_ratio", type=float, default=0.2)
    ap.add_argument("--fanout", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batch_size", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--split", default=str(RAW_MG / "split_idx.pt"))
    ap.add_argument("--txclm_ckpt", default=str(ART / "txclm.pt"))
    ap.add_argument("--magae_ckpt", default=str(ART / "magae.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    log("loading artifacts ...")
    tx_tokens = np.load(ART / "tx_tokens.npy", mmap_mode="r")
    tx_mask = np.load(ART / "tx_mask.npy", mmap_mode="r")
    feats = np.load(ART / "expert_feats.npy")
    adj = sp.load_npz(ART / "adj.npz").tocsr()
    labels = np.load(ART / "labels.npy")
    labels_t = torch.from_numpy(labels).to(device)
    feats_t = torch.from_numpy(feats).to(device)

    split = torch.load(args.split, weights_only=False)
    split_idx = split["split_idx"]
    train_ids_all = split_idx["train"].numpy()
    val_ids = split_idx["valid"].numpy()
    test_ids = split_idx["test"].numpy()
    log(f"split sizes: train={len(train_ids_all)} val={len(val_ids)} test={len(test_ids)}")

    train_pos = train_ids_all[labels[train_ids_all] == 1]
    train_neg = train_ids_all[labels[train_ids_all] == 0]
    log(f"train pos={len(train_pos)} neg={len(train_neg)}")

    ck = torch.load(args.txclm_ckpt, map_location=device, weights_only=False)
    txclm_bert = TinyBert(ck["vocab_size"], ck["hidden"], ck["layers"], ck["heads"], ck["max_len"]).to(device)
    txclm_bert.load_state_dict(ck["enhanced"])
    txclm_bert.eval()
    for p in txclm_bert.parameters():
        p.requires_grad = False

    ck2 = torch.load(args.magae_ckpt, map_location=device, weights_only=False)
    magae_enc = SimpleGAT(ck2["in_dim"], ck2["hidden"], heads=ck2["heads"]).to(device)
    magae_enc.load_state_dict(ck2["encoder"])
    magae_enc.eval()
    for p in magae_enc.parameters():
        p.requires_grad = False

    cafn = CAFN(d_lm=ck["hidden"], d_g=ck2["hidden"]).to(device)
    opt = torch.optim.AdamW(cafn.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    rng = np.random.default_rng(1)
    n_pos = max(1, int(args.batch_size * args.pos_ratio))
    n_neg = args.batch_size - n_pos

    best_val_f1 = -1
    best_state = None
    best_thr = 0.5
    bad_evals = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        cafn.train()
        pos_ids = rng.choice(train_pos, size=n_pos, replace=True)
        neg_ids = rng.choice(train_neg, size=n_neg, replace=False)
        ids = np.concatenate([pos_ids, neg_ids])
        rng.shuffle(ids)

        semantic_seq, mask, interaction_emb = encode_batch(
            txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj, ids, args.fanout, rng, device)
        logits = cafn(semantic_seq, mask, interaction_emb)
        y = labels_t[ids]
        loss = crit(logits, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cafn.parameters(), 5.0)
        opt.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            log(f"step {step}/{args.steps} loss={loss.item():.4f} ({step * args.batch_size / elapsed:.0f} seq/s)")

        if step % args.eval_every == 0:
            val_probs = get_probs(cafn, txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj,
                                   val_ids, args.fanout, args.eval_batch_size, device, rng)
            val_y = labels_t[val_ids].cpu().numpy()
            thr, f1 = best_threshold(val_probs, val_y)
            p, r, _, bacc, npred = metrics_at_threshold(val_probs, val_y, thr)
            log(f"  [val @ step {step}] thr={thr:.2f} P={p:.4f} R={r:.4f} F1={f1:.4f} BAcc={bacc:.4f} n_pred_pos={npred}")
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thr = thr
                best_state = {k: v.detach().clone() for k, v in cafn.state_dict().items()}
                bad_evals = 0
                log(f"  new best val F1={f1:.4f} (thr={thr:.2f})")
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    log("  early stopping (no val F1 improvement)")
                    break

    if best_state is not None:
        cafn.load_state_dict(best_state)
    else:
        best_thr = 0.5
    torch.save({"model": cafn.state_dict(), "threshold": best_thr}, ART / "cafn.pt")

    log(f"evaluating on TEST split (calibrated threshold={best_thr:.2f}) ...")
    p, r, f1, bacc, npred = evaluate(cafn, txclm_bert, magae_enc, tx_tokens, tx_mask, feats_t, adj,
                                      test_ids, labels_t, args.fanout, args.eval_batch_size, device, rng,
                                      thr=best_thr)
    log(f"[TEST] Precision={p:.4f} Recall={r:.4f} F1={f1:.4f} BAcc={bacc:.4f} n_pred_pos={npred} "
        f"n_true_pos={int(labels[test_ids].sum())}")
    log("Paper Table III (MulDiGraph, Ours): Precision=0.9024 Recall=0.8889 F1=0.8960 BAcc=0.9204")


if __name__ == "__main__":
    main()
