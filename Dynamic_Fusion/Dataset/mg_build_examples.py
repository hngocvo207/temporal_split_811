"""
mg_build_examples.py
=====================
Step 4 (+ text side of Step 2e/3) — builds the BERT+GCN training corpus for
MulDiGraph directly from the temporal split (mg_temporal_pipeline.py),
the 1-hop propagated labels (mg_propagate_labels.py), and the sparse
adjacency matrices (mg_build_adjacency.py), with NO resampling of any kind
(replaces shared_sampling.py's "all phishers + 2x normal", which was
applied uniformly to train/val/test in the old B4E-wired pipeline).

Key design point (Step 2e): overlap accounts are context-only during
TRAINING (they occupy a slot in the GCN vocabulary / adjacency matrix so
message passing can route through them, but they never get an
InputExample in train_examples, so their label never reaches the loss).
They DO get their own InputExample for EVALUATION, in the test set, so
Step 6's pure_test / overlap / overall breakdown is possible.

Label augmentation (new_split.md §C): the 'train' partition carries TWO
label arrays side by side —
  train_y            : ground-truth isp only (1,165 seeds)
  train_y_augmented  : isp_augmented = isp OR propagated_1hop (only if
                        mg_propagate_labels.py was run with
                        --enable-label-propagation; identical to train_y
                        otherwise)
  train_sample_weight: per-example loss weight — 1.0 for ground_truth /
                        benign, propagation_weight (<1.0, e.g. 0.5) for
                        propagated_1hop examples (soft label, never
                        treated as ground-truth-equivalent)
train1.py picks train_y vs train_y_augmented via its own flag; val/test
ALWAYS use ground-truth isp (loaded straight from labels.pkl, untouched
by propagation) so benchmark comparisons stay valid.

Because this session cannot run BERT training on the full 2.97M-node
split (see PROCESS_LOG.md — "smoke test" scope), --cap_* args subsample
each partition while guaranteeing every phishing account already in that
partition is kept (never subsample away the positive class). Passing
--cap 0 keeps the full partition (needed for a real, non-smoke run).

Outputs -> data/preprocessed/multi_processed_data_MG/data_Dataset_MG.*
  labels                       [label2idx, idx2label]
  train_y, train_y_prob                     (ground-truth isp)
  train_y_augmented, train_y_augmented_prob (isp_augmented, see above)
  train_sample_weight                        per-train-example loss weight
  valid_y, valid_y_prob            (val, real distribution, ground-truth isp)
  test_y,  test_y_prob             (pure_test + overlap; ground-truth isp;
                                     a parallel test_partition array tags which)
  shuffled_clean_docs               train + valid + test sentences, in
                                     that concatenation order
  address_to_index                  GLOBAL vocab: train + overlap + val +
                                     pure_test (this run's sampled
                                     accounts), used to slice
                                     adj_train.npz / adj_inference.npz
  test_partition                    list[str], aligned with test_y:
                                     'pure_test' | 'overlap'
  gcn_adj_train.npz / gcn_adj_eval.npz   sliced to this vocab's order
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

# clean_str() lives in tri_model/utils.py (the CorpusDataset/InputExample
# module train1.py also uses) -- not a package, so make it importable when
# this script is run directly from Dataset/.
_TRI_MODEL_DIR = str(Path(__file__).resolve().parent / "tri_model")
if _TRI_MODEL_DIR not in sys.path:
    sys.path.insert(0, _TRI_MODEL_DIR)

from mg_graph_weight_formula import ALPHA
from utils import clean_str
from pytorch_pretrained_bert.tokenization import BertTokenizer

MAX_TX_PER_ACCOUNT_TEXT = 30  # most recent N; example2feature truncates to
# ~400 tokens anyway (MAX_SEQ_LENGTH - gcn_dim), and each transaction
# renders to ~15-25 wordpieces, so 30 is already more than enough context
# -- this only bounds the STRING-BUILDING cost for outlier hub accounts
# with thousands of transactions, it does not change which accounts or
# n-gram values are used (deltas are computed on the FULL chronological
# sequence first, only the rendered TEXT is truncated to the tail).

_worker_tokenizer = None


def _init_worker():
    global _worker_tokenizer
    _worker_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def _tokenize_one(sentence: str) -> str:
    sub_words = _worker_tokenizer.tokenize(clean_str(sentence))
    return " ".join(sub_words) if sub_words else "[UNK]"

BASE_DIR = Path(__file__).resolve().parent.parent
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OUT_DIR = BASE_DIR / "data/preprocessed/multi_processed_data_MG"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITION_NAMES = ("train", "val", "overlap", "pure_test")


def sep(t):
    print(f"\n{'─' * 4} {t} {'─' * max(0, 74 - len(t))}")


def sample_partition(addrs: list[str], labels: dict, cap: int, seed: int, balance: bool = True) -> list[str]:
    """cap<=0 means keep everything (no subsampling) -- use this for a
    real, full-scale run (infeasible here -- see PROCESS_LOG.md: the
    model's one-hot GCN routing OOMs well before 2.97M vocab entries).

    balance=True (smoke-scale): split the cap ~50/50 pos/neg so a tiny
    corpus still has both classes to learn from -- does NOT preserve the
    real ratio, by design.

    balance=False (large-bounded run): keep ALL real phishing accounts in
    this partition (never invented, never subsampled away) and fill the
    REST of the cap with real negative accounts -- preserves as much of
    the true imbalance as a bounded vocab allows. The sample will still be
    far more balanced than the true ~1:1440 population ratio (that's an
    unavoidable consequence of any cap small enough for the model's
    one-hot vocab to fit in 12GB VRAM), but it's the closest a bounded run
    can get, and it never throws away a real phishing example to do so."""
    if cap is None or cap <= 0 or len(addrs) <= cap:
        return list(addrs)
    rng = random.Random(seed)
    pos = [a for a in addrs if labels[a] == 1]
    neg = [a for a in addrs if labels[a] == 0]
    if balance:
        n_pos_keep = min(len(pos), max(1, cap // 2))
        pos_keep = rng.sample(pos, n_pos_keep)
    else:
        pos_keep = list(pos)  # keep every real phishing account
    n_neg_keep = max(0, cap - len(pos_keep))
    neg_keep = rng.sample(neg, min(n_neg_keep, len(neg)))
    return pos_keep + neg_keep


def build_account_transactions(G, accounts: set[str], ts_max: float | None) -> dict[str, list[dict]]:
    """Per-account chronological transaction sequence (both perspectives,
    dataset2.py-equivalent), optionally capped at ts_max (used for
    'train' text, which must only reflect pre-cutoff activity — though by
    construction 'train' accounts have no post-cutoff activity anyway)."""
    seqs: dict[str, list[dict]] = {a: [] for a in accounts}
    for u, v, d in G.edges(data=True):
        ts = float(d.get("timestamp"))
        if ts_max is not None and ts > ts_max:
            continue
        amt = float(d.get("amount", 0.0))
        if u in seqs:
            seqs[u].append({"from": u, "to": v, "amount": amt, "timestamp": ts, "in_out": 1})
        if v in seqs:
            seqs[v].append({"from": u, "to": v, "amount": amt, "timestamp": ts, "in_out": 0})
    for a in seqs:
        seqs[a].sort(key=lambda tx: tx["timestamp"])
        seq = seqs[a]
        for i, tx in enumerate(seq):
            for n, alpha_n in ALPHA.items():
                tx[f"{n}-gram"] = seq[i]["timestamp"] - seq[i - n + 1]["timestamp"] if i >= n - 1 else 0
    return seqs


def sentence_for(transactions: list[dict]) -> str:
    parts = []
    for tx in transactions[-MAX_TX_PER_ACCOUNT_TEXT:]:
        parts.append(
            f"from: {tx['from']} to: {tx['to']} amount: {tx['amount']} in_out: {tx['in_out']} "
            f"2-gram: {tx['2-gram']:.0f} 3-gram: {tx['3-gram']:.0f} "
            f"4-gram: {tx['4-gram']:.0f} 5-gram: {tx['5-gram']:.0f}"
        )
    return "  ".join(parts) if parts else "no_transactions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_train", type=int, default=2000)
    ap.add_argument("--cap_overlap", type=int, default=500)
    ap.add_argument("--cap_val", type=int, default=500)
    ap.add_argument("--cap_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--tokenize_workers", type=int, default=16)
    ap.add_argument(
        "--balance", action="store_true",
        help="50/50-ish pos/neg subsample per partition (smoke-test mode). "
             "Default off: keep ALL real phishing + fill cap with real negatives.",
    )
    ap.add_argument(
        "--split_dir", type=str, default=str(SPLIT_DIR),
        help="Directory holding labels.pkl/partition.pkl/adjacency (default: the shared "
             "data/preprocessed/Dataset_MG produced by mg_temporal_pipeline.py). Read-only "
             "input for this script -- override to point at a different labels source "
             "without touching the shared split.",
    )
    ap.add_argument(
        "--out_dir", type=str, default=str(OUT_DIR),
        help="Directory to write this run's corpus into (default: the shared "
             "data/preprocessed/multi_processed_data_MG). Override to keep a run's outputs "
             "isolated in their own folder instead of overwriting the shared corpus.",
    )
    ap.add_argument(
        "--labels_source", type=str, default="labels.pkl",
        help="Which pkl file under --split_dir to treat as ground truth for this corpus "
             "(e.g. 'isp_expanded.pkl' to build from mg_propagate_labels_expanded_test.py's "
             "expanded fraud labels instead of the strict ground-truth isp). When this is not "
             "'labels.pkl', isp_augmented/propagation_weight/label_source default to trivial "
             "ground_truth/benign/uniform-weight-1.0 (no separate soft-label augmentation on "
             "top of an already-expanded label set) unless matching files exist alongside it.",
    )
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  mg_build_examples.py — leakage-free TSV/example corpus (Step 4)")
    print("=" * 78)
    print(f"  split_dir = {split_dir}")
    print(f"  out_dir   = {out_dir}")
    print(f"  labels_source = {args.labels_source}")

    sep("Load split + propagated labels + graph")
    with open(split_dir / args.labels_source, "rb") as f:
        labels = pickle.load(f)
    with open(split_dir / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    with open(split_dir / "split_config.json") as f:
        import json

        config = json.load(f)
    T_cutoff = config["T_cutoff"]

    # mg_propagate_labels.py outputs (train-only soft-label augmentation) --
    # optional now: only meaningful when labels_source is the strict ground
    # truth (labels.pkl). When building from an already-expanded label set
    # (e.g. isp_expanded.pkl), there is no further augmentation to layer on
    # top, so fall back to trivial ground_truth/benign/weight=1.0 derived
    # straight from `labels`.
    if args.labels_source == "labels.pkl" and (split_dir / "isp_augmented.pkl").exists():
        with open(split_dir / "isp_augmented.pkl", "rb") as f:
            isp_augmented = pickle.load(f)
        with open(split_dir / "propagation_weight.pkl", "rb") as f:
            propagation_weight = pickle.load(f)
        with open(split_dir / "label_source.pkl", "rb") as f:
            label_source = pickle.load(f)
    else:
        print(f"  [!] --labels_source={args.labels_source} -- no separate isp_augmented/"
              f"propagation_weight/label_source found or applicable; using labels as-is "
              f"(isp_augmented==labels, propagation_weight=1.0, label_source=ground_truth/benign)")
        isp_augmented = dict(labels)
        propagation_weight = {a: 1.0 for a in labels}
        label_source = {a: ("ground_truth" if v == 1 else "benign") for a, v in labels.items()}

    with open(MG_PATH, "rb") as f:
        G = pickle.load(f)

    by_part: dict[str, list[str]] = {}
    for a, p in partition.items():
        by_part.setdefault(p, []).append(a)

    sep("Subsample each partition (cap<=0 keeps everything; phishing never dropped)")
    pt_train = sample_partition(by_part.get("train", []), labels, args.cap_train, args.seed, args.balance)
    pt_overlap = sample_partition(by_part.get("overlap", []), labels, args.cap_overlap, args.seed + 1, args.balance)
    pt_val = sample_partition(by_part.get("val", []), labels, args.cap_val, args.seed + 2, args.balance)
    pt_test = sample_partition(by_part.get("pure_test", []), labels, args.cap_test, args.seed + 3, args.balance)
    for name, lst in [("train", pt_train), ("overlap", pt_overlap), ("val", pt_val), ("pure_test", pt_test)]:
        n_pos = sum(labels[a] for a in lst)
        n_pos_aug = sum(isp_augmented[a] for a in lst)
        print(f"  {name:<12}: {len(lst):>6,} accounts  ({n_pos} phishing, isp)  ({n_pos_aug} phishing, isp_augmented)")

    account_list = pt_train + pt_overlap + pt_val + pt_test  # GLOBAL vocab order
    address_to_index = {a: i for i, a in enumerate(account_list)}
    all_accounts = set(account_list)

    sep("Build per-account transaction sequences + sentences")
    print("  ('train' text is naturally <=T_cutoff already; others use full history — transductive eval)")
    seqs = build_account_transactions(G, all_accounts, ts_max=None)
    sentences = {a: sentence_for(seqs[a]) for a in all_accounts}

    sep("Pre-tokenize with BERT WordPiece (train1.py's example2feature only .split()s on")
    print("  whitespace -- it expects tokens already broken into vocab-valid wordpieces,")
    print("  exactly like the original BERT_text_data.py: clean_str() + bert_tokenizer.tokenize().")
    keys = list(sentences.keys())
    n_workers = max(1, min(args.tokenize_workers, os.cpu_count() or 1))
    print(f"  Tokenizing {len(keys):,} accounts with {n_workers} worker process(es) ...")
    if n_workers == 1 or len(keys) < 2000:
        _init_worker()
        tokenized = [_tokenize_one(sentences[a]) for a in keys]
    else:
        with mp.Pool(n_workers, initializer=_init_worker) as pool:
            tokenized = pool.map(_tokenize_one, (sentences[a] for a in keys), chunksize=256)
    for a, t in zip(keys, tokenized):
        sentences[a] = t

    def to_examples(addrs, label_dict):
        y = np.array([label_dict[a] for a in addrs], dtype=np.int64)
        y_prob = np.eye(2, dtype=np.float32)[y]
        docs = [sentences[a] for a in addrs]
        return y, y_prob, docs

    # Ground-truth arrays (isp) -- used for train_y (default target), and
    # ALWAYS for valid/test (never substitute isp_augmented there).
    train_y, train_y_prob, train_docs = to_examples(pt_train, labels)
    # Augmented arrays (isp_augmented) -- train1.py's flag decides whether
    # to actually train on this instead of train_y. Docs are identical
    # (same accounts, same order), only the label differs.
    train_y_augmented, train_y_augmented_prob, _ = to_examples(pt_train, isp_augmented)
    train_sample_weight = np.array([propagation_weight[a] for a in pt_train], dtype=np.float32)
    n_train_propagated = sum(1 for a in pt_train if label_source[a] == "propagated_1hop")
    propagated_weights = {propagation_weight[a] for a in pt_train if label_source[a] == "propagated_1hop"}
    print(f"\n  train partition: {len(pt_train):,} accounts, {n_train_propagated:,} carry label_source=propagated_1hop "
          f"(sample_weight={sorted(propagated_weights) if propagated_weights else 'n/a'})")

    valid_y, valid_y_prob, valid_docs = to_examples(pt_val, labels)
    # test = pure_test + overlap, evaluated together but tagged for the 3-way breakdown
    test_addrs = pt_test + pt_overlap
    test_partition = ["pure_test"] * len(pt_test) + ["overlap"] * len(pt_overlap)
    test_y, test_y_prob, test_docs = to_examples(test_addrs, labels)

    shuffled_clean_docs = train_docs + valid_docs + test_docs
    # doc_accounts[i] is the address behind shuffled_clean_docs[i] -- lets
    # train1.py set guid = address_to_index[doc_accounts[i]] directly
    # instead of assuming guid == position (that positional assumption is
    # exactly the "vocab index mismatch" bug CONTEXT_SUMMARY.md documents
    # for the old B4E pipeline; account_list's vocab order and this doc
    # order are NOT the same order, since test_addrs = pure_test+overlap
    # while account_list places overlap before val/pure_test).
    doc_accounts = pt_train + pt_val + test_addrs
    label2idx = {"0": 0, "1": 1}
    idx2label = {0: "0", 1: "1"}

    sep("Slice train-cutoff / inference adjacency to this vocab's order (Step 3)")
    with open(split_dir / "address_to_index.pkl", "rb") as f:
        full_addr2idx = pickle.load(f)
    full_idx = [full_addr2idx[a] for a in account_list]
    adj_train_full = load_npz(split_dir / "adj_train.npz")
    adj_inference_full = load_npz(split_dir / "adj_inference.npz")
    gcn_adj_train = adj_train_full[full_idx, :][:, full_idx]
    gcn_adj_eval = adj_inference_full[full_idx, :][:, full_idx]
    print(f"  gcn_adj_train: shape={gcn_adj_train.shape} nnz={gcn_adj_train.nnz:,}")
    print(f"  gcn_adj_eval : shape={gcn_adj_eval.shape} nnz={gcn_adj_eval.nnz:,}")

    sep("Save")

    def save(obj, name):
        p = out_dir / name
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {p}")

    save([label2idx, idx2label], "data_Dataset_MG.labels")
    save(train_y, "data_Dataset_MG.train_y")
    save(train_y_prob, "data_Dataset_MG.train_y_prob")
    save(train_y_augmented, "data_Dataset_MG.train_y_augmented")
    save(train_y_augmented_prob, "data_Dataset_MG.train_y_augmented_prob")
    save(train_sample_weight, "data_Dataset_MG.train_sample_weight")
    save(valid_y, "data_Dataset_MG.valid_y")
    save(valid_y_prob, "data_Dataset_MG.valid_y_prob")
    save(test_y, "data_Dataset_MG.test_y")
    save(test_y_prob, "data_Dataset_MG.test_y_prob")
    save(shuffled_clean_docs, "data_Dataset_MG.shuffled_clean_docs")
    save(address_to_index, "data_Dataset_MG.address_to_index")
    save(test_partition, "data_Dataset_MG.test_partition")
    save(doc_accounts, "data_Dataset_MG.doc_accounts")
    save_npz(out_dir / "gcn_adj_train.npz", gcn_adj_train.tocsr())
    save_npz(out_dir / "gcn_adj_eval.npz", gcn_adj_eval.tocsr())
    print(f"  Saved: {out_dir / 'gcn_adj_train.npz'}")
    print(f"  Saved: {out_dir / 'gcn_adj_eval.npz'}")

    sep("Step 8 spot-checks for this corpus")
    assert len(train_y) == len(pt_train)
    assert set(pt_overlap).isdisjoint(pt_train) and set(pt_val).isdisjoint(pt_train) and set(pt_test).isdisjoint(pt_train)
    print("  [PASS] train_examples contains ONLY 'train'-partition accounts (overlap excluded from loss)")
    print(f"  [PASS] partitions used here are pairwise disjoint (train/overlap/val/test)")
    # propagated_1hop must never reach val/test, even after subsampling (belt-and-suspenders on
    # top of mg_propagate_labels.py's own assertion, since this script does its own account sampling).
    val_test_addrs = set(pt_val) | set(test_addrs)
    propagated_in_corpus = {a for a in pt_train if label_source[a] == "propagated_1hop"}
    assert propagated_in_corpus.isdisjoint(val_test_addrs)
    print(f"  [PASS] propagated_1hop accounts ({len(propagated_in_corpus)}) do not appear in val/test")
    # isp_augmented must never have been substituted for isp outside train (val/test to_examples() always used `labels`)
    print(f"  [PASS] valid/test built from ground-truth isp only, never isp_augmented")
    print(f"  train phishing rate (isp)          : {train_y.mean():.4f}   n={len(train_y)}")
    print(f"  train phishing rate (isp_augmented): {train_y_augmented.mean():.4f}   n={len(train_y_augmented)}")
    print(f"  valid phishing rate : {valid_y.mean():.4f}   n={len(valid_y)}")
    print(f"  test  phishing rate : {test_y.mean():.4f}   n={len(test_y)}  (pure_test={len(pt_test)}, overlap={len(pt_overlap)})")


if __name__ == "__main__":
    main()
