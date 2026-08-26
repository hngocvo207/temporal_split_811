"""
mg_build_full_test_eval.py
===========================
Builds an evaluation-only corpus covering the ENTIRE real test partition
(overlap + pure_test, 811,704 accounts), for scoring an already-trained
tri_model checkpoint at full scale -- see
dynamic_fusion_leakage_audit/full_scale_test_eval_report.md.

Why this is a separate script from mg_build_examples.py rather than a new
--cap flag combination: mg_build_examples.py's address_to_index (GCN vocab)
is rebuilt from scratch on every run, in an order controlled by per-partition
random sampling. Simply passing --cap_test 0 --cap_overlap 0 would change
which accounts occupy which vocab index, which breaks the already-trained
checkpoint's VocabGraphConvolution.W0_vh -- that tensor is one learned
embedding ROW PER SPECIFIC ACCOUNT in the training vocab (transductive by
construction, see ETH_GBert.py's VocabGraphConvolution), not a generic
lookup table that tolerates reordering.

This script instead keeps the EXISTING vocab (from the current
multi_processed_data_MG build, whatever tri_model/train1.py was just trained
against) as an exact, unchanged PREFIX of a new, bigger vocab, and appends
every real test account not already in it. The checkpoint's W0_vh[:old_size]
then loads verbatim (see tri_model/eval_full_test.py); the appended rows for
newly-added accounts have no learned GCN signal of their own (random init),
which is a stated caveat of this evaluation, not a bug.

Outputs -> data/preprocessed/multi_processed_data_MG_full_test/data_Dataset_MG_full_test.*
  test_y, test_y_prob        ground-truth isp, ALL 811,704 test accounts
  test_partition              'pure_test' | 'overlap' per test account
  doc_accounts                 address behind each shuffled_clean_docs[i]
  shuffled_clean_docs          tokenized sentences, test accounts only
                                (val is unchanged from the existing build --
                                reused directly by eval_full_test.py)
  address_to_index             extended vocab: old vocab (unchanged order/
                                positions) + new test accounts appended
  old_vocab_size                int, the split point for W0_vh loading
  gcn_adj_eval.npz              full transductive adjacency, sliced to the
                                extended vocab's order
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz

_DATASET_DIR = str(Path(__file__).resolve().parent)
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)

# Reuse, don't duplicate: same transaction-sequence / sentence-rendering /
# tokenizer-worker-pool logic mg_build_examples.py already uses for the
# bounded corpus.
from mg_build_examples import (
    build_account_transactions,
    sentence_for,
    _init_worker,
    _tokenize_one,
)
import multiprocessing as mp
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MG_PATH = BASE_DIR / "raw_data/MulDiGraph/MulDiGraph.pkl"
SPLIT_DIR = BASE_DIR / "data/preprocessed/Dataset_MG"
OLD_OUT_DIR = BASE_DIR / "data/preprocessed/multi_processed_data_MG"
NEW_OUT_DIR = BASE_DIR / "data/preprocessed/multi_processed_data_MG_full_test"
NEW_OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZE_WORKERS = 16


def sep(t):
    print(f"\n{'─' * 4} {t} {'─' * max(0, 74 - len(t))}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--old_corpus_dir", type=str, default=str(OLD_OUT_DIR),
        help="Bounded-run corpus whose address_to_index becomes the unchanged PREFIX of the "
             "extended vocab. MUST be the corpus the checkpoint being evaluated was trained "
             "on -- W0_vh is one learned row per specific account, so pairing a checkpoint "
             "with a different corpus's vocab silently misattributes every learned row.",
    )
    ap.add_argument(
        "--out_dir", type=str, default=str(NEW_OUT_DIR),
        help="Where to write this full-test corpus (default: the shared "
             "multi_processed_data_MG_full_test).",
    )
    ap.add_argument(
        "--labels_source", type=str, default="labels.pkl",
        help="Which pkl under data/preprocessed/Dataset_MG to use as this corpus's test_y "
             "(e.g. 'isp_expanded.pkl'). Strict ground truth (labels.pkl) is ALWAYS saved "
             "alongside as test_y_strict so the same predictions can be scored against both.",
    )
    ap.add_argument(
        "--reuse_docs_from", type=str, default="",
        help="Optional: path to an existing full-test corpus dir whose tokenized "
             "shuffled_clean_docs/doc_accounts cover exactly the same test accounts. Tokenized "
             "text depends only on the account's transactions, never on vocab order or labels, "
             "so reusing it skips the graph load + tokenization (the expensive part) when only "
             "the vocab prefix or label source changed. Account sets are asserted equal.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    old_corpus_dir = Path(args.old_corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  mg_build_full_test_eval.py — full 811,704-account test corpus")
    print("=" * 78)
    print(f"  old_corpus_dir = {old_corpus_dir}")
    print(f"  out_dir        = {out_dir}")
    print(f"  labels_source  = {args.labels_source}")
    print(f"  reuse_docs_from= {args.reuse_docs_from or '(none -- tokenizing from scratch)'}")

    sep("Load existing (bounded-run) vocab -- kept as an exact, unchanged prefix")
    with open(old_corpus_dir / "data_Dataset_MG.address_to_index", "rb") as f:
        old_addr2idx: dict[str, int] = pickle.load(f)
    old_vocab_size = len(old_addr2idx)
    account_list_old = [None] * old_vocab_size
    for a, i in old_addr2idx.items():
        account_list_old[i] = a
    assert all(a is not None for a in account_list_old)
    print(f"  Existing vocab: {old_vocab_size:,} accounts (positions 0..{old_vocab_size - 1} preserved as-is)")

    sep("Load global partition + labels")
    with open(SPLIT_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    with open(SPLIT_DIR / args.labels_source, "rb") as f:
        labels = pickle.load(f)
    # Strict ground truth is always kept alongside, so the same predictions can
    # be scored against both label sets in one pass (see eval_full_test.py).
    with open(SPLIT_DIR / "labels.pkl", "rb") as f:
        labels_strict = pickle.load(f)

    pure_test_all = [a for a, p in partition.items() if p == "pure_test"]
    overlap_all = [a for a, p in partition.items() if p == "overlap"]
    test_addrs = pure_test_all + overlap_all  # fixed order: pure_test then overlap
    test_partition = ["pure_test"] * len(pure_test_all) + ["overlap"] * len(overlap_all)
    print(f"  pure_test: {len(pure_test_all):,}  overlap: {len(overlap_all):,}  total test: {len(test_addrs):,}")
    n_pos = sum(labels[a] for a in test_addrs)
    n_pos_strict = sum(labels_strict[a] for a in test_addrs)
    print(f"  positives in full test set ({args.labels_source}): {n_pos:,}")
    print(f"  positives in full test set (strict labels.pkl):    {n_pos_strict:,}")

    sep("Extend vocab: old vocab (unchanged) + new test accounts appended")
    old_vocab_set = set(old_addr2idx.keys())
    new_accounts = sorted(a for a in test_addrs if a not in old_vocab_set)
    account_list_new = account_list_old + new_accounts
    address_to_index_new = {a: i for i, a in enumerate(account_list_new)}
    print(f"  New test accounts appended: {len(new_accounts):,}")
    print(f"  Extended vocab size: {len(account_list_new):,} "
          f"({old_vocab_size:,} old + {len(new_accounts):,} new)")
    assert len(account_list_new) == old_vocab_size + len(new_accounts)
    assert all(a in address_to_index_new for a in test_addrs), "every test account must resolve to a vocab index"

    if args.reuse_docs_from:
        sep("Reuse already-tokenized sentences (text depends on transactions only, not vocab/labels)")
        reuse_dir = Path(args.reuse_docs_from)
        with open(reuse_dir / "data_Dataset_MG_full_test.doc_accounts", "rb") as f:
            reuse_accounts = pickle.load(f)
        with open(reuse_dir / "data_Dataset_MG_full_test.shuffled_clean_docs", "rb") as f:
            reuse_docs = pickle.load(f)
        assert len(reuse_accounts) == len(reuse_docs)
        assert set(reuse_accounts) == set(test_addrs), (
            f"reuse corpus covers a different account set "
            f"({len(set(reuse_accounts))} vs {len(set(test_addrs))} accounts)"
        )
        sentences = dict(zip(reuse_accounts, reuse_docs))
        print(f"  Reused {len(sentences):,} tokenized account sentences from {reuse_dir}")
    else:
        sep("Build per-account transaction sequences + sentences (test accounts only)")
        print("  (full transactional history -- transductive eval, same as mg_build_examples.py)")
        seqs = build_account_transactions(MG_load(), set(test_addrs), ts_max=None)
        sentences = {a: sentence_for(seqs[a]) for a in test_addrs}
        del seqs

        sep("Pre-tokenize with BERT WordPiece")
        keys = list(sentences.keys())
        n_workers = max(1, min(TOKENIZE_WORKERS, os.cpu_count() or 1))
        print(f"  Tokenizing {len(keys):,} accounts with {n_workers} worker process(es) ...")
        if n_workers == 1 or len(keys) < 2000:
            _init_worker()
            tokenized = [_tokenize_one(sentences[a]) for a in keys]
        else:
            with mp.Pool(n_workers, initializer=_init_worker) as pool:
                tokenized = pool.map(_tokenize_one, (sentences[a] for a in keys), chunksize=512)
        for a, t in zip(keys, tokenized):
            sentences[a] = t

    test_y = np.array([labels[a] for a in test_addrs], dtype=np.int64)
    test_y_prob = np.eye(2, dtype=np.float32)[test_y]
    test_y_strict = np.array([labels_strict[a] for a in test_addrs], dtype=np.int64)
    shuffled_clean_docs = [sentences[a] for a in test_addrs]
    doc_accounts = list(test_addrs)

    sep("Slice full-graph adjacency to the extended vocab's order")
    with open(SPLIT_DIR / "address_to_index.pkl", "rb") as f:
        full_addr2idx = pickle.load(f)
    full_idx = [full_addr2idx[a] for a in account_list_new]
    adj_inference_full = load_npz(SPLIT_DIR / "adj_inference.npz")
    gcn_adj_eval = adj_inference_full[full_idx, :][:, full_idx]
    print(f"  gcn_adj_eval: shape={gcn_adj_eval.shape} nnz={gcn_adj_eval.nnz:,}")
    assert gcn_adj_eval.shape[0] == len(account_list_new)

    sep("Save")

    def save(obj, name):
        p = out_dir / name
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {p}")

    save(test_y, "data_Dataset_MG_full_test.test_y")
    save(test_y_prob, "data_Dataset_MG_full_test.test_y_prob")
    save(test_y_strict, "data_Dataset_MG_full_test.test_y_strict")
    save(test_partition, "data_Dataset_MG_full_test.test_partition")
    save(doc_accounts, "data_Dataset_MG_full_test.doc_accounts")
    save(shuffled_clean_docs, "data_Dataset_MG_full_test.shuffled_clean_docs")
    save(address_to_index_new, "data_Dataset_MG_full_test.address_to_index")
    save(old_vocab_size, "data_Dataset_MG_full_test.old_vocab_size")
    save_npz(out_dir / "gcn_adj_eval.npz", gcn_adj_eval.tocsr())
    print(f"  Saved: {out_dir / 'gcn_adj_eval.npz'}")
    with open(out_dir / "build_stats.json", "w") as f:
        json.dump(
            {
                "labels_source": args.labels_source,
                "old_corpus_dir": str(old_corpus_dir),
                "old_vocab_size": old_vocab_size,
                "new_accounts_appended": len(new_accounts),
                "extended_vocab_size": len(account_list_new),
                "test_n": len(test_addrs),
                "test_pos": int(n_pos),
                "test_pos_strict": int(n_pos_strict),
                "pure_test_n": len(pure_test_all),
                "overlap_n": len(overlap_all),
                "gcn_adj_eval_nnz": int(gcn_adj_eval.nnz),
            },
            f,
            indent=2,
        )

    sep("Spot-checks")
    assert len(test_y) == len(test_addrs) == len(test_partition) == len(doc_accounts) == len(shuffled_clean_docs)
    assert int(test_y.sum()) == n_pos
    assert int(test_y_strict.sum()) == n_pos_strict
    print(f"  [PASS] test_y n={len(test_y):,} pos={int(test_y.sum()):,} ({args.labels_source})")
    print(f"  [PASS] test_y_strict pos={int(test_y_strict.sum()):,} (strict isp, expect 529)")
    print(f"  [PASS] extended vocab size {len(account_list_new):,} == "
          f"old {old_vocab_size:,} + appended {len(new_accounts):,}")


_MG_CACHE = None


def MG_load():
    global _MG_CACHE
    if _MG_CACHE is None:
        print(f"  Loading {MG_PATH} ...")
        with open(MG_PATH, "rb") as f:
            _MG_CACHE = pickle.load(f)
    return _MG_CACHE


if __name__ == "__main__":
    main()
