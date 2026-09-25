"""
Builds all tensors needed to train the from-scratch LMAE4Eth reproduction on
MulDiGraph, straight from raw_data/MulDiGraph/{MulDiGraph.pkl, features_output_fullscale.csv}.

Produces (saved under repro/artifacts/):
  - nodes.pkl          : sorted list of addresses (index i <-> address), aligned
                          with Data/split_muldigraph.py's node ordering
  - labels.npy          : int64 [N], 1 = phisher ("isp"/"is_phisher"), 0 = normal
  - expert_feats.npy     : float32 [N, F] expert-engineered node features (Table I),
                          z-score normalized, taken from features_output_fullscale.csv
  - adj.npz             : scipy CSR adjacency, undirected, weight = #transactions
                          between the two accounts (paper: "weight w_uv ... is
                          proportional to the number of transactions")
  - tx_tokens.npy        : int32 [N, 3*K] token ids for each account's transaction
                          sentence (Eq. 2-3): K most recent transactions, 3 tokens
                          each (amount-bucket, direction, timestamp-bucket)
  - tx_mask.npy          : bool [N, 3*K] attention mask (True = real token)
  - vocab.json           : token id -> name, for TxCLM's embedding table

Numeric transaction attributes are discretized into vocabulary tokens (log-scale
amount buckets, global timestamp-quantile buckets, in/out direction) instead of
the WordPiece tokenizer the paper doesn't fully specify -- this keeps the same
"amount / direction / timestamp" linguistic-token structure of Eq. 2 while
staying tractable to train a BERT-style vocab from scratch on ~3M accounts.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data" / "MulDiGraph"
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(exist_ok=True)

K = 32  # max transactions kept per account (most recent)
N_AMOUNT_BUCKETS = 32
N_TIME_BUCKETS = 32

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[MASK]"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_graph():
    log("loading MulDiGraph.pkl ...")
    with open(RAW / "MulDiGraph.pkl", "rb") as f:
        g = pickle.load(f)
    log(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def build_node_table(g):
    addresses = sorted(g.nodes())
    addr2id = {a: i for i, a in enumerate(addresses)}
    labels = np.array([g.nodes[a].get("isp", 0) for a in addresses], dtype=np.int64)
    log(f"nodes={len(addresses)} phishers={labels.sum()}")
    return addresses, addr2id, labels


def build_expert_features(addresses):
    log("loading features_output_fullscale.csv ...")
    df = pd.read_csv(RAW / "features_output_fullscale.csv")
    df = df.drop_duplicates(subset="node").set_index("node")

    # "mg_partition" is a leftover categorical column from an unrelated split scheme
    # (train/val/pure_test/overlap, ratios != 7:1:2) bundled in this CSV -- not used,
    # since we reconstruct the paper's own 7:1:2 random node split separately.
    drop_cols = {"sample_idx", "label", "is_phisher", "mg_partition"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    log(f"expert feature columns ({len(feat_cols)}): {feat_cols}")

    df = df.reindex(addresses)
    n_missing = df[feat_cols[0]].isna().sum()
    if n_missing:
        log(f"WARNING: {n_missing} nodes missing from feature CSV, filling with 0")
    X = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma
    X = np.clip(X, -10, 10)
    return X.astype(np.float32), feat_cols


def build_adjacency(g, addr2id):
    log("collecting edges for adjacency ...")
    n = len(addr2id)
    rows, cols, w = [], [], []
    for u, v in g.edges():
        rows.append(addr2id[u])
        cols.append(addr2id[v])
        w.append(1.0)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    w = np.asarray(w, dtype=np.float32)

    # symmetrize (undirected account-interaction graph) and sum parallel edges -> weight = #tx
    A = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    A = A + A.T
    A = A.tocsr()
    A.sum_duplicates()
    log(f"adjacency: nnz={A.nnz}")
    return A


def build_tx_sentences(g, addr2id, n_nodes):
    log("extracting raw edge records (amount, timestamp) ...")
    src, dst, amt, ts = [], [], [], []
    for u, v, d in g.edges(data=True):
        src.append(addr2id[u])
        dst.append(addr2id[v])
        amt.append(d.get("amount", 0.0))
        ts.append(d.get("timestamp", 0.0))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    amt = np.asarray(amt, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    log(f"edges collected: {len(src)}")

    # each edge u->v = outflow(+1) for u, inflow(-1) for v
    node_id = np.concatenate([src, dst])
    direction = np.concatenate([np.ones_like(src), -np.ones_like(dst)])
    r_amt = np.concatenate([amt, amt])
    r_ts = np.concatenate([ts, ts])
    log(f"total (node,direction) tx records: {len(node_id)}")

    # ---- bucketize amount (log-scale) and timestamp (global quantiles) ----
    log("bucketizing amount / timestamp ...")
    pos_amt = r_amt[r_amt > 0]
    amt_edges = np.quantile(pos_amt, np.linspace(0, 1, N_AMOUNT_BUCKETS - 1)) if len(pos_amt) else np.array([1.0])
    amt_bucket = np.zeros(len(r_amt), dtype=np.int64)
    nz = r_amt > 0
    amt_bucket[nz] = np.digitize(r_amt[nz], amt_edges[1:-1]) + 1  # bucket 0 reserved for amount==0

    ts_valid = r_ts[r_ts > 0]
    ts_edges = np.quantile(ts_valid, np.linspace(0, 1, N_TIME_BUCKETS + 1)[1:-1]) if len(ts_valid) else np.array([])
    ts_bucket = np.digitize(r_ts, ts_edges)

    n_special = len(SPECIAL_TOKENS)
    AMT_OFFSET = n_special
    DIR_OFFSET = AMT_OFFSET + N_AMOUNT_BUCKETS
    TIME_OFFSET = DIR_OFFSET + 2

    amt_tok = AMT_OFFSET + amt_bucket
    dir_tok = DIR_OFFSET + (direction > 0).astype(np.int64)  # 0=out(+1), 1=in(-1)
    time_tok = TIME_OFFSET + ts_bucket

    vocab_size = TIME_OFFSET + N_TIME_BUCKETS
    vocab = {i: SPECIAL_TOKENS[i] for i in range(n_special)}
    for b in range(N_AMOUNT_BUCKETS):
        vocab[AMT_OFFSET + b] = f"amt_{b}"
    vocab[DIR_OFFSET] = "dir_out"
    vocab[DIR_OFFSET + 1] = "dir_in"
    for b in range(N_TIME_BUCKETS):
        vocab[TIME_OFFSET + b] = f"time_{b}"
    log(f"vocab size = {vocab_size}")

    # ---- group per node, keep K most recent by timestamp ----
    log("sorting records by (node, timestamp) ...")
    order = np.lexsort((r_ts, node_id))
    node_sorted = node_id[order]
    amt_tok = amt_tok[order]
    dir_tok = dir_tok[order]
    time_tok = time_tok[order]

    uniq_nodes, start_idx, counts = np.unique(node_sorted, return_index=True, return_counts=True)
    log(f"nodes with >=1 transaction: {len(uniq_nodes)} / {n_nodes}")

    tx_tokens = np.zeros((n_nodes, 3 * K), dtype=np.int32)  # PAD = 0
    tx_mask = np.zeros((n_nodes, 3 * K), dtype=bool)

    log("filling per-account padded token arrays ...")
    t0 = time.time()
    for j, (node, start, cnt) in enumerate(zip(uniq_nodes, start_idx, counts)):
        end = start + cnt
        take = min(cnt, K)
        s = end - take  # keep the most recent `take` transactions
        a = amt_tok[s:end]
        d = dir_tok[s:end]
        t = time_tok[s:end]
        row = np.empty(3 * take, dtype=np.int32)
        row[0::3] = a
        row[1::3] = d
        row[2::3] = t
        tx_tokens[node, :3 * take] = row
        tx_mask[node, :3 * take] = True
        if j % 500000 == 0:
            log(f"  {j}/{len(uniq_nodes)} accounts filled ({time.time() - t0:.1f}s)")

    return tx_tokens, tx_mask, vocab


def main():
    g = load_graph()
    addresses, addr2id, labels = build_node_table(g)
    n = len(addresses)

    with open(OUT / "nodes.pkl", "wb") as f:
        pickle.dump(addresses, f)
    np.save(OUT / "labels.npy", labels)

    expert_feats, feat_cols = build_expert_features(addresses)
    np.save(OUT / "expert_feats.npy", expert_feats)
    with open(OUT / "expert_feat_cols.json", "w") as f:
        json.dump(feat_cols, f)

    A = build_adjacency(g, addr2id)
    sp.save_npz(OUT / "adj.npz", A)

    tx_tokens, tx_mask, vocab = build_tx_sentences(g, addr2id, n)
    np.save(OUT / "tx_tokens.npy", tx_tokens)
    np.save(OUT / "tx_mask.npy", tx_mask)
    with open(OUT / "vocab.json", "w") as f:
        json.dump(vocab, f)

    log("DONE. Artifacts written to " + str(OUT))


if __name__ == "__main__":
    main()
