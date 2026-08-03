"""
adjust_matrix.py
================
Build the weighted transaction adjacency matrix for the GCN.

Key changes vs. the original version
--------------------------------------
1. No independent undersampling.
   The account set now comes from chosen_accounts.pkl produced by
   shared_sampling.py, which is the *same* set used by dataset8.py and
   the subsequent text pipeline.

2. address_to_index is derived from account_list (output of BERT_text_data.py).
   account_list[i] is the Ethereum address of the i-th entry in
   shuffled_clean_docs, so address_to_index[addr] == example.guid for every
   training/validation/test example.  This alignment is required so that the
   GCN adjacency-matrix rows/columns match the example indices that
   CorpusDataset uses in utils.py when it builds gcn_swop_eye.

3. Adjacency matrix is sized [N x N] where N = len(account_list), not the old
   independently-sampled set.

Run order
---------
  1. shared_sampling.py          → chosen_accounts.pkl
  2. dataset8.py … dataset11.py  → TSV files with 'account' column
  3. BERT_text_data.py           → data_Dataset.account_list
  4. THIS script                 → data_Dataset.address_to_index
                                    weighted_adjacency_matrix.pkl
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROC = "/home/ngochv/Dynamic_Feature/data/preprocessed"

ACCOUNT_LIST_PATH = f"{PROC}/multi_processed_data/data_Dataset.account_list"
TRANSACTIONS4_PATH = f"{PROC}/Multigraph/transactions4.pkl"

# Outputs consumed by train1.py
ADDR2IDX_OUT = f"{PROC}/multi_processed_data/data_Dataset.address_to_index"
ADJ_MATRIX_OUT = f"{PROC}/Multigraph/weighted_adjacency_matrix.pkl"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_data(filename: str):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def save_data(data, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as fh:
        pickle.dump(data, fh)


# ---------------------------------------------------------------------------
# Paper-correct α coefficients  (Equation 2)
# ---------------------------------------------------------------------------
# Paper: αₙ = (1/n) / Σⱼ₌₁ᴺ (1/j),  N = 5 (max n-gram order),  n = 2..5
#
#   harmonic_sum = 1 + 1/2 + 1/3 + 1/4 + 1/5 ≈ 2.2833
#
# Resulting values (approx):
#   2-gram: 0.2191   3-gram: 0.1461   4-gram: 0.1096   5-gram: 0.0876

_N_MAX = 5  # paper: N = 5 (maximum n-gram order)
_harmonic_sum: float = sum(1.0 / j for j in range(1, _N_MAX + 1))

ALPHA: Dict[str, float] = {
    f"{n}-gram": (1.0 / n) / _harmonic_sum
    for n in range(2, _N_MAX + 1)   # n is the gram number: 2, 3, 4, 5
}

print("α coefficients (paper Eq. 2, shorter n-gram → larger weight):")
for gram_key, alpha_val in ALPHA.items():
    print(f"  {gram_key}: {alpha_val:.4f}")


# ---------------------------------------------------------------------------
# Edge-weight function  (Equation 3)
# ---------------------------------------------------------------------------


def calculate_weight(transaction: dict) -> float:
    """Return wₖ = valueₖ · Σₙ αₙ · ΔTₙ  (paper Eq. 3).

    Parameters
    ----------
    transaction:
        A single transaction dict containing optional keys
        '2-gram' … '5-gram'  (n-gram temporal gap ΔTₙ, added by dataset4.py)
        and 'value' (ETH amount transferred).

    Returns
    -------
    float
        Scalar edge weight; 0.0 when value is missing/zero.
    """
    value: float = float(transaction.get("value") or 0.0)

    ngram_sum: float = 0.0
    for gram_key, alpha_n in ALPHA.items():
        delta_t = transaction.get(gram_key) or 0
        ngram_sum += alpha_n * float(delta_t)

    return value * ngram_sum


# ---------------------------------------------------------------------------
# Step 1 — Load account_list aligned with shuffled_clean_docs
# ---------------------------------------------------------------------------

print("\nLoading account_list …")
account_list: List[str] = load_data(ACCOUNT_LIST_PATH)
N: int = len(account_list)

# Build address_to_index with lowercase keys for robust hex matching.
# address_to_index[addr] == i  ⟺  account_list[i] == addr  ⟺  example.guid == i
address_to_index: Dict[str, int] = {
    str(addr).lower(): i for i, addr in enumerate(account_list)
}
account_set = set(address_to_index.keys())

print(f"  account_list size : {N}")

# ---------------------------------------------------------------------------
# Step 2 — Load transaction data (transactions4 retains from/to addresses
#           and n-gram gap features added by dataset4.py)
# ---------------------------------------------------------------------------

print("Loading transactions4 …")
transactions4: dict = load_data(TRANSACTIONS4_PATH)

# ---------------------------------------------------------------------------
# Step 3 — Build the N×N weighted adjacency matrix
# ---------------------------------------------------------------------------

print(f"Building {N}×{N} adjacency matrix (paper Eq. 3 weights) …")
adj_matrix = np.zeros((N, N), dtype=np.float32)

edge_count: int = 0
for account, transactions in transactions4.items():
    if str(account).lower() not in account_set:
        continue  # account was removed by shared undersampling

    for tx in transactions:
        from_addr = str(tx.get("from_address", "")).lower()
        to_addr = str(tx.get("to_address", "")).lower()

        if from_addr not in account_set or to_addr not in account_set:
            continue  # one endpoint not in the sampled graph — skip

        fi = address_to_index[from_addr]
        ti = address_to_index[to_addr]

        adj_matrix[fi, ti] += calculate_weight(tx)
        edge_count += 1

print(f"  Transactions processed  : {edge_count}")
print(f"  Non-zero cells in matrix: {np.count_nonzero(adj_matrix)}")

# ---------------------------------------------------------------------------
# Step 4 — Persist outputs
# ---------------------------------------------------------------------------

save_data(address_to_index, ADDR2IDX_OUT)
save_data(adj_matrix, ADJ_MATRIX_OUT)

print(f"\nSaved address_to_index  → {ADDR2IDX_OUT}  ({len(address_to_index)} entries)")
print(f"Saved adjacency matrix  → {ADJ_MATRIX_OUT}  shape={adj_matrix.shape}")

# ---------------------------------------------------------------------------
# Quick sanity check — print the first non-zero edge weight
# ---------------------------------------------------------------------------

print("\nWeight formula spot-check (first non-zero cell):")
found = False
for i in range(N):
    nz_cols = np.nonzero(adj_matrix[i])[0]
    if len(nz_cols):
        j = nz_cols[0]
        src = account_list[i]
        dst = account_list[j]
        print(f"  A[{i},{j}] = {adj_matrix[i, j]:.6f}")
        print(f"    {src}  →  {dst}")
        found = True
        break
if not found:
    print("  WARNING: adjacency matrix is all-zero — check transactions4 path and account overlap.")

print("Done.")
