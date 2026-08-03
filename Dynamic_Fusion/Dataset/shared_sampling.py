"""
shared_sampling.py
==================
Single coordinated undersampling: sample once from transactions7.pkl.

Result is stored in chosen_accounts.pkl and MUST be used by both:
  - dataset8.py   (text pipeline)
  - adjust_matrix.py (graph adjacency matrix pipeline)

Running these two pipelines against different random samples was the root cause
of the account-set mismatch bug where the GCN adjacency matrix and the BERT
text corpus were built on disjoint sets of accounts.

Usage:
    python shared_sampling.py
    python shared_sampling.py --transactions7 /path/to/transactions7.pkl \
                              --output /path/to/chosen_accounts.pkl \
                              --seed 44
"""

import argparse
import os
import pickle
import random

from env_config import env_config

TRANSACTIONS7_DEFAULT = (
    "/home/ngochv/Dynamic_Feature/data/preprocessed/"
    "b4e_processed_data_1/transactions7.pkl"
)
CHOSEN_ACCOUNTS_DEFAULT = (
    "/home/ngochv/Dynamic_Feature/data/preprocessed/"
    "b4e_processed_data_1/chosen_accounts.pkl"
)


def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def sample_once(transactions7_path: str, output_path: str, seed: int) -> set:
    """
    Load transactions7.pkl, perform undersampling ONCE, and persist the result.

    Strategy: keep ALL phisher accounts + 2× randomly selected normal accounts
    (capped at the available normal pool size).

    Returns the chosen set so callers can use it directly without re-loading.
    """
    random.seed(seed)

    data = _load(transactions7_path)

    phisher = [addr for addr, txs in data.items() if txs[0]["tag"] == 1]
    normal  = [addr for addr, txs in data.items() if txs[0]["tag"] == 0]

    n_select = min(2 * len(phisher), len(normal))
    selected_normal = random.sample(normal, n_select)

    chosen = set(phisher + selected_normal)
    _save(chosen, output_path)

    print("=" * 55)
    print("  Shared Undersampling — result")
    print("=" * 55)
    print(f"  Phisher accounts       : {len(phisher)}")
    print(f"  Normal accounts (2×)   : {len(selected_normal)}")
    print(f"  Total chosen           : {len(chosen)}")
    print(f"  Saved → {output_path}")
    print("=" * 55)
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shared undersampling — run before dataset8.py and adjust_matrix.py"
    )
    parser.add_argument("--transactions7", default=TRANSACTIONS7_DEFAULT,
                        help="Path to transactions7.pkl")
    parser.add_argument("--output", default=CHOSEN_ACCOUNTS_DEFAULT,
                        help="Output path for chosen_accounts.pkl")
    parser.add_argument("--seed", type=int, default=env_config.GLOBAL_SEED,
                        help="Random seed (default: GLOBAL_SEED from .env)")
    args = parser.parse_args()

    sample_once(args.transactions7, args.output, args.seed)
