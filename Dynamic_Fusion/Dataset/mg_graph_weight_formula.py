"""
mg_graph_weight_formula.py
===========================
Shared Eq. 1-3 (paper Section III.A.2) implementation, extracted so both
the train-cutoff and full-graph adjacency builders use IDENTICAL weight
math. Values match Dataset/adjust_matrix.py's ALPHA/calculate_weight
(already paper-corrected by commits ec70904 / 6be0eca) — duplicated here
in vectorized/pandas form rather than imported, since adjust_matrix.py's
version operates on a single transaction dict (B4E per-account pipeline
shape) and is not vectorizable as-is.

Eq. 1 (n-gram time gap):      DeltaT_n[i] = ts[i] - ts[i-n+1]   (0 if i < n-1)
Eq. 2 (alpha coefficients):   alpha_n = (1/n) / sum_{j=1..N}(1/j),  N = 5, n = 2..5
Eq. 3 (edge weight):          w = amount * sum_n alpha_n * DeltaT_n
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_MAX = 5
_harmonic_sum = sum(1.0 / j for j in range(1, N_MAX + 1))
ALPHA = {n: (1.0 / n) / _harmonic_sum for n in range(2, N_MAX + 1)}  # {2: .., 3: .., 4: .., 5: ..}


def account_perspective_table(edges_df: pd.DataFrame) -> pd.DataFrame:
    """dataset2.py equivalent: duplicate every edge into the sender's AND
    receiver's personal transaction sequence (in_out flag), vectorized.

    edges_df columns: from_addr, to_addr, amount, timestamp
    Returns a 2x-length frame with an extra 'account' column.
    """
    sent = edges_df.assign(account=edges_df["from_addr"], in_out=1)
    recv = edges_df.assign(account=edges_df["to_addr"], in_out=0)
    return pd.concat([sent, recv], ignore_index=True)


def add_ngram_weight(persp_df: pd.DataFrame) -> pd.DataFrame:
    """dataset3.py (sort) + dataset4.py (Eq.1 ngram) + adjust_matrix.py
    (Eq.2-3 weight), vectorized per-account via groupby.

    Adds a 'weight' column: amount * sum_n alpha_n * DeltaT_n, where
    DeltaT_n is computed against the account's OWN chronological
    transaction sequence (each real edge appears twice, once per
    endpoint's perspective, exactly mirroring the original B4E pipeline's
    per-account transaction lists).
    """
    df = persp_df.sort_values(["account", "timestamp"], kind="mergesort").reset_index(drop=True)
    grp = df.groupby("account", sort=False)["timestamp"]

    ngram_sum = np.zeros(len(df), dtype=np.float64)
    for n, alpha_n in ALPHA.items():
        shifted = grp.shift(n - 1)
        delta = (df["timestamp"] - shifted).fillna(0.0).clip(lower=0.0)
        ngram_sum += alpha_n * delta.to_numpy()

    df["weight"] = df["amount"].to_numpy(dtype=np.float64) * ngram_sum
    return df


def build_edge_weights(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: edges -> per-account perspective -> n-gram weight ->
    aggregated (from_addr, to_addr) -> summed weight (both perspectives
    contribute, matching adjust_matrix.py's accumulation loop)."""
    persp = account_perspective_table(edges_df)
    weighted = add_ngram_weight(persp)
    agg = weighted.groupby(["from_addr", "to_addr"], sort=False)["weight"].sum().reset_index()
    return agg
