# Project Context Summary
## Dynamic Feature Fusion — Blockchain Phishing Detection

---

## 1. Project Overview

**Paper:** *Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Fraud Detection* (arXiv:2501.02032, 2025)

**Goal:** Binary classification — detect phishing (fraud) Ethereum accounts.

**Model:** `ETH_GBertModel` — a 3-stream architecture fusing:
1. **BERT embeddings** — standard token embeddings from `bert-base-uncased`
2. **GCN-enhanced embeddings** — vocab-graph convolution (`VocabGraphConvolution`) over a transaction adjacency matrix
3. **Graph feature embeddings** — top-10 Spearman-selected blockchain graph features projected into BERT hidden space via `FeatureProjector`

A **4-gate `DynamicFusionLayer`** (DiffSoftmax) combines the three streams plus a learned mixed stream → BERT Encoder → Linear classifier.

---

## 2. Repository Structure

```
Dynamic_Fusion/
├── Dataset/
│   ├── shared_sampling.py      ← NEW: one-time coordinated undersampling
│   ├── dataset0.py             ← raw CSV → NetworkX MultiDiGraph
│   ├── dataset1.py             ← full graph with EOA/Contract detection
│   ├── dataset2.py             ← group txs by account (from/to), add in_out flag
│   ├── dataset3.py             ← sort by timestamp
│   ├── dataset4.py             ← add n-gram (2–5) temporal gap features
│   ├── dataset5.py             ← FIXED: only remove timestamp (keep from/to address)
│   ├── dataset6.py             ← shuffle tx order within each account
│   ├── dataset7.py             ← merge all txs per account, keep tag on first
│   ├── dataset8.py             ← FIXED: filter to chosen_accounts.pkl (no re-sampling)
│   ├── dataset9.py             ← remove tag from inner transactions
│   ├── dataset10.py            ← convert tx dicts → text descriptions
│   ├── dataset11.py            ← FIXED: split → TSV with 'account' column
│   ├── BERT_text_data.py       ← FIXED: read account col, save account_list
│   ├── adjust_matrix.py        ← FIXED: aligned address_to_index + adj matrix
│   ├── ETH_GBert.py            ← model definition (unchanged)
│   ├── train1.py               ← training loop (unchanged)
│   ├── utils.py                ← dataset/dataloader utilities (unchanged)
│   ├── feature_pipeline.py     ← Spearman feature selection (top-20 → top-10)
│   ├── select_add_features.py  ← end-to-end: graph → BFS → 26 features → CSV
│   ├── env_config.py           ← loads .env, exposes GLOBAL_SEED etc.
│   └── run_train1.sh           ← launches train1.py inside tmux
├── pytorch_pretrained_bert/    ← vendored HuggingFace BERT v0.6.2
├── raw_data/
│   ├── B4E/                    ← phisher_account.txt + raw CSV transaction files
│   └── MulDiGraph/             ← features_output_top10.csv, phisher_accounts.txt
├── data/preprocessed/
│   ├── Multigraph/             ← weighted_adjacency_matrix.pkl (Git LFS)
│   └── multi_processed_data/   ← data_Dataset.* pkl files used by train1.py
├── .env                        ← GLOBAL_SEED=44, optional offline model paths
└── pyproject.toml              ← Python ≥3.12, torch, bert, wandb, sklearn, etc.
```

---

## 3. Two Bugs Found and Fixed

### Bug 1 — Undersampling Mismatch (account-set divergence)

**Problem:**
- `dataset8.py` sampled accounts from `transactions7.pkl` independently
- `adjust_matrix.py` sampled accounts from `transactions4.pkl` independently
- Different random seeds + different pipeline stages → **different account sets**
- Result: GCN adjacency matrix and BERT text corpus were built on disjoint accounts

**Fix:**
- Created `Dataset/shared_sampling.py` — samples ONCE from `transactions7.pkl` (all phisher + 2× random normal, seeded by `GLOBAL_SEED`), saves `chosen_accounts.pkl`
- `dataset8.py` now only **filters** `transactions7.pkl` to `chosen_accounts.pkl` — no new random sampling
- `adjust_matrix.py` uses the same account set via `account_list` (see Bug 2 fix)

---

### Bug 2 — Vocabulary / Index Mismatch (GCN was a no-op)

This had two sub-problems:

**Sub-problem A — Tokens stripped before text generation:**
- `dataset5.py` removed `from_address`, `to_address`, `timestamp` from every transaction
- After `dataset10.py` converted transactions to text, the corpus only contained tokens like `amount: 0.5  in_out: 1  2-gram: 1234 ...`
- `address_to_index` contains only Ethereum hex keys (`0xabc...`)
- In `example2feature` (utils.py): every `gcn_vocab_map.get(word, -1)` returned `-1` (UNK)
- `gcn_swop_eye` was all-zeros → GCN contributed **nothing** to BERT embeddings

**Fix A:** `dataset5.py` now only removes `timestamp`. `from_address` and `to_address` survive into the text, so tokens like `0xabc123...` appear in the sentence and can be looked up in `address_to_index`.

**Sub-problem B — address_to_index ordering was misaligned with shuffled_clean_docs:**
- `adjust_matrix.py` built `address_to_index` from its own independently-ordered set
- `train1.py` used `example.guid = i` (position in `shuffled_clean_docs`) to look up `index_to_address[i]`
- But `address_to_index[addr] = i` meant "addr is the i-th node in the GRAPH" — a completely different index space
- Result: `graph_features_lookup` always returned `zero_features` (wrong account looked up)

**Fix B:**
- `dataset11.py` now writes an `account` column (first column) to all TSV files — each row carries its Ethereum address
- `BERT_text_data.py` reads `df['account']` → builds `account_list` in the same order as `shuffled_clean_docs` → dumps `data_Dataset.account_list`
- `adjust_matrix.py` loads `account_list` → builds `address_to_index = {addr: i for i, addr in enumerate(account_list)}` → this guarantees `address_to_index[addr] == example.guid` for every example → builds the `N×N` adjacency matrix using this aligned ordering → saves to `multi_processed_data/` for `train1.py`

---

## 4. Correct Pipeline Run Order (after fixes)

```
Step 1:  python shared_sampling.py
         → chosen_accounts.pkl

Step 2:  python dataset8.py        (uses chosen_accounts.pkl)
         → transactions8.pkl

Step 3:  python dataset9.py  → transactions9.pkl
         python dataset10.py → transactions10.pkl
         python dataset11.py → train.tsv / dev.tsv / test.tsv
                                (all with 'account' column)

Step 4:  python BERT_text_data.py
         → data_Dataset.shuffled_clean_docs
         → data_Dataset.train_y / valid_y / test_y / etc.
         → data_Dataset.account_list   ← NEW, aligned with shuffled_clean_docs

Step 5:  python adjust_matrix.py
         (reads account_list → builds aligned address_to_index)
         → data_Dataset.address_to_index   ← saved to multi_processed_data/
         → weighted_adjacency_matrix.pkl   ← saved to Multigraph/

Step 6:  python train1.py
         (no changes needed — picks up the aligned files automatically)
```

---

## 5. Key Invariant After the Fix

For every training/validation/test example `ex`:

```
ex.guid  ==  address_to_index[account_list[ex.guid]]
```

This means:
- `index_to_address[ex.guid]` correctly returns the Ethereum address for that example
- `graph_features_lookup[address]` correctly returns the top-10 graph features
- `gcn_swop_eye` correctly routes address tokens through the GCN branch

---

## 6. Model Architecture Detail (ETH_GBert.py)

```
Input per example:
  - input_ids        : tokenized transaction text (includes 0x... address tokens)
  - graph_features   : [B, 10] top-10 Spearman features (betweenness_centrality,
                        clustering_coefficient, in_degree, freq_in_long, out_degree,
                        freq_out_long, freq_out_short, max_out_amount,
                        in_degree_centrality, active_days)
  - gcn_adj_list     : sparse [N×N] transaction adjacency matrix
  - gcn_swop_eye     : [B, N, seq_len] one-hot selection matrix (token → graph node)

ETH_GBertEmbeddings.forward():
  1. words_embeddings     = BERT word embedding lookup              [B, seq, H]
  2. vocab_input          = gcn_swop_eye @ words_embeddings         [B, H, seq]
     gcn_vocab_out        = VocabGraphConvolution(adj, vocab_input) [B, H, gcn_dim]
     gcn_words_embeddings = words_embeddings + injected gcn output  [B, seq, H]
  3. feature_embeddings   = FeatureProjector(graph_features, seq)   [B, seq, H]
  4. fused                = DynamicFusionLayer(bert, gcn, feat)      [B, seq, H]
  5. final                = fused + position_emb + token_type_emb   [B, seq, H]
  6.                      → LayerNorm → Dropout

DynamicFusionLayer (4-gate DiffSoftmax):
  gate_logits = Linear(concat[bert, gcn, feat]) → 4 gates
  gate_values = DiffSoftmax(gate_logits, tau=1.0)
  embeddings_mixed = softmax(fusion_weights) · [bert, gcn, feat]
  output = gate_bert·bert + gate_gcn·gcn + gate_feat·feat + gate_mixed·mixed
```

---

## 7. Data Sources

| Source | Description |
|--------|-------------|
| `raw_data/B4E/` | Ethereum transaction CSVs (normal_eoa_in/out, phisher_in/out) + `phisher_account.txt` |
| `raw_data/MulDiGraph/` | Pre-built MultiDiGraph pkl + `phisher_accounts.txt` + extracted feature CSVs |
| `data/preprocessed/B4E/` | train.tsv, dev.tsv, test.tsv (generated by dataset11.py) |
| `data/preprocessed/multi_processed_data/` | All `data_Dataset.*` pkl files consumed by train1.py |
| `data/preprocessed/Multigraph/` | `weighted_adjacency_matrix.pkl` (GCN adj, Git LFS) |

---

## 8. Graph Features (top-10 by Spearman correlation, from feature_pipeline.py)

Selected from 26 features across 3 groups extracted by `select_add_features.py` (BFS subgraph depth=2 per account):

```
betweenness_centrality, clustering_coefficient, in_degree,
freq_in_long, out_degree, freq_out_long, freq_out_short,
max_out_amount, in_degree_centrality, active_days
```

These are loaded from `features_output_top10.csv` in `train1.py` and used as a `graph_features_lookup` dict keyed by lowercase Ethereum address.

---

## 9. Training Configuration (train1.py defaults)

| Parameter | Value |
|-----------|-------|
| Model | `ETH_GBertModel` (bert-base-uncased) |
| Batch size | 8 |
| Learning rate | 8e-6 |
| GCN embedding dim | 16 |
| Max seq length | 416 (400 + gcn_dim) |
| Loss | Cross-entropy |
| Optimizer | BertAdam (warmup=0.1) |
| Early stopping patience | 5 epochs |
| Max epochs | 50 |
| Metrics | Weighted F1, Precision, Recall |
| Experiment tracking | WandB (project: Dynamic_Feature_Training) |
