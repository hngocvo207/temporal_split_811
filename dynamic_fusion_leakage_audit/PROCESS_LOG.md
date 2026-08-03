# Dynamic_Fusion — Data-Leakage & Evaluation-Validity Audit

Process log. Updated as work proceeds. Target repo: `/home/ngocvo/Desktop/ngocvo/Dynamic_Fusion`.

---

## STEP 0 — Reconnaissance report (2026-07-23)

### 0. Actual repo layout vs. the assumed layout

The task brief assumed `utils.py` / `train.py` / `ETH_GBert.py` live at the repo root. They don't —
everything lives under `Dataset/`, and the root also contains files not mentioned in the brief:
`CONTEXT_SUMMARY.md` (a prior session's handoff note), `data/` (preprocessed artifacts), and an
**untracked** `muldigraph_preprocessing/` folder. This is a git repo (`git log` shows 20 commits,
current branch `claude/read-github-files-GG2hM`), so there is real history to read, and it shows the
repo has already been touched by at least three different people/machines.

### 1. FOUND: the random 80/10/10 split

`Dataset/dataset11.py:16-19`:
```python
train_rows, temp_rows = train_test_split(rows, train_size=0.8, random_state=42)
val_rows,  test_rows  = train_test_split(temp_rows, test_size=0.5, random_state=42)
```
Pure random split, no time awareness. Confirmed — this is the leakage source Step 2 of the brief
targets.

**However**: this script (and the entire `dataset0.py … dataset11.py` chain feeding it) is wired to
the **B4E dataset**, not MulDiGraph:
- `Dataset/dataset0.py:15` → `default="/home/ngochv/Dynamic_Feature/raw_data/B4E"`
- `Dataset/dataset11.py:5,38-40` → reads `.../b4e_processed_data_1/transactions10.pkl`, writes to
  `.../B4E/train.tsv` etc.
- Every file in that chain (`dataset1.py` … `dataset10.py`, `shared_sampling.py`,
  `BERT_text_data.py`) hardcodes paths under `/home/ngochv/Dynamic_Feature/...` — a different
  project name and a different machine/user than this one (`/home/ngocvo/Desktop/ngocvo/Dynamic_Fusion`).
  None of these scripts can run as-is here without editing every path.

B4E is the paper's *"Transaction Network"* dataset (Wu et al., ~60K nodes) — not the MulDiGraph
dataset (Chen et al., 2,973,489 nodes) the task brief is about. **The split-logic file that's
actually wired up in the repo is for the wrong dataset.**

### 2. FOUND: resampling / balancing, and it touches val/test too

`Dataset/shared_sampling.py:62-68`:
```python
phisher = [addr for addr, txs in data.items() if txs[0]["tag"] == 1]
normal  = [addr for addr, txs in data.items() if txs[0]["tag"] == 0]
n_select = min(2 * len(phisher), len(normal))
selected_normal = random.sample(normal, n_select)
chosen = set(phisher + selected_normal)
```
This undersampling (all phishers + 2× random normals) runs **before** `dataset11.py`'s train/val/test
split. Because the split happens on the already-undersampled pool, **val and test inherit the
~1:2 phisher:normal ratio**, not the real-world distribution. Confirmed empirically: the on-disk
`data/preprocessed/Multigraph/{train,dev,test}.tsv` are only 4,690 / 587 / 588 rows total — nowhere
near what a real split of a 2.97M-node graph should produce, consistent with everything having gone
through this uniform undersampling.

`Dataset/train1.py:89`: `resample_train_set = False` — a *second*, currently-disabled
`WeightedRandomSampler` mechanism exists (`Dataset/train1.py:271-306`) for train-only resampling, but
turning it on/off doesn't touch the upstream leakage above, since that already happened before the
split.

### 3. FOUND: adjacency matrix construction (full-dataset leakage)

`Dataset/adjust_matrix.py:154-172` — zero matrix + accumulation loop, exactly as described:
```python
adj_matrix = np.zeros((N, N), dtype=np.float32)
...
for account, transactions in transactions4.items():
    ...
    adj_matrix[fi, ti] += calculate_weight(tx)
```
Built over the *entire* `account_list` (train+val+test together) with no train/boundary cutoff —
this is the Step 3 leakage target, confirmed.

### 4. FOUND: n-gram time-difference weights (Eq. 1-3)

- Raw ΔTₙ (Eq. 1): `Dataset/dataset4.py:15-23` — `transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']` for n=2..5.
- α coefficients (Eq. 2) and edge weight (Eq. 3): `Dataset/adjust_matrix.py:81-121` —
  `ALPHA[n] = (1/n) / harmonic_sum(1..5)`, `weight = value * Σ αₙ·ΔTₙ`. Two prior commits
  (`ec70904`, `6be0eca`) already corrected this formula to match the paper.

### 5. FOUND: loss function

`Dataset/train1.py`, `cfg_loss_criterion = "cle"` → `F.cross_entropy(...)` at lines ~384-391 and
~561-566 (with an optional pre-computed class-weight tensor). Matches the paper. Confirmed.

### 6. FOUND: threshold / prediction

`Dataset/train1.py:394-395`:
```python
_, predicted = torch.max(logits, -1)
```
Hardcoded argmax over 2-class logits (≡ 0.5 threshold), no calibration step. Confirmed — Step 5
target.

### 7. Critical finding: a second, disconnected "fix" already exists, and it doesn't work here

`muldigraph_preprocessing/` (untracked, never committed) contains `temporal_split.py`,
`temporal_split_pipeline.py`, `master_loader.py`, `fix_labels.py`. This looks like an earlier attempt
at exactly the temporal-split fix this task asks for — `temporal_split_pipeline.py` already does an
80/10/10 **time-window** split by `T_first`, prints per-split illicit-rate warnings, and writes
`data/preprocessed/Dataset/{split_config.json,split_stats.json,train_nodes.pkl,val_nodes.pkl,test_nodes.pkl}`
(all present on disk, dated Jun 16). But:

- `master_loader.py` imports `fraudGT.datasets.mg_dataset`, `fraudGT.graphgym.*`, and
  `torch_geometric.data.HeteroData` — this is code from a **different, unrelated project**
  ("FraudGT"), not something built for this repo's actual model. `pyproject.toml` has no
  `torch-geometric` or `fraudGT` dependency, so this can't run in this environment at all.
- `temporal_split.py`'s own `__main__` block hardcodes
  `MG_PATH = "/home/thegreatestrang/FraudGT-re/FraudGT/FraudGT_pipeline/data/MG/MulDiGraph.pkl"` —
  a path from yet another machine/user entirely, further confirming this file was copied in from
  a different project rather than authored for Dynamic_Fusion.
- Nothing in `Dataset/train1.py` (the actual trainable BERT+GCN model) reads any of
  `muldigraph_preprocessing/`'s outputs. It is disconnected from the model in `ETH_GBert.py`.
- Its label scheme also differs from the paper/task: it adds a "potential fraud" class (any account
  that ever *received* funds from a confirmed-fraud account) on top of confirmed fraud, which is not
  in the task brief and roughly 40-50% inflates the positive class in a way that needs sign-off before
  being treated as ground truth.

**This is not usable as the fix without rework** — it's evidence someone (a prior session) started
the right idea but on the wrong branch of the codebase, using a dependency stack this project doesn't
have.

### 8. Critical finding: the on-disk MulDiGraph artifacts are largely not real data

`data/preprocessed/Multigraph/weighted_adjacency_matrix.pkl` fails to unpickle
(`_pickle.UnpicklingError: invalid load key, 'v'`) — it is a **Git LFS pointer stub**, not the actual
matrix (CONTEXT_SUMMARY.md itself flags this file as "(Git LFS)"). Likewise
`transactions1.pkl` … `transactions10.pkl`, `transactions8.pkl`/`9.pkl`, `adjusted_transactions4.pkl`
under `data/preprocessed/Multigraph/` are all 132-135 bytes — empty/placeholder pickles, not real
transaction data. Only `transactions7_corrected.pkl` (1.6 GB) and `addr_to_node_id.pkl` (148 MB) look
like real artifacts. `data/preprocessed/multi_processed_data/*` (the files `train1.py` actually loads
via `data_%s.%s` names) do look real (`shuffled_clean_docs` 5.4 MB, `address_to_index` 458 KB), but
they total only ~11.5K lines across `train.tsv/dev.tsv/test.tsv` — the undersampled B4E-scale set,
not a real split of 2.97M nodes.

**In short: the current repo cannot actually train end-to-end on MulDiGraph right now** — the script
chain that's committed builds against B4E, the MulDiGraph-specific outputs on disk are partly LFS
stubs, and `train1.py` itself hardcodes a third machine's path
(`/home/iec/uyenvnb/Dynamic_Fusion/...`, e.g. line 140, 187, 238) that doesn't exist here either.

### 9. Dataset provenance (Step 1 of the brief)

Confirmed via `raw_data/MulDiGraph/` (`MulDiGraph.pkl`, `phisher_accounts.txt`,
`features_output*.csv`) and `raw_data/B4E/` (`phisher_account.txt` + raw CSVs) both being present —
this matches the brief's description: MulDiGraph = Chen et al. 2020 (XBlock), B4E = the paper's
separate, smaller "Transaction Network" (Wu et al.) dataset. No README/citation in `Dataset/` or
`raw_data/` currently misattributes MulDiGraph to Wu et al., so there's no citation text to correct —
the provenance *issue* here is architectural (wrong pipeline wired to wrong dataset), not a citation
error.

---

## Where this leaves Step 1 onward

Per the task's own instructions ("if a script isn't in the repo, say so and stop before inventing
one" / "do not proceed until this report is printed"), I stopped here and asked the user two
questions before touching code.

## User decisions (2026-07-23)

1. **Target pipeline**: repoint `Dataset/*.py` to MulDiGraph — i.e. build a MulDiGraph-native
   pipeline rather than reworking the disconnected `muldigraph_preprocessing/` (FraudGT-based) code.
2. **Execution scope**: fix the code, then validate with a **small smoke test** (not a full-scale
   run) — full regeneration of MulDiGraph artifacts + full BERT+GCN retraining is multi-hour/GPU-heavy
   and left for the user to run separately.

## STEP 2 — Temporal split (executed at REAL, full scale)

New file: `Dataset/mg_temporal_pipeline.py`. Loads `raw_data/MulDiGraph/MulDiGraph.pkl` directly (not
from the broken LFS stubs in `data/preprocessed/Multigraph/`) — 2,973,489 nodes, 13,551,303 edges,
1,165 phishing, confirmed identical to the paper's stated numbers. Runtime: ~10s to load + ~1min to
compute (vectorized numpy over the full edge list — no need to subsample for this part).

Design decisions, each recorded so they can be revisited:
- **Label scheme**: confirmed phishing only (`isp==1`). Deliberately did NOT adopt the "potential
  fraud" (any account that ever received funds from a confirmed-fraud account) relabeling found in
  the disconnected `muldigraph_preprocessing/fix_labels.py` — that wasn't asked for in the brief and
  materially changes ground truth (would roughly double the positive class).
- **Split criterion**: position-based 60/20/20 (brief 2c), not the old pipeline's 80/10/10
  time-window split.
- **overlap definition**: relative to T1 (train/val boundary) only, since that's the boundary that
  matters for loss leakage (brief 2e). A node whose activity starts in the val window and drifts past
  T2 into the test window is folded into pure_test rather than given a 5th category — it never
  touches pure_train or the loss either way.

Result (full graph, no subsampling):
- T1 = 2018-03-08 15:40:40 UTC, T2 = 2018-06-03 21:44:48 UTC
- pure_train=1,566,230 (273 phishing) / overlap=217,863 (145) / pure_val=503,382 (270) /
  pure_test=686,014 (477) / isolated=0
- Step 2f diagnostic: no WARNING — every partition has a healthy phishing count.
- Step 8 checks embedded in the script: temporal ordering PASS, mask exclusivity PASS, phishing
  coverage PASS (1165==1165), real-distribution-not-artificially-balanced PASS.

Output: `data/preprocessed/Dataset_MG/{labels,t_first,t_last,partition}.pkl`,
`split_config.json`, `split_stats.json`.

## STEP 3 — Adjacency matrix (executed at REAL, full scale)

New files: `Dataset/mg_graph_weight_formula.py` (Eq. 1-3, vectorized via pandas groupby, reusing the
already paper-corrected ALPHA constants from `adjust_matrix.py`) and `Dataset/mg_build_adjacency.py`.

Key fix beyond leakage: the old `adjust_matrix.py` built a **dense** `np.zeros((N,N))` matrix — at
MulDiGraph's 2.97M-node scale that's ~35 TB, categorically impossible. Rebuilt as **sparse**
(`scipy.sparse.coo_matrix` accumulated directly from edge weights) instead.

Built two matrices over the same full 2,973,489-node index:
- `adj_train.npz` — edges with `timestamp <= T1` only (7,978,336 / 13,551,303 edges, 58.88%),
  nnz=3,002,684. Used for GCN message passing during **training**.
- `adj_inference.npz` — full graph, all edges, nnz=5,355,155. Used for GCN message passing during
  **evaluation** (transductive setting per brief 2e/3b — the leakage this fixes is LABELS crossing
  the boundary, not the GNN seeing later graph structure at inference).

Runtime: ~75s for both matrices, full scale.

## STEP 3c — Feature normalization refit

New file: `Dataset/mg_refit_features.py`. Only 5,655 of 2,973,489 nodes have BFS-depth-2 structural
features extracted already (`features_output_split.csv`) — full-graph extraction
(`select_add_features.py`) is expensive and out of this session's scope, flagged as follow-up work.
Re-derived each of those 5,655 nodes' partition from the NEW split (not the old CSV's own ad-hoc
'split' column), fit `mean`/`std` on pure_train+overlap only (2,740+438=3,178 rows), applied to all.
Printed before/after mean/std for all 10 features; train-only rows verified mean≈0/std≈1 after
scaling. Output: `raw_data/MulDiGraph/features_output_top10_MG_fixed.csv`.

## STEP 4 — Leakage-free corpus (executed at smoke scale, by design)

New file: `Dataset/mg_build_examples.py`. Subsamples each partition (phishing accounts never dropped,
`--cap_* 0` keeps everything for a real run) since full-scale BERT tokenization + training across
1.5M+ accounts is out of the chosen smoke-test scope. Used caps: 800 pure_train / 200 overlap / 200
pure_val / 200 pure_test, rebalanced 50/50-ish within cap so the tiny corpus has both classes to learn
from (explicitly NOT representative of the real 1:1438 ratio — that's what a real, uncapped run is
for).

Important subtlety caught and fixed: `train1.py`'s `CorpusDataset` requires
`ex.guid == address_to_index[own_address]` (an invariant CONTEXT_SUMMARY.md calls out as "Bug 2B" for
the old B4E pipeline). Since this run's account_list (vocab order: pure_train+overlap+pure_val+
pure_test) and doc order (train+valid+test docs) are NOT the same ordering, saved an explicit
`doc_accounts` array so `train1.py` can set each example's guid correctly instead of assuming
guid==position.

Also discovered mid-run: `train1.py`'s tokenizer only `.split()`s on whitespace — it expects
pre-tokenized BERT WordPiece tokens (that's what the original `BERT_text_data.py` did via
`clean_str()` + `bert_tokenizer.tokenize()`), not raw text. Added the same preprocessing to
`mg_build_examples.py` after hitting a `KeyError: 'from:'` on the first run.

Overlap accounts get their own InputExample (for evaluation only — Step 6 breakdown) but never enter
`train_examples` (Step 2e: context-only, no loss). Verified: `[PASS] train_examples contains ONLY
pure_train accounts`.

## Patched `Dataset/train1.py` in place

- Fixed hardcoded paths (`/home/iec/uyenvnb/Dynamic_Fusion/...`, `/home/ngochv/...`) → repo-relative
  via `BASE_DIR`.
- `args.ds` default → `Dataset_MG`; loads the new `data_Dataset_MG.*` files.
- guid now looked up via `doc_accounts`/`address_to_index` instead of assumed == position (see above).
- Two adjacency matrices wired in: `gcn_adj_list_train` (training loop) vs `gcn_adj_list_eval`
  (valid/test `evaluate()` calls) — replaces the old single dense-then-sliced matrix and its silent
  zero-padding/truncation hack (`adjust_matrix_size`).
- Focal loss (γ=2, α from TRAIN class frequency only) via `compute_loss(..., is_training=...)` —
  train loop only; eval loss computation unchanged in spirit (just used for monitoring/checkpointing,
  not gradients).
- `evaluate()` now also returns y_true/y_pred/pos_probs and accepts an optional `threshold` param.
- After training: reload best-by-valid-F1 checkpoint, calibrate threshold on **validation only**
  (precision-recall curve, argmax F1), score test with that threshold (not 0.5/argmax), report
  F1(pos)/AUPRC/G-Mean/Recall@100/Recall@500 broken down by pure_test/overlap/overall, plus Step 8
  runtime checks.
- Installed missing deps into `~/.local` (no venv tooling — no `python3-venv`, no passwordless sudo —
  used `pip install --user --break-system-packages`): scipy, torch, scikit-learn, nltk,
  pytorch-pretrained-bert, wandb, python-dotenv.

## Smoke test run (GPU: RTX 3060, WANDB_MODE=disabled)

`python3 train1.py --max_epochs 4 --patience 4` — 1,400 total accounts (800 train / 200 valid / 400
test = 200 pure_test + 200 overlap). ~5 minutes wall clock. Completed cleanly, no errors. Early
stopped after epoch 0 was never beaten (best valid F1 0.6011 at epoch 0 — expected for a 4-epoch,
tiny-data smoke run, not a claim about model quality). Calibrated threshold: 0.4176. Final test
breakdown:

| Slice | n | pos | F1(pos) | AUPRC | G-Mean | R@100 | R@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 200 | 100 | 0.6644 | 0.5733 | 0.0995 | 0.3400 | 1.0000 |
| overlap | 200 | 100 | 0.6667 | 0.7544 | 0.2412 | 0.5700 | 1.0000 |
| overall | 400 | 200 | 0.6655 | 0.6691 | 0.1852 | 0.3600 | 1.0000 |

Step 8 runtime checks: all PASS.

**This proves the fixed pipeline runs correctly end-to-end** (temporal split → train-cutoff adjacency
→ leakage-free corpus → focal loss → calibration → 3-way metric breakdown), not what the model scores
at full scale/real imbalance — see `preprocessing_and_eval_report.md` §8-9 for what a full run needs.

## Step 7 continuation — full-scale infeasibility + scaling/novelty direction (2026-07-23)

User asked to SSH into `ngocvo@100.106.237.81` and run Step 7 on the original data. Verified this IP
is this same machine's Tailscale address (identical hostname/files/clock) — no separate server, ran
locally instead. User initially chose "literal full scale" (no caps); before burning hours on the
corpus build, checked whether the unmodified model can even hold a ~2.97M-entry GCN vocab: it can't —
`CorpusDataset.pad()` builds a **dense one-hot** `[batch, vocab_size, seq_len]` tensor (`utils.py`)
and `VocabGraphConvolution` holds a dense `[vocab_size, 128]` parameter (`ETH_GBert.py`) — at full
scale the one-hot tensor alone is ~39.6 TB/batch. This is a pre-existing architectural ceiling, not
something the leakage fix introduced. User agreed to fall back to a "large-bounded" run instead
(~35,000-account vocab, ALL 1,165 real phishing accounts kept, `--balance` off so real imbalance is
preserved as much as a bounded sample allows) — added `--balance` flag to `mg_build_examples.py` for
this (default off = keep all real phishing + fill with real negatives, vs. the smoke test's 50/50
`--balance` mode).

Corpus build (35k vocab): 30s. Training launched in background (`train1.py --max_epochs 15
--patience 5`), survived one full machine reboot via checkpoint resume (`--load 1` picked up epoch 1,
valid F1 0.9499, right where it left off) — confirms the existing checkpoint mechanism is adequate for
this kind of unattended, multi-hour run. Second launch used `python3 -u` (unbuffered) after discovering
the first run's stdout was fully block-buffered when redirected to a file, making it look "stuck" when
it wasn't (confirmed via GPU utilization checks, not log output, during the first run). Still running
as of this note (epoch 3/15).

Separately, user asked for a forward-looking direction: how to actually scale `ETH_GBert.py` to the
full dataset, grounded in current SOTA so the eventual write-up has real novelty rather than
duplicating existing work. Researched via WebSearch/WebFetch (10 papers, 2024-2026): found the
one-hot-vs-gather scale fix is a known, mathematically-exact pattern (VGCN-BERT's own scalability
update, InducT-GCN's weighted-sum-instead-of-one-hot fix) and, more importantly, found that
**MulDiGraph is already a standard full-scale benchmark** in 2024-2025 work (TLMG4Eth, KGBERT4Eth,
LMAE4Eth all report on it, F1 80-90%+) but **none of them disclose a temporal split or discuss
leakage** — the same gap this session already fixed for this project's own paper. Full writeup,
citations, and recommended novel direction (leakage-audited full-scale re-evaluation of the BERT+VGCN
fusion family, not yet another new fusion architecture) in
`scaling_and_novelty_direction.md` (this folder).

## Step 7 large-bounded run — final results (2026-07-24)

Training finished: early stopped at epoch 14 (max 15), best checkpoint epoch 13 (valid F1 0.9823).
Total wall clock ~10.3h on one RTX 3060 (survived one machine reboot mid-run via checkpoint resume,
see above). Final numbers, calibrated threshold 0.4864:

| Slice | n | pos | F1(pos) | AUPRC | G-Mean | R@100 | R@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 5,000 | 477 | 0.7753 | 0.8363 | 0.8833 | 0.2075 | 0.7925 |
| overlap | 5,000 | 145 | 0.2609 | 0.4566 | 0.8867 | 0.3241 | 0.7724 |
| overall | 10,000 | 622 | 0.5121 | 0.6419 | 0.8686 | 0.1608 | 0.5129 |

Weighted test F1=91.82 (Pre=94.95, Rec=90.13) — same ballpark as this project's own paper's 94.71% and
TLMG4Eth's 90.41% (both undisclosed-split, likely leaky), but positive-class F1 is much lower and
highly slice-dependent (0.78 pure_test vs 0.26 overlap) — exactly the kind of gap a single weighted
number hides, and exactly what `scaling_and_novelty_direction.md` predicted checking for. All Step 8
runtime checks PASS. Full detail written into
`Dynamic_Fusion/preprocessing_and_eval_report.md` §8b/§9/§11.

This is now the final state of the Step 7 deliverable for this session — literal full-scale
(2,973,489-vocab) training remains blocked on the architecture fix described in
`scaling_and_novelty_direction.md` §2, tracked as follow-up work, not done in this session.

## Final deliverable

`/home/ngocvo/Desktop/ngocvo/Dynamic_Fusion/preprocessing_and_eval_report.md` — the brief's requested
single-file report (Step 0 findings, provenance, split boundaries/counts, feature-normalization
stats, imbalance method, calibrated threshold, Step 6 metrics, Step 8 PASS/FAIL, and what's left for
a full-scale run). Step 7's dual-protocol comparison was explicitly NOT run (would require re-running
the leaky original protocol and a full-scale retrain — out of the chosen smoke-test scope).
