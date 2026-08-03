# Bi-modal (ETH_GBert = BERT + GCN) run — progress report

Continuation of the leakage/evaluation-validity audit (`PROCESS_LOG.md`,
`preprocessing_and_eval_report.md`). Goal: run the **original architecture**
(BERT + GCN only, no top-10 graph-feature stream) under the exact **same
fixed protocol** already validated for the tri-modal model
(`Dataset/tri_model/train1.py`), on the **same corpus and split**, so the two
architectures are comparable head-to-head. Target files:
`Dataset/bi_model/{train_origin.py, ETH_GBert_origin.py, utils_origin.py}`.

---

## 1. Repo state found at start of this session

Between the previous audit session and this one, `Dataset/train1.py`,
`ETH_GBert.py`, `utils.py` had been moved into a new `Dataset/tri_model/`
subfolder, and `Dataset/bi_model/` had been seeded with `train_origin.py`,
`ETH_GBert_origin.py`, `utils_origin.py` — the **pre-audit, original**
versions of the pipeline (random-split-era code, no focal loss, no threshold
calibration, no pure_test/overlap breakdown, no early stopping).

## 2. Bugs found and fixed before touching the bi-modal pipeline

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `Dataset/bi_model/ETH_GBert_origin.py:207` | `ETH_GBertModel.forward()` did `vocab_adj_list = [adj * 0 for adj in vocab_adj_list]` — **zeroed the adjacency matrix on every forward pass**, so the GCN branch never actually received graph structure (only a constant bias survived). Running this as-is would silently score a BERT-only model while claiming "BERT+GCN". | Removed the zeroing line. |
| 2 | `Dataset/tri_model/train1.py` | After the file was moved one directory deeper (`Dataset/train1.py` → `Dataset/tri_model/train1.py`), `BASE_DIR = dirname(dirname(__file__))` now resolved to `Dataset/` instead of the repo root `Dynamic_Fusion/`, and `from env_config import env_config` no longer resolved at all (env_config.py stayed in `Dataset/`, not `Dataset/tri_model/`). The tri-modal script was **broken** as a result of the move. | Added `sys.path.insert(0, Dataset/)` and corrected `BASE_DIR` to point at `Dynamic_Fusion/` again. Verified by re-resolving `env_config` and `data_dir` from `tri_model/`. |
| 3 | `Dataset/tri_model/train1.py:54` | WandB API key was **hardcoded directly in source**, inside a repo with a real GitHub remote (`github.com/hngocvo207/Dynamic_Fusion`) — a real leak risk on next commit/push (confirmed not yet in git history, but sitting in the working tree). | Replaced with `wandb.login()` (reads `~/.netrc`); authenticated once via `wandb login <key>` CLI so the key never lives in a tracked file. Applied the same safe pattern to `bi_model/train_origin.py`. |

Same `sys.path`/`BASE_DIR` fix was applied to `bi_model/train_origin.py` from
the start, since it lives at the same depth (`Dataset/bi_model/`).

## 3. What was rewritten in `bi_model/train_origin.py`

Ported the full fixed protocol from `tri_model/train1.py`, minus the
graph-feature stream:

- Loads `data_Dataset_MG.*` (same 11 pickles, incl. `doc_accounts` and
  `test_partition`) from `Dynamic_Fusion/data/preprocessed/multi_processed_data_MG/`.
- guid resolved via `address_to_index[doc_accounts[i]]` (not position) —
  same vocab-alignment fix as the tri-modal script.
- Two adjacency matrices: `gcn_adj_train.npz` (pre-T1 edges, used during
  training) vs `gcn_adj_eval.npz` (full transductive graph, used for
  valid/test) — loaded with the same shape-assert guard.
- Focal loss (γ=2, α from **train-only** class frequency), used only for the
  training-loop loss.
- Step 5: threshold calibrated on **validation only** (precision-recall
  curve, argmax F1) after reloading the best-by-valid-F1 checkpoint.
- Step 6: test scored with the calibrated threshold, broken down by
  `pure_test` / `overlap` / `overall` (F1-pos, AUPRC, G-Mean, Recall@100/500).
- Step 8 runtime PASS/FAIL checks (train set purity, test class presence).
- Early stopping (`--patience`, default 5) on validation weighted F1.
- WandB: `wandb.login()` + `wandb.init(project="fraud_detection", ...)`,
  per-epoch metrics logged (train/valid/test loss, F1, recall, precision).

`utils_origin.py` needed **no changes** — it was already the correct
"no graph-feature" `CorpusDataset` (6-tuple batches: input_ids, input_mask,
segment_ids, confidence, label, gcn_swop_eye).

## 4. Data used — identical to the tri-modal "large-bounded" run

Same corpus, same split, same vocab (verified by loading the pickles):

| Split | n | phishing |
|---|---:|---:|
| train | 20,000 | 273 |
| valid | 5,000 | 270 |
| test | 10,000 (5,000 pure_test + 5,000 overlap) | 622 |
| vocab (GCN) | 35,000 accounts | all 1,165 real phishing kept |

This is deliberate: same architecture-ablation question ("does the
graph-feature stream help, given the same leakage-free protocol and the same
data"), not a re-run of Step 2's temporal split.

## 5. Validation before the real run

`python3 train_origin.py --validate_program` (1 example/split, 1 epoch) —
completed cleanly end-to-end, WandB logged correctly to
`fraud_detection`, checkpoint saved/reloaded, Step 5/6/8 all executed without
error. Smoke-test checkpoint and local `wandb/` run dir were deleted
afterward (throwaway).

## 6. Real run — launched and in progress

- **Command**: `python3 -u train_origin.py --max_epochs 15 --patience 5`
  (same epoch budget / patience as the completed tri-modal large-bounded run,
  for a fair comparison).
- **Where**: tmux session `bi_model_train`, cwd
  `Dynamic_Fusion/Dataset/bi_model/`.
- **Log file**: `Dataset/bi_model/logs/train_run.log` (also visible via
  `tmux attach -t bi_model_train`).
- **Checkpoint output**: `Dataset/bi_model/output/ETH_GBert16_model_Dataset_MG_cle_sw0.pt`
  (separate directory from the tri-modal checkpoint — no collision).
- **WandB**: project `fraud_detection`,
  https://wandb.ai/hngocvo207-national-economic-university/fraud_detection
  (run `run-20260730_093459-uwnbkkmz`).
- **Started**: 2026-07-30 09:34.
- **Expected duration**: ~40-50 min/epoch (train + valid + test eval on this
  20k/5k/10k corpus), up to 15 epochs unless early stopping fires sooner —
  roughly in the same ~8-10h ballpark as the tri-modal run on the same GPU
  (RTX 3060), possibly a bit faster (bi-modal has fewer parameters: no
  `feature_projector`, 3-gate fusion instead of 4-gate).

### How to check on it later

```bash
tmux attach -t bi_model_train      # live view, Ctrl+B then D to detach again
tail -f Dataset/bi_model/logs/train_run.log
```

## 7. Next step once training finishes

Pull the same Step 6 table (F1-pos / AUPRC / G-Mean / Recall@100/500 by
pure_test/overlap/overall) from the end of `logs/train_run.log` and place it
next to the tri-modal numbers already in
`preprocessing_and_eval_report.md` §8b:

| Slice | n | pos | F1(pos) tri-modal | F1(pos) bi-modal |
|---|---:|---:|---:|---:|
| pure_test | 5,000 | 477 | 0.7753 | *pending* |
| overlap | 5,000 | 145 | 0.2609 | *pending* |
| overall | 10,000 | 622 | 0.5121 | *pending* |

This directly answers whether the added graph-feature stream (the tri-modal
model's differentiator) actually earns its complexity, under a protocol
where both architectures see identical, leakage-free data.
