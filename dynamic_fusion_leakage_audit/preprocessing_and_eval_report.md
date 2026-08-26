# Dynamic_Fusion — Preprocessing & Evaluation-Validity Fix Report

**Revision note (this update):** supersedes the split methodology described in the original version
of this report. That version used a **position-based 60/20/20 split** (sort accounts by `T_first`,
cut by rank). Per `dynamic_fusion_leakage_audit/new_split.md`, the pipeline has been refactored to a
**global 80th-percentile T_cutoff temporal split** (percentile of every transaction timestamp, not of
per-account `T_first`), and a new **controlled 1-hop phishing-label propagation** stage
(`mg_propagate_labels.py`) has been added as a train-only, soft-labeled augmentation signal, kept
strictly separate from ground truth. Sections 3–5, 11 and 13 below are new/rewritten; sections 1–2 and
6–9 (root-cause findings, dataset provenance) are unchanged and kept for continuity.

**Revision note (follow-up update, same session):** the §9/§15 "architecture fix required first" item
has been **implemented and verified** — `CorpusDataset.pad()` and `ETH_GBert.py`'s
`VocabGraphConvolution` were refactored from a dense one-hot scatter-matmul to a sparse
gather/embedding-lookup, removing the memory ceiling that blocked full-vocab (2,973,489-account) runs.
See new §10. This also **corrects a unit error inherited from the original version of this report**:
the dense one-hot tensor was previously stated as "~39.6 TB/batch" — the correct figure, verified by
direct calculation and now by an actual GPU measurement, is **~39.6 GB/batch** (GB, not TB — still far
past a 12 GB card, so the fix was still necessary, just not by three orders of magnitude as previously
written).

**Revision note (training-procedure overhaul, same session):** the large-bounded run under §9's
methodology surfaced a real problem — both runs on the new split selected **epoch 0 as "best"** and
never improved after it. Root-caused to the model-selection metric itself (weighted F1 on a validation
set that is 97%+ majority-class, so it rewards trivially-classifying-benign rather than
phishing-detection quality — see the discussion this triggered, reproduced in §11). Four changes made
in response, all in `Dataset/tri_model/train1.py`, detailed in new §11: (1) `train_dataloader` now uses
the already-present-but-previously-unused `WeightedRandomSampler` path to counter class imbalance
in-batch; (2) best-checkpoint/early-stopping selection switched from weighted F1 to **F1(pos)**
(positive-class-only F1), with weighted F1 still reported alongside, never as the selector; (3)
`evaluate()` now prints F1(pos) as a headline number, not buried inside `classification_report`; (4)
learning rate raised from a hardcoded 8e-6 to `--lr` (default 2e-5), and the optimizer switched from
`BertAdam` (bespoke linear-warmup/decay) to `torch.optim.Adam` + `CosineAnnealingLR`. Sections 7 and 11
below cover this; §9's large-bounded numbers are being re-run under the new setup for direct
comparison (§11).

**Revision note (Attempt 2 result landed, same session):** §11's pending re-run (`Attempt 2 (fixed)`,
wandb run `unzkxl3r`, 20,000/5,000×3 corpus) has finished — table and full slice breakdown added to
§11, §15 item 5 updated. Headline: the sampler/alpha collapse bug is fixed (test F1(pos) overall
0.1005 → 0.5104, back in range with the pre-overhaul 0.5035 baseline), but the problem the §11 overhaul
set out to solve — best checkpoint stuck at epoch 0, no improvement before early stopping — is **not**
fixed; this is now the third run in a row (Before / Attempt 1 / Attempt 2) to select epoch 0. That
question is reopened in §15 item 5.

**Revision note (Attempt 3 result landed — §11's stuck-at-epoch-0 question resolved, same session):**
reverted the optimizer from `Adam`+`CosineAnnealingLR` (no LR warmup) back to `BertAdam` (this project's
original optimizer, warmup+decay baked into the optimizer step, new `--warmup_ratio` flag, default 0.1)
as the only change versus Attempt 2 — sampler, neutral alpha, and F1(pos) selection all kept. Result
(wandb run `h3gxwf4k`): **best epoch = 7**, not 0 — the pattern that held across Before/Attempt 1/
Attempt 2 breaks for the first time. Test F1(pos) overall = **0.6227**, the best number recorded in this
document at 20,000-train scale (vs. 0.5104 for Attempt 2). Missing LR warmup, not the split, the
imbalance, or the sampler/alpha fix, was the actual cause of every prior run's stall. Full detail, the
epoch-by-epoch F1(pos) climb, and the updated comparison table are in §11; §15 item 5 updated
accordingly. In the course of chasing this down, two factual errors from the previous revision note were
also caught and corrected in §11: Attempt 2 actually ran `--max_epochs 15` (not 6), and its corpus vocab
is `35,000` (not `5,400`, which is §9's smaller 3,000-train corpus's vocab size, a different run).

**Revision note (expanded-test propagation + Attempt 3 retrain, follow-up session):** a second,
separate propagation approach was added and run end-to-end — new §17. Unlike `mg_propagate_labels.py`
above (train-only, hub-guarded, soft-weighted, asserted disjoint from val/test),
`Dataset/mg_propagate_labels_expanded_test.py` propagates 1-hop from the same 1,165 verified seeds
**with no hub guard and no partition restriction**, deliberately letting propagated accounts land in
val/test too — done on request, specifically to grow the sparse real test set (529 phishing) for
evaluation. Full graph run: **7,007** newly labeled accounts, total labeled-fraud pool 1,165 → 8,172;
test-partition fraud count 529 → 4,168. The Attempt 3 training config (§11: BertAdam, lr=2e-5,
warmup_ratio=0.1, weighted sampler, neutral focal alpha, F1(pos) selection) was re-run on this expanded
label set at the same 20,000/5,000×3 bounded scale as the original Attempt 3, in a fully isolated
folder (`Dynamic_Fusion/runs/expanded_test_attempt3/`) to avoid checkpoint collision with the existing
`..._attempt3v2.pt` checkpoint. Result: **test F1(pos) overall = 0.8763** — **not comparable** to
Attempt 3's 0.6227, because this run's test set has a 41.68% positive rate (heavily enriched by
propagation) instead of the original ~5.29%, a much easier statistical task. See §17 for the full
breakdown and why the two numbers must not be read as "improvement."

**Revision note (full-scale evaluation of the expanded-test checkpoint, same follow-up session):**
§17's bounded-scale (20,000/5,000×3) checkpoint was then run through the same evaluation-only,
full-811,704-account protocol as `full_scale_test_eval_report.md` — new §17.6. `mg_build_full_test_eval.py`
and `eval_full_test.py` were extended (backward-compatible new flags) to build/score against an
arbitrary checkpoint's own vocab and, when a non-strict `--labels_source` was used to build the
corpus, to score the **same predictions against both label sets** (`isp_expanded` and strict `isp`) with
independently val-calibrated thresholds for each. Headline, scored against **strict ground-truth `isp`**
(directly comparable to `full_scale_test_eval_report.md` §4's 0.0549): this checkpoint gets **F1(pos)
overall = 0.0383** — *worse*, not better, than the original Attempt 3 checkpoint at full scale. Full
detail and interpretation in §17.6.

**Revision note (full-scale feature extraction + a defect that affects every number in this
document, follow-up session):** §15 item 2 is **done** — `select_add_features.py` was run over all
**2,973,489** nodes (coverage 0.19% → 100%), full detail in the companion
`fullscale_feature_extraction_report.md`. In the course of that run, three defects were found that
change how the results already in this document should be read, and one of them is serious:

1. **The graph-feature branch was inert in every experiment recorded here.** `train1.py` read a
   5,655-row feature CSV and falls back to `zero_features` for anything missing, so **99.81% of
   accounts received an all-zero feature vector** — in Attempt 3's 20,000-example corpus, roughly
   **36** examples had real features. Worse, because that CSV came from `select_add_features.py`'s
   5:5 phisher/normal sample, *having* a non-zero vector was itself a **202×** signal for the positive
   class (P(features|phishing)=35.5% vs P(features|benign)=0.18%). So the third modality was either
   silent or emitting a sampling artifact. **Every F1 in §9/§11/§12/§17 was measured under that
   condition** and should not be read as reflecting a working three-modality model.
2. **`TOP10_FEATURE_NAMES` rests on a broken selection.** It came from |Spearman| computed on the OLD
   ad-hoc split (38% of its "TRAIN" rows are val/overlap/pure_test under the current T_cutoff split),
   at 16.0% positive against a real 0.0267% (~600× enrichment), on 0.23% of the train partition, and
   against `phisher_accounts.txt` labels rather than ground truth.
3. **Two disagreeing label sets exist.** `phisher_accounts.txt` (5,480 in graph) vs `labels.pkl`/`isp`
   (1,165) overlap on only **963** addresses — 4,517 only-txt, 202 only-isp, so they disagree in both
   directions rather than one being a superset. The `.txt` never reached the model (labels come from
   `InputExample.label` via `isp`), so no number here is label-corrupted; but it did drive the feature
   selection in (2). The `.txt` is now **retired**, replaced by `raw_data/MulDiGraph/
   phisher_account_muldi.txt` (1,165 addresses exported from `isp`). Which set is correct for this
   dataset remains an open provenance question.

Acted on: `train1.py` and `eval_full_test.py` now consume **all 23 extracted features at full
coverage** (the selection step is dropped rather than re-run, since a feature-only baseline confirms
all-23 > old-top-10). Attempt 3's configuration was re-run unchanged except for the feature branch —
new §18.

**Revision note (feature-only baseline reframes the evaluation, same session):** a plain
`HistGradientBoosting` on the 23 features alone — no BERT, no GCN, no fusion, **4 seconds** to fit —
scores **F1(pos) 0.3479 on the full 811,704-account test partition**, against Attempt 3's **0.0549**
(`full_scale_test_eval_report.md` §4). On the bounded 10,000-node test used throughout §11 the ranking
reverses (baseline 0.4937 vs Attempt 3 0.6227). This is **not** a fair architecture comparison — the
baseline trains on all 1,945,607 train accounts, Attempt 3 on 20,000, a 97× difference — but the
reversal is itself the finding: **the bounded test composition (5.29% positive vs a real 0.065%)
flatters the model**, so §11's numbers overstate real-distribution capability. Full detail in
`fullscale_feature_extraction_report.md` §15.

Scope executed: **Path B — restore & rebuild on the real MulDiGraph dataset.** The real
`raw_data/MulDiGraph/MulDiGraph.pkl` (2,973,489 nodes / 13,551,303 edges / 1,165 phishing accounts,
loaded directly, not from the broken Git-LFS stubs under `data/preprocessed/Multigraph/`) was used to
rebuild the temporal split, the 1-hop propagated labels, and the adjacency matrices from scratch — all
full-scale, on the complete graph, not a sample. Full detail of every command run is in
`/home/ngocvo/Desktop/ngocvo/dynamic_fusion_leakage_audit/PROCESS_LOG.md`; this file is the brief's
requested single-file summary.

---

## 1. Step 0 findings (file:line) — summary (unchanged from prior revision)

| Item | Location | Verdict |
|---|---|---|
| Resampling touching val/test | `Dataset/shared_sampling.py:62-68` (all phishers + 2× normal, applied **before** the split) | Confirmed — inherited by val/test |
| Adjacency built on full dataset | `Dataset/adjust_matrix.py:154-172` (dense `np.zeros((N,N))`, no boundary) | Confirmed leakage target; also **infeasible at MulDiGraph scale** (dense N×N at N=2.97M ≈ 35 TB) |
| n-gram weights (Eq. 1-3) | `Dataset/dataset4.py:15-23` (ΔTₙ) + `Dataset/adjust_matrix.py:81-121` (α, weight) | Correct, already paper-fixed by prior commits `ec70904`/`6be0eca` |
| Loss function | `Dataset/tri_model/train1.py`, `cfg_loss_criterion="cle"` → `F.cross_entropy`/focal loss | Matches paper, extended with focal loss + soft-label sample weighting (§7) |
| Threshold | pre-fix: hardcoded argmax, no calibration | Fixed — calibrated on validation only (§8) |
| On-disk MulDiGraph artifacts | `data/preprocessed/Multigraph/*.pkl` | Mostly Git-LFS pointer stubs / empty placeholders — **not used**; rebuilt from the real `MulDiGraph.pkl` instead |
| Stale `pure_train`/pre-T1 naming | `Dataset/bi_model/train_origin.py:814`, `:218` (BI-MODAL variant, not in the refactor's file list) | Left as-is — separate model variant, not `train1.py`; noted here so it isn't mistaken for an oversight |

Full reasoning and additional findings: see PROCESS_LOG.md Step 0 section.

## 2. Dataset provenance (unchanged)

Confirmed as brief describes: MulDiGraph = Chen et al. 2020 (XBlock), 2,973,489 nodes / 13,551,303
edges / 1,165 phishing (isp=1) — verified by directly loading `MulDiGraph.pkl` (not from stale docs).
No misattributing citation text was found in-repo to correct.

## 3. Temporal split — 80% global T_cutoff (real, full-scale results) — **rewritten**

Script: `Dataset/mg_temporal_pipeline.py`. Ran on the **full graph**, not a sample (~a few seconds,
vectorized numpy). Label scheme unchanged: **confirmed phishing only** (`isp==1`).

**New split standard (new_split.md §A), replacing the old position-based 60/20/20-by-`T_first` split:**

1. `T_cutoff` = the **80th percentile of every edge's timestamp** in the graph (transaction-level, not
   one `T_first` per account). Pre-cutoff graph = `timestamp <= T_cutoff` (~80% of transactions).
2. Every account is classified by its **full activity range** `(t_first, t_last)` relative to
   `T_cutoff`: `pure_train_candidates` (`t_last <= T_cutoff`), `overlap` (`t_first <= T_cutoff < t_last`),
   `pure_test` (`t_first > T_cutoff`).
3. **Validation** = the 10% of `pure_train_candidates` with the **latest** `t_last` (i.e. most recently
   active right before the cutoff). The remaining 90% becomes the final **train** partition.
4. **Test** = `overlap ∪ pure_test`, kept as two distinct labels so the pure_test/overlap/overall
   breakdown (§8, Step 6) is still possible.

**T_cutoff** = `1527988904` = **2018-06-03 01:21:44 UTC** (80.0000% of all 13,551,303 timestamped edges
fall at or before this instant — verified to ±0.1% by a hard assertion in the script, see §13).

| Partition | Accounts | Phishing | Phishing % |
|---|---:|---:|---:|
| train | 1,945,607 | 519 | 0.0267% |
| val | 216,178 | 117 | 0.0541% |
| overlap | 201,931 | 217 | 0.1075% |
| pure_test | 609,773 | 312 | 0.0512% |
| isolated | 0 | 0 | — |
| **test (overlap ∪ pure_test)** | **811,704** | **529** | **0.0652%** |
| **Total** | **2,973,489** | **1,165** | — |

**Diagnostic**: no WARNING raised — every partition has a healthy phishing count (min 117), nowhere
near the "zero or near-zero" threshold that would force a boundary adjustment. No isolated (zero-edge)
accounts exist in this graph, so nothing was dropped from the split.

Real observed test-set imbalance: **529 / 811,704 ≈ 1 : 1534** phishing:normal.

## 4. 1-hop controlled label propagation (`mg_propagate_labels.py`) — **new**

**Goal (new_split.md §C):** from the 1,165 ground-truth phishing seeds, generate a separate
`propagated_1hop` signal for accounts one out-edge downstream (candidate mules/launderers), applied
**only** to the `train` partition, as a soft (down-weighted) label — never mixed into ground truth,
never allowed to touch val/test.

Ran full-scale (`--enable-label-propagation --propagation-weight 0.5 --hub-percentile 99`):

| Stat | Value |
|---|---:|
| Ground-truth seeds | 1,165 |
| Hub-degree threshold (99th pct of in+out degree) | 40.0 |
| Seeds excluded as hubs (do not propagate outward) | 236 |
| Seeds used as propagation sources | 929 |
| Candidates before hub filter (1-hop OUT-neighbors) | 1,743 |
| Candidates excluded as hubs (exchange-like) | 424 |
| Candidates after hub filter | 1,319 |
| Candidates dropped (not in `train` partition) | 800 |
| **Final `propagated_1hop` accounts** | **519** |
| Victims (IN-edge senders to a seed; never a positive) | 22,857 |
| Train positive count: before / after augmentation | 519 → 1,038 (2.0×) |
| Train positive rate: before / after | 0.0267% → 0.0534% |

Safety constraints implemented and verified:
- **Directionality**: only OUT-edges from a seed propagate (money flowing away from a confirmed
  phisher). IN-edges (senders TO a seed) are tagged `is_victim=1` and are never used as a positive.
- **Hub/exchange guard**: any node above the 99th-percentile degree is excluded from propagation on
  *both* sides (as a source and as a candidate) — 236 hub seeds and 424 hub candidates excluded.
- **Label separation**: `label_source ∈ {ground_truth, propagated_1hop, benign}`; `isp` (ground truth)
  is never modified. `isp_augmented = isp OR propagated_1hop`, used for training only.
- **No val/test leakage**: `propagated_1hop` is only ever assigned within the `train` partition;
  asserted in-script (`propagated_1hop ∩ (val ∪ test) == ∅` — **PASS**) and re-asserted independently
  in `mg_build_examples.py` after its own account subsampling.
- **Soft label, not hard label**: propagated examples carry `sample_weight=0.5` (tunable via
  `--propagation-weight`, must be `< 1.0`), consumed as a per-example loss multiplier in
  `train1.py`'s focal loss — never counted as confidently as a real ground-truth positive.
- Seed count re-verified unchanged after processing: **1,165 == 1,165** (no loss/duplication from the
  merge).

## 5. Adjacency matrix (Step 3) — real, full-scale results — **updated (T_cutoff, not T1)**

Script: `Dataset/mg_build_adjacency.py` + shared formula in `Dataset/mg_graph_weight_formula.py`.
Built as **sparse** matrices directly over the full 2,973,489-node index:

- `adj_train.npz`: edges with `timestamp <= T_cutoff` only → 10,841,043 / 13,551,303 edges (80.00%),
  **nnz = 4,162,599**
- `adj_inference.npz`: full graph, all edges (transductive, Step 3b) → **nnz = 5,355,155**

Both reuse the exact Eq. 1-3 weight formula (n-gram ΔTₙ + α coefficients), vectorized per-account via
pandas groupby.

## 6. Feature normalization (Step 3c) — **updated: train-only, not train+overlap**

Script: `Dataset/mg_refit_features.py`. Only 5,655 of 2,973,489 nodes have BFS-depth-2 structural
features extracted (`raw_data/MulDiGraph/features_output_split.csv`; extracting them for the full
graph is the separate, expensive `select_add_features.py` step, out of this session's scope). Per
new_split.md §B, the scaler is now fit **strictly on `mg_partition == "train"`** (previously
`pure_train + overlap`) — no statistics leak in from `overlap`, `val`, or `pure_test`.

This sample's partition breakdown: 3,534 train / 428 overlap / 393 val / 1,300 pure_test rows. Before
(whole-CSV) vs after (train-only-fit) — `betweenness_centrality` mean/std as one example:
`0.004016 / 0.023175` → applied-to-all `0.036827 / 1.117714`, with train-only rows sanity-checked at
mean≈0/std≈1 (max abs mean ≈ 1e-17). Full before/after table for all 10 features is printed by the
script. Output: `raw_data/MulDiGraph/features_output_top10_MG_fixed.csv`, wired into `train1.py`.

## 7. Imbalance handling (Step 4c) — **extended with soft-label sample weighting**

**Focal loss**, γ=2.0, α computed from the **actual training target's** class frequency only (isp or
isp_augmented, whichever `--use_isp_augmented` selected — this run used ground-truth `isp`). On top of
that, every training example now carries a **per-example `sample_weight`** (new_split.md §C.5):
`1.0` for ground-truth/benign examples, `propagation_weight` (`0.5` by default) for `propagated_1hop`
examples — multiplied elementwise into the focal loss before the batch mean, so augmented positives
never contribute to gradients as confidently as real ground truth. This plumbing runs through
`InputExample`/`InputFeatures`/`CorpusDataset` (`tri_model/utils.py`) and `compute_loss`/`focal_loss`
(`tri_model/train1.py`) regardless of whether `--use_isp_augmented` is on — weights are `1.0`
everywhere when it's off, so it's a no-op in that mode.

Val/test loaders and example composition are untouched by any of this (real distribution preserved, no
resampling, no augmented labels — enforced by `mg_build_examples.py`'s own PASS checks).

**Follow-up, this session (§11 for full rationale): in-batch class balancing for `train_dataloader`.**
Focal loss + sample-weighting reshapes the LOSS, but does nothing about which examples land in the same
batch — at `batch_size=8` and a train positive rate of 2-3%, the expected number of positive examples
per batch is ~0.2, meaning the large majority of batches contain **zero** phishing examples. `train1.py`
already had a `WeightedRandomSampler` code path (`shuffle_choice=2`) implemented but never invoked
(train always used plain shuffling); it is now wired up for `train_dataloader` — inverse-class-frequency
weights, sampling with replacement, same epoch size as before — so the minority class appears in most
batches instead of being sparse/absent. val/test are untouched (still `shuffle_choice=0`, real
distribution, no resampling).

## 8. Threshold calibration (Step 5)

Unchanged methodology: computed on the **validation set only** (never test), precision-recall curve →
threshold maximizing F1, applied to test-set predictions. See §9 for this run's calibrated value
(0.4375) — training was interrupted then resumed from checkpoint (§9) to reach this point.

## 9. Verification run — bounded corpus, real full-scale split (this session)

`mg_build_examples.py --cap_train 3000 --cap_overlap 800 --cap_val 800 --cap_test 800` (no `--balance`,
i.e. all real phishing accounts in each partition kept, filled out with real negatives) against the
**real, full-scale T_cutoff split** above:

| Partition | n | phishing (isp) | phishing (isp_augmented) |
|---|---:|---:|---:|
| train | 3,000 | 519 | 520 |
| val | 800 | 117 | 117 (unaffected, always isp) |
| overlap | 800 | 217 | 217 (unaffected, always isp) |
| pure_test | 800 | 312 | 312 (unaffected, always isp) |

(One propagated_1hop account was pulled in among train's negative-fill slots by the random subsample —
expected: only 519/1,945,607 train accounts carry the flag, so a small sample keeps a small share of
them; a full-scale run keeps all 519.) GCN vocabulary for this run: 5,400 accounts.

This is a **verification run** — proves the new split, propagation, adjacency, feature, and training
code paths execute correctly end-to-end together (see §14) — not a claim about model quality at scale.
A `--validate_program` 1-example smoke pass and this bounded run both completed without errors.

Training was launched for real (`--max_epochs 6 --patience 3`, not `--validate_program`) on this
3,000/800/800/800 corpus. It was **manually interrupted by the user after epoch 0** while the
architecture-fix work (§10) took priority, then **resumed from that exact checkpoint** (`--load 1`,
full step-level resume — model, optimizer, epoch/step position, WandB run id all restored, see §8's
checkpointing work) once §10 was verified, and run to completion: **early stopping triggered at epoch
3** (patience=3, no improvement after epoch 0). Best checkpoint = **epoch 0**
(same wandb run throughout, id `tvfhfqad`).

**Training-loop weighted F1 (best epoch = 0):** Valid 87.48% (Pre 90.95%, Rec 89.88%), Test 75.82%
(Pre 83.67%, Rec 79.13%).

**Step 5/6 — calibrated threshold (val only) + extended test metrics, best checkpoint reloaded:**
calibrated threshold = **0.4375** (val positive-class F1 at that point: 0.6901). Scored on test
(pure_test ∪ overlap) at that threshold — weighted F1 73.83%, Pre 76.20%, Rec 73.13%:

`report_slice()` (`train1.py`) now reports **both** F1(pos) (positive-class-only F1 — the metric that
actually reflects phishing-detection quality) **and** F1(weighted) (sklearn `average="weighted"`: each
class's F1 averaged, weighted by its true support in that slice — pulled toward whichever class
dominates the slice's *actual* label count, here the majority benign class) **for every slice**,
`pure_test`/`overlap` included, not just the combined test set:

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 800 | 312 | **0.8157** | 0.8678 | 0.9203 | 0.8380 | 0.3109 | 0.9840 |
| overlap | 800 | 217 | **0.5127** | 0.6113 | 0.6809 | 0.6397 | 0.4147 | 0.8295 |
| overall | 1,600 | 529 | 0.6504 | 0.7383 | 0.7504 | 0.7373 | 0.1871 | 0.5955 |

(`overall`'s F1(weighted)=0.7383 matches the standalone "Test_set(final,calibrated)" weighted-F1 number
above exactly, as expected — same predictions, two code paths, cross-checked.) The gap between the two
columns is the imbalance effect described above: e.g. on `overlap`, F1(weighted)=0.61 looks
"acceptable" in isolation, but F1(pos)=0.51 is what actually says the model catches barely half the
phishing accounts there at this checkpoint/threshold — **F1(pos) is the number to trust for this task**,
F1(weighted) is included for comparability with papers/baselines that only report the weighted figure.

**Same asymmetry as the prior revision's large-bounded run reappears here, at a much smaller scale and
under the new split**: positive-class F1 is markedly worse on `overlap` (0.51) than on genuinely-unseen
`pure_test` (0.82) — true on both the F1(pos) and F1(weighted) columns. Given `overlap` accounts are
excluded from the training loss by design (§3/§9 — only `train`-partition accounts get an
`InputExample`), this is a second, independent data point supporting the earlier hypothesis (original
revision, §8b) that the model struggles specifically with boundary-spanning accounts it never sees a
label for — not just an artifact of the old split or of
scale. All Step 8 runtime checks **PASS** (`train_examples` only from `train` partition; test has both
classes; valid/test scored on ground-truth `isp` only, `use_isp_augmented=False` this run).

This remains a **verification-scale run** (3,000-account train, not the real 1,945,607) — a directly
comparable large-bounded number (§11, §15 item 5) is now available at 20,000-train scale, see §11.

## 10. Scaling fix implemented — sparse gather/embedding-lookup (this session)

**What was fixed:** `CorpusDataset.pad()` (`Dataset/tri_model/utils.py`) built a **dense one-hot**
`gcn_swop_eye` tensor of shape `[batch, vocab_size, seq_len]` via `F.one_hot(...)`, consumed by
`ETH_GBert.py`'s `VocabGraphConvolution.forward` as `X_dv.matmul(H_vh)`. This is mathematically a
**scatter**: every token gets written into a vocab_size-sized slot regardless of how many of those
slots are actually used (at most `seq_len` ≈ 416 out of 2,973,489 per example). Refactored to a
**gather** instead — derived by expanding the matmul algebraically:

```
X_dv[b,h,v]        = sum_l  1[gcn_vocab_ids[b,l]==v] * words_embeddings[b,l,h]
(X_dv @ H_vh)[b,h,k] = sum_v X_dv[b,h,v] * H_vh[v,k]
                     = sum_l words_embeddings[b,l,h] * H_vh[gcn_vocab_ids[b,l], k]
```

i.e. for each token, **gather** the one row of `H_vh` its vocab id addresses, then a small
`[H_bert, L] @ [L, hid_dim]` contraction — never materializes anything with a `vocab_size`-sized
per-batch dimension. Changes:

- `CorpusDataset.pad()`: no longer builds `gcn_swop_eye`; instead returns a plain `[B, seqlen]` long
  tensor `gcn_vocab_ids` (the padded vocab-graph indices, `-1` where a token has no vocab node — the
  same sentinel `example2feature` already produced, just no longer densified into a one-hot).
- `ETH_GBert.py`, `VocabGraphConvolution.forward`: signature changed from `(vocab_adj_list, X_dv,
  add_linear_mapping_term)` to `(vocab_adj_list, words_embeddings, gcn_vocab_ids,
  add_linear_mapping_term)`; internally does `H_vh[safe_ids] * valid_mask` (an embedding-lookup /
  gather) followed by `torch.einsum("blh,blk->bhk", words_embeddings, H_vh_gathered)` instead of the
  one-hot matmul. The (previously dead, `add_linear_mapping_term=False` by default) linear-mapping
  branch was fixed the same way for consistency.
- `ETH_GBertEmbeddings.forward` / `ETH_GBertModel.forward`: parameter renamed `gcn_swop_eye` →
  `gcn_vocab_ids` throughout (same position in the call signature, so this is a rename, not a
  reordering); `vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1,2)` removed —
  `words_embeddings` and `gcn_vocab_ids` are now passed straight to `vocab_gcn(...)`.
- `train1.py`: batch-unpacking and `model(...)` call sites updated for the renamed/reshaped tensor
  (`gcn_swop_eye` → `gcn_vocab_ids`); no change to the training loop's control flow, loss, or
  checkpointing logic.
- **What did *not* change**: `VocabGraphConvolution`'s parameters (`W0_vh`, `fc_hc`), the sparse
  adjacency matrices, `DynamicFusionLayer`, `FeatureProjector`, the BERT encoder, and the
  classifier head are all untouched — this is an input-adapter rewrite, not a model redesign.

**Verification:**

1. *Exact numerical equivalence* — a standalone test (`test_gather_equivalence.py`) built a tiny random
   `VocabGraphConvolution` (V=37, B=4, L=9, dropout disabled) and compared the new gather forward
   against the old one-hot-scatter formula reimplemented literally from the pre-fix code, both fed the
   same weights/inputs. **max abs diff = 2.4e-7** (float32 rounding only) — **PASS**.
2. *End-to-end smoke run* — `train1.py --validate_program` with the new code produced the **exact same
   first-step training loss** (`0.33928459882736206`) as the pre-fix smoke run recorded earlier this
   session, confirming the full BERT+GCN+features model (not just the submodule) is unaffected
   numerically.
3. *Real full-scale memory benchmark* — a forward+backward pass of `VocabGraphConvolution` alone, using
   the **actual** `adj_train.npz` built earlier this session (real full graph, `V=2,973,489`,
   `nnz=4,162,599`) at the real `batch_size=8` / `MAX_SEQ_LENGTH=416`: **peak GPU memory = 7,362 MB**
   (`test_full_scale_memory.py`). For comparison, the OLD dense one-hot tensor alone, at that same
   batch/seq/vocab size, would require **~39.6 GB** (`8 × 2,973,489 × 416 × 4` bytes) — and the very
   next tensor in the old chain, `X_dv` (`[B, H_bert=768, V]`), would need **~73 GB** — both far past a
   12 GB card, and past most single consumer GPUs. The corrected figure (§ revision note above) is GB,
   not the "~39.6 TB" stated in the original version of this report, but the practical conclusion is
   unchanged: the old path was infeasible on this hardware, and the new path measurably is not
   (7.4 GB fits with room to spare for the rest of BERT-base + activations).

**What this does and does not unblock:** the per-batch **memory** ceiling that made a full 2,973,489-
vocab run impossible is gone. It does **not** change per-step **compute time** — BERT-base
forward/backward cost is a function of `batch_size`/`seq_len`, not vocab size, and was already
independent of vocab before this fix. Wall-clock for a literal full-scale run is therefore now a
step-count problem, not a memory problem — see the updated estimate in §15, item 1.

## 11. Training-procedure overhaul — imbalance handling, F1(pos) selection, optimizer/warmup — **resolved**

**The problem, found by running §9's methodology at large-bounded scale (20,000 train, real full-scale
split):** two separate runs on the new T_cutoff split both selected **epoch 0** as the best checkpoint
and never improved past it before early-stopping triggered — unusual (more training normally helps at
least a little before overfitting sets in). Root cause: the model-selection / early-stopping metric
(`perform_metrics` in `train1.py`) was **weighted F1 on validation**, and the validation set here is
~97%+ majority-class (benign). Weighted F1 rewards a model for correctly classifying the *easy* majority
class; a model that is barely learning anything about phishing detection can still post a high weighted
F1 just by defaulting toward "benign." §9's own results table already demonstrated this gap directly
(F1(weighted)=0.7383 vs F1(pos)=0.6504 "overall" in the 3,000-train run) — the selection criterion was
using exactly the metric shown to be misleading for this task.

**Four changes, all in `Dataset/tri_model/train1.py`, no model-architecture changes:**

1. **In-batch class balancing** (§7): `train_dataloader` switched from plain shuffling to the
   previously-unused `WeightedRandomSampler` path, so most batches now contain at least one positive
   example instead of the ~0.2-expected-positives-per-batch that plain shuffling gives at this
   imbalance ratio and `batch_size=8`.
2. **Selection metric switched to F1(pos)**: `perform_metrics` (drives "save new best checkpoint" and
   the early-stopping counter) is now the positive-class-only F1 on validation, not weighted F1.
   Weighted F1 is still computed and reported everywhere (console, WandB, checkpoints) for
   comparability — it is simply no longer what decides which epoch "wins."
3. **F1(pos) reported as a headline number**: `evaluate()`'s per-epoch print now shows
   `F1(pos): .. F1(weighted): ..` explicitly, instead of only the weighted figure with F1(pos) buried
   inside the `classification_report` block's "1" row.
4. **Learning rate raised, optimizer switched**: `--lr` default raised to `2e-5` (a standard BERT-
   fine-tuning scale) and a hardcoded `8e-6` override — applied regardless of `--lr` — removed. Only
   BERT itself is pretrained; the GCN/feature-projector/fusion/classifier layers are randomly
   initialized, and 8e-6 is likely too small for them to learn much within a handful of epochs.
   Optimizer switched from `BertAdam` (bespoke, bakes in its own linear-warmup/linear-decay schedule)
   to `torch.optim.Adam` (adaptive per-parameter LR) + `torch.optim.lr_scheduler.CosineAnnealingLR`
   (`T_max=total_train_steps`, stepped once per optimizer step). Resume checkpoints now also save/load
   the scheduler's state; a checkpoint saved by the old `BertAdam` run is model-compatible (parameter
   shapes are unaffected) but its *optimizer* state cannot be restored into the new `Adam` object —
   `train1.py` catches this and falls back to a fresh optimizer/scheduler with a warning rather than
   crashing.

**A fifth change surfaced a second, separate bug (caught mid-run, not shipped silently):** the first
attempt at re-running with changes 1-4 above collapsed the model to predicting the positive class for
almost every example — observed as `Accuracy` on validation/test landing almost exactly on the eval
set's true positive rate (2.34% / 5.29%), the signature of an always-predict-positive collapse, and it
never recovered across 4 epochs (identical F1(pos)=4.573%/10.048% every single epoch). Root cause:
change 1 (`WeightedRandomSampler`) and the **pre-existing** focal-loss `alpha` term were both correcting
for the same imbalance, stacked. `alpha` was still derived from the original (pre-sampling) class
frequency — `alpha=[0.026, 0.974]` — a ~37x extra multiplier applied on top of batches the sampler had
*already* rebalanced to roughly 50/50, which is what pushed the model into the collapse. **Fix**: `alpha`
is now neutral (`[0.5, 0.5]`) — pick ONE rebalancing mechanism (the sampler), not two. The focal loss's
`(1-pt)^gamma` focusing term is kept (that part still usefully down-weights already-easy examples
independent of class balance). This is exactly the kind of interaction worth calling out explicitly:
each of the two mechanisms is individually reasonable and was individually tested (§7, §9) — the bug
only appears when they're combined, which is precisely why this session re-ran and checked rather than
assuming the individually-good pieces compose safely.

**Comparison, same 20,000/5,000/5,000/5,000 corpus (§9), same real T_cutoff split:**

| Config | Optimizer | LR | Warmup | Sampler | Focal α | Selection metric | Best epoch | Test F1(pos) overall |
|---|---|---:|---|---|---|---|---:|---:|
| Before (§9-scale) | BertAdam | 8e-6 | 0.1 (nominal) | plain shuffle | class-freq | weighted F1 | 0 | 0.5035 |
| Attempt 1 (bug) | Adam+Cosine | 2e-5 | none | weighted | class-freq | F1(pos) | 0 (stuck) | 0.1005 (collapsed) |
| Attempt 2 (fixed) | Adam+Cosine | 2e-5 | none | weighted | **neutral 0.5/0.5** | F1(pos) | 0 (stuck) | 0.5104 |
| **Attempt 3 (warmup)** | **BertAdam** | 2e-5 | **0.1** | weighted | neutral 0.5/0.5 | F1(pos) | **7** | **0.6227** |

("Before"'s warmup=0.1 is marked *nominal*: it ran under the pre-§11 code, which also hardcoded LR to
8e-6 regardless of `--lr` and used plain shuffling — its validation weighted-F1/Pre/Rec are bit-for-bit
identical across epochs 0-2, log-verified, meaning the model barely moved for three straight epochs. Its
epoch-0 stall has a different, already-diagnosed cause (weighted-F1 selection metric + near-zero
effective LR from the tiny 8e-6 + almost-no-positives-per-batch under plain shuffling), not the missing-
warmup mechanism Attempt 3 isolates below — Attempt 2 vs Attempt 3 is the clean, single-variable
comparison for that.)

**Attempt 2 (fixed) — full result, same 20,000/5,000×3 corpus, wandb run `unzkxl3r`:**

Ran with `--max_epochs 15 --patience 5` (verified from this run's `wandb-metadata.json`, correcting an
earlier draft of this section that misstated it as `--max_epochs 6`), wall clock **177.2 minutes
(~2.95h)** on the RTX 3060 for the 6 epochs it actually completed (0-5) before early stopping cut it
off — much faster per-epoch than the prior revision's 35,000-vocab run (~41 min/epoch there vs ~29.5
min/epoch here). **Not because of a smaller vocab** — checked directly against
`data_Dataset_MG.address_to_index` on disk: this run's own 20,000/5,000×3 corpus also lands at
`gcn_vocab_size=35,000` (an earlier draft of this section wrongly assumed 5,400, which is actually the
vocab size of §9's much smaller 3,000-train corpus, a different run). The real reason is the axis §10
changed: the prior revision's 10.3h run predates the sparse-gather fix (dense one-hot, so its per-step
compute scaled with `vocab_size`), while every run in this section (Before/Attempt 1/Attempt 2, and the
BertAdam run below) runs post-§10 (gather-based, vocab-size-independent compute) — consistent with §10's
own claim that the fix leaves compute time unchanged at fixed batch/seq while removing the old path's
vocab-dependent cost entirely. Since `CosineAnnealingLR`'s `T_max` is set from `--max_epochs` (15 × 2,500
steps = 37,500), the schedule was only ~40% decayed by the time training stopped — LR was still ~1.6e-5
(80% of peak) at epoch 4, not near-zero. **Early stopping triggered at epoch 5** (5/5 epochs with no
improvement) — best checkpoint remained **epoch 0** (best
valid F1(pos) 0.4390). Calibrated threshold (val only, argmax F1): **0.9013** (val F1(pos) at that
point: 0.4557).

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 5,000 | 312 | 0.4867 | 0.9471 | 0.6588 | 0.5688 | 0.3141 | 0.7276 |
| overlap | 5,000 | 217 | 0.5425 | 0.9663 | 0.4600 | 0.6181 | 0.4009 | 0.4700 |
| overall | 10,000 | 529 | 0.5104 | 0.9568 | 0.4857 | 0.5895 | 0.1853 | 0.4045 |

Step 8 runtime checks: all PASS (`train_examples` n=20,000, `train`-partition only; test has both
classes at `phishing_rate=0.0529`; valid/test scored on ground-truth `isp` only).

**Reading this result:** the collapse bug (Attempt 1) is fixed — F1(pos) overall recovers from 0.1005
to 0.5104, back in the same range as the pre-overhaul "Before" run (0.5035). But the problem the §11
overhaul was built to fix — **best epoch stuck at 0, no improvement before early stopping** — is still
present, now for a third run in a row (Before, Attempt 1, Attempt 2 all select epoch 0). Total Train
Loss is close to flat across epochs 0-5 (180.9 → 175.4, no clear downward trend), so this doesn't look
like classic overfitting-after-a-good-epoch; it looks more like the model reaching most of what it will
learn within epoch 0 (~2,500 steps) at this LR/batch-size/architecture, with little added by further
epochs. The four §11 fixes were necessary (Attempt 1 shows what breaks without them) but evidently not
sufficient on their own to produce multi-epoch improvement — reopened as an item in §15.

**A second thing worth flagging directly, since it complicates §9's finding rather than confirming
it:** at this 20,000-train scale, `overlap` (F1(pos)=0.5425) scores *higher* than `pure_test`
(F1(pos)=0.4867) — the **opposite** of §9's 3,000-train result (pure_test 0.8157 » overlap 0.5127,
written up in §9 as "the model struggles specifically with boundary-spanning accounts"). Both are real,
correctly-computed numbers from the same split methodology and the same held-out slices; the ranking
between the two slices is not stable across corpus scale/checkpoint. This is reported as-is rather than
reconciled — with best-epoch stuck at 0 in both cases, neither run is evidence of a converged model, so
drawing a general conclusion about pure_test-vs-overlap difficulty from either one (or their
disagreement) would be over-reading a single early-epoch snapshot.

**Attempt 3 (BertAdam + warmup) — this session, follow-up, wandb run `h3gxwf4k`:**

Three runs in a row (Before, Attempt 1, Attempt 2) shared one property regardless of their other
differences: whenever the optimizer was `Adam` + `CosineAnnealingLR`, peak `--lr` hit the randomly-
initialized GCN/feature-projector/fusion/classifier layers from step 1, no warmup (only BERT itself is
pretrained). To test this directly, `train1.py` was reverted to this project's original optimizer,
**BertAdam** (bakes a linear-warmup/linear-decay schedule into the optimizer step;
`schedule="warmup_linear"`), exposed via a new `--warmup_ratio` flag (default `0.1`) — the **only**
variable changed versus Attempt 2. Sampler (weighted), neutral alpha (0.5/0.5), and F1(pos)-based
selection (§11 changes 1-3) were all kept identical. Same 20,000/5,000×3 corpus
(`gcn_vocab_size=35,000`): `--lr 2e-5 --warmup_ratio 0.1 --max_epochs 15 --patience 5`.

| Epoch | Valid F1(pos) | Valid F1(weighted) | Outcome |
|---:|---:|---:|---|
| 0 | 0.1863 | 0.8789 | new best |
| 1 | 0.2022 | 0.8929 | new best |
| 2 | 0.4764 | 0.9651 | new best |
| 3 | 0.5090 | 0.9691 | new best |
| 4 | 0.5939 | 0.9770 | new best |
| 5 | 0.6464 | 0.9824 | new best |
| 6 | (0.6464) | — | no improvement, 1/5 |
| **7** | **0.6667** | 0.9835 | **new best (final)** |
| 8-12 | ≤0.6667 | — | no improvement, 2/5 → 5/5, early stop |

**Best epoch = 7** — the stuck-at-0 pattern breaks for the first time in this document at this scale.
Early stopping triggered at epoch 12 (patience 5 exhausted after epoch 7). Total wall clock **398.9
minutes (~6.65h)** — more than double Attempt 2's 2.95h, because this run actually used its epoch/
patience budget to explore 13 epochs of real improvement instead of idling through 6. Calibrated
threshold (val only, argmax F1): **0.8134** (val F1(pos) at that point: 0.6838).

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 5,000 | 312 | **0.7406** | 0.9701 | 0.8194 | 0.7927 | 0.3141 | 0.8782 |
| overlap | 5,000 | 217 | 0.5107 | 0.9510 | 0.6285 | 0.7946 | 0.4055 | 0.7742 |
| overall | 10,000 | 529 | **0.6227** | 0.9594 | 0.6904 | 0.7922 | 0.1834 | 0.6049 |

Step 8 runtime checks: all PASS (`train_examples` n=20,000, `train`-partition only; test has both
classes at `phishing_rate=0.0529`; valid/test scored on ground-truth `isp` only).

**Reading this result:** this resolves the open question reopened at the end of the Attempt 2 writeup
above and tracked as §15 item 5. The missing LR warmup — not the split, not the imbalance, not a flaw
in the sampler/alpha fix — was what kept every prior run pinned to epoch 0. With a 10%-of-steps linear
warmup restored, F1(pos) climbs for 8 straight epochs (0.19 → 0.67) before plateauing: an ordinary-
looking training curve, not the flat-loss/no-improvement pattern every earlier run in this section
showed. Test F1(pos) overall = 0.6227 is the best result recorded anywhere in this document at 20,000-
train scale (vs. 0.5104 Attempt 2, 0.5035 Before), and `pure_test` alone reaches 0.7406, back in range
with §9's smaller-corpus `pure_test` number (0.8157) rather than Attempt 2's degraded 0.4867.

One thing this does **not** resolve: `overlap` (0.5107) is still clearly behind `pure_test` (0.7406)
here — matching §9's original gap direction (not Attempt 2's reversed one). This is the first run in
this document with an actually-converged checkpoint (best epoch found via real multi-epoch improvement,
not a stall), so this pure_test-vs-overlap gap is the first data point that can be read as a real signal
about boundary-spanning accounts rather than a training-stall artifact — consistent with §9's original
hypothesis that the model struggles specifically with `overlap` accounts, which never receive an
`InputExample` in the training loss by construction (§3/§9).

## 12. Step 7 — dual-protocol / scaling discussion

`CorpusDataset.pad()`'s dense one-hot GCN routing (§10) was the memory blocker for literal full-scale
(2,973,489-vocab) training; that blocker is now removed and verified (§10). What remains is wall-clock:
the prior revision of this report's large-bounded run (20,000 train / 5,000×3 val+test+overlap,
~35,000-vocab, **under the OLD 60/20/20 position split**) reached 91.82% weighted F1 / 77.53%
positive-class F1 on pure_test after ~10.3h wall clock on an RTX 3060 — that number is **not directly
comparable** to future runs under the new T_cutoff split, since the split boundary, val/test
membership, and train positive count are all different now. A literal apples-to-apples re-run of that
scale under the new split, now that the architecture fix removes the memory ceiling, is the concrete
next step (§15).

## 13. Validation checks — **new assertion set, all real full-scale results**

| Check | Result |
|---|---|
| T_cutoff is the 80th percentile of all edge timestamps (±0.1%) | **PASS** (80.0000% exactly) |
| Boundary consistency: max(t_last, train∪val) ≤ T_cutoff < min(t_first, pure_test) | **PASS** |
| pure-train/overlap/pure-test pairwise disjoint | **PASS** |
| pure-train ∪ overlap ∪ pure-test ∪ isolated == all accounts | **PASS** |
| train ∩ val = ∅ | **PASS** |
| train ∩ test(overlap∪pure_test) = ∅ | **PASS** |
| val ∩ test(overlap∪pure_test) = ∅ | **PASS** |
| Phishing coverage: sum(labels) across all partitions == original 1,165 | **PASS** (1165 == 1165) |
| Real distribution check, val/pure_test not artificially balanced | **PASS** (0.0541% / 0.0512%) |
| `propagated_1hop` ∩ (val ∪ test) = ∅ | **PASS** (in both `mg_propagate_labels.py` and re-checked in `mg_build_examples.py`) |
| No address is both `ground_truth` and `propagated_1hop` | **PASS** |
| Ground-truth seed count preserved through propagation | **PASS** (1,165 == 1,165) |
| Scaler fit only on `train`-partition rows (excludes overlap/val/pure_test) | **PASS** (3,534 of 5,655 rows) |
| Training corpus: `train_examples` contains only `train`-partition accounts | **PASS** |
| valid/test built from ground-truth `isp` only, never `isp_augmented` | **PASS** |

## 14. Post-refactor repo-wide audit (new_split.md §5)

- **Grep for stale names**: `pure_train`/`pure_val`/`T1`/`T2` no longer appear in the active `mg_*.py`
  / `tri_model/train1.py` pipeline (all renamed to `train`/`val`/`T_cutoff`). Two references remain in
  `Dataset/bi_model/train_origin.py` (a separate BI-MODAL model variant, not `train1.py`, and not in
  new_split.md's file list) — left untouched deliberately, noted in §1's findings table rather than
  silently fixed, so it isn't mistaken for an oversight.
- **Column/partition-name consistency**: `mg_partition` (train/val/overlap/pure_test/isolated),
  `label_source` (ground_truth/propagated_1hop/benign), `isp`, `isp_augmented` verified consistent
  across `mg_propagate_labels.py` → `mg_build_examples.py` → `train1.py` (exact filenames/keys cross-
  checked, no schema drift).
- **Data-dependency order**: verified `mg_temporal_pipeline.py → mg_propagate_labels.py →
  mg_build_adjacency.py → mg_refit_features.py → mg_build_examples.py → train1.py`; each stage reads
  exactly the filenames/columns the previous stage wrote (traced by hand, confirmed by the full run in
  §9 completing without a schema/shape mismatch).
- **Leakage re-audit**: `propagated_1hop` confirmed absent from val/test (§13); folding `overlap`'s
  pre-cutoff edges into `adj_train.npz` confirmed to not leak `overlap`/test **labels** into the loss
  (only `train`-partition accounts get an `InputExample` in `train_examples` — structure vs. label
  leakage are handled separately, as in the prior revision); final reporting confirmed to always score
  against ground-truth `isp`, never `isp_augmented`, regardless of the `--use_isp_augmented` training
  flag (`valid_y`/`test_y` are built from `labels.pkl` only, upstream of that flag).
- **Two real runtime conflicts found and fixed** (pipeline stage): `mg_build_examples.py`'s `from utils
  import clean_str` had no working import path when run from `Dataset/` (that module lives in
  `Dataset/tri_model/utils.py`) — fixed by inserting `tri_model/` onto `sys.path` at the top of
  `mg_build_examples.py`. A leftover 464MB "best" checkpoint from the prior 60/20/20-split,
  ~35,000-vocab run collided by filename with this session's new, differently-sized vocab
  (`RuntimeError: size mismatch ... [35000, 128] vs [5400, 128]`) — resolved by moving the old
  checkpoint (and its stale resume file) to `Dataset/output/_backup_prior_run_vocab35000/` rather than
  deleting it; the same collision then happened **two more times**: once after the §10 architecture fix
  changed what a checkpoint's tensors mean (moved to
  `Dataset/output/_backup_prior_run_dense_onehot/`), and again after §11's optimizer swap (a
  `BertAdam`-saved checkpoint at the same vocab size loaded its *model* weights fine into the new
  `Adam`-based run but not its *optimizer* state — handled gracefully this time by the try/except added
  in §11, moved to `Dataset/output/_backup_run_train20k_bertadam_weightedF1/`). `train1.py`'s checkpoint
  naming scheme includes neither vocab size nor an architecture/optimizer-version tag, so this is a
  **latent, now three-times-confirmed conflict** — called out again in §15 as an open item.
- **Sparse gather/embedding-lookup fix re-audit** (§10): confirmed the refactor touches only
  `CorpusDataset.pad()` (`utils.py`) and `VocabGraphConvolution`/`ETH_GBertEmbeddings`/`ETH_GBertModel`
  (`ETH_GBert.py`)'s input-adapter code — no change to any learned parameter's shape, the loss
  function, the checkpointing/resume logic (§8's WandB/checkpoint work), or `mg_build_examples.py`'s
  output schema. Verified via exact numerical equivalence against the pre-fix formula (§10) rather than
  by inspection alone.

## 15. What still needs a follow-up, literal full-scale run

1. **Wall-clock, not memory, is now the remaining blocker** (§10 removed the memory ceiling). BERT-base
   forward/backward cost scales with `batch_size`/`seq_len`, not vocab size — so per-step time should
   stay close to what this session already measured (§9: ~0.6s/train-step, ~0.18s/eval-step at
   batch=8). Extrapolating to the full `train` partition (1,945,607 accounts ÷ 8 ≈ 243,201 steps/epoch)
   gives a **rough order-of-magnitude of ~1–2 days per epoch**; multi-epoch convergence (this session's
   partial run was still improving after epoch 0) means a literal full-scale training run is a
   multi-day-to-multi-week undertaking on a single RTX 3060, even though it would no longer OOM. Full
   **evaluation-only** passes over the real test partition (811,704 accounts ÷ 8 ≈ 101,463 steps) are
   much more tractable — roughly **5 hours** at the measured eval-step rate — and are the more
   realistic next full-scale deliverable (score an already-trained-at-bounded-scale checkpoint against
   the entire real test set, rather than train from scratch at full scale).
2. ~~`select_add_features.py` needs to be re-run over the full 2,973,489-node graph so
   `mg_refit_features.py`'s train-only scaler is fit on real coverage, not a 0.2% sample.~~
   — **done** (follow-up session): all 2,973,489 nodes extracted, coverage 0.19% → **100%**, 22.13 h on
   16 CPU workers via a `scipy.sparse` rewrite of the BFS/ball construction (the original networkx path
   needs ~15 GB resident, which makes process parallelism impossible). Fidelity verified against the
   unmodified original script — group 1+2 exact on all clean seeds, 7 of 8 centralities exact to
   0.0e+00, `betweenness` matched only within the original's own un-seeded sampling spread. Scaler now
   fit on the real 1,945,607 train rows; the 0.2% fit it replaces was inflating
   `betweenness_centrality` by ~7× (train std 0.023175 vs the true 0.003276). This also surfaced the
   inert-feature-branch defect described in the revision notes at the top of this document. Full
   detail: `fullscale_feature_extraction_report.md`.
   **Newly open in its place:** run `eval_full_test.py --feature_set all23` on the §18 checkpoint over
   all 811,704 test accounts — §18's numbers are bounded-test only and therefore not decisive.
3. `mg_build_examples.py --cap_train 0 --cap_overlap 0 --cap_val 0 --cap_test 0` for the real,
   non-subsampled corpus (1,945,607 / 216,178 / 201,931 / 609,773 accounts) — even after (1), expect
   multi-day wall clock on a single GPU at this scale.
4. ~~This session's own bounded run (§9) was manually stopped after epoch 0~~ — **done**: resumed from
   the epoch-0 checkpoint and run to convergence (early stopping at epoch 3, §9). Next smallest step is
   now item 5 below (a larger corpus), not a rerun of this one.
5. ~~A rigorous large-bounded run (comparable to the prior revision's 20,000/5,000×3, ~10.3h run) under
   the new T_cutoff split~~ — **done**, see §11 (also triggered the §11 training-procedure overhaul).
   ~~Still open: the SAME comparison but with the §11 fixes applied from the start of a *fresh* run~~
   — **done**: Attempt 2 (§11) ran the full 20,000/5,000×3 corpus with all four §11 fixes from the
   start, to completion (early stopping at epoch 5, patience exhausted). Result: the collapse bug was
   fixed (F1(pos) overall 0.1005 → 0.5104) but genuine multi-epoch convergence still did not happen.
   ~~Newly open: why training loss stays flat past epoch 0~~ — **resolved, same session**: reverting the
   optimizer from `Adam`+`CosineAnnealingLR` (no warmup) to `BertAdam` (its own linear warmup+decay,
   `--warmup_ratio 0.1`) — the only variable changed, everything else identical to Attempt 2 — produced
   **Attempt 3** (§11): best epoch **7** (not 0), F1(pos) climbing monotonically for 8 epochs
   (0.19 → 0.67 on valid) before plateauing, test F1(pos) overall **0.6227** (best number recorded in
   this document at 20,000-train scale). Missing LR warmup, not the split/imbalance/sampler/alpha, was
   the actual cause. Remaining open item, not yet investigated: whether this generalizes to the full,
   non-subsampled `train` partition (1,945,607 accounts) — Attempt 3 is still a 20,000-train bounded
   run, and BertAdam's `t_total`-driven schedule is computed from `--max_epochs` × steps/epoch, which
   will need re-tuning (or a step-count-based warmup, not epoch-count-based) at that scale.
6. ~~`model_file_4save` / `resume_ckpt_path` naming should include vocab size and/or a code-version
   tag~~ — **done, this session (follow-up)**: `model_file_4save` in `train1.py` now embeds the actual
   `gcn_vocab_size` (e.g. `..._sw0_vocab5400.pt`), computed from `len(address_to_index)` right after
   the dataset pickles load, plus an optional `--run_tag` free-text suffix for cases vocab size alone
   can't distinguish (an architecture/optimizer change at the *same* vocab size — the §10 gather-
   rewrite and §11 BertAdam→Adam collisions were exactly this case). Required moving the
   filename/resume-checkpoint/`wandb.init` block from its old position (right after `args =
   parser.parse_args()`, before `gcn_vocab_size` exists) to right after `gcn_vocab_size = len
   (address_to_index)` is computed — `wandb.init` isn't otherwise touched until the training loop, so
   this reorder has no other effect. **Not retroactive**: the Attempt 2 run already in flight when this
   edit landed (§11, PID launched before this change, `train1_large_v3.log`) loaded the old code at
   process start and will still save under the old (no-`vocab`-suffix) filename — this only protects
   *future* runs from colliding with each other, not that in-progress one.
7. `Dataset/dataset0.py` … `dataset11.py`, `shared_sampling.py`, `adjust_matrix.py`,
   `BERT_text_data.py`, `feature_pipeline.py`, `select_add_features.py`, and
   `Dataset/bi_model/train_origin.py` still use the old naming/paths and are out of this fix's scope
   (parallel MulDiGraph-native `mg_*.py` pipeline built instead of repointing them). `bi_model/`'s own
   `utils_origin.py`/`ETH_GBert_origin.py` still use the old dense one-hot `gcn_swop_eye` path too —
   the §10 fix was applied only to `tri_model/` (the one `train1.py` runs), per new_split.md's file
   list.

See also `dynamic_fusion_leakage_audit/scaling_and_novelty_direction.md` for the full scalability
root-cause analysis (its §2 proposed exactly this gather/embedding-lookup fix — now implemented — and
its ~39.6 TB figure has the same unit error corrected by this revision, see the top-of-file revision
note) and the literature-grounded novelty direction.

## 16. Files added/changed this revision

- `Dataset/mg_temporal_pipeline.py` — **rewritten**: 80th-percentile global `T_cutoff` split (was:
  position-based 60/20/20 by `T_first`)
- `Dataset/mg_propagate_labels.py` — **new**: controlled 1-hop phishing label propagation
- `Dataset/mg_build_adjacency.py` — updated to `T_cutoff` naming/boundary
- `Dataset/mg_refit_features.py` — updated: scaler now fit strictly on `train` (was `pure_train +
  overlap`)
- `Dataset/mg_build_examples.py` — updated: new partition names, wires in `isp_augmented` /
  `propagation_weight` / `label_source`, outputs both `train_y` and `train_y_augmented` +
  `train_sample_weight`; fixed a broken `utils` import when run from `Dataset/`
- `Dataset/tri_model/utils.py` — `InputExample`/`InputFeatures`/`CorpusDataset` extended to carry a
  per-example `sample_weight` through to the collated batch (soft-label loss weighting); **and**
  (§10, this follow-up) `CorpusDataset.pad()` no longer builds the dense one-hot `gcn_swop_eye` —
  returns a plain `[B, seqlen]` `gcn_vocab_ids` index tensor instead
- `Dataset/tri_model/ETH_GBert.py` — **new to this revision's changed-file list** (§10):
  `VocabGraphConvolution.forward` rewritten from a one-hot scatter-matmul to a gather/embedding-lookup;
  `ETH_GBertEmbeddings.forward`/`ETH_GBertModel.forward` parameter renamed `gcn_swop_eye` →
  `gcn_vocab_ids`. No parameter shapes, no other module (`DynamicFusionLayer`, `FeatureProjector`,
  BERT encoder, classifier) touched — verified by exact numerical equivalence against the pre-fix
  formula (§10)
- `Dataset/tri_model/train1.py` — `--use_isp_augmented` flag (train on `isp` vs `isp_augmented`,
  reporting always on ground-truth `isp`); weighted focal loss; WandB run-id persistence/resume and
  step-level resume checkpointing (every 250 steps + epoch boundaries, ported from
  `bi_model/train_origin.py`'s more robust harness, at the user's request — execution infra only,
  model architecture untouched by that particular change); Step 5/6/8 final metrics now pushed to
  WandB, not just per-epoch loop metrics; (§10) batch-unpacking/`model(...)` call sites updated for the
  renamed `gcn_vocab_ids` tensor; **and** (§11, this follow-up) `train_dataloader` now uses
  `WeightedRandomSampler`; model-selection/early-stopping metric switched from weighted F1 to F1(pos)
  (`evaluate()` now returns and prints both); `--lr` default raised to 2e-5 and the hardcoded 8e-6
  override removed; optimizer switched from `BertAdam` to `torch.optim.Adam` +
  `torch.optim.lr_scheduler.CosineAnnealingLR`, with scheduler state added to the resume-checkpoint
  save/load path and a guarded fallback if an old-format (`BertAdam`) checkpoint's optimizer state
  can't be restored into the new optimizer; **and** (§11 Attempt 3, this follow-up) **reverted back to
  `BertAdam`** after Attempt 2 showed the Adam+Cosine setup never escaped epoch 0 — new `--warmup_ratio`
  flag (default 0.1) feeds `BertAdam(warmup=..., t_total=total_train_steps)`; the `scheduler` object,
  its `scheduler.step()` call, and its checkpoint state are removed (BertAdam has no separate scheduler
  — LR is read per-step via `optimizer.get_lr()`, used for the training-loop LR printout); the guarded
  optimizer-state-restore fallback now guards the reverse direction (an Adam-saved checkpoint's
  optimizer state failing to load into BertAdam)
- `Dataset/mg_graph_weight_formula.py` — unchanged, reused as before

## 17. Expanded-test label propagation + Attempt 3 retrain (follow-up session) — **new**

Separate experiment, additive to everything above — does not modify or replace `mg_propagate_labels.py`,
the existing `Dataset_MG` split/corpus, or the existing Attempt 3 checkpoint (`..._attempt3v2.pt`). Full
narrative detail also in `dynamic_fusion_leakage_audit/expanded_test_propagation_report.md` (first,
hub-guarded version of the idea, superseded by the simpler approach below at the user's request).

### 17.1 Why

The real test set has only 529 verified-phishing accounts out of 811,704 (§3: ≈1:1534) — very sparse
for evaluation. This experiment grows the labeled-fraud pool available for both train and test via
1-hop propagation from the same 1,165 verified seeds, **on request allowing propagated labels into
val/test** (the opposite of §4's train-only, no-leakage design, which stays unchanged and is still
what the main benchmark numbers in this report use).

### 17.2 Method — `Dataset/mg_propagate_labels_expanded_test.py` (new script)

Basic/no-guard version, per explicit follow-up instruction to simplify: for every 1,165 verified seed,
every 1-hop **out**-neighbor (money flowing away from the seed) is labeled fraud too — no hub/exchange
degree filter, no victim tracking, no soft-label weighting, no PASS/FAIL assertions. Result stored as
`isp_expanded` (separate from `isp`, which is never modified), landing wherever the account's existing
partition already placed it (train/val/overlap/pure_test).

Full-graph run (2,973,489 nodes / 13,551,303 edges):

| Partition | Accounts | Fraud before (`isp`) | Fraud after (`isp_expanded`) |
|---|---:|---:|---:|
| train | 1,945,607 | 519 | 3,541 |
| val | 216,178 | 117 | 463 |
| overlap | 201,931 | 217 | 2,587 |
| pure_test | 609,773 | 312 | 1,581 |
| **test (overlap+pure_test)** | **811,704** | **529** | **4,168** |
| **Total** | **2,973,489** | **1,165** | **8,172** |

1-hop propagation found 7,007 new accounts total (much larger than §4's 1,319 — no hub guard means a
few high-degree seeds fan out to many accounts each). Output: `data/preprocessed/Dataset_MG/isp_expanded.pkl`.

### 17.3 Attempt 3 retrain on the expanded labels — isolated folder

Pipeline re-executed end-to-end, reusing every label-independent artifact as-is (T_cutoff split,
`adj_train.npz`/`adj_inference.npz`, `features_output_top10_MG_fixed.csv` — none of these depend on
which label file is used) and rebuilding only the corpus step against `isp_expanded`:

1. `mg_build_examples.py` — given new `--split_dir`/`--out_dir`/`--labels_source` flags (added this
   session, backward-compatible, default behavior unchanged) so it can build from
   `Dataset_MG/isp_expanded.pkl` without touching the shared `Dataset_MG`/`multi_processed_data_MG`
   folders. Same caps as the original Attempt 3 (`--cap_train 20000 --cap_overlap 5000 --cap_val 5000
   --cap_test 5000`), output written to `Dynamic_Fusion/runs/expanded_test_attempt3/corpus/`.
   Resulting corpus: 35,000 vocab, train phishing rate 17.71% (3,541/20,000 — all real
   `isp_expanded`-positive train accounts kept, capped fill), valid phishing rate 9.26% (463/5,000),
   test phishing rate 41.68% (4,168/10,000: pure_test 1,581/5,000 + overlap 2,587/5,000).
2. `train1.py` — given new `--data_dir`/`--output_dir` flags (same session, backward-compatible) so
   checkpoints land in `Dynamic_Fusion/runs/expanded_test_attempt3/checkpoints/` instead of the shared
   `Dataset/tri_model/output/`. Launched with the exact Attempt 3 config (§11): `--lr 2e-5
   --warmup_ratio 0.1 --max_epochs 15 --patience 5`, plus `--run_tag expanded_test_attempt3` so the
   checkpoint filename (`ETH_GBert16_model_Dataset_MG_cle_sw0_vocab35000_expanded_test_attempt3.pt`)
   cannot collide with the existing `..._attempt3v2.pt` even by accident. Run in a detached tmux
   session (survives disconnects, matching this project's established convention for multi-hour runs).

**Training result:** ran the full 15 epochs (0–14), never hit early-stopping's patience=5 (best epoch
12, 2/5 epochs without improvement when the epoch budget ran out) — total wall clock **454.6 minutes
(~7.58h)**, close to Attempt 3's original 398.9 minutes given the larger positive class made every
epoch's `WeightedRandomSampler`-driven batches somewhat different in composition.

| Epoch | Valid F1(pos) | Valid F1(weighted) | Outcome |
|---:|---:|---:|---|
| 0 | 0.6801 | 0.9254 | new best |
| 1 | 0.7621 | 0.9487 | new best |
| 2 | 0.7704 | 0.9510 | new best |
| 3 | (0.7704) | — | no improvement, 1/5 |
| 4 | 0.7926 | 0.9567 | new best |
| 5 | (0.7926) | — | no improvement, 1/5 |
| 6 | (0.7926) | — | no improvement, 2/5 |
| 7 | 0.7986 | 0.9581 | new best |
| 8 | (0.7986) | — | no improvement, 1/5 |
| 9 | 0.8069 | 0.9603 | new best |
| 10 | (0.8069) | — | no improvement, 1/5 |
| 11 | (0.8069) | — | no improvement, 2/5 |
| **12** | **0.8109** | **0.9620** | **new best (final)** |
| 13 | 0.8098 | 0.9627 | no improvement, 1/5 |
| 14 | 0.8074 | 0.9623 | no improvement, 2/5 (epoch budget exhausted) |

Calibrated threshold (val only, argmax F1): **0.6976** (val F1(pos) at that point: 0.8168).

**Step 6 — final test breakdown, best checkpoint (epoch 12) reloaded, calibrated threshold applied:**

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 5,000 | 1,581 | **0.9038** | 0.9408 | 0.9474 | 0.9208 | 0.0607 | 0.3125 |
| overlap | 5,000 | 2,587 | **0.8605** | 0.8526 | 0.8846 | 0.8515 | 0.0371 | 0.1774 |
| overall | 10,000 | 4,168 | **0.8763** | 0.8971 | 0.9155 | 0.8937 | 0.0228 | 0.1173 |

Step 8 runtime checks: all PASS (`train_examples` n=20,000, train-partition only; test has both
classes, phishing_rate=0.4168; reporting used the label file loaded as this run's `labels` —
i.e. `isp_expanded`, not the original strict `isp` — see §17.4 caveat).

### 17.4 Why this is NOT an "improvement over Attempt 3" — read together with §11, not against it

Test F1(pos) overall = **0.8763** vs. Attempt 3's **0.6227** looks like a large jump, but the two runs
are scoring against **different test sets with different difficulty**:

- Attempt 3 (§11): test positive rate ≈ 5.29% (529 real phishing / 10,000, real-distribution sample of
  the strict ground-truth `isp`).
- This run: test positive rate = 41.68% (4,168 / 10,000), because the un-guarded 1-hop propagation
  added mostly-inferred "suspected fraud" labels directly into the test partitions.

A test set that is ~40% positive is a categorically easier classification problem than one that is
~5% positive — most of the F1(pos) gain here is attributable to that shift in class balance, not to
the model learning anything new. Additionally, roughly (4,168-529)/4,168 ≈ 87% of this run's test
positives are **propagated (inferred), not verified** — the model is being scored substantially against
its own family of heuristic labels (1-hop out-neighbor of a seed), which inflates apparent performance
further since propagated accounts are structurally closer to a seed than a random negative is.

**Correct reading:** this result answers "can the model learn well when given a denser, propagation-
expanded label set" (yes — F1(pos) 0.88, Recall 91%, Precision 91% on that expanded target), not
"is the model better at detecting real phishing than Attempt 3 was." The original strict-`isp`
benchmark numbers (§9, §11 Attempt 3: test F1(pos) 0.6227 on the real 529-positive test set) remain the
valid comparison point for actual phishing-detection quality and are unaffected by this run — `isp`
itself was never modified, only read from a differently-sourced `labels.pkl`-equivalent (`isp_expanded`)
for this one isolated experiment.

### 17.5 Isolation / no checkpoint overlap

All new artifacts for this experiment live under `Dynamic_Fusion/runs/expanded_test_attempt3/`:

- `corpus/` — this run's `data_Dataset_MG.*` + `gcn_adj_{train,eval}.npz` (35,000-vocab, built from
  `isp_expanded`)
- `checkpoints/` — `ETH_GBert16_model_Dataset_MG_cle_sw0_vocab35000_expanded_test_attempt3.pt` (+
  `..._resume.pt`), distinct filename (via `--run_tag`) and distinct folder (via `--output_dir`) from
  the existing `Dataset/tri_model/output/..._attempt3v2.pt`
- `logs/build_examples.log`, `logs/train1_attempt3.log` — full console output

Nothing under the shared `data/preprocessed/Dataset_MG/`, `data/preprocessed/multi_processed_data_MG/`,
or `Dataset/tri_model/output/` was overwritten. `mg_build_examples.py` (`--split_dir`/`--out_dir`/
`--labels_source`) and `train1.py` (`--data_dir`/`--output_dir`) gained new optional flags to make this
possible; all default to the original shared paths when omitted, so every prior command in this report
still reproduces unchanged.

### 17.6 Full-scale evaluation (all 811,704 real test accounts) — **new**

Same evaluation-only protocol as `full_scale_test_eval_report.md` (that report's §1(b)/§3 vocab-
extension-as-prefix technique, reused as-is), applied to the §17.3 checkpoint instead of `attempt3v2`.
`mg_build_full_test_eval.py` and `tri_model/eval_full_test.py` were extended with new, backward-
compatible flags (`--old_corpus_dir`/`--out_dir`/`--labels_source`/`--reuse_docs_from` on the corpus
builder; `--old_data_dir`/`--full_test_dir` on the evaluator) so an arbitrary checkpoint/vocab/label
combination can be pointed at without touching `full_scale_test_eval_report.md`'s original run or
files. Already-tokenized sentences for all 811,704 accounts were reused via `--reuse_docs_from`
(text depends only on an account's transactions, never on vocab order or label source) — skips the
expensive tokenization pass entirely, only the vocab prefix/adjacency slice was rebuilt.

**Corpus**: extended vocab = this run's own 35,000-account training vocab (unchanged, exact prefix) +
801,704 newly-appended real test accounts = **836,704** total, `gcn_adj_eval` nnz=2,145,493. Built with
`--labels_source isp_expanded.pkl`, which also writes a parallel `test_y_strict` array (`labels.pkl`)
so the identical set of predictions can be scored against **both** label sets in one evaluation pass —
no second forward pass needed.

**Checkpoint load verified** (same methodology as `full_scale_test_eval_report.md` §4): `missing=
['embeddings.vocab_gcn.W0_vh']`, `unexpected=[]`, `W0_vh[:35000]` bit-identical to the checkpoint,
epoch=12/`perform_metrics`=0.8109 loaded correctly.

**Two independent threshold calibrations on the same validation set** (5,000 accounts, all in-vocab):

| Calibrated against | Val positives | Threshold | Val F1(pos) at threshold |
|---|---:|---:|---:|
| this run's own labels (`isp_expanded`) | 463 | **0.6976** | 0.8168 |
| strict ground-truth `isp` | 117 | **0.9020** | 0.3233 |

The strict-`isp` calibration is markedly less confident (F1(pos)=0.32 vs 0.82) — expected, since this
checkpoint was trained to recognize the denser, propagation-augmented positive class, not the sparser
true-phishing signal specifically.

**Runtime**: 214.5 minutes for the full test pass (811,704 accounts) + ~2 minutes for val calibration
≈ **3.6h total**, matching `full_scale_test_eval_report.md`'s ~3.6h for the same account count (same
per-step rate, ~126ms/step at the end vs. the ~110-140ms/step range reported there).

**Results — scored against `isp_expanded` (this run's own training target, threshold 0.6976):**

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 609,773 | 1,581 | 0.1608 | 0.9859 | 0.1816 | 0.9220 | 0.0032 | 0.0342 |
| overlap | 201,931 | 2,587 | 0.1151 | 0.8941 | 0.1067 | 0.8516 | 0.0012 | 0.0738 |
| **overall** | **811,704** | **4,168** | **0.1290** | 0.9643 | 0.1386 | 0.9066 | 0.0005 | 0.0228 |

**Same predictions, re-scored against strict ground-truth `isp` (threshold 0.9020) — directly
comparable to `full_scale_test_eval_report.md` §4's tri_model numbers:**

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 609,773 | 312 | 0.0632 | 0.9954 | 0.0377 | 0.7287 | 0.0000 | 0.0481 |
| overlap | 201,931 | 217 | 0.0261 | 0.9722 | 0.0597 | 0.7850 | 0.0000 | 0.4009 |
| **overall** | **811,704** | **529** | **0.0383** | 0.9898 | 0.0408 | 0.7558 | 0.0000 | 0.0605 |

Step 8 checks: **PASS** — `n`/`pos` exact for both label sets (811,704/811,704; 4,168/4,168 for
`isp_expanded`, 529/529 for strict `isp`).

**Direct comparison — this checkpoint vs. the original Attempt 3 checkpoint (`attempt3v2`), same full
811,704-account test set, both scored against strict ground-truth `isp`:**

| Metric (overall, strict isp) | attempt3v2 (`full_scale_test_eval_report.md` §4) | expanded_test_attempt3 (§17.6) |
|---|---:|---:|
| F1(pos) | 0.0549 | **0.0383** |
| F1(weighted) | 0.9921 | 0.9898 |
| AUPRC | 0.1241 | **0.0408** |
| G-Mean | 0.7960 | 0.7558 |
| Recall@500 | 0.2590 | **0.0605** |

**Reading this: the expanded-label checkpoint is not better at real phishing detection at full scale —
it is worse, on every metric, than the original strict-`isp`-trained checkpoint**, most sharply on
AUPRC (0.041 vs 0.124, a 3× drop) and Recall@500 (0.061 vs 0.259, a 4.3× drop). This is consistent
with — and sharpens — §17.4's caveat: project-wide, 7,007 of the 8,172 total labeled-fraud accounts
(≈86%) are propagated/inferred "suspected fraud" (1-hop out-neighbors of a seed, no hub guard), not
verified phishing — and this checkpoint's 20,000-account train partition mirrors that skew (3,541
train positives vs. only 519 real ground-truth phishing accounts in `train`, §3). The model learned to recognize
*that* broader, noisier pattern well (§17.3: F1(pos)=0.88 against its own `isp_expanded` target at
bounded scale), but that pattern is evidently a worse proxy for actual phishing accounts than training
on the smaller, clean, strictly-verified seed set was — full-scale evaluation against real ground
truth is what exposes this; the bounded-scale numbers (§17.3-17.4, scored against the same expanded
labels used for training) could not have shown it, since they were never checked against strict `isp`
at all before this section.

**What this does and doesn't establish:** it is one data point, from one propagation configuration
(no hub guard, no soft-label down-weighting, `--propagation-weight` not applicable since this script
doesn't have that mechanism — see §17.2) and one training run. It does not by itself prove that *no*
label-propagation-based augmentation can help real phishing detection — `mg_propagate_labels.py`'s
original, hub-guarded, soft-weighted, train-only design (§4) is a materially different, more
conservative mechanism that has not been evaluated at full scale in this document. What it does
establish is that this specific expanded-test approach, evaluated properly at full scale against
strict ground truth, underperforms the existing best full-scale checkpoint — the bounded-scale
"0.88 F1(pos)" number from §17.3 was measuring fit to an inferred label, not phishing-detection
quality, exactly as flagged in §17.4 before this run existed to confirm it.

**Files**: `runs/expanded_test_attempt3/full_test_corpus/` (corpus), `runs/expanded_test_attempt3/logs/
{build_full_test,eval_full_test}.log` (full console output), wandb run
[`yz0kayvm`](https://wandb.ai/hngocvo207-national-economic-university/fraud_detection/runs/yz0kayvm).

## 18. Full feature coverage + all 23 features — Attempt 3 re-run (follow-up session) — **new**

Companion document: `fullscale_feature_extraction_report.md`.

### 18.1 What changed and why

The graph-feature branch had never actually functioned (see the revision notes at the top of this
document): `train1.py` read a 5,655-row feature CSV against a 2,973,489-account graph, so
`zero_features` fired for **99.81%** of accounts, and because that CSV came from a 5:5 phisher/normal
sample, merely having a non-zero vector was a **202×** signal for the positive class. Two changes
were applied together:

| file | change |
|---|---|
| `tri_model/train1.py` | `GRAPH_FEATURE_NAMES` = all 23 extracted features; reads `features_output_all23_MG_fullscale.csv` (full 2,973,489-account coverage, `StandardScaler` fit on the `train` partition only) |
| `tri_model/eval_full_test.py` | new `--feature_set {all23,legacy10}`, default `all23` |

The univariate top-10 selection was **dropped rather than re-run**. Its basis is broken three ways
(old split / 600× enrichment / wrong label set), a re-run would rest on 519 train positives, and
23 features against 1.9M training rows is not a dimensionality problem. A feature-only baseline
confirms the choice empirically: on the full test partition, all-23 beats the old top-10 for both
model families (HistGBM F1(pos) 0.3479 vs 0.3168; LogReg 0.0657 vs 0.0514).

`legacy10` is retained deliberately: a checkpoint bakes the `FeatureProjector`'s input dimension, so
every pre-existing checkpoint — including the one behind `full_scale_test_eval_report.md` — can only
be re-scored under the configuration it was trained with. Scoring an old checkpoint as `all23` fails
on a shape mismatch rather than silently producing wrong numbers.

A performance defect had to be fixed first: the feature lookup built **one torch tensor per account**
via `df.iterrows()` — free at 5,655 rows, but >7 GB RSS and unfinished after 5 minutes at full
coverage. Replaced in both files by a single `[N, 23]` tensor plus a `str→row` index: **21 s and
2.86 GB** for a complete smoke run. Left unfixed it would have recurred on every full-scale
811,704-example evaluation.

### 18.2 Result — same corpus, same config, only the feature branch changed

BertAdam, `--lr 2e-5 --warmup_ratio 0.1 --max_epochs 15 --patience 5`, weighted sampler, neutral focal
alpha, F1(pos) selection; same 20,000/5,000×3 corpus, same `gcn_vocab_size=35,000`. Checkpoint
`..._vocab35000_all23_fullcov.pt`. Wall clock **240.2 min (~4.0 h)** vs Attempt 3's 6.65 h; early
stopping at epoch 7, **best epoch 2** (Attempt 3: epoch 7).

| epoch | Attempt 3 valid F1(pos) | all-23 valid F1(pos) |
|---:|---:|---:|
| 0 | 0.1863 | **0.4849** |
| 1 | 0.2022 | **0.5812** |
| 2 | 0.4764 | **0.6732** ← best |
| 3 | 0.5090 | 0.5324 |
| 4 | 0.5939 | 0.4912 |
| 5 | 0.6464 | 0.5808 |
| 6 | (0.6464) | 0.5767 |
| 7 | **0.6667** ← best | 0.5385 → early stop |

Calibrated threshold (val only, argmax F1): **0.8562** (val F1(pos) 0.7984).

| Slice | n | pos | F1(pos) | F1(weighted) | AUPRC | G-Mean | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_test | 5,000 | 312 | **0.7582** | 0.9718 | 0.7818 | 0.8122 | 0.2821 | 0.8526 |
| overlap | 5,000 | 217 | **0.5987** | 0.9640 | 0.4652 | 0.7886 | 0.2903 | 0.7788 |
| overall | 10,000 | 529 | **0.6852** | 0.9675 | 0.5822 | 0.8019 | 0.1304 | 0.6711 |

| | Attempt 3 (§11) | all-23 full coverage | Δ |
|---|---:|---:|---:|
| overall F1(pos) | 0.6227 | **0.6852** | **+0.0625** |
| pure_test F1(pos) | 0.7406 | **0.7582** | +0.0176 |
| overlap F1(pos) | 0.5107 | **0.5987** | **+0.0880** |
| overall AUPRC | 0.6904 | 0.5822 | **−0.1082** |

### 18.3 Reading this result — not a clean win

F1(pos) improves on every slice, and the largest gain is on `overlap` (+0.088) — the slice §11 and §9
both identified as this model's persistent weakness. That is what one would predict if the feature
branch went from inert to informative.

But **AUPRC moved the other way** (0.6904 → 0.5822). AUPRC is threshold-free; F1(pos) depends on a
threshold calibrated on 117 validation positives. The two metrics disagreeing means the new model
**ranks slightly worse overall but sits better at the chosen operating point**. With 529 test
positives a 0.06 F1 difference is also within the range a different random seed could produce. The
honest summary is *no worse, probably better at the operating point, ranking quality unresolved* — not
a clean improvement.

Two further caveats:

1. **Best epoch fell from 7 to 2**, after which validation F1(pos) degrades monotonically. With real
   features present the model peaks much sooner and then overfits — plausible, but it also means only
   three epochs of signal informed the selection.
2. **This is the bounded 10,000-node test at 5.29% positive**, the composition the feature-only
   baseline showed to be flattering (HistGBM 0.4937 here vs 0.3479 on the real distribution; Attempt 3
   0.6227 here vs 0.0549 there). **No conclusion about real-world capability follows from §18.2.**

The decisive measurement is the full-scale evaluation of this checkpoint —
`eval_full_test.py --ckpt ..._all23_fullcov.pt --feature_set all23` over all 811,704 test accounts,
against Attempt 3's 0.0549 and the feature-only baseline's 0.3479. **Not yet run**; tracked as the new
§15 item 2.

**Files**: `Dataset/tri_model/output/ETH_GBert16_model_Dataset_MG_cle_sw0_vocab35000_all23_fullcov.pt`,
`Dataset/logs/retrain_all23_fullcov.log`, `Dataset/fullscale_features/` (extraction + baseline code),
`raw_data/MulDiGraph/features_output_all23_MG_fullscale.csv`,
`raw_data/MulDiGraph/phisher_account_muldi.txt`.
