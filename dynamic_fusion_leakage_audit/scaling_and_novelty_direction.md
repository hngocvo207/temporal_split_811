# Scaling ETH_GBert to full MulDiGraph + novelty positioning against current SOTA

Companion to `PROCESS_LOG.md` and `Dynamic_Fusion/preprocessing_and_eval_report.md`. Those documents
fixed the leakage/evaluation issues; this one answers two follow-up questions: (1) what's actually
needed to run `ETH_GBert.py` on the **full** 2,973,489-node MulDiGraph without changing its
architecture's math, and (2) given where the field already is (surveyed below), what's a genuinely
novel contribution rather than a re-derivation of published work.

---

## 1. Root cause of the scale ceiling (precise, not just "it's big")

`Dataset/utils.py`, `CorpusDataset.pad()`:
```python
batch_gcn_swop_eye = F.one_hot(indices, num_classes=gcn_vocab_size + 1).float()
...
batch_gcn_swop_eye = batch_gcn_swop_eye.view(len(batch_core), -1, gcn_vocab_size).transpose(1, 2)
```
`Dataset/ETH_GBert.py`, `VocabGraphConvolution`:
```python
self.W%d_vh = nn.Parameter(torch.randn(voc_dim, hid_dim))          # [vocab_size, 128]
H_dh = X_dv.matmul(H_vh)                                            # X_dv carries a vocab_size dim
```
Two structures scale with `gcn_vocab_size` (= number of distinct accounts in the corpus):
- `gcn_swop_eye`: **dense** one-hot tensor, shape `[batch, vocab_size, seq_len]`. At
  `vocab_size=2,973,489`, `batch=8`, `seq_len=416`: **≈39.6 TB per batch**. This is what actually
  crashes — not "slow," instantly unallocatable.
- `W_vh`: dense `[vocab_size, 128]` parameter. At full scale: ~1.5 GB (×~3 with gradient + Adam
  state ≈ 4.6 GB) — survivable alone, but not next to the tensor above.

Critically, the **sparse graph convolution itself is not the bottleneck**:
`torch.sparse.mm(vocab_adj_list[i], W_vh)` is `O(nnz)` — with `adj_inference.npz` at nnz≈5.35M (built
in this session, §3 of `preprocessing_and_eval_report.md`), this multiply is cheap even at full scale.
The blowup is specifically in how `CorpusDataset.pad()` turns "which vocab row does this token map to"
into a tensor: it materializes a one-hot indicator instead of using an index.

## 2. The fix is a known, mathematically-exact pattern — not a new technique

`gcn_swop_eye_token @ words_embeddings` (building the GCN input) and `X_dv.matmul(H_vh)` (routing GCN
output back per-token) are both **one-hot matrix multiplications used to implement a gather** — this
is exactly what `nn.Embedding` / `torch.index_select` do without ever materializing the one-hot
tensor. Concretely:
- Building `vocab_input`: replace `gcn_swop_eye @ words_embeddings` with a `scatter_add`/segment-sum
  over the (batch, token) → vocab-id pairs already computed in `example2feature`'s `gcn_vocab_ids`
  (no dense intermediate).
- Routing `H_vh` back to tokens: replace `X_dv.matmul(H_vh)` with `H_vh[gcn_vocab_ids]` (a plain
  index/gather), memory `O(batch·seq_len·hid_dim)` instead of `O(batch·seq_len·vocab_size)`.

This is provably identical output — one-hot-matmul-by-a-matrix *is* row-gather from that matrix, just
computed the expensive way. External confirmation this is the standard fix, not a guess:
- **VGCN-BERT** (the architecture family `ETH_GBert.py`'s GCN-injection is derived from) shipped a
  scalability update to its own vocab-graph-conv computation for exactly this reason — the maintained
  repo notes "greatly speeds up the calculation speed of embedding vocabulary graph convolutional
  network" with an "updated subgraph selection algorithm," though the exact internals aren't
  documented in the README ([Louis-udm/VGCN-BERT](https://github.com/Louis-udm/VGCN-BERT); original
  paper: [Lu et al., 2020](https://arxiv.org/abs/2004.05707)).
- **TLMG4Eth** ([Ethereum Fraud Detection via Joint Transaction Language Model and Graph
  Representation Learning, 2024](https://arxiv.org/pdf/2409.07494)) — architecturally the closest
  published relative of `ETH_GBert.py` (BERT transaction-sentences + vocabulary co-occurrence graph +
  GCN + account-interaction graph + fusion) — reports handling MulDiGraph at its real full size
  (2,973,489 nodes) via "batch updates and dictionary-based embedding tracking," i.e. a lookup-table
  pattern, not a dense one-hot tensor. This is direct evidence the same architectural family scales
  to this exact dataset once the routing is done via lookup instead of one-hot.
- General precedent for the identical problem in the plain-text-GCN literature: **TextGCN**'s original
  one-hot/transductive design doesn't scale past small corpora; **InducT-GCN**
  ([Wang et al., 2022](https://arxiv.org/abs/2206.00265)) fixes this by representing documents as a
  weighted sum of word vectors instead of one-hot rows, specifically to reduce time/space complexity
  for larger vocabularies.

**What this does NOT require**: touching `ETH_GBertModel`'s fusion logic, the BERT encoder, or the
DynamicFusionLayer — the fix is local to `CorpusDataset.pad()` (utils.py) and the two matmuls in
`VocabGraphConvolution.forward()` (ETH_GBert.py), replacing a wasteful-but-equivalent computation with
a cheap-and-equivalent one. It is a scale fix, not an architecture redesign.

## 3. Second-order scale concern: raw compute time, not memory

Even with the one-hot fix, full-scale training (1,566,230 train examples) is a genuine compute-time
problem: at this session's observed ~45-50 min/epoch for a 20,000-example subsample on a single RTX
3060 (see `PROCESS_LOG.md`, Step 7 run), a naive linear scale-up to 1.56M examples implies **~1-2
days/epoch** on this GPU. Two standard, literature-backed levers, applicable without touching
`ETH_GBert.py`'s math:
- **Mixed precision + larger batch + multiple GPUs** — mechanical, no methodology change.
- **Graph-side mini-batching for the GCN component** — Cluster-GCN
  ([Chiang et al., 2019](https://github.com/GraphSAINT/GraphSAINT)) or GraphSAINT (same repo) style
  subgraph sampling would let the vocabulary graph itself be mini-batched instead of doing one
  full-graph sparse conv per step; likely unnecessary here since §1 already showed the sparse conv is
  cheap at this graph's actual nnz (~5.35M), but relevant if a future dataset/threshold makes it not.
- **BERT-side**: the real per-step cost driver is BERT-base forward/backward on `MAX_SEQ_LENGTH=416`,
  independent of vocab size — standard mitigations (gradient checkpointing, sequence-length reduction
  via the existing `MAX_TX_PER_ACCOUNT_TEXT` cap already added in `mg_build_examples.py` this session,
  FP16) apply directly.

## 4. SOTA landscape (2024-2026) — where this project already sits, and where it doesn't

| Work | Year | Method | MulDiGraph? | Split method disclosed? | Leakage/temporal validity discussed? |
|---|---|---|---|---|---|
| Dynamic Feature Fusion (this project's paper) | 2025 | BERT + VGCN + graph-features fusion | Yes (2,973,489 / 13,551,303 / 1,165) | Random 80/10/10 + 5:5 resample (per this session's Step 0 audit) | No |
| [TLMG4Eth](https://arxiv.org/pdf/2409.07494) | 2024 | BERT transaction-LM + vocab co-occurrence GCN + account-interaction GCN + fusion | Yes, same exact counts | **Not disclosed** | **Not discussed** |
| [KGBERT4Eth](https://arxiv.org/pdf/2509.03860) | 2025 | BERT + knowledge-graph embeddings, multi-task | Yes (xblock/Etherscan-sourced) | Not clearly disclosed | Not discussed |
| [LMAE4Eth](https://arxiv.org/pdf/2509.03939) | 2025 | Masked graph autoencoder + transaction semantics | Yes — reports ~+10.24% F1 over strongest baseline on MulDiGraph | Not extracted/unclear | Not discussed |
| [DiT-SGCR](https://arxiv.org/pdf/2506.20123) | 2025 | Directed **temporal** structural representation, global-cluster aware | Not confirmed | Uses temporal *features*; split methodology unclear | **Not explicitly addressed**, despite "temporal" framing |
| [FraudGT](https://research.ibm.com/publications/fraudgt-a-simple-effective-and-efficient-graph-transformer-for-financial-fraud-detection) | 2024 | Graph transformer, edge-gated message passing, edge-attribute attention bias | No (different financial fraud graphs) | N/A | N/A — but explicitly scalability-first (2.4x throughput) |
| PEAE-GNN, ETGNN, TTGAN, PDGNN | 2024-2025 | Ego-graph augmentation / edge-topology / temporal-attention / Chebyshev-GCN variants | Mixed | Mixed, generally not the focus | Generally not the focus |

**The pattern that stands out**: MulDiGraph is now a standard, actively-used full-scale benchmark
(confirmed by TLMG4Eth, KGBERT4Eth, LMAE4Eth all reporting on it with F1 in the 80-90%+ range) — so
"train at full scale" per se is not novel; multiple 2024-2025 papers already do it. What's consistently
**absent across every one of them**, including this project's own source paper, is any disclosed
temporal split or discussion of label/adjacency leakage across a train/test boundary — exactly the
class of issue this session's Steps 2-6 fixed for this specific architecture. TLMG4Eth is the closest
architectural relative and reports the largest MulDiGraph jump (+20.12% F1 over BERT4ETH) with **zero**
split-methodology disclosure — i.e., there is a real, live possibility that a meaningful fraction of
the field's current MulDiGraph leaderboard is inflated by the same random-split leakage this project's
own paper had.

## 5. Recommended novel direction (avoids re-deriving published ideas)

Not: "a new BERT+GCN fusion architecture for Ethereum fraud" — that space is now crowded (TLMG4Eth,
KGBERT4Eth, LMAE4Eth, ETGNN, PEAE-GNN, TTGAN, PDGNN all occupy it, 2024-2025).

Instead: **a leakage-audited, full-scale re-evaluation of the BERT+VGCN fusion family on MulDiGraph**,
positioned as an evaluation-methodology contribution:

1. Apply the one-hot→gather scalability fix (§2) to `ETH_GBert.py`'s vocab routing — enables genuine
   full-scale (or much-larger-than-currently-possible) training without changing what the model
   computes.
2. Run it under the **already-built** leakage-free temporal protocol from this session
   (`mg_temporal_pipeline.py` position-based 60/20/20 split, train-cutoff adjacency, no val/test
   resampling, focal loss, calibrated threshold, pure_test/overlap/overall breakdown) — this pipeline
   already exists and was validated end-to-end (`PROCESS_LOG.md`, Step 7 large-bounded run).
3. Report the **gap** between (a) this project's/TLMG4Eth-style random-split, resampled numbers and
   (b) the temporal, real-imbalance numbers — for the *same* architecture family. This is the "Step 7
   dual-protocol comparison table" already scaffolded in `preprocessing_and_eval_report.md` §9, just
   extended to explicitly cite and, ideally, reproduce TLMG4Eth's reported 90.41% MulDiGraph F1 as the
   "random-split, undisclosed-methodology" reference point being interrogated.
4. If the gap is large (this project's own Step 0 audit already found +1.7 to +4.4 F1 points of
   overlap-label leakage alone, before even accounting for resampling) — that is itself the
   contribution: evidence that a substantial share of current Ethereum-fraud-detection SOTA on this
   *specific, standard, widely-cited* benchmark needs re-evaluation under realistic temporal
   evaluation, with a concrete, open pipeline for doing so.

This directly extends work already done in this session (not a restart), is falsifiable/reproducible,
and targets a gap that a real literature pass (10 papers surveyed here) shows is genuinely open rather
than assumed.

## 6. Concrete next steps

1. Implement the `gcn_swop_eye`/`W_vh` gather-based rewrite in `utils.py` + `ETH_GBert.py` (§2) —
   estimated a few hours of focused work + regression testing against the existing smoke-test numbers
   in `preprocessing_and_eval_report.md` §8 (must reproduce them, since the rewrite is mathematically
   exact, not approximate).
2. Re-run `mg_build_examples.py --cap_train 0 ...` (or a much larger bound than 20,000) now that the
   vocab-size ceiling is gone — re-check wall-clock feasibility per §3 before committing to a literal
   1.56M-example run.
3. Extend `train1.py`'s Step 7 scaffold to also load/report TLMG4Eth's published MulDiGraph number as
   an external reference row in the comparison table (already has a slot for "ORIGINAL protocol,
   ~94.71% F1" — add a second reference row for TLMG4Eth's 90.41%).
4. Write up the gap as the paper's actual contribution, citing the works surveyed in §4.
