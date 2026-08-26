"""
eval_full_test.py
==================
Evaluation-only pass of an already-trained tri_model checkpoint against the
ENTIRE real test partition (811,704 accounts: overlap + pure_test) -- see
dynamic_fusion_leakage_audit/full_scale_test_eval_report.md for the full
write-up and why this needs its own script rather than reusing train1.py
directly.

Two things this script does that train1.py's normal path can't:

1. Loads a checkpoint trained at one vocab size (e.g. 35,000 accounts) into a
   model built at a BIGGER vocab size (the full test set, ~836k accounts).
   VocabGraphConvolution.W0_vh (ETH_GBert.py) is one learned embedding row per
   specific account in the training vocab -- transductive by construction, so
   a strict state_dict load would fail on shape mismatch. This script instead
   loads every OTHER weight (BERT, feature_projector, dynamic_fusion_layer,
   classifier -- none of which depend on vocab size) verbatim, and copies the
   checkpoint's W0_vh into the FIRST `old_vocab_size` rows of the new, bigger
   W0_vh (built by mg_build_full_test_eval.py as an exact prefix of the new
   vocab). The remaining rows (newly-added test accounts the checkpoint never
   saw) keep their random init -- no per-account learned GCN signal for those
   specific accounts, though their trained neighbors still propagate real
   signal through the adjacency. This is a stated methodology caveat, not a
   bug: BERT-text and graph-feature streams are fully trained either way.

2. Runs evaluation only -- no training loop, no gradient steps -- reusing
   train1.py's calibrate-on-val / score-test-at-calibrated-threshold protocol
   (Step 5/6) and utils.py's report_slice/g_mean/recall_at_k for the
   pure_test/overlap/overall breakdown.
"""

import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("WANDB_START_METHOD", "thread")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.dirname(_THIS_DIR)
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, precision_recall_curve

from pytorch_pretrained_bert.tokenization import BertTokenizer

from env_config import env_config
from ETH_GBert import ETH_GBertModel
from utils import (
    InputExample, CorpusDataset, sparse_scipy2torch, normalize_adj,
    report_slice, g_mean, recall_at_k,
)

BASE_DIR = os.path.dirname(_DATASET_DIR)
OLD_DATA_DIR = os.path.join(BASE_DIR, "data/preprocessed/multi_processed_data_MG")
FULL_TEST_DATA_DIR = os.path.join(BASE_DIR, "data/preprocessed/multi_processed_data_MG_full_test")

GCN_EMBEDDING_DIM = 16
MAX_SEQ_LENGTH = 400 + GCN_EMBEDDING_DIM
BATCH_SIZE = 8

# Two feature configurations. They are NOT interchangeable at load time: the
# FeatureProjector's input dimension is baked into a checkpoint, so a checkpoint
# must be scored with the configuration it was trained under.
#
#   all23   -- current default. All 23 extracted features, full 2,973,489-account
#              coverage (features_output_all23_MG_fullscale.csv).
#   legacy10 -- the old top-10 over the 5,655-row file, i.e. zero vectors for
#              99.81% of accounts. Required to reproduce or re-score any
#              checkpoint from before this change, including the numbers in
#              full_scale_test_eval_report.md.
#
# NOTE (transductive tradeoff, symmetric to gcn_adj_eval's full-graph
# adjacency): both CSVs' raw feature VALUES are computed once over the full
# graph timeline (no T_cutoff boundary) and reused unchanged for val/test
# here, same as they were for train in train1.py -- only the normalization
# stats were train-only-fit. See train1.py's features_csv_path comment and
# dynamic_fusion_leakage_audit/pipeline_review_vs_code_check.md §D/§H.
FEATURE_SETS = {
    "all23": (
        "raw_data/MulDiGraph/features_output_all23_MG_fullscale.csv",
        ["out_degree", "in_degree", "direction_ratio",
         "max_out_amount", "min_out_amount", "avg_out_amount",
         "max_in_amount", "min_in_amount", "avg_in_amount",
         "account_balance", "lifetime_days", "active_days",
         "freq_out_short", "freq_in_short", "freq_out_long", "freq_in_long",
         "short_long_out_ratio", "short_long_in_ratio",
         "betweenness_centrality", "degree_centrality", "clustering_coefficient",
         "in_degree_centrality", "out_degree_centrality"],
    ),
    "legacy10": (
        "raw_data/MulDiGraph/features_output_top10_MG_fixed.csv",
        ["betweenness_centrality", "clustering_coefficient", "in_degree", "freq_in_long",
         "out_degree", "freq_out_long", "freq_out_short", "max_out_amount",
         "in_degree_centrality", "active_days"],
    ),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to the trained tri_model checkpoint (.pt)")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only score the first N test examples (timing/correctness check, not the real run)")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--run_name", type=str, default="full_test_eval_811704")
    ap.add_argument("--feature_set", choices=sorted(FEATURE_SETS), default="all23",
                    help="Must match what the checkpoint was trained with. Checkpoints from "
                         "before the full-coverage switch need --feature_set legacy10; scoring "
                         "them as all23 fails on the FeatureProjector's input dimension.")
    ap.add_argument(
        "--old_data_dir", type=str, default=OLD_DATA_DIR,
        help="Bounded-run corpus the checkpoint was trained on -- supplies the validation set "
             "used for threshold calibration. Must match the checkpoint (and the corpus whose "
             "vocab was used as the extended vocab's prefix).",
    )
    ap.add_argument(
        "--full_test_dir", type=str, default=FULL_TEST_DATA_DIR,
        help="Full-test corpus directory built by mg_build_full_test_eval.py.",
    )
    return ap.parse_args()


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    args = parse_args()
    old_data_dir = args.old_data_dir
    full_test_dir = args.full_test_dir
    cuda_yes = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_yes else "cpu")
    print(f"Device: {device}")
    print(f"  old_data_dir  = {old_data_dir}")
    print(f"  full_test_dir = {full_test_dir}")

    print("\n----- Load val (unchanged, from the bounded-run corpus) -----")
    train_y = load_pkl(os.path.join(old_data_dir, "data_Dataset_MG.train_y"))
    valid_y = load_pkl(os.path.join(old_data_dir, "data_Dataset_MG.valid_y"))
    valid_y_prob = load_pkl(os.path.join(old_data_dir, "data_Dataset_MG.valid_y_prob"))
    shuffled_clean_docs_old = load_pkl(os.path.join(old_data_dir, "data_Dataset_MG.shuffled_clean_docs"))
    doc_accounts_old = load_pkl(os.path.join(old_data_dir, "data_Dataset_MG.doc_accounts"))
    train_size = len(train_y)
    valid_size = len(valid_y)
    valid_docs = shuffled_clean_docs_old[train_size: train_size + valid_size]
    valid_doc_accounts = doc_accounts_old[train_size: train_size + valid_size]
    assert len(valid_docs) == valid_size
    print(f"  valid: n={valid_size} pos={int(valid_y.sum())}")

    print("\n----- Load full test (811,704 accounts, from mg_build_full_test_eval.py) -----")
    test_y = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.test_y"))
    test_y_prob = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.test_y_prob"))
    test_partition = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.test_partition"))
    test_docs = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.shuffled_clean_docs"))
    test_doc_accounts = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.doc_accounts"))
    address_to_index = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.address_to_index"))
    old_vocab_size = load_pkl(os.path.join(full_test_dir, "data_Dataset_MG_full_test.old_vocab_size"))
    gcn_vocab_size = len(address_to_index)
    print(f"  test: n={len(test_y)} pos={int(test_y.sum())}  gcn_vocab_size={gcn_vocab_size:,} (old={old_vocab_size:,})")

    # Optional second label set (written by mg_build_full_test_eval.py whenever
    # --labels_source isn't the strict ground truth): the SAME predictions get
    # scored against both, so an expanded-label run stays directly comparable
    # to the strict-isp full-scale numbers in full_scale_test_eval_report.md.
    strict_path = os.path.join(full_test_dir, "data_Dataset_MG_full_test.test_y_strict")
    test_y_strict = load_pkl(strict_path) if os.path.exists(strict_path) else None
    if test_y_strict is not None:
        print(f"  strict-isp labels also present: pos={int(test_y_strict.sum())} "
              f"-- will report both label sets on the same predictions")

    if args.smoke > 0:
        print(f"  [SMOKE] truncating test set to first {args.smoke} examples")
        test_y = test_y[: args.smoke]
        test_y_prob = test_y_prob[: args.smoke]
        test_partition = test_partition[: args.smoke]
        test_docs = test_docs[: args.smoke]
        test_doc_accounts = test_doc_accounts[: args.smoke]
        if test_y_strict is not None:
            test_y_strict = test_y_strict[: args.smoke]

    print("\n----- Build InputExamples -----")
    valid_examples = [
        InputExample(address_to_index[valid_doc_accounts[i]], valid_docs[i].strip(), confidence=valid_y_prob[i], label=int(valid_y[i]))
        for i in range(valid_size)
    ]
    test_examples = [
        InputExample(address_to_index[test_doc_accounts[i]], test_docs[i].strip(), confidence=test_y_prob[i], label=int(test_y[i]))
        for i in range(len(test_y))
    ]
    assert len(test_examples) == len(test_partition)

    print("\n----- Load extended adjacency (full transductive graph, sliced to this vocab) -----")
    adj_mat = load_npz(os.path.join(FULL_TEST_DATA_DIR, "gcn_adj_eval.npz"))
    assert adj_mat.shape[0] == gcn_vocab_size, f"adjacency shape {adj_mat.shape} != gcn_vocab_size {gcn_vocab_size}"
    gcn_adj_list_eval = [sparse_scipy2torch(normalize_adj(csr_matrix(adj_mat)).tocoo()).to(device)]

    rel_csv, TOP10_FEATURE_NAMES = FEATURE_SETS[args.feature_set]
    NUM_GRAPH_FEATURES = len(TOP10_FEATURE_NAMES)
    print(f"\n----- Load graph features: feature_set={args.feature_set} "
          f"({NUM_GRAPH_FEATURES} features) -----")
    features_csv_path = os.path.join(BASE_DIR, rel_csv)
    features_df = pd.read_csv(features_csv_path)
    zero_features = torch.zeros(NUM_GRAPH_FEATURES, dtype=torch.float)

    class GraphFeatureLookup:
        """One [N, F] tensor + str->row index, not one tensor per account.

        The per-account dict comprehension this replaces was fine at the old
        5,655-row coverage but costs >7 GB and minutes at the full 2,973,489.
        """

        __slots__ = ("_m", "_i", "_default")

        def __init__(self, df, cols, default):
            self._m = torch.from_numpy(np.ascontiguousarray(
                df[cols].to_numpy(dtype=np.float32)))
            self._i = {a: k for k, a in enumerate(df["node"].astype(str).str.lower())}
            self._default = default

        def get(self, key, default=None):
            k = self._i.get(key)
            return self._default if k is None else self._m[k]

        def __contains__(self, key):
            return key in self._i

        def __len__(self):
            return len(self._i)

    graph_features_lookup = GraphFeatureLookup(
        features_df, TOP10_FEATURE_NAMES, zero_features)
    del features_df
    cov = 100.0 * len(graph_features_lookup) / 2_973_489
    print(f"  {len(graph_features_lookup):,} accounts have real graph features "
          f"({cov:.2f}% of the graph); the rest fall back to a zero vector")

    bert_model_scale = "bert-base-uncased"
    if env_config.TRANSFORMERS_OFFLINE == 1:
        bert_model_scale = os.path.join(env_config.HUGGING_LOCAL_MODEL_FILES_PATH, f"hf-maintainers_{bert_model_scale}")
    tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=True)

    def get_dataloader(examples, batch_size):
        ds = CorpusDataset(
            examples, tokenizer, address_to_index, MAX_SEQ_LENGTH, GCN_EMBEDDING_DIM,
            graph_features_lookup=graph_features_lookup, zero_features=zero_features,
        )
        return torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds.pad)

    valid_dataloader = get_dataloader(valid_examples, args.batch_size)
    test_dataloader = get_dataloader(test_examples, args.batch_size)

    print("\n----- Build model + load checkpoint (vocab-extended partial load) -----")
    model = ETH_GBertModel.from_pretrained(
        bert_model_scale, gcn_adj_dim=gcn_vocab_size, gcn_adj_num=1,
        gcn_embedding_dim=GCN_EMBEDDING_DIM, num_labels=2, num_graph_features=NUM_GRAPH_FEATURES,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ckpt["model_state"])
    gcn_key = "embeddings.vocab_gcn.W0_vh"
    old_W = sd.pop(gcn_key)
    assert old_W.shape[0] == old_vocab_size, f"checkpoint W0_vh rows ({old_W.shape[0]}) != old_vocab_size ({old_vocab_size}) -- wrong checkpoint/corpus pairing"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={missing}  unexpected={unexpected}")
    assert missing == [gcn_key], f"expected only {gcn_key} missing, got {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    with torch.no_grad():
        model.embeddings.vocab_gcn.W0_vh.data[: old_W.shape[0]] = old_W
    assert torch.equal(model.embeddings.vocab_gcn.W0_vh.data[: old_W.shape[0]].cpu(), old_W), "W0_vh prefix copy verification failed"
    print(f"  Checkpoint epoch={ckpt.get('epoch')} perform_metrics={ckpt.get('perform_metrics')} -- loaded, W0_vh[:{old_W.shape[0]}] verified bit-identical to checkpoint")
    model.to(device)
    model.eval()

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.login()
        wandb.init(project="fraud_detection", name=args.run_name, config=vars(args))

    def evaluate(dataloader, name, threshold=None, log_every=2000):
        predict_out, pos_probs, all_label_ids = [], [], []
        total, correct = 0, 0
        n_batches = len(dataloader)
        start = time.time()
        with torch.no_grad():
            # gcn_adj_list_eval and the model weights are frozen for this whole
            # call (eval mode, no_grad) -- compute the batch-invariant
            # sparse_mm(adj, W) ONCE instead of once per batch inside
            # model.forward() (see VocabGraphConvolution.compute_H_vh). This is
            # the dominant cost at full scale: ~101k batches (8 examples/batch
            # over 811,704 accounts) were each redoing the same full-graph
            # sparse matmul before this change.
            precomputed_H_vh = model.compute_gcn_H_vh(gcn_adj_list_eval)
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_vocab_ids, graph_features, _sw) = batch
                logits = model(
                    gcn_adj_list_eval, gcn_vocab_ids, input_ids, graph_features, segment_ids, input_mask,
                    precomputed_H_vh=precomputed_H_vh,
                )
                probs = F.softmax(logits, dim=-1)
                batch_pos_prob = probs[:, 1]
                if threshold is None:
                    _, predicted = torch.max(logits, -1)
                else:
                    predicted = (batch_pos_prob >= threshold).long()
                predict_out.extend(predicted.tolist())
                pos_probs.extend(batch_pos_prob.tolist())
                all_label_ids.extend(label_ids.tolist())
                correct += predicted.eq(label_ids).sum().item()
                total += len(label_ids)
                if step > 0 and step % log_every == 0:
                    elapsed = time.time() - start
                    rate = elapsed / step
                    eta_min = rate * (n_batches - step) / 60.0
                    print(f"  [{name}] step {step}/{n_batches}  {rate*1000:.1f}ms/step  elapsed={elapsed/60:.1f}m  ETA={eta_min:.1f}m")
                    sys.stdout.flush()
        y_true_arr = np.array(all_label_ids).reshape(-1)
        y_pred_arr = np.array(predict_out).reshape(-1)
        f1_pos = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted")
        print(f"[{name}] n={total} F1(pos)={f1_pos:.4f} F1(weighted)={f1_weighted:.4f} Acc={correct/total:.4f}  "
              f"Spend {(time.time()-start)/60.0:.1f}m")
        print(classification_report(y_true_arr, y_pred_arr, digits=4))
        return y_true_arr, y_pred_arr, np.array(pos_probs)

    def calibrate(y_true, pos_probs, label_name):
        precisions, recalls, thresholds = precision_recall_curve(y_true, pos_probs)
        f1s = np.where((precisions + recalls) > 0, 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12), 0.0)
        best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else None
        thr = float(thresholds[best_idx]) if best_idx is not None else 0.5
        print(f"  Calibrated threshold on val ({label_name}, argmax F1): {thr:.4f}  "
              f"(val F1(pos) at this point: {f1s[best_idx]:.4f})")
        return thr

    print("\n----- Step 5: calibrate threshold on VALIDATION only -----")
    valid_y_true, _, valid_pos_probs = evaluate(valid_dataloader, "Valid_set(calibration)")
    calibrated_threshold = calibrate(valid_y_true, valid_pos_probs, "this corpus' labels")

    # Second calibration against STRICT ground truth on the same val probabilities
    # (no extra forward pass) -- so the strict-isp comparison below follows the
    # exact protocol full_scale_test_eval_report.md's numbers were produced with:
    # calibrate on val strict, score test strict.
    strict_threshold = None
    if test_y_strict is not None:
        import pickle as _pkl
        split_labels_path = os.path.join(BASE_DIR, "data/preprocessed/Dataset_MG/labels.pkl")
        with open(split_labels_path, "rb") as f:
            _strict_labels = _pkl.load(f)
        valid_y_strict = np.array([_strict_labels[a] for a in valid_doc_accounts], dtype=np.int64)
        del _strict_labels
        print(f"  val strict-isp positives: {int(valid_y_strict.sum())} (vs {int(valid_y_true.sum())} in this corpus' labels)")
        if 0 < valid_y_strict.sum() < len(valid_y_strict):
            strict_threshold = calibrate(valid_y_strict, valid_pos_probs, "strict isp")

    print(f"\n----- Step 6: score ALL {len(test_examples):,} test accounts at the calibrated threshold -----")
    test_y_true, test_y_pred, test_pos_probs = evaluate(test_dataloader, "Test_set(full,calibrated)", threshold=calibrated_threshold, log_every=2000)

    test_partition_arr = np.array(test_partition)
    metrics_pure_test = report_slice("pure_test", test_partition_arr == "pure_test", test_y_true, test_y_pred, test_pos_probs)
    metrics_overlap = report_slice("overlap", test_partition_arr == "overlap", test_y_true, test_y_pred, test_pos_probs)
    metrics_overall = report_slice("overall (pure_test+overlap)", np.ones(len(test_y_true), dtype=bool), test_y_true, test_y_pred, test_pos_probs)

    # ── Same predictions, scored against STRICT ground-truth isp ─────────────
    # Directly comparable to full_scale_test_eval_report.md §4's numbers (which
    # used strict isp throughout). Nothing is re-inferred: identical pos_probs,
    # only the label vector and the threshold differ.
    metrics_strict = {}
    if test_y_strict is not None and strict_threshold is not None:
        print(f"\n----- Step 6b: SAME predictions re-scored against STRICT ground-truth isp -----")
        print(f"  strict labels: n={len(test_y_strict):,} pos={int(test_y_strict.sum()):,}  "
              f"threshold={strict_threshold:.4f} (calibrated on val strict-isp)")
        test_y_pred_strict = (test_pos_probs >= strict_threshold).astype(np.int64)
        metrics_strict["pure_test"] = report_slice("pure_test [strict isp]", test_partition_arr == "pure_test", test_y_strict, test_y_pred_strict, test_pos_probs)
        metrics_strict["overlap"] = report_slice("overlap [strict isp]", test_partition_arr == "overlap", test_y_strict, test_y_pred_strict, test_pos_probs)
        metrics_strict["overall"] = report_slice("overall [strict isp]", np.ones(len(test_y_strict), dtype=bool), test_y_strict, test_y_pred_strict, test_pos_probs)

    print("\n----- Step 8: runtime validation checks -----")
    print(f"  [{'PASS' if len(test_y_true) == len(test_y) else 'FAIL'}] test n matches expected ({len(test_y_true)} vs {len(test_y)})")
    print(f"  [{'PASS' if int(test_y_true.sum()) == int(test_y.sum()) else 'FAIL'}] test positives match expected ({int(test_y_true.sum())} vs {int(test_y.sum())})")

    if use_wandb:
        final_metrics = {"final/calibrated_threshold": calibrated_threshold, "final/test_n": len(test_y_true), "final/test_pos": int(test_y_true.sum())}
        slices = [("pure_test", metrics_pure_test), ("overlap", metrics_overlap), ("overall", metrics_overall)]
        slices += [(f"strict_{k}", v) for k, v in metrics_strict.items()]
        for slice_name, m in slices:
            if m is None:
                continue
            for k, v in m.items():
                if k == "recall_at":
                    for kk, vv in v.items():
                        final_metrics[f"final/{slice_name}_recall_at_{kk}"] = vv
                else:
                    final_metrics[f"final/{slice_name}_{k}"] = v
        wandb.log(final_metrics)
        for k, v in final_metrics.items():
            wandb.summary[k] = v
        wandb.finish()


if __name__ == "__main__":
    main()
