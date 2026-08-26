"""
eval_full_test_origin.py
=========================
bi_model counterpart of tri_model/eval_full_test.py -- evaluation-only pass of
the already-trained bi_model (BERT+GCN, no graph-feature stream) checkpoint
against the ENTIRE real test partition (811,704 accounts: overlap + pure_test).
See dynamic_fusion_leakage_audit/full_scale_test_eval_report.md for the
tri_model write-up this mirrors, and why this needs its own script rather than
train_origin.py's normal path (same two reasons: vocab-locked GCN branch,
eval-only vs. a training loop).

Reuses the SAME full-test corpus tri_model's run built
(data/preprocessed/multi_processed_data_MG_full_test/) -- bi_model's best
checkpoint (bi_modal_run_report.md §9) was trained on the exact same
20,000/5,000x3, gcn_vocab_size=35,000 corpus as tri_model's Attempt 3, so the
old-vocab-as-prefix extension already built is valid for this checkpoint too;
no need to rebuild it.
"""

import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("WANDB_START_METHOD", "thread")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.dirname(_THIS_DIR)
_TRI_MODEL_DIR = os.path.join(_DATASET_DIR, "tri_model")
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)
if _TRI_MODEL_DIR not in sys.path:
    sys.path.insert(0, _TRI_MODEL_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import f1_score, classification_report, precision_recall_curve

from pytorch_pretrained_bert.tokenization import BertTokenizer

from env_config import env_config
from ETH_GBert_origin import ETH_GBertModel
from utils_origin import InputExample, CorpusDataset, sparse_scipy2torch, normalize_adj
# Pure metric helpers, architecture-independent -- reuse tri_model's, don't
# duplicate (see tri_model/utils.py's report_slice/g_mean/recall_at_k).
from utils import report_slice, g_mean, recall_at_k

BASE_DIR = os.path.dirname(_DATASET_DIR)
OLD_DATA_DIR = os.path.join(BASE_DIR, "data/preprocessed/multi_processed_data_MG")
FULL_TEST_DATA_DIR = os.path.join(BASE_DIR, "data/preprocessed/multi_processed_data_MG_full_test")

GCN_EMBEDDING_DIM = 16
MAX_SEQ_LENGTH = 400 + GCN_EMBEDDING_DIM
BATCH_SIZE = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to the trained bi_model checkpoint (.pt)")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only score the first N test examples")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--run_name", type=str, default="full_test_eval_811704_bimodal")
    return ap.parse_args()


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    args = parse_args()
    cuda_yes = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_yes else "cpu")
    print(f"Device: {device}")

    print("\n----- Load val (unchanged, from the bounded-run corpus) -----")
    train_y = load_pkl(os.path.join(OLD_DATA_DIR, "data_Dataset_MG.train_y"))
    valid_y = load_pkl(os.path.join(OLD_DATA_DIR, "data_Dataset_MG.valid_y"))
    valid_y_prob = load_pkl(os.path.join(OLD_DATA_DIR, "data_Dataset_MG.valid_y_prob"))
    shuffled_clean_docs_old = load_pkl(os.path.join(OLD_DATA_DIR, "data_Dataset_MG.shuffled_clean_docs"))
    doc_accounts_old = load_pkl(os.path.join(OLD_DATA_DIR, "data_Dataset_MG.doc_accounts"))
    train_size = len(train_y)
    valid_size = len(valid_y)
    valid_docs = shuffled_clean_docs_old[train_size: train_size + valid_size]
    valid_doc_accounts = doc_accounts_old[train_size: train_size + valid_size]
    assert len(valid_docs) == valid_size
    print(f"  valid: n={valid_size} pos={int(valid_y.sum())}")

    print("\n----- Load full test (811,704 accounts, from mg_build_full_test_eval.py) -----")
    test_y = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.test_y"))
    test_y_prob = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.test_y_prob"))
    test_partition = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.test_partition"))
    test_docs = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.shuffled_clean_docs"))
    test_doc_accounts = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.doc_accounts"))
    address_to_index = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.address_to_index"))
    old_vocab_size = load_pkl(os.path.join(FULL_TEST_DATA_DIR, "data_Dataset_MG_full_test.old_vocab_size"))
    gcn_vocab_size = len(address_to_index)
    print(f"  test: n={len(test_y)} pos={int(test_y.sum())}  gcn_vocab_size={gcn_vocab_size:,} (old={old_vocab_size:,})")
    assert old_vocab_size == 35000, (
        f"old_vocab_size={old_vocab_size} != 35000 -- this corpus was built from a different bounded "
        f"run than bi_model's checkpoint; re-check data/preprocessed/multi_processed_data_MG/ matches "
        f"the corpus bi_modal_run_report.md §9 trained on before reusing it here."
    )

    if args.smoke > 0:
        print(f"  [SMOKE] truncating test set to first {args.smoke} examples")
        test_y = test_y[: args.smoke]
        test_y_prob = test_y_prob[: args.smoke]
        test_partition = test_partition[: args.smoke]
        test_docs = test_docs[: args.smoke]
        test_doc_accounts = test_doc_accounts[: args.smoke]

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

    bert_model_scale = "bert-base-uncased"
    if env_config.TRANSFORMERS_OFFLINE == 1:
        bert_model_scale = os.path.join(env_config.HUGGING_LOCAL_MODEL_FILES_PATH, f"hf-maintainers_{bert_model_scale}")
    tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=True)

    def get_dataloader(examples, batch_size):
        ds = CorpusDataset(examples, tokenizer, address_to_index, MAX_SEQ_LENGTH, GCN_EMBEDDING_DIM)
        return torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds.pad)

    valid_dataloader = get_dataloader(valid_examples, args.batch_size)
    test_dataloader = get_dataloader(test_examples, args.batch_size)

    print("\n----- Build model + load checkpoint (vocab-extended partial load) -----")
    model = ETH_GBertModel.from_pretrained(
        bert_model_scale, gcn_adj_dim=gcn_vocab_size, gcn_adj_num=1,
        gcn_embedding_dim=GCN_EMBEDDING_DIM, num_labels=2,
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
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_vocab_ids) = batch
                logits = model(gcn_adj_list_eval, gcn_vocab_ids, input_ids, segment_ids, input_mask)
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

    print("\n----- Step 5: calibrate threshold on VALIDATION only -----")
    valid_y_true, _, valid_pos_probs = evaluate(valid_dataloader, "Valid_set(calibration)")
    precisions, recalls, thresholds = precision_recall_curve(valid_y_true, valid_pos_probs)
    f1s = np.where((precisions + recalls) > 0, 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12), 0.0)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else None
    calibrated_threshold = float(thresholds[best_idx]) if best_idx is not None else 0.5
    print(f"  Calibrated threshold (val, argmax F1): {calibrated_threshold:.4f}  (val F1(pos) at this point: {f1s[best_idx]:.4f})")

    print(f"\n----- Step 6: score ALL {len(test_examples):,} test accounts at the calibrated threshold -----")
    test_y_true, test_y_pred, test_pos_probs = evaluate(test_dataloader, "Test_set(full,calibrated)", threshold=calibrated_threshold, log_every=2000)

    test_partition_arr = np.array(test_partition)
    metrics_pure_test = report_slice("pure_test", test_partition_arr == "pure_test", test_y_true, test_y_pred, test_pos_probs)
    metrics_overlap = report_slice("overlap", test_partition_arr == "overlap", test_y_true, test_y_pred, test_pos_probs)
    metrics_overall = report_slice("overall (pure_test+overlap)", np.ones(len(test_y_true), dtype=bool), test_y_true, test_y_pred, test_pos_probs)

    print("\n----- Step 8: runtime validation checks -----")
    print(f"  [{'PASS' if len(test_y_true) == len(test_y) else 'FAIL'}] test n matches expected ({len(test_y_true)} vs {len(test_y)})")
    print(f"  [{'PASS' if int(test_y_true.sum()) == int(test_y.sum()) else 'FAIL'}] test positives match expected ({int(test_y_true.sum())} vs {int(test_y.sum())})")

    if use_wandb:
        final_metrics = {"final/calibrated_threshold": calibrated_threshold, "final/test_n": len(test_y_true), "final/test_pos": int(test_y_true.sum())}
        for slice_name, m in (("pure_test", metrics_pure_test), ("overlap", metrics_overlap), ("overall", metrics_overall)):
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
