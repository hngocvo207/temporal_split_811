"""
E2 v2 -- sua 2 loi cua E2 v1 (xem STATUS.md muc "CANH BAO QUAN TRONG: E2 dung
nham nhan lan truyen..."):
  1. Train corpus MOI, 100,000 account (toan bo 519 duong that + 99,481 am
     that), nhan STRICT (labels.pkl, KHONG phai isp_expanded.pkl nhu
     Attempt-3) -- xem data_prep/e2_train_corpus.py,
     runs/inductive_e2_corpus_100k/build.log.
  2. Val la mau con CO DINH (seed co dinh) rut tu overlap population THAT
     (201,931 account, 217 duong that), khong con la slice 5k thien lech cua
     Attempt-3 (da xac nhan oversample duong ~43x) -- xem
     data_prep/e2_val_subsample.py.

BERT_debug.md (xem STATUS.md muc "RA SOAT CODE TIM NGUYEN NHAN"): ban retrain
dau tien (2026-09-13/14, checkpoint output/e2_v2_best_checkpoint.pt cu) sap
AUPRC full pure_test con 0.0495 (graph-only thuan dat 0.303) -- nguyen nhan
goc la nhanh BERT hoc thuoc dia chi vi CHINH CHU lap lai trong van ban (khong
phai bug logic). Ban nay sua theo BERT_debug.md Task 1-4:
  - Task 1+2 (data_prep/text_rendering.py, mg_build_examples.py): corpus gio
    la raw_records (khong self-address); moi EPOCH render lai (dedup +
    anonymize dia chi doi tac + tokenize) VOI SEED KHAC -- cung 1 document
    duong sinh ra 1 chuoi token KHAC moi epoch, chan memorization dia chi.
  - Task 4: giam pos:neg oversample 1:4 -> 1:2 (--pos-neg-ratio), tuy chon
    them class-weighted loss (--pos-weight, mac dinh tat), VA doi tieu chi
    chon checkpoint/early-stop tu AUPRC sang P@100 tren val (--selection-metric,
    mac dinh 'p_at_100') -- theo dung yeu cau "dung tieu chi lech AUC/loss".

Sampling mat can bang: WeightedRandomSampler voi ty le muc tieu 1 duong :
--pos-neg-ratio am. Loss: CrossEntropyLoss, co the them --pos-weight (Task 4).

Script nay CHI train + validate + chon best checkpoint (theo --selection-metric
tren VAL) + tim threshold F1 toi uu tren VAL. KHONG danh gia pure_test/overlap
THAT trong qua trinh nay (tranh nhin trom test). Sau khi checkpoint + threshold
da co dinh, chay eval_full_test.py --checkpoint output/e2_v2_best_checkpoint.pt
MOT LAN DUY NHAT tren toan bo pure_test (609,773) + overlap (201,931) that.

Log wandb project fraud_detection_inductive, group "e2_v2_full_fusion" (group
rieng, khong ghi de log E2 v1 "e2_full_fusion").
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import wandb
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_prep.e2_train_corpus import load_e2_train_examples
from data_prep.e2_val_subsample import build_fixed_val_subsample
from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import PREPROC_DIR
from data_prep.text_rendering import render_examples_inplace
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import compute_metrics, find_best_f1_threshold
from train_eval.train_e2 import build_models, compute_probs_labels, evaluate, make_train_loader, train_one_epoch

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"
GLOBAL_SEED = 44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--max-examples", type=int, default=None, help="gioi han train corpus, dung de smoke-test nhanh")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=4, help="epoch khong cai thien selection-metric truoc khi dung som")
    ap.add_argument("--pos-neg-ratio", type=float, default=2.0,
                     help="ty le 1 duong : N am muc tieu khi sample train (BERT_debug.md Task 4: giam tu 4.0 -> 2.0 "
                          "mac dinh de giam so lan lap lai/epoch cua 519 document duong, xem STATUS.md)")
    ap.add_argument("--pos-weight", type=float, default=None,
                     help="Task 4 tuy chon: them class-weighted CE loss (vd 2.0) DOC LAP voi --pos-neg-ratio cua "
                          "sampler -- mac dinh None (tat, hanh vi CU: CrossEntropyLoss khong trong so)")
    ap.add_argument("--selection-metric", type=str, default="p_at_100", choices=["p_at_100", "auprc"],
                     help="Task 4: tieu chi chon checkpoint + early-stop tren VAL -- mac dinh p_at_100 (bam sat "
                          "muc tieu thuc chien 'top-K account nghi van'), 'auprc' de doi chieu voi ban truoc")
    ap.add_argument("--val-target-n", type=int, default=20000)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group="e2_v2_full_fusion", name="e2_v2_bert_graphsage_fusion", config=vars(args)
    )

    print("loading E2 v2 train corpus (100,000 account, nhan strict labels.pkl)...")
    train_examples = load_e2_train_examples(max_examples=args.max_examples)
    n_pos = sum(e.label for e in train_examples)
    print(f"train examples={len(train_examples)} n_pos={n_pos} n_neg={len(train_examples) - n_pos}")

    val_examples = build_fixed_val_subsample(target_n=args.val_target_n, seed=GLOBAL_SEED)
    all_examples = train_examples + val_examples

    node_features = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    labels_tensor = torch.zeros(node_features.shape[0], dtype=torch.long)
    for e in train_examples:
        labels_tensor[e.global_idx] = e.label
    sampler = LabelAwareNeighborSampler(adj_train, labels_tensor, seed=GLOBAL_SEED)

    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    graph_encoder, classifier = build_models(device)
    optimizer = torch.optim.AdamW([
    {"params": classifier.text_encoder.parameters(), "lr": 2e-5},
    {"params": graph_encoder.parameters(), "lr": 1e-3},
    {"params": classifier.fusion.parameters(), "lr": 1e-3},
    {"params": classifier.classifier.parameters(), "lr": 1e-3},
])

    train_loader = make_train_loader(train_examples, args.batch_size, pos_neg_ratio=args.pos_neg_ratio)

    best_metric = -1.0
    best_state = None
    best_epoch = -1
    best_val_metrics = None
    epochs_since_improve = 0
    history = []

    for epoch in range(args.epochs):
        # BERT_debug.md Task 1-2: render lai (dedup+anonymize+tokenize) TOAN
        # BO 100k document train VOI SEED KHAC moi epoch -- cung 1 document
        # duong sinh ra 1 phep gan dia chi doi tac (addr0/addr1/...) KHAC
        # NHAU, chan model hoc thuoc 1 chuoi sub-token co dinh. Gan input_ids/
        # attention_mask TAI CHO tren tung Example (train_loader da tro toi
        # cung cac object nay nen khong can tao lai DataLoader).
        t_render = time.time()
        render_examples_inplace(train_examples, seed=f"train_epoch{epoch}", tokenizer=text_tokenizer)
        print(f"epoch={epoch} re-rendered {len(train_examples)} train docs in {time.time()-t_render:.1f}s")

        t0 = time.time()
        train_loss = train_one_epoch(
            graph_encoder, classifier, train_loader, sampler, node_features, optimizer, device,
            max_steps=args.max_steps_per_epoch, pos_weight=args.pos_weight,
        )
        dt = time.time() - t0
        val_metrics = evaluate(graph_encoder, classifier, all_examples, device, split_filter="val", k_list=[100, 1000])
        val_metric_value = val_metrics[args.selection_metric]
        improved = val_metric_value > best_metric
        if improved:
            best_metric = val_metric_value
            best_val_metrics = val_metrics
            best_state = {
                "graph_encoder": {k: v.detach().cpu().clone() for k, v in graph_encoder.state_dict().items()},
                "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
            }
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(f"epoch={epoch} train_loss={train_loss:.4f} time={dt:.1f}s "
              f"selection_metric({args.selection_metric})={val_metric_value:.4f} val={val_metrics}"
              f"{' *best*' if improved else ''}")
        history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt, "val": val_metrics})
        if run is not None:
            run.log({"train_loss": train_loss, "epoch_time_s": dt,
                      **{f"val/{k}": v for k, v in val_metrics.items()},
                      f"best_{args.selection_metric}_so_far": best_metric}, step=epoch)

        if args.patience and epochs_since_improve >= args.patience:
            print(f"early stop at epoch={epoch} (khong cai thien {args.selection_metric} sau {args.patience} epoch)")
            break

    graph_encoder.load_state_dict(best_state["graph_encoder"])
    classifier.load_state_dict(best_state["classifier"])

    print("\ntim threshold F1 toi uu tren VAL (chi val subsample, khong dung test)...")
    val_probs, val_labels = compute_probs_labels(graph_encoder, classifier, all_examples, device, split_filter="val")
    best_threshold = find_best_f1_threshold(val_labels, val_probs)
    # Bao cao CA HAI: threshold mac dinh (0.5) va threshold toi uu (quet tren
    # val) -- de so sanh cong bang voi GBM/graph-only (cung lam 2 ban nay,
    # xem gbm_baseline.py / train_graph_only_hypothesis.py).
    val_metrics_default_t = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_metrics_best_t = compute_metrics(val_labels, val_probs, threshold=best_threshold)
    print(f"best_threshold={best_threshold:.4f}")
    print(f"val F1(pos) @ threshold=0.5 (mac dinh)   : {val_metrics_default_t['f1_pos']:.4f}")
    print(f"val F1(pos) @ threshold={best_threshold:.4f} (toi uu): {val_metrics_best_t['f1_pos']:.4f}")

    checkpoint = {
        "graph_encoder": best_state["graph_encoder"],
        "classifier": best_state["classifier"],
        "threshold": best_threshold,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_metric_value": best_metric,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(checkpoint, OUTPUT_DIR / "e2_v2_best_checkpoint.pt")
    print(f"\n=== E2 v2 checkpoint saved -> output/e2_v2_best_checkpoint.pt "
          f"(best_epoch={best_epoch}, {args.selection_metric}={best_metric:.4f}, threshold={best_threshold:.4f}) ===")
    print("KHONG eval pure_test/overlap that o day -- chay "
          "`python -m train_eval.eval_full_test --checkpoint output/e2_v2_best_checkpoint.pt "
          "--out-json e2_v2_full_eval_result.json --group e2_v2_full_eval` MOT LAN DUY NHAT sau buoc nay.")

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary[f"best_{args.selection_metric}"] = best_metric
        run.summary["best_threshold"] = best_threshold
        for k, v in val_metrics_default_t.items():
            run.summary[f"val_default_t/{k}"] = v
        for k, v in val_metrics_best_t.items():
            run.summary[f"val_best_t/{k}"] = v
        run.finish()

    with open(OUTPUT_DIR / "e2_v2_train_history.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "selection_metric": args.selection_metric,
                    "best_metric_value": best_metric, "best_threshold": best_threshold,
                    "val_metrics_at_default_threshold": val_metrics_default_t,
                    "val_metrics_at_best_threshold": val_metrics_best_t,
                    "args": vars(args)}, f, indent=2)
    print(f"Saved -> {OUTPUT_DIR / 'e2_v2_train_history.json'}")


if __name__ == "__main__":
    main()
