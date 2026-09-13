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

Sampling mat can bang: WeightedRandomSampler voi ty le muc tieu 1 duong :
--pos-neg-ratio am (mac dinh 4), KHONG can bang day du 1:1 nhu E2 v1 (can
bang day du se lap 519 duong ~96 lan/epoch, de overfit dung 519 example nay).
Loss: CrossEntropyLoss KHONG trong so (sampler da xu ly imbalance -- cong
them class-weighted loss se bu 2 lan).

Script nay CHI train + validate + chon best checkpoint (theo val AUPRC) + tim
threshold F1 toi uu tren VAL. KHONG danh gia pure_test/overlap THAT trong qua
trinh nay (tranh nhin trom test). Sau khi checkpoint + threshold da co dinh,
chay eval_full_test.py --checkpoint output/e2_v2_best_checkpoint.pt MOT LAN
DUY NHAT tren toan bo pure_test (609,773) + overlap (201,931) that.

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

from data_prep.e2_train_corpus import load_e2_train_examples
from data_prep.e2_val_subsample import build_fixed_val_subsample
from data_prep.io_utils import load_graph, load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import find_best_f1_threshold
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
    ap.add_argument("--patience", type=int, default=4, help="epoch khong cai thien val AUPRC truoc khi dung som")
    ap.add_argument("--pos-neg-ratio", type=float, default=4.0, help="ty le 1 duong : N am muc tieu khi sample train")
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

    graph_encoder, classifier = build_models(device)
    optimizer = torch.optim.AdamW([
    {"params": classifier.text_encoder.parameters(), "lr": 2e-5},
    {"params": graph_encoder.parameters(), "lr": 1e-3},
    {"params": classifier.fusion.parameters(), "lr": 1e-3},
    {"params": classifier.classifier.parameters(), "lr": 1e-3},
])

    train_loader = make_train_loader(train_examples, args.batch_size, pos_neg_ratio=args.pos_neg_ratio)

    best_val_auprc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            graph_encoder, classifier, train_loader, sampler, node_features, optimizer, device,
            max_steps=args.max_steps_per_epoch,
        )
        dt = time.time() - t0
        val_metrics = evaluate(graph_encoder, classifier, all_examples, device, split_filter="val")
        val_auprc = val_metrics["auprc"]
        improved = val_auprc > best_val_auprc
        if improved:
            best_val_auprc = val_auprc
            best_state = {
                "graph_encoder": {k: v.detach().cpu().clone() for k, v in graph_encoder.state_dict().items()},
                "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
            }
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(f"epoch={epoch} train_loss={train_loss:.4f} time={dt:.1f}s val={val_metrics}"
              f"{' *best*' if improved else ''}")
        history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt, "val": val_metrics})
        if run is not None:
            run.log({"train_loss": train_loss, "epoch_time_s": dt,
                      **{f"val/{k}": v for k, v in val_metrics.items()},
                      "best_val_auprc_so_far": best_val_auprc}, step=epoch)

        if args.patience and epochs_since_improve >= args.patience:
            print(f"early stop at epoch={epoch} (khong cai thien val AUPRC sau {args.patience} epoch)")
            break

    graph_encoder.load_state_dict(best_state["graph_encoder"])
    classifier.load_state_dict(best_state["classifier"])

    print("\ntim threshold F1 toi uu tren VAL (chi val subsample, khong dung test)...")
    val_probs, val_labels = compute_probs_labels(graph_encoder, classifier, all_examples, device, split_filter="val")
    best_threshold = find_best_f1_threshold(val_labels, val_probs)
    from train_eval.metrics import compute_metrics
    val_metrics_at_best_t = compute_metrics(val_labels, val_probs, threshold=best_threshold)
    print(f"best_threshold={best_threshold:.4f} val_metrics_at_best_t={val_metrics_at_best_t}")

    checkpoint = {
        "graph_encoder": best_state["graph_encoder"],
        "classifier": best_state["classifier"],
        "threshold": best_threshold,
        "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc,
    }
    torch.save(checkpoint, OUTPUT_DIR / "e2_v2_best_checkpoint.pt")
    print(f"\n=== E2 v2 checkpoint saved -> output/e2_v2_best_checkpoint.pt "
          f"(best_epoch={best_epoch}, val_auprc={best_val_auprc:.4f}, threshold={best_threshold:.4f}) ===")
    print("KHONG eval pure_test/overlap that o day -- chay "
          "`python -m train_eval.eval_full_test --checkpoint output/e2_v2_best_checkpoint.pt "
          "--out-json e2_v2_full_eval_result.json --group e2_v2_full_eval` MOT LAN DUY NHAT sau buoc nay.")

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_auprc"] = best_val_auprc
        run.summary["best_threshold"] = best_threshold
        for k, v in val_metrics_at_best_t.items():
            run.summary[f"val_best_t/{k}"] = v
        run.finish()

    with open(OUTPUT_DIR / "e2_v2_train_history.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_auprc": best_val_auprc,
                    "best_threshold": best_threshold, "val_metrics_at_best_t": val_metrics_at_best_t,
                    "args": vars(args)}, f, indent=2)
    print(f"Saved -> {OUTPUT_DIR / 'e2_v2_train_history.json'}")


if __name__ == "__main__":
    main()
