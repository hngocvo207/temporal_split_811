"""
Learning-curve ablation: train tu dau (encoder ngau nhien, giong E2 v2) voi
25/50/75/100% cua 519 duong xac thuc, am giu CO DINH 20,000 (xem
data_prep/e2_ablation_corpus.py). Cung val subsample co dinh (overlap that,
217 duong) + cung ky luat (nhan strict, pos:neg sampling ratio, threshold tim
tren val, khong nhin trom test luc chon checkpoint) nhu train_e2_v2.py --
CHI khac o corpus train (nho hon, am co dinh, duong bien doi theo fraction).

Sau khi train xong 1 fraction, chay them 1 eval o quy mo TRUNG BINH (100,000
mau dau cua pure_test -- xem full_test_corpus.py, pure_test luon dung truoc
overlap trong thu tu doc) de co duong cong AUPRC-vs-%duong dang tin hon val
don thuan (van co the bi thoi phong precision do undersample am, nhung it
nghiem trong hon val 20k vi 100k mau lon hon 5x) -- xem STATUS.md muc
"Eval full pure_test/overlap" de biet ly do can buoc nay.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import wandb

from data_prep.e2_ablation_corpus import build_ablation_train_examples
from data_prep.e2_val_subsample import build_fixed_val_subsample
from data_prep.full_test_corpus import load_full_test_examples
from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import find_best_f1_threshold, compute_metrics
from train_eval.train_e2 import build_models, compute_probs_labels, evaluate, make_train_loader, train_one_epoch

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"
GLOBAL_SEED = 44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-fraction", type=float, required=True, choices=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--pos-neg-ratio", type=float, default=4.0)
    ap.add_argument("--val-target-n", type=int, default=20000)
    ap.add_argument("--pure-test-check-n", type=int, default=100000,
                     help="mau pure_test o quy mo trung binh de kiem tra sau khi train (0 = bo qua)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    tag = f"pos{int(args.pos_fraction * 100)}"
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} tag={tag}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group="e2_ablation_pos_fraction", name=f"e2_ablation_{tag}",
        config={**vars(args), "tag": tag}
    )

    train_examples = build_ablation_train_examples(args.pos_fraction)
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
    optimizer = torch.optim.AdamW(
        list(graph_encoder.parameters()) + list(classifier.parameters()), lr=args.lr
    )
    train_loader = make_train_loader(train_examples, args.batch_size, pos_neg_ratio=args.pos_neg_ratio)

    best_val_auprc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(graph_encoder, classifier, train_loader, sampler, node_features,
                                      optimizer, device)
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

        print(f"[{tag}] epoch={epoch} train_loss={train_loss:.4f} time={dt:.1f}s val={val_metrics}"
              f"{' *best*' if improved else ''}")
        history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt, "val": val_metrics})
        if run is not None:
            run.log({"train_loss": train_loss, "epoch_time_s": dt,
                      **{f"val/{k}": v for k, v in val_metrics.items()},
                      "best_val_auprc_so_far": best_val_auprc}, step=epoch)

        if args.patience and epochs_since_improve >= args.patience:
            print(f"[{tag}] early stop at epoch={epoch}")
            break

    graph_encoder.load_state_dict(best_state["graph_encoder"])
    classifier.load_state_dict(best_state["classifier"])

    val_probs, val_labels = compute_probs_labels(graph_encoder, classifier, all_examples, device, split_filter="val")
    best_threshold = find_best_f1_threshold(val_labels, val_probs)
    val_metrics_at_best_t = compute_metrics(val_labels, val_probs, threshold=best_threshold)
    print(f"[{tag}] best_threshold={best_threshold:.4f} val_metrics_at_best_t={val_metrics_at_best_t}")

    checkpoint = {
        "graph_encoder": best_state["graph_encoder"], "classifier": best_state["classifier"],
        "threshold": best_threshold, "best_epoch": best_epoch, "best_val_auprc": best_val_auprc,
        "pos_fraction": args.pos_fraction, "n_pos_train": n_pos,
    }
    ckpt_path = OUTPUT_DIR / f"e2_ablation_{tag}_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"[{tag}] checkpoint saved -> {ckpt_path} (best_epoch={best_epoch}, val_auprc={best_val_auprc:.4f})")

    result = {
        "pos_fraction": args.pos_fraction, "n_pos_train": n_pos, "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc, "best_threshold": best_threshold,
        "val_metrics_at_best_t": val_metrics_at_best_t, "history": history,
    }

    if args.pure_test_check_n:
        print(f"[{tag}] chay eval kiem tra pure_test ({args.pure_test_check_n} mau dau, khong phai full)...")
        pt_examples = load_full_test_examples(max_examples=args.pure_test_check_n)
        pt_metrics = evaluate(graph_encoder, classifier, pt_examples, device, split_filter="pure_test",
                               batch_size=16, threshold=best_threshold)
        print(f"[{tag}] pure_test_check({args.pure_test_check_n})={pt_metrics}")
        result["pure_test_check"] = pt_metrics
        if run is not None:
            for k, v in pt_metrics.items():
                run.summary[f"pure_test_check/{k}"] = v

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_auprc"] = best_val_auprc
        run.summary["best_threshold"] = best_threshold
        for k, v in val_metrics_at_best_t.items():
            run.summary[f"val_best_t/{k}"] = v
        run.finish()

    out_path = OUTPUT_DIR / f"e2_ablation_{tag}_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] Saved -> {out_path}")


if __name__ == "__main__":
    main()
