"""
Chan doan: tach 2 nguyen nhan gop lai gay sap ket qua full-scale (F1=0.031 vs
0.919 o mau 5k) -- (1) corpus Attempt-3 dung nhan LAN TRUYEN (isp_expanded,
KHONG PHAI nhan xac nhan doc lap -- da phat hien va xac nhan, xem STATUS.md)
va (2) ty le duong cuc thap o full scale (0.05%) khac han mau 5k (31.6%).

Chay 1 lan forward tren 1 subsample (~100k, van la mau con NGAU NHIEN vi doc
da "shuffled" tu truoc) roi tinh metric cho CA HAI loai nhan tu cung 1 pass --
re hon nhieu so voi eval full 2 lan.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wandb

from data_prep.full_test_corpus import load_full_test_examples
from train_eval.metrics import compute_metrics
from data_prep.io_utils import load_graph
from train_eval.train_e2 import build_models

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(OUTPUT_DIR / "e2_best_checkpoint.pt"))
    ap.add_argument("--max-examples", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group="e2_diagnostic", name="label_scheme_vs_scale_diagnostic",
        config=vars(args),
    )

    graph_encoder, classifier = build_models(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    graph_encoder.load_state_dict(ckpt["graph_encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    graph_encoder.eval()
    classifier.eval()
    print("checkpoint loaded")

    print(f"loading {args.max_examples} examples (pure_test prefix, van la mau ngau nhien vi da shuffled)...")
    examples = load_full_test_examples(max_examples=args.max_examples)
    n_prop_pos = sum(e.label for e in examples)
    n_strict_pos = sum(e.label_strict for e in examples)
    print(f"n={len(examples)} n_pos_propagated={n_prop_pos} n_pos_strict={n_strict_pos}")

    print("computing h_graph for all nodes (full_graph_forward, CPU)...")
    graph_encoder_device = next(graph_encoder.parameters()).device
    graph_encoder.to("cpu")
    graph_inference = load_graph("inference")
    h_graph_all = graph_encoder.full_graph_forward(graph_inference, torch.device("cpu"))
    graph_encoder.to(graph_encoder_device)

    probs, labels_prop, labels_strict = [], [], []
    bs = args.batch_size
    for i in range(0, len(examples), bs):
        batch = examples[i:i + bs]
        input_ids = torch.stack([e.input_ids for e in batch]).to(device)
        attention_mask = torch.stack([e.attention_mask for e in batch]).to(device)
        global_idx = torch.tensor([e.global_idx for e in batch], dtype=torch.long)
        h_graph = h_graph_all[global_idx].to(device)
        logits = classifier(input_ids, h_graph, attention_mask=attention_mask)
        prob_pos = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(prob_pos.cpu().tolist())
        labels_prop.extend([e.label for e in batch])
        labels_strict.extend([e.label_strict for e in batch])
        if (i // bs) % 500 == 0:
            print(f"  {i}/{len(examples)}")

    probs = np.array(probs)
    metrics_propagated = compute_metrics(np.array(labels_prop), probs)
    metrics_strict = compute_metrics(np.array(labels_strict), probs)

    result = {
        "n": len(examples),
        "propagated_labels (isp_expanded, giong corpus E2 dung)": metrics_propagated,
        "strict_labels (xac nhan doc lap)": metrics_strict,
    }
    print("\n=== DIAGNOSTIC RESULT ===")
    print(json.dumps(result, indent=2))
    print("\nSo sanh:")
    print(f"  E2 5k-sample (propagated): F1=0.919 AUPRC=0.973")
    print(f"  Day, {len(examples)}-sample (propagated): F1={metrics_propagated['f1_pos']:.4f} AUPRC={metrics_propagated['auprc']:.4f}")
    print(f"  Day, {len(examples)}-sample (strict):     F1={metrics_strict['f1_pos']:.4f} AUPRC={metrics_strict['auprc']:.4f}")
    print(f"  Full 609,773 (strict, da chay that): F1=0.031 AUPRC=0.060")

    if run is not None:
        run.summary["propagated"] = metrics_propagated
        run.summary["strict"] = metrics_strict
        run.finish()

    with open(OUTPUT_DIR / "e2_diagnostic_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
