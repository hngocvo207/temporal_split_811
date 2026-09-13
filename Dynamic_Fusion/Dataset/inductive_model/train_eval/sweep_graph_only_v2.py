"""
Sweep graph-only SAU KHI SUA BUG HUONG CANH (2026-09-06, xem STATUS.md muc
"BUG QUAN TRONG" + "SUA LAI LAN 2") -- lap lai dung tinh than
sweep_graph_only.py (12 config goc, luc do con bug huong canh) nhung MO RONG
luoi learning_rate (theo yeu cau "sua Direction + LR") vi huong aggregate da
doi hoan toan (tu payee sang payer), khong the gia dinh LR cu (1e-3) van con
toi uu.

Luoi: learning_rate x {3e-4, 1e-3, 3e-3, 1e-2} x (hidden,out) x {(64,64),
(128,128),(256,128)} x mlp_classifier x {False,True} = 24 config.

Ky luat GIONG HET sweep_graph_only.py: chon config theo VAL AUPRC (khong nhin
pure_test luc chon).
"""
import json
import time
from argparse import Namespace
from pathlib import Path

import scipy.sparse as sp
import torch
import wandb

from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from train_eval.train_graph_only_hypothesis import build_partition_indices, evaluate, train

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"

GRID = [
    {"lr": lr, "hidden": h, "out": o, "mlp_classifier": mlp}
    for lr in (3e-4, 1e-3, 3e-3, 1e-2)
    for (h, o) in ((64, 64), (128, 128), (256, 128))
    for mlp in (False, True)
]

BASE_ARGS = dict(
    epochs=150, batch_size=32, neg_ratio=3, weight_decay=1e-5, dropout=0.3,
    eval_every=5, patience=30,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading shared data once for {len(GRID)} configs (huong canh DA SUA -- xem STATUS.md)...")
    idx, labels = build_partition_indices()
    node_features = load_node_features()
    adj_train = sp.load_npz(PREPROC_DIR / "adj_train.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for i, cfg in enumerate(GRID):
        args = Namespace(**BASE_ARGS, **cfg)
        run_name = f"sweepv2_{i:02d}_lr{cfg['lr']}_h{cfg['hidden']}_o{cfg['out']}_mlp{int(cfg['mlp_classifier'])}"
        print(f"\n[{i+1}/{len(GRID)}] config={cfg}")
        run = wandb.init(
            project=WANDB_PROJECT, group="graph_only_sweep_v2_fixed_direction", name=run_name,
            config={**cfg, **BASE_ARGS}, reinit=True,
        )
        t0 = time.time()
        model, history, _, _, best_epoch, best_val_auprc = train(
            args, idx=idx, labels=labels, node_features=node_features, adj_train=adj_train, verbose=False,
            wandb_run=run,
        )
        final_eval = evaluate(model, idx, labels, device)
        dt = time.time() - t0
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_auprc"] = best_val_auprc
        run.summary["time_s"] = dt
        for split, m in final_eval.items():
            for k, v in m.items():
                run.summary[f"final_{split}/{k}"] = v
        run.finish()

        row = {
            "config": cfg, "best_epoch": best_epoch, "best_val_auprc": best_val_auprc,
            "time_s": dt, "n_epochs_ran": len(history), "eval": final_eval,
        }
        results.append(row)
        print(f"  best_epoch={best_epoch} val_auprc={best_val_auprc:.4f} "
              f"pure_test_auprc={final_eval['pure_test']['auprc']:.4f} "
              f"pure_test_f1={final_eval['pure_test']['f1_pos']:.4f} "
              f"pure_test_recall@1000={final_eval['pure_test']['recall_at_1000']:.4f} ({dt:.1f}s)")

    results.sort(key=lambda r: r["best_val_auprc"], reverse=True)
    print("\n=== SWEEP V2 RESULTS (huong canh da sua, sorted by val AUPRC) ===")
    for r in results:
        print(f"val_auprc={r['best_val_auprc']:.4f} | pure_test_auprc={r['eval']['pure_test']['auprc']:.4f} "
              f"pure_test_f1={r['eval']['pure_test']['f1_pos']:.4f} | overlap_auprc={r['eval']['overlap']['auprc']:.4f} "
              f"| config={r['config']} (epoch {r['best_epoch']}, {r['time_s']:.0f}s)")

    best = results[0]
    print(f"\n=== BEST CONFIG (by val AUPRC, huong canh da sua): {best['config']} ===")
    print(json.dumps(best["eval"], indent=2))

    with open(OUTPUT_DIR / "graph_only_sweep_v2_fixed_direction_results.json", "w") as f:
        json.dump({"grid": GRID, "base_args": BASE_ARGS, "results": results}, f, indent=2)
    print(f"\nSaved -> {OUTPUT_DIR / 'graph_only_sweep_v2_fixed_direction_results.json'}")


if __name__ == "__main__":
    main()
