"""
Eval checkpoint E2 (đã chọn theo val AUPRC, xem STATUS.md) trên TOÀN BỘ
pure_test (609,773 account) + overlap (201,931 account) thật -- thay cho mẫu
5,000/609,773 quá nhỏ đã dùng ở E2, để có kết quả đáng tin hơn nhiều mà không
cần train thêm gì (dùng lại corpus 811,704 account đã tokenized sẵn từ trước,
xem data_prep/full_test_corpus.py).

Không train -- chỉ load checkpoint + eval. Log lên wandb (project
fraud_detection_inductive, group "e2_full_eval").
"""
import argparse
import json
import time
from pathlib import Path

import torch
import wandb

from data_prep.full_test_corpus import load_full_test_examples
from train_eval.train_e2 import build_models, evaluate

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(OUTPUT_DIR / "e2_best_checkpoint.pt"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=None, help="giới hạn corpus, dùng để smoke-test nhanh")
    ap.add_argument("--threshold", type=float, default=None,
                     help="mặc định: lấy từ checkpoint['threshold'] nếu có (E2 v2, chọn trên VAL), "
                          "fallback 0.5 nếu checkpoint không có (E2 v1)")
    ap.add_argument("--out-json", type=str, default="e2_full_eval_result.json",
                     help="tên file kết quả trong output/ -- đổi tên khi eval checkpoint khác để không đè kết quả cũ")
    ap.add_argument("--group", type=str, default="e2_full_eval")
    ap.add_argument("--name", type=str, default="e2_eval_full_pure_test")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group=args.group, name=args.name,
        config={"checkpoint": args.checkpoint, "batch_size": args.batch_size},
    )

    print(f"loading checkpoint {args.checkpoint} ...")
    graph_encoder, classifier = build_models(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    graph_encoder.load_state_dict(ckpt["graph_encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    threshold = args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5)
    print(f"checkpoint loaded (threshold={threshold:.4f}, "
          f"best_epoch={ckpt.get('best_epoch')}, best_val_auprc={ckpt.get('best_val_auprc')})")

    print("tokenizing full_test_corpus (811,704 docs, dùng lại text đã tokenized sẵn)...")
    t0 = time.time()
    examples = load_full_test_examples(max_examples=args.max_examples)
    print(f"loaded {len(examples)} examples in {time.time()-t0:.1f}s")

    results = {}
    for split in ("pure_test", "overlap"):
        n_split = sum(1 for e in examples if e.split == split)
        print(f"\nevaluating {split} ({n_split} account thật -- không phải mẫu con)...")
        t0 = time.time()
        metrics = evaluate(graph_encoder, classifier, examples, device, split_filter=split,
                            batch_size=args.batch_size, threshold=threshold)
        dt = time.time() - t0
        print(f"[{split}] {metrics} ({dt:.1f}s, {metrics['n']/dt:.1f} account/s)")
        results[split] = {**metrics, "time_s": dt}
        if run is not None:
            run.log({f"{split}/{k}": v for k, v in metrics.items() if k != "threshold"})

    print(f"\n=== FULL-SCALE EVAL RESULT (609,773 pure_test + 201,931 overlap thật, threshold={threshold:.4f}) ===")
    print(json.dumps(results, indent=2))
    print("\nBaseline Attempt-3 gốc (10k test, calibrated): F1(pos)=87.63%, AUPRC overall=0.9155")

    if run is not None:
        run.summary["threshold"] = threshold
        for split, m in results.items():
            for k, v in m.items():
                run.summary[f"{split}/{k}"] = v
        run.finish()

    out_path = OUTPUT_DIR / args.out_json
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "threshold": threshold, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
