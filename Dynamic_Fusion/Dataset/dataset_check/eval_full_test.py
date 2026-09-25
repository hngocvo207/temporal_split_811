"""
Eval checkpoint E2 (đã chọn theo val AUPRC, xem STATUS.md) trên TOÀN BỘ
pure_test (609,773 account) + overlap (201,931 account) thật -- thay cho mẫu
5,000/609,773 quá nhỏ đã dùng ở E2, để có kết quả đáng tin hơn nhiều mà không
cần train thêm gì (dùng lại corpus 811,704 account đã tokenized sẵn từ trước,
xem data_prep/full_test_corpus.py).

Không train -- chỉ load checkpoint + eval. Log lên wandb (project
fraud_detection_inductive, group "e2_full_eval").

So sánh với GBM (theo yêu cầu): thêm P@100/P@1000/Recall@1000 (k_list, giống
hệt metric GBM đã báo cáo) và in bảng đối chiếu trực tiếp với con số GBM đã
TUNE (output/gbm_hparam_sweep_result.json, chạy trên ĐÚNG cùng pure_test đầy
đủ 609,773 account của split cũ -- chỉ so sánh công bằng khi checkpoint E2
cũng train/eval trên split cũ, xem STATUS.md).

Theo yêu cầu bổ sung (so sánh F1-positive ở ngưỡng mặc định VÀ ngưỡng tối ưu):
eval CẢ HAI threshold=0.5 (mặc định) và threshold=checkpoint['threshold']
(tối ưu, đã quét trên val lúc train -- xem train_e2_v2.py) cho mỗi split, và
đọc GBM_TUNED_REFERENCE trực tiếp từ output/gbm_hparam_sweep_result.json (đã
tự lưu cả 2 ngưỡng từ gbm_baseline.py) thay vì hardcode, để luôn khớp lần
chạy GBM gần nhất."""
import argparse
import json
import time
from pathlib import Path

import torch
import wandb

from data_prep.full_test_corpus import load_full_test_examples
from train_eval.metrics import compute_metrics
from train_eval.train_e2 import build_models, compute_probs_labels

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"


def _load_gbm_reference():
    """Doc output/gbm_hparam_sweep_result.json (ghi boi gbm_baseline.py) thay
    vi hardcode -- luon khop lan chay GBM gan nhat, ca 2 threshold (mac
    dinh/toi uu). None neu file chua ton tai (chua chay gbm_baseline.py)."""
    p = OUTPUT_DIR / "gbm_hparam_sweep_result.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        "default_t": d["final_full_test_metrics"],
        "best_t": d.get("final_full_test_metrics_best_threshold"),
        "best_threshold": d.get("best_threshold"),
    }


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
        # 1 lan forward DUY NHAT (full_graph_forward + BERT tren toan bo
        # split -- phan ton thoi gian nhat, ~54-60 acc/s) roi tinh metric o
        # CA HAI threshold tu CUNG probs -- tranh forward 2 lan (se gap doi
        # ~3-4h eval that neu goi evaluate() rieng cho tung threshold).
        probs, y_true = compute_probs_labels(graph_encoder, classifier, examples, device,
                                              split_filter=split, batch_size=args.batch_size)
        metrics_best_t = compute_metrics(y_true, probs, threshold=threshold, k_list=[100, 1000])
        metrics_default_t = compute_metrics(y_true, probs, threshold=0.5, k_list=[100, 1000]) \
            if threshold != 0.5 else metrics_best_t
        dt = time.time() - t0
        print(f"[{split}] @best_t={threshold:.4f}: f1={metrics_best_t['f1_pos']:.4f} "
              f"auprc={metrics_best_t['auprc']:.4f} | @default_t=0.5: f1={metrics_default_t['f1_pos']:.4f} "
              f"({dt:.1f}s, {metrics_best_t['n']/dt:.1f} account/s)")
        results[split] = {**metrics_best_t, "time_s": dt,
                           "metrics_at_default_threshold": metrics_default_t,
                           "metrics_at_best_threshold": metrics_best_t}
        if run is not None:
            run.log({f"{split}/{k}": v for k, v in metrics_best_t.items() if k != "threshold"})
            run.log({f"{split}/default_t/{k}": v for k, v in metrics_default_t.items() if k != "threshold"})

    print(f"\n=== FULL-SCALE EVAL RESULT (609,773 pure_test + 201,931 overlap thật, threshold_toi_uu={threshold:.4f}) ===")
    print(json.dumps(results, indent=2))
    print("\nBaseline Attempt-3 gốc (10k test, calibrated): F1(pos)=87.63%, AUPRC overall=0.9155")

    gbm_ref = _load_gbm_reference()
    if "pure_test" in results and gbm_ref is not None:
        e_best, e_default = results["pure_test"]["metrics_at_best_threshold"], results["pure_test"]["metrics_at_default_threshold"]
        g_best = gbm_ref["best_t"] or gbm_ref["default_t"]
        g_default = gbm_ref["default_t"]
        print("\n=== SO SÁNH TRỰC TIẾP VỚI GBM ĐÃ TUNE (cùng full pure_test, 609,773 account, split cũ) ===")
        print("--- threshold TỐI ƯU (quét trên val, mỗi model tự chọn threshold riêng) ---")
        print(f"{'metric':<14} {'E2 (fusion)':>14} {'GBM (tuned)':>14} {'chênh lệch':>14}")
        for key, label in [
            ("auprc", "AUPRC"), ("roc_auc", "ROC-AUC"), ("f1_pos", "F1(pos)"),
            ("precision_pos", "Precision"), ("recall_pos", "Recall"),
            ("p_at_100", "P@100"), ("p_at_1000", "P@1000"), ("recall_at_1000", "Recall@1000"),
        ]:
            ev, gv = e_best.get(key), g_best.get(key)
            if ev is None or gv is None:
                continue
            diff = ev - gv
            arrow = "▲ E2 thắng" if diff > 0 else ("▼ GBM thắng" if diff < 0 else "= hoà")
            print(f"{label:<14} {ev:>14.4f} {gv:>14.4f} {diff:>+9.4f}  {arrow}")
        print("--- threshold MẶC ĐỊNH (0.5, cả 2 model) ---")
        print(f"F1(pos)        {e_default['f1_pos']:>14.4f} {g_default['f1_pos']:>14.4f} "
              f"{e_default['f1_pos'] - g_default['f1_pos']:>+9.4f}")
        print(f"Precision      {e_default['precision_pos']:>14.4f} {g_default['precision_pos']:>14.4f} "
              f"{e_default['precision_pos'] - g_default['precision_pos']:>+9.4f}")
        print(f"Recall         {e_default['recall_pos']:>14.4f} {g_default['recall_pos']:>14.4f} "
              f"{e_default['recall_pos'] - g_default['recall_pos']:>+9.4f}")
        print("\n(GBM tuned: output/gbm_hparam_sweep_result.json -- lr=0.3/max_leaf_nodes=63/l2=1.0, "
              "chỉ 23 đặc trưng tabular, KHÔNG graph/BERT, train+eval trên ĐÚNG split cũ để so sánh công bằng.)")
        results["gbm_tuned_reference"] = gbm_ref
    elif "pure_test" in results:
        print("\n[!] output/gbm_hparam_sweep_result.json chưa tồn tại -- chạy `python -m train_eval.gbm_baseline` "
              "trước để có bảng so sánh với GBM.")

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
