"""
E2 v3 -- ket hop 2 huong da chon khi so sanh cach training cua LMAE4Eth voi
train_e2_v2.py (xem cuoc trao doi so sanh, luu lai o STATUS.md muc "E2 v3"):

  1. Warm-start GraphSAGEEncoder tu checkpoint pretrain khong-nhan (MAGAE-style
     masked feature reconstruction, TOAN BO ~2.97M node -- xem model/graph_mae.py,
     train_eval/pretrain_graph_encoder.py), thay vi khoi tao ngau nhien nhu v2.
     Ly do: STATUS.md muc "Thi nghiem toi gian" da do that -- rieng nhanh graph
     cham tran AUPRC ~0.01-0.02 du tune capacity, ket luan la thieu TIN HIEU
     (519 mau duong xac thuc qua it), khong phai thieu capacity. Pretrain
     khong nhan giai quyet dung diem nghen do vi khong can nhan.
  2. Cho phep tro toi corpus finetune lon hon 100k cua v2 (--corpus-dir, xem
     data_prep/e2_v3_train_corpus.py) -- TUY CHON, chi co tac dung neu da chay
     lai mg_build_examples.py --cap_train <N> voi N > 100000 truoc do. Mac
     dinh van dung dung corpus 100k co san cua v2 de script chay duoc ngay.

Phan con lai (nhan strict labels.pkl, val subsample co dinh tu overlap that,
WeightedRandomSampler pos:neg=--pos-neg-ratio, khong danh gia pure_test/overlap
that trong luc train, chi chay eval_full_test.py MOT LAN DUY NHAT sau khi da
chon xong checkpoint+threshold) giu NGUYEN ky luat cua E2 v2 -- xem
train_e2_v2.py va STATUS.md muc "CANH BAO QUAN TRONG" de biet ly do ky luat
nay ton tai (E2 v1 tung sap vi nhan lan truyen + val thien lech).

KHONG sua train_e2_v2.py/output/e2_v2_best_checkpoint.pt -- output cua v3 nam
rieng (output/e2_v3_best_checkpoint.pt, wandb group "e2_v3_full_fusion"), giu
nguyen kha nang doi chieu lai v2 bat cu luc nao.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import wandb

from data_prep.e2_v3_train_corpus import DEFAULT_CORPUS_DIR, load_e2_v3_train_examples
from data_prep.e2_val_subsample import build_fixed_val_subsample
from data_prep.io_utils import load_node_features
from data_prep.labels_io import PREPROC_DIR
from model.label_aware_sampler import LabelAwareNeighborSampler
from train_eval.metrics import find_best_f1_threshold, compute_metrics
from train_eval.train_e2 import D_GRAPH, build_models, compute_probs_labels, evaluate, make_train_loader, \
    train_one_epoch

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
WANDB_PROJECT = "fraud_detection_inductive"
GLOBAL_SEED = 44


def load_pretrained_graph_encoder(graph_encoder: torch.nn.Module, ckpt_path: str) -> None:
    """Nap encoder_state_dict tu checkpoint cua pretrain_graph_encoder.py vao
    graph_encoder (da build boi build_models(), kien truc GraphSAGEEncoder
    giong het build_models: in_channels=23, out_channels=D_GRAPH). Assert
    kien truc khop truoc, tranh loi kho hieu load_state_dict o giua chung
    (dung tinh than "assert som" da thay o FraudClassifier.__init__)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["encoder_config"]
    assert cfg["in_channels"] == 23, f"pretrain in_channels={cfg['in_channels']} != 23 (build_models co dinh 23)"
    assert cfg["out_channels"] == D_GRAPH, f"pretrain out_channels={cfg['out_channels']} != D_GRAPH={D_GRAPH}"
    graph_encoder.load_state_dict(ckpt["encoder_state_dict"])
    print(f"warm-started graph_encoder tu {ckpt_path} "
          f"(pretrain best_epoch={ckpt.get('best_epoch')}, best_val_recon_loss={ckpt.get('best_val_recon_loss')})")


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
    ap.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR),
                     help="thu muc corpus da build boi mg_build_examples.py -- doi sang corpus --cap_train lon hon "
                          "100k neu da rebuild (xem data_prep/e2_v3_train_corpus.py)")
    ap.add_argument("--pretrained-graph-encoder", type=str, default=None,
                     help="duong dan checkpoint tu pretrain_graph_encoder.py de warm-start graph_encoder; "
                          "bo trong = khoi tao ngau nhien giong het E2 v2")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run = None if args.no_wandb else wandb.init(
        project=WANDB_PROJECT, group="e2_v3_full_fusion", name="e2_v3_bert_graphsage_fusion", config=vars(args)
    )

    print(f"loading E2 v3 train corpus tu {args.corpus_dir} (nhan strict labels.pkl)...")
    train_examples = load_e2_v3_train_examples(corpus_dir=Path(args.corpus_dir), max_examples=args.max_examples)
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
    if args.pretrained_graph_encoder:
        load_pretrained_graph_encoder(graph_encoder, args.pretrained_graph_encoder)
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
    val_metrics_at_best_t = compute_metrics(val_labels, val_probs, threshold=best_threshold)
    print(f"best_threshold={best_threshold:.4f} val_metrics_at_best_t={val_metrics_at_best_t}")

    checkpoint = {
        "graph_encoder": best_state["graph_encoder"],
        "classifier": best_state["classifier"],
        "threshold": best_threshold,
        "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc,
        "pretrained_graph_encoder": args.pretrained_graph_encoder,
        "corpus_dir": args.corpus_dir,
    }
    torch.save(checkpoint, OUTPUT_DIR / "e2_v3_best_checkpoint.pt")
    print(f"\n=== E2 v3 checkpoint saved -> output/e2_v3_best_checkpoint.pt "
          f"(best_epoch={best_epoch}, val_auprc={best_val_auprc:.4f}, threshold={best_threshold:.4f}) ===")
    print("KHONG eval pure_test/overlap that o day -- chay "
          "`python -m train_eval.eval_full_test --checkpoint output/e2_v3_best_checkpoint.pt "
          "--out-json e2_v3_full_eval_result.json --group e2_v3_full_eval` MOT LAN DUY NHAT sau buoc nay.")

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_auprc"] = best_val_auprc
        run.summary["best_threshold"] = best_threshold
        for k, v in val_metrics_at_best_t.items():
            run.summary[f"val_best_t/{k}"] = v
        run.finish()

    with open(OUTPUT_DIR / "e2_v3_train_history.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_auprc": best_val_auprc,
                    "best_threshold": best_threshold, "val_metrics_at_best_t": val_metrics_at_best_t,
                    "args": vars(args)}, f, indent=2)
    print(f"Saved -> {OUTPUT_DIR / 'e2_v3_train_history.json'}")


if __name__ == "__main__":
    main()
