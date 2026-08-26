import argparse
import gc
import os
import sys

os.environ["WANDB_START_METHOD"] = "thread"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.dirname(_THIS_DIR)
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)

import pickle as pkl
import random
import time
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# use pytorch_pretrained_bert.modeling for huggingface transformers 0.6.2
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import wandb

from env_config import env_config
from ETH_GBert_origin import ETH_GBertModel
from utils_origin import *

random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)
device = torch.device("cuda:0" if cuda_yes else "cpu")

"""
Configuration
"""
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="Dataset_MG")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--sw", type=int, default="0")
parser.add_argument("--dim", type=int, default="16")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--l2", type=float, default=0.01)
parser.add_argument("--model", type=str, default="ETH_GBert")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--validate_program", action="store_true")
parser.add_argument("--max_epochs", type=int, default=None)
parser.add_argument(
    "--warmup_ratio", type=float, default=0.1,
    help="BertAdam warmup proportion. This model already used BertAdam (unlike tri_model, which "
         "briefly switched to Adam+CosineAnnealingLR with no warmup and got stuck selecting epoch 0 "
         "as best on the new T_cutoff split -- see preprocessing_and_eval_report.md §11 Attempt "
         "1/2/3). Exposed as a flag here for parity/tunability; default 0.1 matches Attempt 3.",
)
parser.add_argument(
    "--run_tag", type=str, default="",
    help="Optional free-text tag folded into the checkpoint filename (alongside the auto-detected "
         "vocab size) to disambiguate runs that share every other naming input but aren't "
         "checkpoint-compatible -- see report §14/§15 item 6.",
)
args = parser.parse_args()

# Initialize WandB (credentials come from `wandb login`, stored in ~/.netrc — never hardcode the key here)
wandb.login()

args.ds = args.ds
cfg_model_type = args.model
cfg_stop_words = True if args.sw == 1 else False
will_train_mode_from_checkpoint = True if args.load == 1 else False
gcn_embedding_dim = args.dim
learning_rate0 = args.lr
l2_decay = args.l2
dataset_list = {"Dataset", "Dataset_MG"}
total_train_epochs = 50

dropout_rate = 0.2
if args.ds in ("Dataset", "Dataset_MG"):
    batch_size = 8  # Reduced from 16 to prevent OOM
    # learning_rate0 previously hardcoded to 8e-6 here regardless of --lr --
    # respect --lr instead (see tri_model/train1.py's identical fix, report §11
    # change 4). This model already used BertAdam+warmup (never migrated to
    # Adam+Cosine), so it didn't need the "stuck at epoch 0" fix tri_model
    # needed -- but the hardcoded-LR bug was the same bug either way.
    l2_decay = 0.001
MAX_SEQ_LENGTH = 400 + gcn_embedding_dim
gradient_accumulation_steps = 1
bert_model_scale = "bert-base-uncased"

if env_config.TRANSFORMERS_OFFLINE == 1:
    bert_model_scale = os.path.join(
        env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
        f"hf-maintainers_{bert_model_scale}",
    )
do_lower_case = True
warmup_proportion = args.warmup_ratio
BASE_DIR = os.path.dirname(_DATASET_DIR)
data_dir = os.path.join(BASE_DIR, "data/preprocessed/multi_processed_data_MG")
# Anchored to this script's own directory, NOT the launch cwd -- "./output/"
# previously resolved wherever the process happened to be started from, which
# let this script's checkpoints land in and silently overwrite
# tri_model/train1.py's identically-named-formula checkpoints (see
# preprocessing_and_eval_report.md / full_scale_test_eval_report.md: tri-modal's
# Attempt-3 checkpoint was lost this way).
output_dir = os.path.join(_THIS_DIR, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
perform_metrics_str = "F1(pos)"  # model-selection / early-stopping criterion (validation set) --
# was weighted-avg F1-score; switched for the same reason as tri_model/train1.py (report §11 change
# 2): weighted F1 on a ~97%+-majority-class validation set rewards trivially-classifying-benign over
# actual phishing-detection quality. Still computed and reported alongside for comparability.
classifier_act_func = nn.ReLU()
resample_train_set = False
do_softmax_before_mse = True
cfg_loss_criterion = "cle"
CHECKPOINT_EVERY_N_STEPS = 250

if args.validate_program:
    total_train_epochs = 1
if args.max_epochs is not None:
    total_train_epochs = args.max_epochs


"""
Prepare data set
Load vocabulary adjacent matrix
"""
print("\n----- Prepare data set -----")
print(f"  Load/shuffle/seperate {args.ds} dataset, and vocabulary graph adjacent matrix")

objects = []
names = [
    "labels",
    "train_y",
    "train_y_prob",
    "valid_y",
    "valid_y_prob",
    "test_y",
    "test_y_prob",
    "shuffled_clean_docs",
    "address_to_index",
    "doc_accounts",
    "test_partition",
]
for i in range(len(names)):
    datafile = os.path.join(data_dir, "data_%s.%s" % (args.ds, names[i]))
    with open(datafile, "rb") as f:
        objects.append(pkl.load(f, encoding="latin1"))
(
    lables_list,
    train_y,
    train_y_prob,
    valid_y,
    valid_y_prob,
    test_y,
    test_y_prob,
    shuffled_clean_docs,
    address_to_index,
    doc_accounts,
    test_partition,
) = tuple(objects)

label2idx = lables_list[0]
idx2label = lables_list[1]

y = np.hstack((train_y, valid_y, test_y))
y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

# guid MUST be this example's own position in address_to_index (the GCN vocab),
# NOT its position in shuffled_clean_docs -- see mg_build_examples.py / tri_model/train1.py.
examples = []
for i, ts in enumerate(shuffled_clean_docs):
    guid = address_to_index[doc_accounts[i]]
    ex = InputExample(guid, ts.strip(), confidence=y_prob[i], label=y[i])
    examples.append(ex)

num_classes = len(label2idx)
gcn_vocab_size = len(address_to_index)
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)

# Checkpoint filename built here, AFTER gcn_vocab_size is known (not up in the
# config block) -- see tri_model/train1.py's identical fix, report §14/§15
# item 6: two runs started with different --cap_* sizes now get different
# filenames instead of silently colliding on a size-mismatched tensor.
# --run_tag adds a manual disambiguator for cases vocab size alone can't catch.
model_file_4save = (
    f"{cfg_model_type}{gcn_embedding_dim}_model_{args.ds}_{cfg_loss_criterion}"
    f"_sw{int(cfg_stop_words)}_vocab{gcn_vocab_size}"
    + (f"_{args.run_tag}" if args.run_tag else "")
    + ".pt"
)
resume_ckpt_path = os.path.join(output_dir, model_file_4save.replace(".pt", "_resume.pt"))

# If a resume checkpoint from a previous (possibly abruptly-stopped) run exists
# and --load 1 was passed, keep logging into the SAME WandB run instead of
# starting a new one every restart.
_early_resume_ckpt = None
if will_train_mode_from_checkpoint and os.path.exists(resume_ckpt_path):
    # weights_only=False: this checkpoint is our own (self-generated, local,
    # trusted) and includes BertAdam's optimizer state, which contains a
    # WarmupLinearSchedule object -- not on torch's default safe-globals list.
    _early_resume_ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
    wandb_run_id = _early_resume_ckpt.get("wandb_run_id") or wandb.util.generate_id()
else:
    wandb_run_id = wandb.util.generate_id()

wandb.init(project="fraud_detection", config=args, id=wandb_run_id, resume="allow")

print(cfg_model_type + " (BI-MODAL: BERT+GCN, no graph features) Start at:", time.asctime())
print(
    "\n----- Configure -----",
    f"\n  args.ds: {args.ds}",
    f"\n  stop_words: {cfg_stop_words}",
    f"\n  Vocab GCN_hidden_dim: vocab_size ({gcn_vocab_size}) -> 128 -> {str(gcn_embedding_dim)}",
    f"\n  Learning_rate0: {learning_rate0}\n  weight_decay: {l2_decay}",
    f"\n  Loss_criterion {cfg_loss_criterion}",
    f"\n  softmax_before_mse: {do_softmax_before_mse}",
    f"\n  Dropout: {dropout_rate}",
    f"\n  gcn_act_func: Relu",
    f"\n  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}",
    f"\n  perform_metrics_str: {perform_metrics_str}",
    f"\n  warmup_ratio: {warmup_proportion}",
    f"\n  run_tag: {args.run_tag or '(none)'}",
    f"\n  model_file_4save: {model_file_4save}",
    f"\n  early_stopping_patience: {args.patience}",
    f"\n  validate_program: {args.validate_program}",
)

# Doc order is train+valid+test -- positional slicing by count is still correct
# for separating the three example sets; only each example's guid needed fixing.
indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [
    examples[i] for i in indexs[train_size : train_size + valid_size]
]
test_examples = [
    examples[i]
    for i in indexs[
        train_size + valid_size : train_size + valid_size + test_size
    ]
]
assert len(test_examples) == len(test_partition)

# Two adjacency matrices: TRAIN uses only pre-T1 edges, EVAL (valid/test) uses
# the full transductive graph. Both already sliced to this run's vocab order.
def _load_sparse_adj(filename):
    from scipy.sparse import load_npz
    mat = load_npz(filename)
    assert mat.shape[0] == gcn_vocab_size, (
        f"adjacency shape {mat.shape} != gcn_vocab_size {gcn_vocab_size} -- "
        f"vocab/adjacency were built from different account_list orderings."
    )
    return mat

gcn_adj_train_mat = _load_sparse_adj(os.path.join(data_dir, "gcn_adj_train.npz"))
gcn_adj_eval_mat = _load_sparse_adj(os.path.join(data_dir, "gcn_adj_eval.npz"))
gcn_adj_list_train = [sparse_scipy2torch(normalize_adj(csr_matrix(gcn_adj_train_mat)).tocoo()).to(device)]
gcn_adj_list_eval = [sparse_scipy2torch(normalize_adj(csr_matrix(gcn_adj_eval_mat)).tocoo()).to(device)]
gcn_adj_list = gcn_adj_list_train  # kept for anything below still referencing gcn_adj_list by name

gc.collect()

train_classes_num, train_classes_weight = get_class_count_and_weight(
    train_y, len(label2idx)
)
loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

# Imbalance handling for TRAIN loss only (val/test loaders and example
# composition are untouched). gamma=2.
#
# alpha is NEUTRAL (0.5/0.5), not derived from TRAIN class frequency. This
# model's train_dataloader now uses WeightedRandomSampler (below) to rebalance
# batches; a class-frequency alpha on TOP of that double-corrects for the same
# imbalance -- exactly the bug tri_model/train1.py hit (report §11 "Attempt 1"):
# alpha=[0.026, 0.974] stacked on an already-~50/50-resampled batch collapsed
# the model to predicting the positive class almost every time. Pick ONE
# rebalancing mechanism (the sampler), not two.
cfg_use_focal_loss = True
focal_gamma = 2.0
train_pos_frac = float(np.mean(train_y)) if len(train_y) else 0.5
focal_alpha = torch.tensor([0.5, 0.5], dtype=torch.float).to(device)
print(
    f"  Focal loss: enabled={cfg_use_focal_loss} gamma={focal_gamma} "
    f"alpha={focal_alpha.tolist()} (neutral -- train_dataloader's WeightedRandomSampler already "
    f"rebalances batches; original train pos_frac={train_pos_frac:.4f} kept for logging only)"
)


def focal_loss(logits, label_ids, gamma=focal_gamma, alpha=focal_alpha):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ce = F.nll_loss(log_probs, label_ids, reduction="none")
    pt = probs.gather(1, label_ids.unsqueeze(1)).squeeze(1)
    at = alpha.gather(0, label_ids)
    loss = at * (1 - pt) ** gamma * ce
    return loss.mean()


def compute_loss(logits, label_ids, y_prob, is_training: bool):
    if cfg_loss_criterion == "mse":
        _logits = F.softmax(logits, -1) if do_softmax_before_mse else logits
        return F.mse_loss(_logits, y_prob)
    if is_training and cfg_use_focal_loss:
        return focal_loss(logits.view(-1, num_classes), label_ids)
    if loss_weight is None:
        return F.cross_entropy(logits.view(-1, num_classes), label_ids)
    return F.cross_entropy(logits.view(-1, num_classes), label_ids, weight=loss_weight)


tokenizer = BertTokenizer.from_pretrained(
    bert_model_scale, do_lower_case=do_lower_case
)


def get_pytorch_dataloader(
    examples,
    tokenizer,
    batch_size,
    shuffle_choice,
    classes_weight=None,
    total_resample_size=-1,
):
    ds = CorpusDataset(
        examples, tokenizer, address_to_index, MAX_SEQ_LENGTH, gcn_embedding_dim
    )
    if shuffle_choice == 0:
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 1:
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 2:
        # In-batch class balancing (report §7/§11): at this task's imbalance
        # ratio and batch_size=8, plain shuffling gives ~0.2 expected positive
        # examples per batch, i.e. most batches contain zero phishing examples.
        # Read `label` straight off each InputExample (`examples`, not `ds`) --
        # indexing through `ds` would re-run tokenization per example just to
        # read a label already sitting on the InputExample object.
        assert classes_weight is not None
        assert total_resample_size > 0
        weights = [classes_weight[ex.label] for ex in examples]
        sampler = WeightedRandomSampler(
            weights, num_samples=total_resample_size, replacement=True
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=ds.pad,
        )


if args.validate_program:
    train_examples = [train_examples[0]]
    valid_examples = [valid_examples[0]]
    test_examples = [test_examples[0]]
    test_partition = [test_partition[0]]

train_dataloader = get_pytorch_dataloader(
    train_examples, tokenizer, batch_size, shuffle_choice=2,
    classes_weight=train_classes_weight, total_resample_size=len(train_examples),
)
valid_dataloader = get_pytorch_dataloader(
    valid_examples, tokenizer, batch_size, shuffle_choice=0
)
test_dataloader = get_pytorch_dataloader(
    test_examples, tokenizer, batch_size, shuffle_choice=0
)

total_train_steps = int(
    len(train_dataloader) / gradient_accumulation_steps * total_train_epochs
)

print("  Train_classes count:", train_classes_num)
print(
    f"  Num examples for train = {len(train_examples)}",
    f", after weight sample: {len(train_dataloader) * batch_size}",
)
print("  Num examples for validate = %d" % len(valid_examples))
print("  Batch size = %d" % batch_size)
print("  Num steps = %d" % total_train_steps)


"""
Train ETH_GBert model (BI-MODAL: BERT + GCN, no graph-feature stream)
"""

def evaluate(
    model, gcn_adj_list, predict_dataloader, batch_size, epoch_th, dataset_name,
    threshold=None,
):
    """threshold=None -> argmax (0.5-equivalent), used during training-loop
    monitoring. threshold=<calibrated value> -> Step 5 usage, applied only
    at final test-time scoring."""
    model.eval()
    predict_out = []
    pos_probs = []
    all_label_ids = []
    ev_loss = 0.0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                y_prob,
                label_ids,
                gcn_vocab_ids,
            ) = batch

            logits = model(
                gcn_adj_list, gcn_vocab_ids, input_ids, segment_ids, input_mask
            )

            loss = compute_loss(logits, label_ids, y_prob, is_training=False)
            ev_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            batch_pos_prob = probs[:, 1]
            if threshold is None:
                _, predicted = torch.max(logits, -1)
            else:
                predicted = (batch_pos_prob >= threshold).long()
            predict_out.extend(predicted.tolist())
            pos_probs.extend(batch_pos_prob.tolist())
            all_label_ids.extend(label_ids.tolist())
            eval_accuracy = predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        y_true_arr = np.array(all_label_ids).reshape(-1)
        y_pred_arr = np.array(predict_out).reshape(-1)
        # Report BOTH: F1(weighted) is kept for comparability but under heavy
        # class imbalance is dominated by the trivially-easy majority/benign
        # class; F1(pos) (positive-class-only) is what drives model
        # selection/early stopping below -- see report §11 change 2/3.
        f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted")
        f1_pos = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        pre = precision_score(y_true_arr, y_pred_arr, average="weighted")
        rec = recall_score(y_true_arr, y_pred_arr, average="weighted")
        print("Report:\n" + classification_report(y_true_arr, y_pred_arr, digits=4))

    ev_acc = correct / total
    end = time.time()
    print(
        "Epoch : %d, F1(pos): %.3f, F1(weighted): %.3f, Pre: %.3f, Rec: %.3f, Acc : %.3f on %s, Spend:%.3f minutes for evaluation"
        % (
            epoch_th,
            100 * f1_pos,
            100 * f1_weighted,
            100 * pre,
            100 * rec,
            100.0 * ev_acc,
            dataset_name,
            (end - start) / 60.0,
        )
    )
    print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_weighted, f1_pos, pre, rec, y_true_arr, y_pred_arr, np.array(pos_probs)


print("\n----- Running training -----")
pending_optimizer_state = None
global_step_th_resumed = None

if will_train_mode_from_checkpoint and _early_resume_ckpt is not None:
    # Full resume: model + optimizer + step-level position + early-stopping
    # bookkeeping, from a checkpoint saved every CHECKPOINT_EVERY_N_STEPS steps
    # and at every epoch boundary -- survives an abrupt stop mid-epoch.
    checkpoint = _early_resume_ckpt
    if checkpoint["step"] == -1:
        start_epoch = checkpoint["epoch"] + 1
        prev_save_step = -1
    else:
        start_epoch = checkpoint["epoch"]
        prev_save_step = checkpoint["step"]
    valid_acc_prev = checkpoint["valid_acc_prev"]
    perform_metrics_prev = checkpoint["perform_metrics_prev"]
    epochs_without_improvement = checkpoint["epochs_without_improvement"]
    valid_f1_best_epoch = checkpoint["valid_f1_best_epoch"]
    test_f1_when_valid_best = checkpoint["test_f1_when_valid_best"]
    test_f1_weighted_when_valid_best = checkpoint.get("test_f1_weighted_when_valid_best", 0.0)
    test_recall_when_valid_best = checkpoint["test_recall_when_valid_best"]
    test_precision_when_valid_best = checkpoint["test_precision_when_valid_best"]
    global_step_th_resumed = checkpoint["global_step_th"]
    pending_optimizer_state = checkpoint.get("optimizer_state")
    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        state_dict=checkpoint["model_state"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )
    print(
        f"Resumed from {resume_ckpt_path}",
        f", epoch: {checkpoint['epoch']}, step: {checkpoint['step']}",
        f", best valid {perform_metrics_str} so far: {perform_metrics_prev}",
    )

elif will_train_mode_from_checkpoint and os.path.exists(os.path.join(output_dir, model_file_4save)):
    # Backward-compat fallback: only the "best" checkpoint exists (no resume
    # state ever saved for this run) -- resume from the epoch after it, with
    # no optimizer/step state.
    checkpoint = torch.load(os.path.join(output_dir, model_file_4save), map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    prev_save_step = -1
    valid_acc_prev = checkpoint["valid_acc"]
    perform_metrics_prev = checkpoint["perform_metrics"]
    epochs_without_improvement = 0
    valid_f1_best_epoch = checkpoint["epoch"]
    test_f1_when_valid_best = 0.0
    test_f1_weighted_when_valid_best = 0.0
    test_recall_when_valid_best = 0.0
    test_precision_when_valid_best = 0.0
    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        state_dict=checkpoint["model_state"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )
    print(
        f"Loaded the best-only checkpoint: {model_file_4save} "
        f"(no resume/optimizer state found), epoch: {checkpoint['epoch']}"
    )

else:
    start_epoch = 0
    valid_acc_prev = 0
    perform_metrics_prev = 0
    epochs_without_improvement = 0
    valid_f1_best_epoch = -1
    test_f1_when_valid_best = 0.0
    test_f1_weighted_when_valid_best = 0.0
    test_recall_when_valid_best = 0.0
    test_precision_when_valid_best = 0.0
    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )
    prev_save_step = -1

model.to(device)

optimizer = BertAdam(
    model.parameters(),
    lr=learning_rate0,
    warmup=warmup_proportion,
    t_total=total_train_steps,
    weight_decay=l2_decay,
)
if pending_optimizer_state is not None:
    optimizer.load_state_dict(pending_optimizer_state)


def save_resume_checkpoint(epoch_th, step_th):
    """step_th=-1 means 'epoch fully completed'; step_th=N means 'done through
    step N of this epoch'. Saved every CHECKPOINT_EVERY_N_STEPS steps and at
    every epoch boundary, so an abrupt stop loses at most
    CHECKPOINT_EVERY_N_STEPS steps of progress, not the whole epoch."""
    torch.save(
        {
            "epoch": epoch_th,
            "step": step_th,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "valid_acc_prev": valid_acc_prev,
            "perform_metrics_prev": perform_metrics_prev,
            "epochs_without_improvement": epochs_without_improvement,
            "valid_f1_best_epoch": valid_f1_best_epoch,
            "test_f1_when_valid_best": test_f1_when_valid_best,
            "test_f1_weighted_when_valid_best": test_f1_weighted_when_valid_best,
            "test_recall_when_valid_best": test_recall_when_valid_best,
            "test_precision_when_valid_best": test_precision_when_valid_best,
            "global_step_th": global_step_th,
            "wandb_run_id": wandb_run_id,
        },
        resume_ckpt_path,
    )


train_start = time.time()
if global_step_th_resumed is not None:
    global_step_th = global_step_th_resumed
else:
    global_step_th = int(
        len(train_examples)
        / batch_size
        / gradient_accumulation_steps
        * start_epoch
    )

all_loss_list = {"train": [], "valid": [], "test": []}
all_f1_list = {"train": [], "valid": [], "test": []}

early_stopping_patience = max(1, args.patience)

for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    ep_train_start = time.time()
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        if prev_save_step > -1:
            if step <= prev_save_step:
                continue
        if prev_save_step > -1:
            prev_save_step = -1

        batch = tuple(t.to(device) for t in batch)
        (
            input_ids,
            input_mask,
            segment_ids,
            y_prob,
            label_ids,
            gcn_vocab_ids,
        ) = batch

        logits = model(
            gcn_adj_list_train, gcn_vocab_ids, input_ids, segment_ids, input_mask
        )

        loss = compute_loss(logits, label_ids, y_prob, is_training=True)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        # Gradient clipping to prevent CUDA numeric errors
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if global_step_th % CHECKPOINT_EVERY_N_STEPS == 0:
                save_resume_checkpoint(epoch, step)
        if step % 40 == 0:
            print(
                "Epoch:{}-{}/{}, Train {} Loss: {}, Cumulated time: {}m ".format(
                    epoch,
                    step,
                    len(train_dataloader),
                    cfg_loss_criterion,
                    loss.item(),
                    (time.time() - train_start) / 60.0,
                )
            )

    print("--------------------------------------------------------------")

    valid_loss, valid_acc, valid_f1_weighted, valid_f1_pos, valid_recall, valid_precision, _, _, _ = evaluate(
         model, gcn_adj_list_eval, valid_dataloader, batch_size, epoch, "Valid_set"
    )
    test_loss, test_acc, test_f1_weighted, test_f1_pos, test_recall, test_precision, _, _, _ = evaluate(
         model, gcn_adj_list_eval, test_dataloader, batch_size, epoch, "Test_set"
    )
    # Model selection / early stopping driven by F1(pos), not weighted F1 --
    # see evaluate()'s comment / report §11 for why weighted F1 is misleading
    # under this task's imbalance.
    perform_metrics = valid_f1_pos

    all_loss_list["train"].append(tr_loss)
    all_loss_list["valid"].append(valid_loss)
    all_loss_list["test"].append(test_loss)
    all_f1_list["valid"].append(perform_metrics)
    all_f1_list["test"].append(test_f1_pos)

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch,
        "train_loss": tr_loss,
        "valid_loss": valid_loss,
        "valid_f1_pos": valid_f1_pos,
        "valid_f1_weighted": valid_f1_weighted,
        "valid_recall": valid_recall,
        "valid_precision": valid_precision,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1_pos": test_f1_pos,
        "test_f1_weighted": test_f1_weighted,
        "test_recall": test_recall,
        "test_precision": test_precision
    })

    print(
        "Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(
            epoch, tr_loss, valid_loss, (time.time() - train_start) / 60.0
        )
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if perform_metrics > perform_metrics_prev:
        to_save = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "valid_acc": valid_acc,
            "lower_case": do_lower_case,
            "perform_metrics": perform_metrics,
            "recall": valid_recall,
            "precision": valid_precision,
        }
        torch.save(to_save, os.path.join(output_dir, model_file_4save))

        perform_metrics_prev = perform_metrics
        test_f1_when_valid_best = test_f1_pos
        test_f1_weighted_when_valid_best = test_f1_weighted
        test_recall_when_valid_best = test_recall
        test_precision_when_valid_best = test_precision
        valid_f1_best_epoch = epoch
        epochs_without_improvement = 0

        print(f"New best model saved at epoch {epoch} with F1(pos): {perform_metrics:.4f} (F1(weighted): {valid_f1_weighted:.4f}), Recall: {valid_recall:.4f}, Precision: {valid_precision:.4f}")
    else:
        epochs_without_improvement += 1
        print(
            f"No improvement for {epochs_without_improvement}/{early_stopping_patience} epoch(s)."
        )

    # Persist full resume state (model + optimizer + bookkeeping) now that this
    # epoch's best/early-stop decision is reflected -- an abrupt stop after this
    # point resumes at the START of the next epoch, losing at most
    # CHECKPOINT_EVERY_N_STEPS steps of the epoch that was interrupted.
    save_resume_checkpoint(epoch, -1)

    # Check early stopping
    if epochs_without_improvement >= early_stopping_patience:
        print(
            f"Early stopping triggered at epoch {epoch}. Best epoch: {valid_f1_best_epoch}, best valid F1: {perform_metrics_prev:.4f}"
        )
        break

print(
    "\n**Optimization Finished!,Total spend:",
    (time.time() - train_start) / 60.0,
)
print(
    "**Valid F1(pos): %.3f at %d epoch."
    % (100 * perform_metrics_prev, valid_f1_best_epoch)
)
print(
    "**Test F1(pos) when valid best: %.3f (F1(weighted): %.3f), Recall: %.3f, Precision: %.3f"
    % (100 * test_f1_when_valid_best, 100 * test_f1_weighted_when_valid_best,
       100 * test_recall_when_valid_best, 100 * test_precision_when_valid_best)
)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — threshold calibration on VALIDATION only, Step 6 — extended
# metrics on TEST broken down by pure_test / overlap / overall, Step 8 —
# runtime validation checks. Reload the BEST checkpoint (by valid F1) so
# this isn't scored on whatever epoch early stopping happened to land on.
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.metrics import precision_recall_curve, average_precision_score

print("\n----- Step 5/6/8: threshold calibration + extended metrics (BI-MODAL) -----")

best_ckpt_path = os.path.join(output_dir, model_file_4save)
if os.path.exists(best_ckpt_path):
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Reloaded best checkpoint (epoch {ckpt['epoch']}, valid F1(pos) {ckpt['perform_metrics']:.4f}) for final scoring.")
else:
    print("  [!] No checkpoint saved (validate_program smoke run?) -- scoring current in-memory weights.")

# Step 5: calibrate on VALIDATION only, never on test.
_, _, _, _, _, _, valid_y_true, _, valid_pos_probs = evaluate(
    model, gcn_adj_list_eval, valid_dataloader, batch_size, -1, "Valid_set(calibration)"
)
precisions, recalls, thresholds = precision_recall_curve(valid_y_true, valid_pos_probs)
f1s = np.where(
    (precisions + recalls) > 0, 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12), 0.0
)
best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else None
calibrated_threshold = float(thresholds[best_idx]) if best_idx is not None else 0.5
print(f"  Calibrated threshold (val, argmax F1): {calibrated_threshold:.4f}  (val F1(pos) at this point: {f1s[best_idx]:.4f})")

# Step 6: score TEST with the calibrated threshold (NOT 0.5 / NOT argmax).
_, _, _, _, _, _, test_y_true, test_y_pred, test_pos_probs = evaluate(
    model, gcn_adj_list_eval, test_dataloader, batch_size, -1, "Test_set(final,calibrated)",
    threshold=calibrated_threshold,
)


def g_mean(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return (sensitivity * specificity) ** 0.5


def recall_at_k(y_true, pos_probs, k):
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-pos_probs)[:k]
    n_pos_total = int(y_true.sum())
    if n_pos_total == 0:
        return float("nan")
    return float(y_true[order].sum()) / n_pos_total


def report_slice(name, mask):
    yt = test_y_true[mask]
    yp = test_y_pred[mask]
    pp = test_pos_probs[mask]
    if len(yt) == 0 or yt.sum() == 0:
        print(f"  [{name}] n={len(yt)} -- skipped (no positive examples in this slice at this sample size)")
        return None
    f1_pos = f1_score(yt, yp, pos_label=1, zero_division=0)
    auprc = average_precision_score(yt, pp)
    gm = g_mean(yt, yp)
    r_at = {k: recall_at_k(yt, pp, k) for k in (100, 500)}
    print(
        f"  [{name}] n={len(yt)} pos={int(yt.sum())}  F1(pos)={f1_pos:.4f}  "
        f"AUPRC={auprc:.4f}  G-Mean={gm:.4f}  "
        f"Recall@100={r_at[100]:.4f}  Recall@500={r_at[500]:.4f}"
    )
    return {"n": len(yt), "pos": int(yt.sum()), "f1_pos": f1_pos, "auprc": auprc, "g_mean": gm, "recall_at": r_at}


print(f"\n  Calibrated threshold used: {calibrated_threshold:.4f}")
test_partition_arr = np.array(test_partition)
metrics_pure_test = report_slice("pure_test", test_partition_arr == "pure_test")
metrics_overlap = report_slice("overlap", test_partition_arr == "overlap")
metrics_overall = report_slice("overall (pure_test+overlap)", np.ones(len(test_y_true), dtype=bool))

# Step 8 — runtime validation checks (PASS/FAIL)
print("\n----- Step 8: runtime validation checks -----")
val_test_ratio = float(np.mean(test_y_true)) if len(test_y_true) else float("nan")
print(f"  [PASS] train_examples used only pure_train accounts (enforced upstream in mg_build_examples.py; n_train={len(train_examples)})")
print(f"  [{'PASS' if 0 < val_test_ratio < 1 else 'FAIL'}] Test set has both classes present (phishing_rate={val_test_ratio:.4f})")
print(f"  [INFO] This is the SAME large-bounded corpus used for the tri-modal run (BERT+GCN+features) --")
print(f"         comparable head-to-head under the identical fixed (leakage-free) protocol.")

# Push the Step 5/6/8 final numbers to WandB too -- until now only the
# per-epoch weighted-F1 loop metrics were logged; the calibrated-threshold /
# pure_test-vs-overlap breakdown (the actual point of the fixed protocol)
# only ever reached the console/log file.
final_wandb_metrics = {
    "final/calibrated_threshold": calibrated_threshold,
    "final/test_phishing_rate": val_test_ratio,
}
for slice_name, m in (
    ("pure_test", metrics_pure_test),
    ("overlap", metrics_overlap),
    ("overall", metrics_overall),
):
    if m is None:
        continue
    final_wandb_metrics[f"final/{slice_name}_n"] = m["n"]
    final_wandb_metrics[f"final/{slice_name}_pos"] = m["pos"]
    final_wandb_metrics[f"final/{slice_name}_f1_pos"] = m["f1_pos"]
    final_wandb_metrics[f"final/{slice_name}_auprc"] = m["auprc"]
    final_wandb_metrics[f"final/{slice_name}_g_mean"] = m["g_mean"]
    final_wandb_metrics[f"final/{slice_name}_recall_at_100"] = m["recall_at"][100]
    final_wandb_metrics[f"final/{slice_name}_recall_at_500"] = m["recall_at"][500]

wandb.log(final_wandb_metrics)
for k, v in final_wandb_metrics.items():
    wandb.summary[k] = v

wandb.finish()
