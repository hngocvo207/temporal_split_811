# -*- coding: utf-8 -*-
import argparse
import gc
import os
import sys
import pickle as pkl
import random
import time
from tqdm import tqdm

import dgl
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from sklearn.metrics import classification_report, f1_score

from env_config import env_config
from model.TLM import VGCNBertModel
from utils import *
from model import BertGCN, BertGAT, BertSAGE

def update_feature(g, model, dataloader, device):
    with torch.no_grad():
        g = g.to(device)
        model = model.to(device)
        model.eval()
        for _, batch in tqdm(enumerate(dataloader), desc="update graph"):
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                y_prob,
                label_ids,
                gcn_swop_eye,
                guid,
            ) = batch
            _, output = model.bert_model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
            idx = [guid_to_index[item.item()] for item in guid]
            g.ndata['cls_feats'][idx] = output
    return g


random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

cuda_yes = torch.cuda.is_available()

"""
Configuration
"""

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="Dataset")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--sw", type=int, default="0")
parser.add_argument("--dim", type=int, default="16")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--l2", type=float, default=0.01)
parser.add_argument("--validate_program", action="store_true")
parser.add_argument("--m", type=float, default=0.7, help="The m parameter value")
parser.add_argument("--model_name", type=str, default='BertGCN')
parser.add_argument("--use_baseline", type=bool, default=False)
parser.add_argument("--seq_len", type=int, default=200)
parser.add_argument("--vocab", type=str, default='tf')
parser.add_argument("--threshold", type=float, default=0.2)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--epoch", type=int, default="100")
parser.add_argument("--batch_size", type=int, default="64")
args = parser.parse_args()
if cuda_yes:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)
device = torch.device(args.device if cuda_yes else "cpu")
m = args.m
# cfg_add_linear_mapping_term=False
# cfg_vocab_adj = "pmi"
# cfg_vocab_adj='all'
# cfg_vocab_adj='tf'
cfg_vocab_adj = args.vocab
# cfg_adj_npmi_threshold = 0.2
# cfg_adj_tf_threshold = 0
cfg_adj_npmi_threshold = args.threshold
cfg_adj_tf_threshold = args.threshold
classifier_act_func = nn.ReLU()
# if not args.use_baseline:
#     sys.stdout = open(
#         './output/vgcndim=' + str(args.dim) + '_zsedata_m=' + str(m) +
#         "vocab=" + cfg_vocab_adj
#         +"threshold="+str(args.threshold)+ args.model_name +
#         "len=" + str(args.seq_len) + '.txt', 'w')
# else:
#     sys.stdout = open('./output/vgcn_output_zsdata_baseline' + args.model_name + '.txt', 'w')

if not args.use_baseline:
    sys.stdout = open(
        './b4e_output/testvgcndim=' + str(args.dim) + '_b4edata_m=' + str(m) +
        "vocab=" + cfg_vocab_adj
        +"threshold="+str(args.threshold)+ args.model_name +
        "len=" + str(args.seq_len) + '.txt', 'w')
else:
    sys.stdout = open('./b4e_output/testvgcn_output_b4edata_baseline' + args.model_name + '.txt', 'w')

print("m=", m)
print("use_bseline:", args.use_baseline)
print("model_name:", args.model_name)
args.ds = args.ds
cfg_model_type = "TLGNN"
cfg_stop_words = True if args.sw == 1 else False
will_train_mode_from_checkpoint = True if args.load == 1 else False
gcn_embedding_dim = args.dim
learning_rate0 = args.lr
l2_decay = args.l2

total_train_epochs = args.epoch
dropout_rate = 0.2

if args.ds == "Dataset":
    batch_size = args.batch_size  # 12
    learning_rate0 = 8e-6  # 2e-5
    # learning_rate0 = 2e-5
    l2_decay = 0.001

MAX_SEQ_LENGTH = args.seq_len + gcn_embedding_dim
gradient_accumulation_steps = 1

# bert_model_scale='bert-large-uncased'
bert_model_scale = "bert-base-uncased"
if env_config.TRANSFORMERS_OFFLINE == 1:
    bert_model_scale = os.path.join(
        env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
        f"hf-maintainers_{bert_model_scale}",
    )

do_lower_case = True
warmup_proportion = 0.1

data_dir = f"gen_b4e_seq/data_train_b4e"
output_dir = "b4e_output/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

perform_metrics_str = ["weighted avg", "f1-score"]

resample_train_set = False  # if mse and resample, then do resample
do_softmax_before_mse = True
cfg_loss_criterion = "cle"
model_file_4save = (
    f"{cfg_model_type}{gcn_embedding_dim}_model_{args.ds}_{cfg_loss_criterion}"
    f"_sw{int(cfg_stop_words)}.pt"
)

if args.validate_program:
    total_train_epochs = 1

print(cfg_model_type + " Start at:", time.asctime())
print(
    "\n----- Configure -----",
    f"\n  args.ds: {args.ds}",
    f"\n  stop_words: {cfg_stop_words}",
    # '\n  Vocab GCN_hidden_dim: 768 -> 1152 -> 768',
    f"\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> {str(gcn_embedding_dim)}",
    f"\n  Learning_rate0: {learning_rate0}" f"\n  weight_decay: {l2_decay}",
    f"\n  Loss_criterion {cfg_loss_criterion}"
    f"\n  softmax_before_mse: {do_softmax_before_mse}",
    f"\n  Dropout: {dropout_rate}"
    f"\n  Run_adj: {cfg_vocab_adj}"
    f"\n  gcn_act_func: Relu",
    f"\n  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}",  # 'valid_data_taux',valid_data_taux
    f"\n  perform_metrics_str: {perform_metrics_str}",
    f"\n  model_file_4save: {model_file_4save}",
    f"\n  validate_program: {args.validate_program}",
)

"""
Prepare data set
Load vocabulary adjacent matrix
"""
print("\n----- Prepare data set -----")
print(
    f"  Load/shuffle/seperate {args.ds} dataset, and vocabulary graph adjacent matrix"
)
with open("./" + data_dir + "/adj.pkl", "rb") as f:
    gcn_adj = pkl.load(f)
adj_norm = normalize_adj(gcn_adj + sp.eye(gcn_adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata["cls_feats"] = (torch.rand(gcn_adj.shape[0], 64) - 0.5) * 2
g = g.to(device)

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
    "vocab_adj_tf",
    "vocab_adj_pmi",
    "vocab_map",
    "guid_to_index",
    "index_to_guid",
]
for i in range(len(names)):
    datafile = "./" + data_dir + "/data_%s.%s" % (args.ds, names[i])
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
    gcn_vocab_adj_tf,
    gcn_vocab_adj,
    gcn_vocab_map,
    guid_to_index,
    index_to_guid,
) = tuple(objects)

# a=guid_to_index[5154]
label2idx = lables_list[0]
idx2label = lables_list[1]

y = np.hstack((train_y, valid_y, test_y))
y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

examples = []
for i, ts in enumerate(shuffled_clean_docs):
    ex = InputExample(i, ts.strip(), confidence=y_prob[i], label=y[i])
    examples.append(ex)

num_classes = len(label2idx)
gcn_vocab_size = len(gcn_vocab_map)
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)

indexs = np.arange(0, len(examples))
all_examples = [examples[i] for i in indexs]
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [
    examples[i] for i in indexs[train_size: train_size + valid_size]
]
test_examples = [
    examples[i]
    for i in indexs[
             train_size + valid_size: train_size + valid_size + test_size
             ]
]

if cfg_adj_tf_threshold > 0:
    gcn_vocab_adj_tf.data *= gcn_vocab_adj_tf.data > cfg_adj_tf_threshold
    gcn_vocab_adj_tf.eliminate_zeros()
if cfg_adj_npmi_threshold > 0:
    gcn_vocab_adj.data *= gcn_vocab_adj.data > cfg_adj_npmi_threshold
    gcn_vocab_adj.eliminate_zeros()

if cfg_vocab_adj == "pmi":
    gcn_vocab_adj_list = [gcn_vocab_adj]
elif cfg_vocab_adj == "tf":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf]
elif cfg_vocab_adj == "all":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj]

norm_gcn_vocab_adj_list = []
for i in range(len(gcn_vocab_adj_list)):
    adj = gcn_vocab_adj_list[i]  # .tocsr()
    print(
        "  Zero ratio(?>66%%) for vocab adj %dth: %.8f"
        % (i, 100 * (1 - adj.count_nonzero() / (adj.shape[0] * adj.shape[1])))
    )
    adj = normalize_adj(adj)
    norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
gcn_adj_list = norm_gcn_vocab_adj_list

del gcn_vocab_adj_tf, gcn_vocab_adj, gcn_vocab_adj_list
gc.collect()

train_classes_num, train_classes_weight = get_class_count_and_weight(
    train_y, len(label2idx)
)
loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

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
        examples, tokenizer, gcn_vocab_map, MAX_SEQ_LENGTH, gcn_embedding_dim
    )
    if shuffle_choice == 0:  # shuffle==False
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 1:  # shuffle==True
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 2:  # weighted resampled
        assert classes_weight is not None
        assert total_resample_size > 0
        weights = [
            classes_weight[0]
            if label == 0
            else classes_weight[1]
            if label == 1
            else classes_weight[2]
            for _, _, _, _, label in dataset
        ]
        sampler = WeightedRandomSampler(
            weights, num_samples=total_resample_size, replacement=True
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=ds.pad,
        )


# ds size=1 for validating the program
if args.validate_program:
    train_examples = [train_examples[0]]
    valid_examples = [valid_examples[0]]
    test_examples = [test_examples[0]]

all_dataloader = get_pytorch_dataloader(
    all_examples, tokenizer, batch_size, shuffle_choice=0
)
train_dataloader = get_pytorch_dataloader(
    train_examples, tokenizer, batch_size, shuffle_choice=0
)
valid_dataloader = get_pytorch_dataloader(
    valid_examples, tokenizer, batch_size, shuffle_choice=0
)
test_dataloader = get_pytorch_dataloader(
    test_examples, tokenizer, batch_size, shuffle_choice=0
)

# total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
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
Train vgcn_bert model
"""


def predict(model, examples, tokenizer, batch_size):
    dataloader = get_pytorch_dataloader(
        examples, tokenizer, batch_size, shuffle_choice=0
    )
    predict_out = []
    confidence_out = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                _,
                label_ids,
                gcn_swop_eye,
            ) = batch
            score_out = model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
            )
            if cfg_loss_criterion == "mse" and do_softmax_before_mse:
                score_out = torch.nn.functional.softmax(score_out, dim=-1)
            predict_out.extend(score_out.max(1)[1].tolist())
            confidence_out.extend(score_out.max(1)[0].tolist())

    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(
        -1
    )


def evaluate(
        model, gcn_adj_list, predict_dataloader, batch_size, epoch_th, dataset_name
):
    # print("***** Running prediction *****")
    model.eval()
    predict_out = []
    all_label_ids = []
    ev_loss = 0
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
                gcn_swop_eye,
                guid,
            ) = batch
            # the parameter label_ids is None, model return the prediction score
            idx = [guid_to_index[item.item()] for item in guid]
            # logits = model(
            #     gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask,
            # )
            if use_baseline_model:
                logits = model(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
            else:
                logits = model(
                    g, idx, gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask,
                )

            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
            else:
                if loss_weight is None:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
                else:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            eval_accuracy = predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        f1_metrics = f1_score(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            # average="weighted",
            average="binary"
        )
        print(
            "Report:\n"
            + classification_report(
                np.array(all_label_ids).reshape(-1),
                np.array(predict_out).reshape(-1),
                digits=4,
            )
        )

    ev_acc = correct / total
    end = time.time()
    print(
        "Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation"
        % (
            epoch_th,
            " ".join(perform_metrics_str),
            100 * f1_metrics,
            100.0 * ev_acc,
            dataset_name,
            (end - start) / 60.0,
        )
    )
    print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_metrics


print("\n----- Running training -----")
if will_train_mode_from_checkpoint and os.path.exists(
        os.path.join(output_dir, model_file_4save)
):
    checkpoint = torch.load(
        os.path.join(output_dir, model_file_4save), map_location="cpu"
    )
    if "step" in checkpoint:
        prev_save_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
    else:
        prev_save_step = -1
        start_epoch = checkpoint["epoch"] + 1
    valid_acc_prev = checkpoint["valid_acc"]
    perform_metrics_prev = checkpoint["perform_metrics"]
    model = VGCNBertModel.from_pretrained(
        bert_model_scale,
        state_dict=checkpoint["model_state"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
        output_attentions=True,
    )
    pretrained_dict = checkpoint["model_state"]
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {
        k: v for k, v in pretrained_dict.items() if k in net_state_dict
    }
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print(
        f"Loaded the pretrain model: {model_file_4save}",
        f", epoch: {checkpoint['epoch']}",
        f"step: {prev_save_step}",
        f"valid acc: {checkpoint['valid_acc']}",
        f"{' '.join(perform_metrics_str)}_valid: {checkpoint['perform_metrics']}",
    )

else:
    start_epoch = 0
    valid_acc_prev = 0
    perform_metrics_prev = 0
    model = VGCNBertModel.from_pretrained(
        bert_model_scale,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
        output_attentions=True,
    )
    prev_save_step = -1

if args.model_name == "BertGCN":
    model = BertGCN(model, m=m)
elif args.model_name == "BertGAT":
    model = BertGAT(model, m=m)
elif args.model_name == "BertSAGE":
    model = BertSAGE(model, m=m)
model.to(device)

optimizer = BertAdam(
    # model.bert_model.parameters(),
    model.parameters(),
    lr=learning_rate0,
    warmup=warmup_proportion,
    t_total=total_train_steps,
    weight_decay=l2_decay,
)

train_start = time.time()
global_step_th = int(
    len(train_examples)
    / batch_size
    / gradient_accumulation_steps
    * start_epoch
)

all_loss_list = {"train": [], "valid": [], "test": []}
all_f1_list = {"train": [], "valid": [], "test": []}
update_feature(g, model, all_dataloader, device)
use_baseline_model = args.use_baseline
if use_baseline_model:
    model = model.gcn
    model.to(device)
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    ep_train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}"):
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
            gcn_swop_eye,
            guid,
        ) = batch
        idx = [guid_to_index[item.item()] for item in guid]
        # logits = model(
        #     gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask,
        # )
        if use_baseline_model:
            logits = model(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        else:
            logits = model(
                g, idx, gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask,
            )

        if cfg_loss_criterion == "mse":
            if do_softmax_before_mse:
                logits = F.softmax(logits, -1)
            loss = F.mse_loss(logits, y_prob)
        else:
            if loss_weight is None:
                loss = F.cross_entropy(logits, label_ids)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, num_classes), label_ids, loss_weight
                )

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        g.ndata['cls_feats'].detach_()
        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
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
    if not use_baseline_model:
        update_feature(g, model, all_dataloader, device)
    print("--------------------------------------------------------------")
    valid_loss, valid_acc, perform_metrics = evaluate(
        model, gcn_adj_list, valid_dataloader, batch_size, epoch, "Valid_set"
    )
    test_loss, _, test_f1 = evaluate(
        model, gcn_adj_list, test_dataloader, batch_size, epoch, "Test_set"
    )
    all_loss_list["train"].append(tr_loss)
    all_loss_list["valid"].append(valid_loss)
    all_loss_list["test"].append(test_loss)
    all_f1_list["valid"].append(perform_metrics)
    all_f1_list["test"].append(test_f1)
    print(
        "Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(
            epoch, tr_loss, valid_loss, (time.time() - train_start) / 60.0
        )
    )
    # Save a checkpoint
    # if valid_acc > valid_acc_prev:
    if perform_metrics > perform_metrics_prev:
        to_save = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "valid_acc": valid_acc,
            "lower_case": do_lower_case,
            "perform_metrics": perform_metrics,
        }
        torch.save(to_save, os.path.join(output_dir, model_file_4save))
        perform_metrics_prev = perform_metrics
        test_f1_when_valid_best = test_f1
        valid_f1_best_epoch = epoch

print(
    "\n**Optimization Finished!,Total spend:",
    (time.time() - train_start) / 60.0,
)
print(
    # "**Valid weighted F1: %.3f at %d epoch."
    "**Valid binary F1: %.3f at %d epoch."
    % (100 * perform_metrics_prev, valid_f1_best_epoch)
)
print(
    # "**Test weighted F1 when valid best: %.3f"
    "**Test binary F1 when valid best: %.3f"
    % (100 * test_f1_when_valid_best)
)
sys.stdout.close()
sys.stdout = sys.__stdout__
