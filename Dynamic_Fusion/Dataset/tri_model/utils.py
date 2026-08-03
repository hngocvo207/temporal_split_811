import re
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from nltk.tokenize import TweetTokenizer
from torch.utils import data
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

"""
General functions
"""

def del_http_user_tokenize(tweet):
    space_pattern = r"\s+"
    url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"
    tweet = re.sub(space_pattern, " ", tweet)
    tweet = re.sub(url_regex, "", tweet)
    tweet = re.sub(mention_regex, "", tweet)
    return tweet


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_tweet_tokenize(string):
    tknzr = TweetTokenizer(
        reduce_len=True, preserve_case=False, strip_handles=False
    )
    tokens = tknzr.tokenize(string.lower())
    return " ".join(tokens).strip()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse):
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        weight.append(len(y) / (n_classes * count))
    return classes_count, weight


"""
Functions and Classes for read and organize data set
"""

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None, sample_weight=1.0):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.confidence = confidence
        self.label = label
        # Per-example loss weight (new_split.md §C.5): 1.0 for ground-truth
        # / benign examples, < 1.0 (e.g. 0.5) for soft propagated_1hop
        # labels so they never count as confidently as real ground truth.
        self.sample_weight = sample_weight


class InputFeatures(object):
    def __init__(
        self,
        guid,
        tokens,
        input_ids,
        gcn_vocab_ids,
        input_mask,
        segment_ids,
        confidence,
        label_id,
        sample_weight=1.0,
    ):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence = confidence
        self.label_id = label_id
        self.sample_weight = sample_weight


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def example2feature(example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim):
    tokens_a = example.text_a.split()
    assert example.text_b is None
    if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
        tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]

    gcn_vocab_ids = []
    for word in tokens_a:
        if word in gcn_vocab_map:
            gcn_vocab_ids.append(gcn_vocab_map[word])
        else:
            gcn_vocab_ids.append(gcn_vocab_map.get("UNK", -1))

    tokens = ["[CLS]"] + tokens_a + ["[SEP]" for _ in range(gcn_embedding_dim + 1)]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
        sample_weight=getattr(example, "sample_weight", 1.0),
    )
    return feat


class CorpusDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        address_to_index,            
        max_seq_len,
        gcn_embedding_dim,
        graph_features_lookup=None,  
        zero_features=None,          
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = address_to_index          
        
        # SỬA LỖI: Tạo từ điển ánh xạ ngược (Index -> Chuỗi địa chỉ 0x...)
        self.index_to_address = {v: str(k).lower() for k, v in address_to_index.items()}

        self.graph_features_lookup = graph_features_lookup or {}
        self.zero_features = (
            zero_features if zero_features is not None
            else torch.zeros(10)
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        feat = example2feature(
            ex,
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )

        # SỬA LỖI: Ánh xạ guid (số nguyên) thành địa chỉ ví (0x...) để Lookup
        node_idx = int(ex.guid)
        node_addr = self.index_to_address.get(node_idx, "")
        
        graph_feat = self.graph_features_lookup.get(node_addr, self.zero_features)

        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
            graph_feat,              # ← tensor [num_graph_features]
            feat.sample_weight,      # ← per-example loss weight (soft propagated labels)
        )

    def pad(self, batch):
        # Bóc 2 phần tử cuối (graph_features, sample_weight) ra trước
        *others, graph_features, sample_weights = zip(*batch)
        batch_core = list(zip(*others))

        seqlen_list = [len(sample[0]) for sample in batch_core]
        maxlen = np.array(seqlen_list).max()

        f_collect = lambda x: [sample[x] for sample in batch_core]
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch_core
        ]
        f_pad2 = lambda x, seqlen: [
            [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
            for sample in batch_core
        ]

        batch_input_ids      = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask     = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids    = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences    = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids      = torch.tensor(f_collect(4), dtype=torch.long)

        # Sparse gather/embedding-lookup representation (replaces the old
        # dense one-hot "gcn_swop_eye" [B, vocab_size, seqlen] matrix, which
        # required an O(batch * vocab_size * seqlen) tensor -- ~40 TB/batch
        # at MulDiGraph's 2,973,489-account vocab, i.e. infeasible past a
        # tiny subsampled vocab. -1 marks "no vocab-graph node at this
        # position" (padding / non-address token); ETH_GBert.py's
        # VocabGraphConvolution gathers the relevant [vocab_size, hid_dim]
        # row per token instead of scattering every token into a dense
        # per-vocab-id slot -- mathematically identical, O(batch*seqlen)
        # memory instead of O(batch*vocab_size*seqlen).
        batch_gcn_vocab_ids = torch.tensor(f_pad2(5, maxlen), dtype=torch.long)

        # Stack graph_features: list of tensor[10] → tensor[B, 10]
        batch_graph_features = torch.stack(list(graph_features))
        batch_sample_weights = torch.tensor(sample_weights, dtype=torch.float)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_vocab_ids,     # ← was batch_gcn_swop_eye (dense one-hot); now a sparse index tensor
            batch_graph_features,    # ← Đảm bảo trả về đủ 7 phần tử
            batch_sample_weights,    # ← per-example loss weight (soft propagated labels)
        )