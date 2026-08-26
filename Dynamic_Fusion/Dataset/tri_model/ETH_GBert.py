import math
import inspect
from networkx import degree_centrality
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# for huggingface transformers 0.6.2;
from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)

# ---------------------------------------------------------
# THAY ĐỔI: Cập nhật đúng 10 đặc trưng từ file CSV mới nhất
# ---------------------------------------------------------
TOP10_FEATURE_NAMES = [
    "betweenness_centrality",
    "clustering_coefficient",
    "in_degree",
    "freq_in_long",
    "out_degree",
    "freq_out_long",
    "freq_out_short",
    "max_out_amount",
    "in_degree_centrality",
    "active_days",
    # "degree_centrality",
    # "avg_out_amount",
    # "freq_in_short",
    # "out_degree_centrality",
    # "max_in_amount",
    # "lifetime_days",
    # "closeness_centrality",
    # "avg_in_amount",
    # "short_long_in_ratio",
    # "account_balance"
]
NUM_GRAPH_FEATURES = len(TOP10_FEATURE_NAMES)   # = 10


class VocabGraphConvolution(nn.Module):
    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if (
                    n.startswith("W")
                    or n.startswith("a")
                    or n in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def compute_H_vh(self, vocab_adj_list):
        """H_vh = dropout(sparse_mm(adj, W)) for every adjacency in
        vocab_adj_list -- the one genuinely batch-invariant piece of
        forward(). Adjacency and weights are frozen for an entire eval()
        pass (model.eval() + torch.no_grad(), same vocab_adj_list argument
        on every call), so this only needs to run ONCE per evaluate() call,
        not once per batch. Call this before the batch loop and pass the
        result to forward(precomputed_H_vh=...) -- see full_scale_test_eval
        _report.md: recomputing this per-batch across ~101k batches (8
        examples/batch over 811,704 accounts) was the dominant cost of the
        213-minute full-scale eval run.

        Returns a list of [vocab_size, hid_dim] tensors, one per adjacency.
        """
        H_vh_list = []
        for i in range(self.num_adj):
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")
            W_i = getattr(self, "W%d_vh" % i)
            H_vh = torch.sparse.mm(vocab_adj_list[i].float(), W_i)  # [vocab_size, hid_dim]
            H_vh_list.append(self.dropout(H_vh))
        return H_vh_list

    def forward(self, vocab_adj_list, words_embeddings, gcn_vocab_ids, add_linear_mapping_term=False, precomputed_H_vh=None):
        """Sparse gather/embedding-lookup form (replaces the dense one-hot
        "gcn_swop_eye" scatter-matmul: X_dv.matmul(H_vh) where X_dv was a
        [B, H_bert, vocab_size] tensor built from a [B, vocab_size, seqlen]
        one-hot -- infeasible at MulDiGraph's 2,973,489-account vocab
        (~40 TB/batch). Mathematically identical:

          X_dv[b,h,v] = sum_l 1[gcn_vocab_ids[b,l]==v] * words_embeddings[b,l,h]
          (X_dv @ H_vh)[b,h,k] = sum_v X_dv[b,h,v] * H_vh[v,k]
                               = sum_l words_embeddings[b,l,h] * H_vh[gcn_vocab_ids[b,l], k]

        i.e. GATHER the H_vh row addressed by each token's vocab id, then a
        small [H_bert,L] @ [L,hid_dim] contraction -- never materializes
        anything with a vocab_size-sized dimension per batch item.

        words_embeddings: [B, L, H_bert]
        gcn_vocab_ids:    [B, L] long, -1 where the token has no vocab-graph node
        precomputed_H_vh: optional list of tensors from compute_H_vh(), one
            per adjacency -- when given, skips the sparse_mm(adj, W) below
            (use this during eval; see compute_H_vh's docstring). Leave None
            during training, where W_i changes every optimizer step.
        """
        valid_mask = (gcn_vocab_ids >= 0).unsqueeze(-1).to(words_embeddings.dtype)  # [B, L, 1]
        safe_ids = gcn_vocab_ids.clamp(min=0)                                        # [B, L]

        for i in range(self.num_adj):
            if precomputed_H_vh is not None:
                H_vh = precomputed_H_vh[i]
            else:
                if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                    raise TypeError("Expected a PyTorch sparse tensor")
                W_i = getattr(self, "W%d_vh" % i)
                H_vh = torch.sparse.mm(vocab_adj_list[i].float(), W_i)  # [vocab_size, hid_dim] -- one-time, small
                H_vh = self.dropout(H_vh)

            H_vh_gathered = H_vh[safe_ids] * valid_mask             # [B, L, hid_dim]
            H_dh = torch.einsum("blh,blk->bhk", words_embeddings, H_vh_gathered)  # [B, H_bert, hid_dim]

            if add_linear_mapping_term:
                W_i = getattr(self, "W%d_vh" % i)
                W_gathered = W_i[safe_ids] * valid_mask              # [B, L, hid_dim]
                H_linear = torch.einsum("blh,blk->bhk", words_embeddings, W_gathered)
                H_linear = self.dropout(H_linear)
                H_dh = H_dh + H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out = self.fc_hc(fused_H)
        return out


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Triển khai DiffSoftmax, dùng để sử dụng nhãn mềm hoặc nhãn cứng trong huấn luyện.
    - tau: tham số nhiệt độ, kiểm soát độ mịn của đầu ra softmax
    - hard: có sử dụng nhãn cứng hay không
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class FeatureProjector(nn.Module):
    """
    Project top-10 Spearman graph features → BERT hidden space.

    Input : raw_features  [B, num_features]   (giá trị số, đã normalize)
    Output: feat_emb      [B, seq_len, hidden_size]
    """
    def __init__(self, num_features: int, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(num_features, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, raw_features: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            raw_features : [B, num_features]  – top-10 graph features (float)
            seq_len      : int                – độ dài chuỗi token (từ input_ids)
        Returns:
            feat_emb     : [B, seq_len, hidden_size]
        """
        feat_emb = self.projector(raw_features)          # [B, hidden_size]
        feat_emb = feat_emb.unsqueeze(1).expand(         # [B, 1, hidden_size]
            -1, seq_len, -1                              # → [B, seq_len, hidden_size]
        )
        return feat_emb


# class DynamicFusionLayer(nn.Module):
#     """
#     Fuse 3 embedding streams với gate động (DiffSoftmax):
#       - Stream 0: bert_embeddings          (BERT token embedding thuần)
#       - Stream 1: gcn_enhanced_embeddings  (BERT + GCN vocab graph)
#       - Stream 2: feature_embeddings       (top-10 Spearman graph features)

#     Gate network nhận concat([bert, gcn, feat]) → [B, seq_len, 3 gates]
#     """
#     def __init__(self, hidden_dim: int, tau: float = 1.0, hard_gate: bool = False):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.tau = tau
#         self.hard_gate = hard_gate

#         # Input: concat 3 streams → hidden_dim * 3
#         self.gate_network = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim),   # ← hidden_dim*3
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 3),
#         )

#         self.fusion_weight = nn.Parameter(torch.tensor(0.5))

#     def forward(
#         self,
#         bert_embeddings: torch.Tensor,           # [B, seq_len, hidden_dim]
#         gcn_enhanced_embeddings: torch.Tensor,   # [B, seq_len, hidden_dim]
#         feature_embeddings: torch.Tensor,        # [B, seq_len, hidden_dim]
#     ) -> torch.Tensor:
        
#         # 1. Gate computation
#         concat_embeddings = torch.cat(
#             [bert_embeddings, gcn_enhanced_embeddings, feature_embeddings], dim=-1
#         )                                                # [B, seq_len, hidden*3]

#         gate_logits = self.gate_network(concat_embeddings)   # [B, seq_len, 3]
#         gate_values = DiffSoftmax(
#             gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1
#         )                                                # [B, seq_len, 3]

#         # 2. Tách 3 gate scalar per token
#         gate_bert     = gate_values[:, :, 0].unsqueeze(-1)   # [B, seq_len, 1]
#         gate_gcn      = gate_values[:, :, 1].unsqueeze(-1)
#         gate_feat     = gate_values[:, :, 2].unsqueeze(-1)

#         # 3. Weighted fusion
#         fused_embeddings = (
#             gate_bert * bert_embeddings +
#             gate_gcn  * gcn_enhanced_embeddings +
#             gate_feat * feature_embeddings
#         )                                                # [B, seq_len, hidden_dim]

#         return fused_embeddings

# class DynamicFusionLayer(nn.Module):
#     def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
#         super(DynamicFusionLayer, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.tau = tau
#         self.hard_gate = hard_gate

#         # Mạng gate gốc của tác giả (chỉ nhận BERT và GCN)
#         self.gate_network = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 3),
#         )
#         self.fusion_weight = nn.Parameter(torch.tensor(0.5))

#         # Lớp học phụ để gộp Features tĩnh vào
#         self.feature_gate = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Sigmoid() # Trả về giá trị từ 0 -> 1
#         )

#     def forward(self, bert_embeddings, gcn_enhanced_embeddings, feature_embeddings):
#         # ---------- BƯỚC 1: CODE GỐC CỦA TÁC GIẢ ----------
#         concat_bg = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)
#         gate_logits = self.gate_network(concat_bg)
#         gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

#         gate_bert = gate_values[:, :, 0].unsqueeze(-1)
#         gate_gcn  = gate_values[:, :, 1].unsqueeze(-1)
#         gate_mixed = gate_values[:, :, 2].unsqueeze(-1)

#         embeddings_mixed = self.fusion_weight * bert_embeddings + (1 - self.fusion_weight) * gcn_enhanced_embeddings

#         original_fused = (
#             gate_bert * bert_embeddings +
#             gate_gcn * gcn_enhanced_embeddings +
#             gate_mixed * embeddings_mixed
#         )

#         # ---------- BƯỚC 2: CỘNG HƯỞNG FEATURE ----------
#         # Mạng gate sẽ xem xét original_fused và feature_embeddings để quyết định
#         # nên cộng bao nhiêu % tính năng đồ thị vào kết quả cuối cùng.
#         concat_all = torch.cat([original_fused, feature_embeddings], dim=-1)
#         alpha = self.feature_gate(concat_all) # alpha có shape [B, seq_len, hidden_dim]

#         # Kết hợp mềm
#         final_fused = (1 - alpha) * original_fused + alpha * feature_embeddings

#         return final_fused

class DynamicFusionLayer(nn.Module):
    def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
        super(DynamicFusionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4), # 4 cổng
        )

        # 3 trọng số có thể học để pha trộn luồng thứ 4
        self.fusion_weights = nn.Parameter(torch.ones(3)) 

    def forward(self, bert_embeddings, gcn_enhanced_embeddings, feature_embeddings):
        concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings, feature_embeddings], dim=-1)

        gate_logits = self.gate_network(concat_embeddings)
        gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

        # Tách 4 cổng
        gate_bert  = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn   = gate_values[:, :, 1].unsqueeze(-1)
        gate_feat  = gate_values[:, :, 2].unsqueeze(-1)
        gate_mixed = gate_values[:, :, 3].unsqueeze(-1)

        # Tạo luồng mixed bằng cách softmax trọng số (để tổng = 1)
        w = torch.softmax(self.fusion_weights, dim=0)
        embeddings_mixed = (
            w[0] * bert_embeddings + 
            w[1] * gcn_enhanced_embeddings + 
            w[2] * feature_embeddings
        )

        fused_embeddings = (
            gate_bert * bert_embeddings +
            gate_gcn * gcn_enhanced_embeddings +
            gate_feat * feature_embeddings +
            gate_mixed * embeddings_mixed
        )

        return fused_embeddings


class ETH_GBertEmbeddings(BertEmbeddings):
    def __init__(
        self,
        config,
        gcn_adj_dim: int,
        gcn_adj_num: int,
        gcn_embedding_dim: int,
        num_graph_features: int = NUM_GRAPH_FEATURES,
    ):
        super().__init__(config)
        assert gcn_embedding_dim >= 0

        self.gcn_embedding_dim = gcn_embedding_dim

        # GCN vocab graph
        self.vocab_gcn = VocabGraphConvolution(
            gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim
        )

        # FeatureProjector: top-10 Spearman features → hidden_size
        self.feature_projector = FeatureProjector(
            num_features=num_graph_features,
            hidden_size=config.hidden_size,
        )

        # DynamicFusionLayer
        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(
        self,
        vocab_adj_list,
        gcn_vocab_ids,
        input_ids,
        graph_features,              # [B, num_graph_features]
        token_type_ids=None,
        attention_mask=None,
        precomputed_H_vh=None,
    ):
        # BERT word embeddings
        words_embeddings = self.word_embeddings(input_ids)   # [B, seq_len, hidden]

        # GCN-enhanced embeddings (sparse gather -- see VocabGraphConvolution.forward)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, words_embeddings, gcn_vocab_ids, precomputed_H_vh=precomputed_H_vh)

        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (
                attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
            ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        # Graph feature embeddings 
        seq_len = input_ids.size(1)
        feature_embeddings = self.feature_projector(graph_features, seq_len)

        # Dynamic fusion (3 streams) 
        new_words_embeddings = self.dynamic_fusion_layer(
            bert_embeddings=words_embeddings,
            gcn_enhanced_embeddings=gcn_words_embeddings,
            feature_embeddings=feature_embeddings,
        )                                                    # [B, seq_len, hidden]

        # Position + token-type embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = new_words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ETH_GBertModel(BertModel):
    def __init__(
        self,
        config,
        gcn_adj_dim: int,
        gcn_adj_num: int,
        gcn_embedding_dim: int,
        num_labels: int,
        num_graph_features: int = NUM_GRAPH_FEATURES,
        output_attentions: bool = False,
        keep_multihead_output: bool = False,
    ):
        super().__init__(config)
        self.embeddings = ETH_GBertEmbeddings(
            config,
            gcn_adj_dim,
            gcn_adj_num,
            gcn_embedding_dim,
            num_graph_features=num_graph_features,
        )
        self.encoder  = BertEncoder(config)
        self.pooler   = BertPooler(config)
        self.num_labels = num_labels
        self.dropout  = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.output_attentions    = getattr(config, 'output_attentions', False)
        self.keep_multihead_output = getattr(config, 'keep_multihead_output', False)
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.apply(self.init_bert_weights)

    def compute_gcn_H_vh(self, vocab_adj_list):
        """See VocabGraphConvolution.compute_H_vh -- call once before an
        eval batch loop (model.eval() + torch.no_grad(), adjacency/weights
        frozen for the whole pass) and pass the result to forward(
        precomputed_H_vh=...) for every batch, instead of letting each
        forward() call recompute the same sparse_mm(adj, W)."""
        return self.embeddings.vocab_gcn.compute_H_vh(vocab_adj_list)

    def forward(
        self,
        vocab_adj_list,
        gcn_vocab_ids,
        input_ids,
        graph_features,                  # [B, num_graph_features]
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
        precomputed_H_vh=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Embedding layer (BERT + GCN + Graph features)
        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_vocab_ids,
            input_ids,
            graph_features,              # ← Truyền feature vào embedding
            token_type_ids,
            attention_mask,
            precomputed_H_vh=precomputed_H_vh,
        )

        # Extended attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand_as(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Encoder
        encoder_args = {}
        if 'head_mask' in inspect.signature(self.encoder.forward).parameters:
            encoder_args['head_mask'] = head_mask

        if self.output_attentions:
            output_all_encoded_layers = True

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args,
        )

        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        # Pooler → classifier
        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits