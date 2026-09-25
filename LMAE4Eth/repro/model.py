"""
From-scratch reimplementation of LMAE4Eth's three components, following the
equations in the paper (Sections III-IV):

  - TxCLM   : Eq. 1-6   (transaction-token contrastive language model)
  - MAGAE   : Eq. 7-16  (masked account graph autoencoder w/ LABOR sampling)
  - CAFN    : Eq. 17-21, Algorithm 1 (cross-attention fusion network)

Written in plain PyTorch (no dgl -- dgl ships no wheel for the Python 3.14
runtime in this environment, see conversation notes). The GAT/GCN backbones
in model/gat.py, model/gcn.py depend on dgl and are therefore not reused;
`SimpleGAT` below is a compact from-scratch replacement with the same role
(single-hop graph attention over a sampled neighborhood).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# LABOR sampling (Eq. 7-9)
# ----------------------------------------------------------------------------
def labor_sample(adj_csr, batch_nodes, fanout, rng, shared_keys=None):
    """One layer of LABOR (layer-neighbor) sampling for a mini-batch.

    Eq. 8: N_LABOR(v) = {u in N(v) | r_u <= c_v * pi_u}, with r_u a *shared*
    per-neighbor random variable (not redrawn per v) -- this is what lets
    overlapping neighborhoods reuse the same sampled nodes across the batch
    and shrink the sampled-edge budget (the paper's whole point of LABOR vs.
    plain node-wise neighbor sampling).

    We realize this with weighted sampling-without-replacement via random
    keys: key_u = pi_u / r_u, take the top-`fanout` neighbors by key. This is
    a standard exponential/uniform-key weighted-sampling construction and is
    rank-equivalent to the inclusion rule above, while being easy to
    vectorize and to share `r_u` (via `shared_keys`) across an entire batch.
    """
    n = adj_csr.shape[0]
    if shared_keys is None:
        r = rng.random(n).astype(np.float32)
        r = np.clip(r, 1e-6, 1.0)
    else:
        r = shared_keys

    sampled = {}
    neighbor_union = set(batch_nodes.tolist())
    indptr, indices, data = adj_csr.indptr, adj_csr.indices, adj_csr.data
    for v in batch_nodes:
        s, e = indptr[v], indptr[v + 1]
        neigh = indices[s:e]
        if len(neigh) == 0:
            sampled[v] = neigh
            continue
        w = data[s:e].astype(np.float32)
        pi = w / w.sum()
        key = pi / r[neigh]
        if len(neigh) > fanout:
            top = np.argpartition(-key, fanout - 1)[:fanout]
            neigh = neigh[top]
        sampled[v] = neigh
        neighbor_union.update(neigh.tolist())
    return sampled, r


def build_padded_block(sampled, batch_nodes, fanout):
    """Pack a dict{v: neighbor_ids} into a padded [B, fanout+1] index tensor
    (slot 0 = self-loop) plus a boolean mask, for vectorized attention."""
    B = len(batch_nodes)
    idx = np.zeros((B, fanout + 1), dtype=np.int64)
    mask = np.zeros((B, fanout + 1), dtype=bool)
    for i, v in enumerate(batch_nodes):
        idx[i, 0] = v
        mask[i, 0] = True
        neigh = sampled[v]
        k = min(len(neigh), fanout)
        idx[i, 1:1 + k] = neigh[:k]
        mask[i, 1:1 + k] = True
    return torch.from_numpy(idx), torch.from_numpy(mask)


class SimpleGAT(nn.Module):
    """Single-hop multi-head graph attention: aggregates a padded, sampled
    neighborhood (self-loop + LABOR-sampled neighbors) into a new embedding
    for each center node. Stands in for the dgl-based GAT/GCN backbones."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.dh = max(4, math.ceil(out_dim / heads))
        internal_dim = self.heads * self.dh
        self.W = nn.Linear(in_dim, internal_dim, bias=False)
        self.attn_l = nn.Parameter(torch.empty(heads, self.dh))
        self.attn_r = nn.Parameter(torch.empty(heads, self.dh))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(internal_dim, out_dim)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)

    def forward(self, feats, block_idx, block_mask):
        # feats: [N_all, in_dim]; block_idx/mask: [B, S] (S = fanout+1, slot0=self)
        B, S = block_idx.shape
        x = self.W(feats[block_idx])                       # [B, S, internal_dim]
        x = x.view(B, S, self.heads, self.dh)               # [B, S, H, dh]
        center = x[:, 0:1]                                  # [B, 1, H, dh]

        el = (center * self.attn_l).sum(-1)                 # [B, 1, H]
        er = (x * self.attn_r).sum(-1)                       # [B, S, H]
        e = self.leaky(el + er)                              # [B, S, H]
        e = e.masked_fill(~block_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)
        alpha = self.drop(alpha)

        out = (alpha.unsqueeze(-1) * x).sum(1)                # [B, H, dh]
        return self.out_proj(out.reshape(B, self.heads * self.dh))


# ----------------------------------------------------------------------------
# MAGAE (Eq. 10-16)
# ----------------------------------------------------------------------------
class MAGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4, mask_rate=0.5, alpha=3.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_rate = mask_rate
        self.alpha = alpha  # SCE scaling factor gamma, Eq. 16

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.encoder = SimpleGAT(in_dim, hidden_dim, heads=heads)
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = SimpleGAT(hidden_dim, in_dim, heads=heads)
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)

    @staticmethod
    def sce_loss(x, y, alpha):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return ((1 - (x * y).sum(-1)).pow(alpha)).mean()

    def forward(self, feats, block_idx, block_mask, mask_bool):
        """feats: full [N_all, in_dim] table (already gathered for this batch's
        node universe); block_idx/mask describe the LABOR-sampled 1-hop block
        per target node (center = slot 0); mask_bool: [N_all] True at nodes
        whose *input* feature should be replaced with [MASK] (Eq. 10)."""
        x_in = feats.clone()
        x_in[mask_bool] = 0.0
        x_in[mask_bool] += self.enc_mask_token

        h = self.encoder(x_in, block_idx, block_mask)         # Eq. 11
        h_proj = self.enc_to_dec(h)

        h_dec_in = h_proj.clone()
        h_dec_in[mask_bool[block_idx[:, 0]]] = 0.0
        h_dec_in[mask_bool[block_idx[:, 0]]] += self.dec_mask_token  # Eq. 13 (remask)

        # decode: reuse the same block structure, but node features are now h_dec_in per center
        full_h = torch.zeros(feats.size(0), self.hidden_dim, device=feats.device)
        full_h[block_idx[:, 0]] = h_dec_in
        recon = self.decoder(full_h, block_idx, block_mask)   # Eq. 14

        centers = block_idx[:, 0]
        target_mask = mask_bool[centers]
        x_orig = feats[centers][target_mask]
        x_rec = recon[target_mask]
        loss = self.sce_loss(x_orig, x_rec, self.alpha)        # Eq. 16
        return loss

    @torch.no_grad()
    def embed(self, feats, block_idx, block_mask):
        return self.encoder(feats, block_idx, block_mask)


# ----------------------------------------------------------------------------
# TxCLM (Eq. 1-6)
# ----------------------------------------------------------------------------
class TinyBert(nn.Module):
    """Compact BERT-style encoder (token+position embeddings + Transformer
    encoder stack), standing in for the pretrained BERT checkpoint the paper
    initializes TxCLM's anchor/enhanced models from -- there is no meaningful
    pretrained-English-BERT prior for these discretized transaction tokens,
    so both models here start from the *same* random init instead (still
    satisfying "anchor and enhanced initialized with the same weights")."""

    def __init__(self, vocab_size, hidden=128, layers=4, heads=4, max_len=128, ff=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=ff,
                                            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.hidden = hidden

    def forward(self, token_ids, attn_mask):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        x = self.drop(self.ln(x))
        pad_mask = ~attn_mask  # True = ignore, for TransformerEncoder's src_key_padding_mask
        return self.encoder(x, src_key_padding_mask=pad_mask)


class TxCLM(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=4, heads=4, max_len=128, tau=0.1):
        super().__init__()
        self.anchor = TinyBert(vocab_size, hidden, layers, heads, max_len)
        self.enhanced = TinyBert(vocab_size, hidden, layers, heads, max_len)
        self.enhanced.load_state_dict(self.anchor.state_dict())
        for p in self.anchor.parameters():
            p.requires_grad = False
        self.mlm_head = nn.Linear(hidden, vocab_size)
        self.tau = tau
        self.hidden = hidden

    def mask_tokens(self, token_ids, attn_mask, mask_id, mask_rate=0.3):
        maskable = attn_mask.clone()
        maskable[:, 0] = False  # never mask [CLS]
        probs = torch.rand(token_ids.shape, device=token_ids.device)
        mask_bool = maskable & (probs < mask_rate)
        masked_input = token_ids.clone()
        masked_input[mask_bool] = mask_id
        return masked_input, mask_bool

    def forward(self, token_ids, attn_mask, mask_id):
        masked_input, mask_bool = self.mask_tokens(token_ids, attn_mask, mask_id)

        h_tilde = self.enhanced(masked_input, attn_mask)      # [B, L, H]
        with torch.no_grad():
            h = self.anchor(token_ids, attn_mask)

        # ---- MLM loss (Eq. 5) ----
        logits = self.mlm_head(h_tilde)
        mlm_loss = F.cross_entropy(logits[mask_bool], token_ids[mask_bool]) if mask_bool.any() \
            else torch.tensor(0.0, device=token_ids.device)

        # ---- token-aware contrastive loss (Eq. 4), per-sequence softmax over positions ----
        hn = F.normalize(h, dim=-1)
        htn = F.normalize(h_tilde, dim=-1)
        sim = torch.bmm(htn, hn.transpose(1, 2)) / self.tau    # [B, L, L]
        sim = sim.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        logp = F.log_softmax(sim, dim=-1)
        target = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0).expand_as(mask_bool)
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, L]
        nll = torch.where(mask_bool, nll, torch.zeros_like(nll))  # avoid 0*inf=nan on padded/unmasked rows
        ta_loss = nll.sum() / mask_bool.sum().clamp(min=1)

        return mlm_loss + ta_loss, mlm_loss.detach(), ta_loss.detach()

    @torch.no_grad()
    def embed(self, token_ids, attn_mask):
        return self.enhanced(token_ids, attn_mask)  # [B, L, H], full sequence (Eq. 17's "s")


# ----------------------------------------------------------------------------
# CAFN (Eq. 17-21, Algorithm 1)
# ----------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_out, heads=4):
        super().__init__()
        assert d_out % heads == 0
        self.heads = heads
        self.dh = d_out // heads
        self.Wq = nn.Linear(d_q, d_out, bias=False)
        self.Wk = nn.Linear(d_kv, d_out, bias=False)
        self.Wv = nn.Linear(d_kv, d_out, bias=False)

    def forward(self, q_tokens, kv, kv_mask=None):
        # q_tokens: [B, kq, d_q] (broadcast learnable tokens) ; kv: [B, Nkv, d_kv]
        B = kv.size(0)
        q = self.Wq(q_tokens).expand(B, -1, -1) if q_tokens.dim() == 2 else self.Wq(q_tokens)
        k = self.Wk(kv)
        v = self.Wv(kv)
        q = q.view(B, -1, self.heads, self.dh).transpose(1, 2)
        k = k.view(B, -1, self.heads, self.dh).transpose(1, 2)
        v = v.view(B, -1, self.heads, self.dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, kq, dh]
        out = out.transpose(1, 2).reshape(B, -1, self.heads * self.dh)
        return out


class CAFN(nn.Module):
    """Algorithm 1: semantic aggregation -> cross-perspective fusion ->
    fusion-token cross-attention -> classification MLP."""

    def __init__(self, d_lm, d_g, d_s=64, d_f=64, k_s=4, k_f=4, n_classes=2):
        super().__init__()
        self.agg_tokens = nn.Parameter(torch.randn(k_s, d_s) * 0.02)   # A^s
        self.fusion_tokens = nn.Parameter(torch.randn(k_f, d_f) * 0.02)  # A^f

        self.semantic_attn = CrossAttention(d_s, d_lm, d_s, heads=4)   # Eq. 17
        self.Ws = nn.Linear(d_s, d_f)
        self.Wg = nn.Linear(d_g, d_f)
        self.fusion_attn = CrossAttention(d_f, d_f, d_f, heads=4)      # Eq. 20

        self.classifier = nn.Sequential(
            nn.Linear(k_f * d_f, d_f), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_f, n_classes),
        )

    def forward(self, semantic_seq, semantic_mask, interaction_emb):
        # semantic_seq: [B, N, d_lm], interaction_emb: [B, d_g]
        Zs = self.semantic_attn(self.agg_tokens, semantic_seq, semantic_mask)  # [B, k_s, d_s]  Eq. 17
        Zs_pool = Zs.mean(dim=1)                                                # [B, d_s]

        Zsg = torch.sigmoid(self.Ws(Zs_pool) + self.Wg(interaction_emb))        # [B, d_f]  Eq. 18
        Zsg = Zsg.unsqueeze(1).expand(-1, Zs.size(1), -1)                       # broadcast per agg token -> [B, k_s, d_f]

        F_out = self.fusion_attn(self.fusion_tokens, Zsg)                       # [B, k_f, d_f]  Eq. 20
        logits = self.classifier(F_out.reshape(F_out.size(0), -1))
        return logits
