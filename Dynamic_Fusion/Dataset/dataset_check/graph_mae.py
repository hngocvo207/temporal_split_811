"""
Pretrain module khong-nhan cho GraphSAGEEncoder (B1) -- bo sung theo huong da
chon khi so sanh LMAE4Eth (MAGAE: masked graph autoencoder, pretrain tren
TOAN BO node do thi, khong can nhan) voi train_e2_v2.py (GraphSAGEEncoder
khoi tao ngau nhien, chi hoc duoc tu 519 mau duong xac thuc). Xem
train_eval/pretrain_graph_encoder.py cho ly do cu the (STATUS.md muc "Thi
nghiem toi gian": tune capacity nhanh graph rieng le khong nang duoc tran
AUPRC ~0.01-0.02, ket luan la thieu tin hieu chu khong phai thieu capacity).

Cong thuc mask + remask + SCE loss lay dung y tuong GraphMAE (Hou et al.,
2022) ma LMAE4Eth cung dung (xem loss_func.py:sce_loss va
model/edcoder.py:encoding_mask_noise/mask_attr_prediction cua LMAE4Eth) --
KHONG copy code (khac thu vien, DGL+GAT o LMAE4Eth vs PyG+GraphSAGE o day, xem
A3_library_decision.md), va CO CHU DICH don gian hoa: bo phan BYOL/EMA-teacher
+ multi-remask cua PreModel goc (LMAE4Eth ho tro ca loss_fn='sce' LAN 'byol';
o day chi lam nhanh 'sce' -- don gian hon, it tham so hon, phu hop quy mo
pretrain 1 lan cho encoder nho (2 lop, ~128 chieu) chu khong phai mot kien
truc can nghien cuu rieng).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gnn_encoder import GraphSAGEEncoder


def sce_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    """Scaled cosine error -- cung cong thuc loss_func.py:sce_loss cua
    LMAE4Eth (Hou et al., GraphMAE): (1 - cos_sim)^alpha, on dinh hon MSE voi
    dac trung da chuan hoa/heterogenous scale (23-dim cua A1)."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow(alpha).mean()


class GraphMAE(nn.Module):
    """Boc GraphSAGEEncoder thanh 1 masked-feature-autoencoder khong-nhan:
    mask ngau nhien mot phan node CUA CA SUBGRAPH duoc truyen vao (khong
    rieng seed -- dung y encoding_mask_noise cua LMAE4Eth, moi node trong
    subgraph deu co co hoi bi mask va duoc tai thiet lap trong cung 1 luot
    forward), encode, remask h tai dung vi tri masked bang 1 token hoc duoc
    truoc khi decode -- BAT BUOC remask, neu khong decoder co the "gian lan"
    bang cach copy lai h da encode (WeightedSAGEConv co duong self lin_self
    cong thang vao h, con mang du residual cua x goc) thay vi thuc su hoc tu
    ngu canh hang xom.

    self.encoder chinh la GraphSAGEEncoder dung de finetune sau (xem
    train_eval/train_e2_v3.py, load state_dict cua encoder nay de warm-start
    thay vi khoi tao ngau nhien) -- kien truc/sieu tham so PHAI khop voi
    build_models() trong train_eval/train_e2.py (in_channels=23, out=D_GRAPH)
    de checkpoint nap thang vao duoc."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, mask_rate: float = 0.5, alpha: float = 3.0):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
        # decoder 1 lop (khong phai GAT nhieu lop + remasking lap lai nhu
        # PreModel goc) -- GraphMAE gi nhan decoder nhe van du, xem docstring
        # module o tren ve pham vi don gian hoa co chu dich.
        self.decoder = GraphSAGEEncoder(out_channels, hidden_channels, in_channels, num_layers=1, dropout=dropout)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, out_channels))
        self.mask_rate = mask_rate
        self.alpha = alpha

    def _sample_mask_idx(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        num_mask = max(1, int(self.mask_rate * num_nodes))
        perm = torch.randperm(num_nodes, device=device)
        return perm[:num_mask]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Tra ve truc tiep SCE reconstruction loss (khong phai h_graph) --
        module nay chi dung de pretrain, khong dung trong forward cua
        FraudClassifier (xem model/fraud_model.py, h_graph van tinh rieng
        boi GraphSAGEEncoder thuan qua encoder.forward()/full_graph_forward())."""
        mask_idx = self._sample_mask_idx(x.size(0), x.device)

        x_masked = x.clone()
        x_masked[mask_idx] = self.enc_mask_token

        h = self.encoder(x_masked, edge_index, edge_weight)

        h_remasked = h.clone()
        h_remasked[mask_idx] = self.dec_mask_token
        x_rec = self.decoder(h_remasked, edge_index, edge_weight)

        return sce_loss(x_rec[mask_idx], x[mask_idx], alpha=self.alpha)
