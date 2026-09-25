"""GraphSAGE using DGL nn package
References
----------
Paper: https://arxiv.org/abs/1706.02216
Author's code: https://github.com/williamleif/GraphSAGE
"""

import argparse
import dgl
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, agg_type, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type=agg_type, activation=activation))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type=agg_type, activation=activation))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type=agg_type, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, g ,edge_weight):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
