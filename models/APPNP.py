"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import torch.nn as nn
from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 num_hops,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        # self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens))
        # hidden layers
        for i in range(1, num_hops):
            self.layers.append(nn.Linear(hiddens, hiddens))
        # output layer
        self.layers.append(nn.Linear(hiddens, n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        if edge_drop:
            self.propagate = APPNPConv(k, alpha, edge_drop)
        else:
            self.propagate = APPNPConv(k, alpha)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h
