import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)



class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hops):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hidden_dim))

        for i in range(1, num_hops):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, g, features):

        x = features
        for gcn_layer in self.layers:
            x = F.relu(gcn_layer(g, x))

        logits = self.fc(x)
        return logits
