import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class EGNNLayer(nn.Module):
    def __init__(self, in_dim, p, out_dim):
        super(EGNNLayer, self).__init__()
        
        self.in_dim = in_dim
        self.p = p
        self.linear = nn.Linear(p * in_dim, out_dim)

    def forward(self, g, features, edge_inputs):
        g.edata['eft'] = edge_inputs
        g.ndata['h'] = features

        def message_func(edges):

            e = edges.data['eft']
            h = edges.src['h']

            h = e.unsqueeze(-1) * h.unsqueeze(-2).repeat(1, self.p, 1)
            h = h.reshape(-1, self.p * self.in_dim)

            h = self.linear(h)
            return {'m' : h}   

        g.update_all(message_func, gcn_reduce)
        h = g.ndata['h']

        return h



class EGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim, num_hops):
        super(EGNN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(EGNNLayer(node_in_dim, edge_in_dim, hidden_dim))

        for i in range(1, num_hops):
            self.layers.append(EGNNLayer(hidden_dim, edge_in_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, g, features):

        e = g.edata['edge_features']
        h = features

        for gcn_layer in self.layers:
            h = F.relu(gcn_layer(g, h, e))

        logits = self.fc(h)
        return logits
