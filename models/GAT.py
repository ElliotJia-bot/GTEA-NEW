import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn


class NodeUpdate(nn.Module):
    def __init__(self, in_dim, out_dim, test=False):
        super(NodeUpdate, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.test = test

        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, node):
        h = node.data['h']
        self_h = node.data['self_h']

        # print('hhhhhhhhhhhhh self', self_h.shape)

        if self.test:
            h = (h - self_h)
        else:
            h = (h - self_h)

        h = torch.cat((self_h, h), dim=1)
        h = F.relu(self.layer(h))
        return {'activation': h}

class GATTrain(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, num_layers, dropout=None):
        super(GATTrain, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(NodeUpdate(2 * hidden_dim, hidden_dim))


        self.feat_layers = nn.ModuleList()
        self.feat_layers.append(nn.Linear(in_dim, hidden_dim))

        self.attn_layers = nn.ModuleList()
        self.attn_layers.append(nn.Linear(2 * hidden_dim, 1, bias=False))


        for i in range(1, self.num_layers):
            self.gcn_layers.append(NodeUpdate(2*hidden_dim, hidden_dim))
            self.feat_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.attn_layers.append(nn.Linear(2 * hidden_dim, 1, bias=False))

        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (feat_layer, attn_layer, gcn_layer) in enumerate(zip(self.feat_layers, self.attn_layers, self.gcn_layers)):
            h = nf.layers[i].data.pop('activation')

            h = feat_layer(h)
            h = self.dropout_layer(h)
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]


            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h


            def message_func(edges):
                # print(edges.dst.keys(), edges.src.keys())
                h = edges.src['h']

                e = F.leaky_relu(attn_layer(torch.cat((h, edges.dst['self_h']), dim=1)))                

                return {'m': h, 'e': e}

            def reduce_func(nodes):
                m = nodes.mailbox['m']
                e = nodes.mailbox['e']

                alpha = F.softmax(e, dim=1)

                h = alpha * m

                h = torch.sum(h, dim=1)
                return {'h': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func,
                            gcn_layer)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h

class GATInfer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, num_layers, dropout=None):
        super(GATInfer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(NodeUpdate(2 * hidden_dim, hidden_dim, test=True))


        self.feat_layers = nn.ModuleList()
        self.feat_layers.append(nn.Linear(in_dim, hidden_dim))

        self.attn_layers = nn.ModuleList()
        self.attn_layers.append(nn.Linear(2 * hidden_dim, 1, bias=False))


        for i in range(1, self.num_layers):
            self.gcn_layers.append(NodeUpdate(2*hidden_dim, hidden_dim, test=True))
            self.feat_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.attn_layers.append(nn.Linear(2 * hidden_dim, 1, bias=False))

        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (feat_layer, attn_layer, gcn_layer) in enumerate(zip(self.feat_layers, self.attn_layers, self.gcn_layers)):
            h = nf.layers[i].data.pop('activation')

            h = feat_layer(h)
            h = self.dropout_layer(h)
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]


            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h


            def message_func(edges):
                # print(edges.dst.keys(), edges.src.keys())
                h = edges.src['h']

                e = F.leaky_relu(attn_layer(torch.cat((h, edges.dst['self_h']), dim=1)))                

                return {'m': h, 'e': e}

            def reduce_func(nodes):
                m = nodes.mailbox['m']
                e = nodes.mailbox['e']

                alpha = F.softmax(e, dim=1)

                h = alpha * m

                h = torch.sum(h, dim=1)
                return {'h': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func,
                            gcn_layer)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h