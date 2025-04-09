import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
# gcn_reduce = fn.mean(msg='m', out='activation')


class MiniBatchEGNNTrain(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_hidden_dim, num_class, num_layers, device, dropout=None):
        super(MiniBatchEGNNTrain, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(nn.Linear(edge_in_dim * node_in_dim, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.node_layers.append(nn.Linear(edge_in_dim * node_hidden_dim, node_hidden_dim))


        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (node_layer) in enumerate(self.node_layers):

            e = nf.blocks[i].data['edge_features']
            
            e = self.dropout_layer(e)
            nf.blocks[i].data['e'] = e

            h = nf.layers[i].data.pop('activation')
            h = self.dropout_layer(h)
            

            '''
                Get the last layer's node feature for each target node.
                Remove the influence of the self loop by substract self_h_tmp.
                See detail in the doc of DGL.
            '''
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]   

                  
            # tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            # self_h_tmp = torch.cat((self_h, tmp), dim=1)
            # self_h_tmp = edge_out_layer(self_h_tmp)
            nf.layers[i].data['h'] = h
            # nf.layers[i+1].data['self_h'] = self_h
            # nf.layers[i+1].data['self_h_tmp'] = self_h_tmp

            
            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    

                h = e.unsqueeze(-1) * h.unsqueeze(-2).repeat(1, self.edge_in_dim, 1)
                if i == 0:
                    h = h.reshape(-1, self.edge_in_dim * self.node_in_dim)
                else:
                    h = h.reshape(-1, self.edge_in_dim * self.node_hidden_dim)

                h = node_layer(h)
                h = F.relu(h)
                return {'m' : h}

            def reduce_func(nodes):
                m = nodes.mailbox['m'] 

                h = torch.mean(m, 1)
                h = F.relu(h)

                return {'activation':h}

            nf.block_compute(i,
                            message_func,
                            reduce_func)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h


class MiniBatchEGNNInfer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_hidden_dim, num_class, num_layers, device, dropout=None):
        super(MiniBatchEGNNInfer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(nn.Linear(edge_in_dim * node_in_dim, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.node_layers.append(nn.Linear(edge_in_dim * node_hidden_dim, node_hidden_dim))


        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (node_layer) in enumerate(self.node_layers):

            e = nf.blocks[i].data['edge_features']
            
            e = self.dropout_layer(e)
            nf.blocks[i].data['e'] = e

            h = nf.layers[i].data.pop('activation')
            h = self.dropout_layer(h)
            

            '''
                Get the last layer's node feature for each target node.
                Remove the influence of the self loop by substract self_h_tmp.
                See detail in the doc of DGL.
            '''
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]   

                  
            # tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            # self_h_tmp = torch.cat((self_h, tmp), dim=1)
            # self_h_tmp = edge_out_layer(self_h_tmp)
            nf.layers[i].data['h'] = h
            # nf.layers[i+1].data['self_h'] = self_h
            # nf.layers[i+1].data['self_h_tmp'] = self_h_tmp

            
            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    

                h = e.unsqueeze(-1) * h.unsqueeze(-2).repeat(1, self.edge_in_dim, 1)
                if i == 0:
                    h = h.reshape(-1, self.edge_in_dim * self.node_in_dim)
                else:
                    h = h.reshape(-1, self.edge_in_dim * self.node_hidden_dim)

                h = node_layer(h)
                h = F.relu(h)
                return {'m' : h}

            def reduce_func(nodes):
                m = nodes.mailbox['m'] 

                h = torch.mean(m, 1)
                h = F.relu(h)

                return {'activation':h}

            nf.block_compute(i,
                            message_func,
                            reduce_func)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h