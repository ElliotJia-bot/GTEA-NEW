import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from .TimeModel import LSTM

# import logging


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
        self_h_tmp = node.data['self_h_tmp']
        if self.test:
            h = (h - self_h_tmp) * node.data['norm']   
            # h = h * node.data['norm']
        else:
            h = (h - self_h_tmp) * node.data['subg_norm']
            # h = h * node.data['subg_norm']

        h = torch.cat((self_h, h), dim=1)

        # print('hhhhhhhhhh', h.shape)
        h = F.relu(self.layer(h))
        return {'activation': h}


class PEdgeLSTMTrain(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, num_class, num_layers, num_lstm_layers, bidirectional, device, dropout=None):
        super(PEdgeLSTMTrain, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_lstm_layers = num_lstm_layers
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x


        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(LSTM(edge_in_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))

        self.node_project_w = []
        self.node_project_w.append(torch.nn.Parameter(data=torch.Tensor(node_in_dim, 1), requires_grad=True))

        self.edge_out_layers = nn.ModuleList()
        self.edge_out_layers.append(nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim))


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim + node_hidden_dim, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.edge_layers.append(LSTM(edge_in_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))
            self.node_project_w.append(torch.nn.Parameter(data=torch.Tensor(node_hidden_dim, 1), requires_grad=True))
            self.edge_out_layers.append(nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            self.node_layers.append(NodeUpdate(2 * node_hidden_dim, node_hidden_dim))

        # init weights
        for i in range(num_layers):
            nn.init.xavier_normal_(self.node_project_w[i].data, gain=0.01)

        for i in range(num_layers):
            self.node_project_w[i] = self.node_project_w[i].to(device)
        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (edge_layer, node_p_layer, edge_out_layer, node_layer) in enumerate(zip(self.edge_layers, self.node_project_w, self.edge_out_layers, self.node_layers)):

            e = nf.blocks[i].data['edge_features']
            e_len = nf.blocks[i].data['edge_len']
            e_times = nf.blocks[i].data['seq_times']
            
            e = self.dropout_layer(e)
            nf.blocks[i].data['e'] = e
            nf.blocks[i].data['e_len'] = e_len
            nf.blocks[i].data['e_times'] = e_times

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

            tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            self_h_tmp = torch.cat((self_h, tmp), dim=1)
            self_h_tmp = edge_out_layer(self_h_tmp)
            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h
            nf.layers[i+1].data['self_h_tmp'] = self_h_tmp

            
            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times']     
                
                e = torch.cat((e, e_times.unsqueeze(-1)), dim=-1)
                e = edge_layer(e, e_len)

                h = h - torch.matmul(torch.matmul(h, node_p_layer), node_p_layer.t())
                # h = self.feat_drop(h)        
                h = edge_out_layer(torch.cat((h, e), dim=1))

                h = F.relu(h)
                return {'m': h}

            def reduce_func(nodes):
                m = nodes.mailbox['m']                

                h = torch.sum(m, dim=1)
                return {'h': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func,
                            node_layer)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h

'''
    Main difference between train model and infer model is in NodeUpdate layer, by setting test=True
'''
class PEdgeLSTMInfer(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, num_class, num_layers, num_lstm_layers, bidirectional, device, dropout=None):
        super(PEdgeLSTMInfer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_lstm_layers = num_lstm_layers
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x


        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(LSTM(edge_in_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))

        self.node_project_w = []
        self.node_project_w.append(torch.nn.Parameter(data=torch.Tensor(node_in_dim, 1), requires_grad=True))

        self.edge_out_layers = nn.ModuleList()
        self.edge_out_layers.append(nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim))


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim + node_hidden_dim, node_hidden_dim, test=True))

        for i in range(1, self.num_layers):
            self.edge_layers.append(LSTM(edge_in_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))
            self.node_project_w.append(torch.nn.Parameter(data=torch.Tensor(node_hidden_dim, 1), requires_grad=True))
            self.edge_out_layers.append(nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            self.node_layers.append(NodeUpdate(2 * node_hidden_dim, node_hidden_dim, test=True))

        # init weights
        for i in range(num_layers):
            nn.init.xavier_normal_(self.node_project_w[i].data, gain=0.01)


        for i in range(num_layers):
            self.node_project_w[i] = self.node_project_w[i].to(device)

        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (edge_layer, node_p_layer, edge_out_layer, node_layer) in enumerate(zip(self.edge_layers, self.node_project_w, self.edge_out_layers, self.node_layers)):

            e = nf.blocks[i].data['edge_features']
            e_len = nf.blocks[i].data['edge_len']
            e_times = nf.blocks[i].data['seq_times']
            
            e = self.dropout_layer(e)
            nf.blocks[i].data['e'] = e
            nf.blocks[i].data['e_len'] = e_len
            nf.blocks[i].data['e_times'] = e_times

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

            tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            self_h_tmp = torch.cat((self_h, tmp), dim=1)
            self_h_tmp = edge_out_layer(self_h_tmp)
            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h
            nf.layers[i+1].data['self_h_tmp'] = self_h_tmp

            
            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times']     
                
                e = torch.cat((e, e_times.unsqueeze(-1)), dim=-1)
                e = edge_layer(e, e_len)

                h = h - torch.matmul(torch.matmul(h, node_p_layer), node_p_layer.t())
                # h = self.feat_drop(h)        
                h = edge_out_layer(torch.cat((h, e), dim=1))

                h = F.relu(h)
                return {'m': h}

            def reduce_func(nodes):
                m = nodes.mailbox['m']                

                h = torch.sum(m, dim=1)
                return {'h': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func,
                            node_layer)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h