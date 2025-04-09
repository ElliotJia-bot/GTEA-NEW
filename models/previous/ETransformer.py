import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from .TimeModel import TransformerModel

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


class ETransformerTrain(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, num_class, num_layers, num_heads, num_encoder_layers, device, dropout=None):
        super(ETransformerTrain, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x


        self.edge_in_layers = nn.ModuleList()
        self.edge_in_layers.append(nn.Linear(edge_in_dim, node_hidden_dim))

        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(TransformerModel(node_hidden_dim, num_heads, node_hidden_dim, num_encoder_layers, device, dropout))


        self.edge_out_layers = nn.ModuleList()
        self.edge_out_layers.append(nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim))


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim + node_hidden_dim, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.edge_in_layers.append(nn.Linear(edge_in_dim, node_hidden_dim))
            self.edge_layers.append(TransformerModel(node_hidden_dim, num_heads, node_hidden_dim, num_encoder_layers, device, dropout))
            self.edge_out_layers.append(nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            self.node_layers.append(NodeUpdate(2 * node_hidden_dim, node_hidden_dim))


        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (edge_in_layer, edge_layer, edge_out_layer, node_layer) in enumerate(zip(self.edge_in_layers, self.edge_layers, self.edge_out_layers, self.node_layers)):

            e = nf.blocks[i].data['edge_features']
            e_len = nf.blocks[i].data['edge_len']
            e_times = nf.blocks[i].data['seq_times']
            e_mask = nf.blocks[i].data['edge_mask']
            
            e = self.dropout_layer(e)
            nf.blocks[i].data['e'] = e
            nf.blocks[i].data['e_len'] = e_len
            nf.blocks[i].data['e_times'] = e_times
            nf.blocks[i].data['e_mask'] = e_mask

            h = nf.layers[i].data.pop('activation')
            h = self.dropout_layer(h)
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]            

            tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            self_h_tmp = torch.cat((self_h, tmp), dim=1)
            self_h_tmp = edge_out_layer(self_h_tmp)
            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h
            nf.layers[i+1].data['self_h_tmp'] = self_h_tmp

            # print('self_h', self_h.shape, self_h_tmp.shape)


            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times']   
                e_mask = edges.data['e_mask']  

                # e_times = e_times.reshape(num_edges, -1, 1)  
                e_times = e_times.unsqueeze(-1)

                e = torch.cat((e, e_times), dim=-1)
                
                bs, s_len, feat_dim = e.shape                

                e = edge_in_layer(e.reshape(-1, feat_dim))
                e = F.relu(e).reshape(bs, s_len, -1)
             
                e = edge_layer(e, e_len, e_mask)

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

class ETransformerInfer(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, num_class, num_layers, num_heads, num_encoder_layers, device, dropout=None):
        super(ETransformerInfer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x


        self.edge_in_layers = nn.ModuleList()
        self.edge_in_layers.append(nn.Linear(edge_in_dim, node_hidden_dim))

        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(TransformerModel(node_hidden_dim, num_heads, node_hidden_dim, num_encoder_layers, device, dropout))


        self.edge_out_layers = nn.ModuleList()
        self.edge_out_layers.append(nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim))


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim + node_hidden_dim, node_hidden_dim, test=True))

        for i in range(1, self.num_layers):
            self.edge_in_layers.append(nn.Linear(edge_in_dim, node_hidden_dim))
            self.edge_layers.append(TransformerModel(node_hidden_dim, num_heads, node_hidden_dim, num_encoder_layers, device, dropout))
            self.edge_out_layers.append(nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            self.node_layers.append(NodeUpdate(2 * node_hidden_dim, node_hidden_dim, test=True))


        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (edge_in_layer, edge_layer, edge_out_layer, node_layer) in enumerate(zip(self.edge_in_layers, self.edge_layers, self.edge_out_layers, self.node_layers)):

            e = nf.blocks[i].data['edge_features']
            e_len = nf.blocks[i].data['edge_len']
            e_times = nf.blocks[i].data['seq_times']
            e_mask = nf.blocks[i].data['edge_mask']
            
            e = self.dropout_layer(e)
            nf.blocks[i].data['e'] = e
            nf.blocks[i].data['e_len'] = e_len
            nf.blocks[i].data['e_times'] = e_times
            nf.blocks[i].data['e_mask'] = e_mask

            h = nf.layers[i].data.pop('activation')
            h = self.dropout_layer(h)
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]            

            tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            self_h_tmp = torch.cat((self_h, tmp), dim=1)
            self_h_tmp = edge_out_layer(self_h_tmp)
            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h
            nf.layers[i+1].data['self_h_tmp'] = self_h_tmp

            # print('self_h', self_h.shape, self_h_tmp.shape)


            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times']   
                e_mask = edges.data['e_mask']  

                e = torch.cat((e, e_times.unsqueeze(-1)), dim=-1)


                bs = e.shape[0]
                s_len = e.shape[1]
                feat_dim = e.shape[2]

                e = edge_in_layer(e.reshape(-1, feat_dim))
                e = F.relu(e).reshape(bs, s_len, -1)                
                
                e = edge_layer(e, e_len, e_mask)

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