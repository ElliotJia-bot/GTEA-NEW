import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        self.register_parameter('basis_frea', self.basis_freq)
        self.register_parameter('phase', self.phase)
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)



class GATPlusTTrain(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_heads, device, dropout=None):
        super(GATPlusTTrain, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.time_layers = nn.ModuleList()
        self.time_layers.append(TimeEncode(time_hidden_dim))

        self.attn_layers = nn.ModuleList()
        self.attn_layers.append(nn.Linear(node_in_dim + time_hidden_dim + edge_in_dim - 1, 1))

        self.node_layers =  nn.ModuleList()
        self.node_layers.append(nn.Linear(node_in_dim + node_in_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.time_layers.append(TimeEncode(time_hidden_dim))
            self.attn_layers.append(nn.Linear(node_hidden_dim + time_hidden_dim + edge_in_dim - 1, 1))
            self.node_layers.append(nn.Linear(node_hidden_dim + time_hidden_dim + edge_in_dim - 1 + node_hidden_dim, node_hidden_dim))

        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (time_layer, attn_layer, node_layer) in enumerate(zip(self.time_layers, self.attn_layers, self.node_layers)):

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
                prepare for substract the self-loop, DGL need self-loop to convert 
                the node features of the target node of the last layer
            '''
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]            

            t_tmp = time_layer(torch.zeros((self_h.shape[0], 1), device=self.device))   # Timeencoder(t=0) for target node 
            t_tmp = t_tmp.reshape(self_h.shape[0], -1)
            e_tmp = (torch.zeros((self_h.shape[0], self.edge_in_dim-1), device=self.device))
            

            self_h_tmp = torch.cat((self_h, e_tmp, t_tmp), dim=1)
            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h
            nf.layers[i+1].data['self_h_tmp'] = self_h_tmp           


            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times'] 

                num_edges = e_times.shape[0]                

                e_times = time_layer(e_times)   # temporal encoder to get time embedding                

                e_times = e_times.reshape(num_edges, -1, self.time_hidden_dim)  
           
                e = torch.cat((e, e_times), dim=-1)
                h = h.unsqueeze(1).repeat(1, e_times.shape[1], 1)             
                h = torch.cat((h, e), dim=-1)

                h = h.reshape(h.shape[0], -1)
                return {'m': h}

            def reduce_func(nodes):
                m = nodes.mailbox['m']    

                self_h = nodes.data['self_h']
                self_h_tmp = nodes.data['self_h_tmp']
                num_nodes, num_edges = m.shape[0], m.shape[1]
                feat_dim = self_h_tmp.shape[-1]

                m = m.reshape(num_nodes, -1, feat_dim)
                a = F.leaky_relu(attn_layer(m))
                h = a * m
                h = torch.sum(h, dim=1)
      
                # substract from self-loop
               
                tmp_a = F.leaky_relu(attn_layer(self_h_tmp))
                tmp = tmp_a * self_h_tmp

                h -= tmp
                h = F.relu(node_layer(torch.cat((h, self_h), dim=1))) 
            

                return {'activation': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h


'''
    The model not include node update layer, there's no difference between train and infer model
'''

class GATPlusTInfer(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_heads, device, dropout=None):
        super(GATPlusTInfer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.time_layers = nn.ModuleList()
        self.time_layers.append(TimeEncode(time_hidden_dim))

        self.attn_layers = nn.ModuleList()
        self.attn_layers.append(nn.Linear(node_in_dim + time_hidden_dim + edge_in_dim - 1, 1))

        self.node_layers =  nn.ModuleList()
        self.node_layers.append(nn.Linear(node_in_dim + node_in_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.time_layers.append(TimeEncode(time_hidden_dim))
            self.attn_layers.append(nn.Linear(node_hidden_dim + time_hidden_dim + edge_in_dim - 1, 1))
            self.node_layers.append(nn.Linear(node_hidden_dim + time_hidden_dim + edge_in_dim - 1 + node_hidden_dim, node_hidden_dim))

        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (time_layer, attn_layer, node_layer) in enumerate(zip(self.time_layers, self.attn_layers, self.node_layers)):

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
                prepare for substract the self-loop, DGL need self-loop to convert 
                the node features of the target node of the last layer
            '''
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]            

            t_tmp = time_layer(torch.zeros((self_h.shape[0], 1), device=self.device))   # Timeencoder(t=0) for target node 
            t_tmp = t_tmp.reshape(self_h.shape[0], -1)
            e_tmp = (torch.zeros((self_h.shape[0], self.edge_in_dim-1), device=self.device))
            

            self_h_tmp = torch.cat((self_h, e_tmp, t_tmp), dim=1)
            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h
            nf.layers[i+1].data['self_h_tmp'] = self_h_tmp           


            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times'] 

                num_edges = e_times.shape[0]                

                e_times = time_layer(e_times)   # temporal encoder to get time embedding                

                e_times = e_times.reshape(num_edges, -1, self.time_hidden_dim)  
           
                e = torch.cat((e, e_times), dim=-1)
                h = h.unsqueeze(1).repeat(1, e_times.shape[1], 1)             
                h = torch.cat((h, e), dim=-1)

                h = h.reshape(h.shape[0], -1)
                return {'m': h}

            def reduce_func(nodes):
                m = nodes.mailbox['m']    

                self_h = nodes.data['self_h']
                self_h_tmp = nodes.data['self_h_tmp']
                num_nodes, num_edges = m.shape[0], m.shape[1]
                feat_dim = self_h_tmp.shape[-1]

                m = m.reshape(num_nodes, -1, feat_dim)
                a = F.leaky_relu(attn_layer(m))
                h = a * m
                h = torch.sum(h, dim=1)
      
                # substract from self-loop
               
                tmp_a = F.leaky_relu(attn_layer(self_h_tmp))
                tmp = tmp_a * self_h_tmp

                h -= tmp
                h = F.relu(node_layer(torch.cat((h, self_h), dim=1))) 
            

                return {'activation': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h