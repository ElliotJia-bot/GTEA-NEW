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

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, k_=None, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, _ = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        

        '''
            substarct from self-loop
        '''
        if k_ is not None:

            q = self.w_qs(k_).view(sz_b, 1, n_head, d_k)
            k = self.w_ks(k_).view(sz_b, 1, n_head, d_k)
            v = self.w_vs(k_).view(sz_b, 1, n_head, d_v)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k) # (n*b) x lq x dk
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k) # (n*b) x lk x dk
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_v) # (n*b) x lv x dv

            
            tmp, _ = self.attention(q, k, v, mask=mask)
            tmp = tmp.view(n_head, sz_b, 1, d_v)

            tmp = tmp.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
            

            output -= tmp

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output



class TGATTrain(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_heads, device, dropout=None):
        super(TGATTrain, self).__init__()

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
        self.attn_layers.append(MultiHeadAttention(num_heads, node_in_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim, node_hidden_dim))

        self.node_layers =  nn.ModuleList()
        self.node_layers.append(nn.Linear(node_in_dim + node_in_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.time_layers.append(TimeEncode(time_hidden_dim))
            self.attn_layers.append(MultiHeadAttention(num_heads, node_hidden_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim, node_hidden_dim))
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
                self_h_tmp = nodes.data['self_h_tmp'].unsqueeze(1)
                num_nodes, num_edges = m.shape[0], m.shape[1]
                feat_dim = self_h_tmp.shape[-1]
                m = m.reshape(num_nodes, -1, feat_dim)

                '''
                    h (N, 2 + L, D), L is the number of interactions of the target node
                    2 is the target node, 1 is for q, the other is k and v for substracting self-loop
                '''           
                # h = torch.sum(m, dim=1)

                k = v = m
                q = self_h_tmp.reshape(num_nodes, 1, -1)


                out = attn_layer(q, k ,v, q)


                h = torch.sum(out, dim=1) 

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

class TGATInfer(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_heads, device, dropout=None):
        super(TGATInfer, self).__init__()

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
        self.attn_layers.append(MultiHeadAttention(num_heads, node_in_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim, node_hidden_dim))

        self.node_layers =  nn.ModuleList()
        self.node_layers.append(nn.Linear(node_in_dim + node_in_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.time_layers.append(TimeEncode(time_hidden_dim))
            self.attn_layers.append(MultiHeadAttention(num_heads, node_hidden_dim + time_hidden_dim + edge_in_dim - 1, node_hidden_dim, node_hidden_dim))
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
                self_h_tmp = nodes.data['self_h_tmp'].unsqueeze(1)
                num_nodes, num_edges = m.shape[0], m.shape[1]
                feat_dim = self_h_tmp.shape[-1]
                m = m.reshape(num_nodes, -1, feat_dim)

                '''
                    h (N, 2 + L, D), L is the number of interactions of the target node
                    2 is the target node, 1 is for q, the other is k and v for substracting self-loop
                '''           
                # h = torch.sum(m, dim=1)

                k = v = m
                q = self_h_tmp.reshape(num_nodes, 1, -1)

                out = attn_layer(q, k ,v, q)
                h = torch.sum(out, dim=1)
                h = F.relu(node_layer(torch.cat((h, self_h), dim=1))) 
            

                return {'activation': h}


            nf.block_compute(i,
                            message_func,
                            reduce_func)

        h = nf.layers[-1].data.pop('activation')
        h = self.fc(h)

        return h
