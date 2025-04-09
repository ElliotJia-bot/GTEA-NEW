import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeUpdate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodeUpdate, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.test = test

        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, node):
        h = node.data['h']
        self_h = node.data['self_h']        
        ''' self_h already not include in h'''

        h += F.relu(self.layer(self_h))
        return {'activation': h}


class MiniBatchECConvTrain(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_class, num_layers, dropout=None):
        super(MiniBatchECConvTrain, self).__init__()

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.node_layers = nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim, hidden_dim))

        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(nn.Linear(edge_in_dim, hidden_dim * node_in_dim))


        for i in range(1, self.num_layers):
            self.node_layers.append(NodeUpdate(hidden_dim, hidden_dim))
            self.edge_layers.append(nn.Linear(edge_in_dim, hidden_dim * hidden_dim))

        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (edge_layer,  node_layer) in enumerate(zip(self.edge_layers, self.node_layers)):
            h = nf.layers[i].data.pop('activation')

            e = nf.blocks[i].data['edge_features']

            e = F.relu(edge_layer(e))

            nf.blocks[i].data['e'] = e
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]


            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h


            def message_func(edges):

                h = edges.src['h']
                e = edges.data['e']
                e = e.reshape(e.shape[0], self.hidden_dim, -1)
                
                m = torch.bmm(e, h.unsqueeze(-1)).squeeze(-1)

                return {'m': m}

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


class MiniBatchECConvInfer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_class, num_layers, dropout=None):
        super(MiniBatchECConvInfer, self).__init__()

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.node_layers = nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim, hidden_dim))

        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(nn.Linear(edge_in_dim, hidden_dim * node_in_dim))


        for i in range(1, self.num_layers):
            self.node_layers.append(NodeUpdate(hidden_dim, hidden_dim))
            self.edge_layers.append(nn.Linear(edge_in_dim, hidden_dim * hidden_dim))

        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['node_features']


        for i, (edge_layer,  node_layer) in enumerate(zip(self.edge_layers, self.node_layers)):
            h = nf.layers[i].data.pop('activation')

            e = nf.blocks[i].data['edge_features']

            e = F.relu(edge_layer(e))

            nf.blocks[i].data['e'] = e
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            self_h = h[layer_nid]


            nf.layers[i].data['h'] = h
            nf.layers[i+1].data['self_h'] = self_h


            def message_func(edges):

                h = edges.src['h']
                e = edges.data['e']
                e = e.reshape(e.shape[0], self.hidden_dim, -1)
                
                m = torch.bmm(e, h.unsqueeze(-1)).squeeze(-1)

                return {'m': m}

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