import numpy as np
import pandas as pd
import dgl
import pickle
import logging
import sys
import os
import torch
from sklearn.model_selection import KFold



# logging.basicConfig(level=logging.INFO)

class Dataset(object):
    def __init__(self, data_dir, max_event, use_K=None, K=5, load_static_edge=False, remove_node_features=False):

        self.dgl_pickle_path = os.path.join(data_dir, 'dynamic_dgl_graph.pkl')
        self.labels_pickle_path = os.path.join(data_dir, 'labels.pkl')
        self.feature_path = os.path.join(data_dir, 'features.npz')
        self.static_edge_feature_path = os.path.join(data_dir, 'static_edge_features.npy')
        self.max_event = max_event

        self.use_K = use_K
        self.K = K
        self.load_static_edge = load_static_edge
        self.remove_node_features = remove_node_features


        self.load()

    def load(self):
        with open(self.dgl_pickle_path, 'rb') as f:
            self.g = pickle.load(f)

        with open(self.labels_pickle_path, 'rb') as f:
            self.labels, self.train_mask, self.val_mask, self.test_mask = pickle.load(f)

        
        # print(self.labels)

        if self.use_K is not None:
            self.labels_id = np.concatenate([np.nonzero(self.train_mask), np.nonzero(self.val_mask), np.nonzero(self.test_mask)], axis=1).reshape(-1)
            self.labels_id = np.array(self.labels_id).astype(np.int64)
            kf = KFold(n_splits=self.K, shuffle=True, random_state=0)
            assert self.use_K >=0 and self.use_K < self.K, "K fold out of range"
            count = 0
            for train_index, val_index in kf.split(self.labels_id):
                # print(self.labels_id[train_index], self.labels_id[val_index])
                if count == self.use_K:
                    break
                else:
                    count += 1

            train_index = self.labels_id[train_index]
            val_index = self.labels_id[val_index]


            self.train_mask = np.zeros((self.g.number_of_nodes()))
            self.val_mask = np.zeros((self.g.number_of_nodes()))
            self.test_mask = np.zeros((self.g.number_of_nodes()))

            self.train_mask[train_index] = 1
            self.val_mask[val_index] = 1
            self.test_mask[val_index] = 1   # in k-fold, we do not use test, so pad with val_mask

            # print(self.train_mask)


        data = np.load(self.feature_path)

        node_features = data['node_features']
        edge_features = data['edge_features']
        edge_len = data['edge_len']
        seq_times = data['seq_times']

        edge_features = edge_features[:, :self.max_event, :]
        seq_times = seq_times[:, :self.max_event]

        delta_t = np.zeros((seq_times.shape[0], self.max_event), dtype=np.float32)
        delta_t[:, 1:] = seq_times[:, :-1]
        delta_t = seq_times - delta_t
        delta_t[:, 0] = 0

        
        for i in range(seq_times.shape[0]):
            delta_t[i, edge_len[i]:] = 0

        '''
            node_features: (num_nodes, feat_dim)
            edge_features: with padding max length(num_edges, time_max_len, feat_dim)
            edge_len: length(without padding) for each edge time sequence: (num_edges, )
            seq_times: timestamp dimension features sequence, (num_edges, time_max_len) 
            delta_t: time span, (num_edges, time_max_len)
        '''

        if self.remove_node_features:
            node_features = torch.zeros_like(torch.from_numpy(node_features))
            # print(node_features)

        if not self.load_static_edge:


            self.g.ndata['node_features'] = node_features
            # print(self.g.ndata['node_features'])
            self.g.edata['edge_features'] = torch.tensor(edge_features, dtype=torch.float32)
            self.g.edata['edge_len'] = torch.clamp(torch.tensor(edge_len, dtype=torch.long), min=0, max=self.max_event)
            self.g.edata['seq_times'] = torch.tensor(seq_times, dtype=torch.float32)
            self.g.edata['delta_t'] = torch.tensor(delta_t, dtype=torch.float32)


            # sequence length mask for transformer
            mask = torch.ones(len(edge_len), self.max_event)
            for i in range(len(edge_len)):
                mask[i, :edge_len[i]] = 0

            self.g.edata['edge_mask'] = mask.bool()

            edge_in_dim = self.g.edata['edge_features'].shape[2]
            edge_timestep_len = self.g.edata['edge_features'].shape[1] 

            self_loop_mask = torch.ones((self.g.number_of_nodes(), edge_timestep_len), dtype=torch.bool)
            self_loop_mask[:, 0] = False

            print('Adding self-loop')
            self.g.add_edges(self.g.nodes(), self.g.nodes(), 
                        data={'edge_features': torch.zeros((self.g.number_of_nodes(), edge_timestep_len, edge_in_dim), dtype=torch.float32),
                              'edge_len': torch.ones((self.g.number_of_nodes()), dtype=torch.long),
                              'seq_times': torch.zeros((self.g.number_of_nodes(), edge_timestep_len), dtype=torch.float32),
                              'edge_mask': self_loop_mask,
                              'delta_t': torch.zeros(self.g.number_of_nodes(), edge_timestep_len)})
            
            self.features = self.g.ndata['node_features']
            
            self.num_classes = len(np.unique(self.labels))
            self.edge_in_dim = edge_in_dim + 1
            self.edge_timestep_len = edge_timestep_len

        else:
            edge_features = np.load(self.static_edge_feature_path)

            self.g.ndata['node_features'] = node_features
            self.g.edata['edge_features'] = torch.tensor(edge_features, dtype=torch.float32)


            edge_in_dim = self.g.edata['edge_features'].shape[-1]
            edge_timestep_len = 0 


            print('Adding self-loop')
            self.g.add_edges(self.g.nodes(), self.g.nodes(), 
                        data={'edge_features': torch.zeros((self.g.number_of_nodes(), edge_in_dim), dtype=torch.float32)})
                              
            
            self.features = self.g.ndata['node_features']
            
            self.num_classes = len(np.unique(self.labels))
            self.edge_in_dim = edge_in_dim
            self.edge_timestep_len = edge_timestep_len