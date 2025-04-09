import numpy as np
import pandas as pd
import dgl
import os
from sklearn import preprocessing
import torch
import pickle
from datetime import datetime

max_event = 30
num_edge_feats = 5

start_date = datetime(2015, 12, 26, 0, 0).timestamp()
end_date = datetime(2020, 12, 3, 0, 0).timestamp()

def read_lines(fname):
    with open(fname) as f:
        return f.readlines()

def padding_tensor(sequences, max_len=None):
    """
    :param sequences: list of tensors
    :return:
    """
    sequences = torch.tensor(sequences, dtype=torch.float32)
    num = len(sequences)
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask

def padding_sequences(sequences, max_len):

    # sequences: numpy array

    new_s = np.zeros((max_len, sequences.shape[1]))

    new_s[:len(sequences), :] = sequences

    return new_s, len(sequences)

def agg_fun(x):

    x = x.sort_values(['time'])
    # times = list(x['time'].map(datetime.fromtimestamp)) 
    # x['time'] = x['time'] - start_date

    length = min(len(x), max_event)

    # delta_t = np.zeros(length)
    # for i in range(1, length):
    #     delta_t[i] = ((times[i]) - (times[i-1])).days # delta seconds
    
    x = x.to_numpy()[:length, 2:]
    # for i in range(len(x)):
    #     x[i][-1] = delta_t[i]

    return x


def process_adj(num_nodes, labels):
    
    print('Start reading adj')
    adj = pd.read_csv('adj.csv', sep=',')

    adj = adj.rename(columns={'from_address':'srcId', 'to_address':'dstId', 'block_timestamp':'time'})


    print('Start filter adj')
    print('Before filter: {}'.format(len(adj)))

    print('Filtering time')
    
    adj = adj.loc[(adj['time'] >= start_date) & (adj['time'] < end_date)]
    adj['time'] = adj['time'] - start_date

    # print(adj)

    labels_nodes = set(labels['id'])


    cond1 = adj['srcId'].isin(labels_nodes)
    cond2 = ~adj['dstId'].isin(labels_nodes)
    adj_s = adj.loc[cond1 & cond2]
    adj_d = adj[adj['dstId'].isin(labels_nodes)]

    adj = pd.concat([adj_s, adj_d])

    print('Filtering nodes without node features')

    features_nodes = set(range(num_nodes))

    cond1 = adj['srcId'].isin(features_nodes)
    cond2 = adj['dstId'].isin(features_nodes)

    adj = adj.loc[cond1 & cond2]

    print('After filter: {}'.format(len(adj)))

    print('Start standard edge features')

    # scaler = preprocessing.StandardScaler().fit(adj[adj.columns[2:-1]].values)
    # adj[adj.columns[2:-1]] = scaler.transform(adj[adj.columns[2:-1]])
    
    scaler = preprocessing.StandardScaler().fit(adj[adj.columns[2:]].values)
    adj[adj.columns[2:]] = scaler.transform(adj[adj.columns[2:]])

    print('Start groupby')

    adj_reverse = adj.rename(columns={'srcId':'dstId', 'dstId':'srcId'})

    adj['direction'] = 0
    adj_reverse['direction'] = 1

    adj_ = pd.concat([adj, adj_reverse])

    print(adj_)  

    adj_ = adj_.groupby(['srcId', 'dstId']).apply(agg_fun)
    # adj_reverse_ = adj_reverse.groupby(['srcId', 'dstId']).apply(agg_fun)

      

    # edges = dict()
    # for edge in adj_.index:
    #     if edge in edges:
    #         edges[edge][0] = adj_[edge]
    #     else:
    #         edges[edge] = [[], []]
    #         edges[edge][0] = adj_[edge]


    # for edge in adj_reverse_.index:
    #     if edge in edges:
    #         edges[edge][1] = adj_reverse_[edge]
    #     else:
    #         edges[edge] = [[], []]
    #         edges[edge][1] = adj_reverse_[edge]

    # for edge in edges:
    #     feats = edges[edge]
    #     for i in range(2):
    #         if len(feats[i]) == 0:
    #             feats[i] = np.zeros((max_event, num_edge_feats))
    #         else:
    #             feats[i] = np.array(feats[i])

    edges = dict()
    for edge in adj_.index:
        edges[edge] = adj_[edge]

    return edges

def process_features():

    print('Loading node features')
    features = pd.read_csv('features.csv', sep=',')
    print('Start standard node features')
    features = features.to_numpy()    

    scaler = preprocessing.StandardScaler().fit(features)
    features = scaler.transform(features)
    print('Finish loading node features', features.shape)
    return features

def process_labels():
    labels = pd.read_csv('labels.csv', sep=',')

    return labels

def create_features():

    features = process_features() 
    num_nodes = features.shape[0]

    labels = process_labels()
    edges = process_adj(num_nodes, labels)
      

    print('Collecting edge features') 

    # edge_features_from = []
    # edge_features_to = []

    edge_features = []
    edge_len = []
    edge_from_id = []
    edge_to_id = []
    seq_times = []

    for edge in edges:
        e = edges[edge]
        e, length = padding_sequences(e, max_event)
        t = e[:, -2]
        e = np.delete(e, -2, axis=1)

        edge_from_id.append(edge[0])
        edge_to_id.append(edge[1])
        edge_features.append(e)
        edge_len.append(length)
        seq_times.append(t)

    edge_features = np.stack(edge_features)
    edge_len = np.array(edge_len).astype(np.int32)
    seq_times = np.stack(seq_times)
    # print(edge_features_from)

    # edge_features_from = np.stack(edge_features_from)
    # edge_features_to = np.stack(edge_features_to)

    # print(edge_features_from.shape, edge_features_to.shape)

    # print(zip(edges.keys()))
    # (edge_from_id, edge_to_id) = zip(*edges.keys())

    print('Edge information', edge_features.shape, len(edge_from_id), len(edge_to_id), len(edge_len), seq_times.shape)

    return features, labels, edges, edge_from_id, edge_to_id, edge_features, edge_len, seq_times


def create_dgl_graph(features, labels, edges, edge_from_id, edge_to_id, edge_features, edge_len, seq_times):
    
    number_of_nodes = features.shape[0]
    node_in_dim = features.shape[1]
    number_of_edges = len(edges)
    edge_in_dim = num_edge_feats + 1


    g = dgl.DGLGraph()
    g.add_nodes(number_of_nodes)
    # g.add_edges(u=edge_from_id, 
    #             v=edge_to_id, 
    #             data={'edge_features_from': torch.tensor(edge_features_from, dtype=torch.float32),
    #                   'edge_features_to': torch.tensor(edge_features_to, dtype=torch.float32)})

    g.add_edges(u=edge_from_id, v=edge_to_id)


    # g.ndata['node_features'] = torch.tensor(features, dtype=torch.float32)

    print('number of nodes:', g.number_of_nodes())
    print('number of edges (multi-direction):', g.number_of_edges())
    print('node features shape: ', features.shape)
    print('edge features shape: ', edge_features.shape, edge_len.shape)

    # print('Adding self-loop')
    # g.add_edges(g.nodes(), g.nodes(), 
    #             data={'edge_features_from': torch.zeros((g.number_of_nodes(), max_event, edge_in_dim), dtype=torch.float32),
    #                   'edge_features_to': torch.zeros((g.number_of_nodes(), max_event, edge_in_dim), dtype=torch.float32)})



    # print('Transpose edge features')

    # g.edata['edge_features_from'] = g.edata['edge_features_from'].transpose(1, 2)
    # g.edata['edge_features_to'] = g.edata['edge_features_to'].transpose(1, 2)
    # print('edge features shape: ', g.edata['edge_features_from'].shape, g.edata['edge_features_to'].shape)

    print('Writing graph to pkl')
    with open('dynamic_dgl_graph.pkl', 'wb') as f:
        pickle.dump(g, f)


    print('Train test split')
    num_labels = len(labels)

    train_index, val_index, test_index = np.split(labels.sample(frac=1, random_state=10), [int(.6*num_labels), int(.8*num_labels)])
    train_index, val_index, test_index = list(train_index['id']), list(val_index['id']), list(test_index['id'])

    train_mask = np.zeros((g.number_of_nodes()))
    val_mask = np.zeros((g.number_of_nodes()))
    test_mask = np.zeros((g.number_of_nodes()))

    train_mask[train_index] = 1
    val_mask[val_index] = 1
    test_mask[test_index] = 1
    new_labels = np.zeros((g.number_of_nodes()))
    for _, row in labels.iterrows():
        new_labels[row['id']] = row['label']

    print('Writing labels')
    with open('labels.pkl', 'wb') as f:
        pickle.dump((new_labels, train_mask, val_mask, test_mask), f)

    # print(features.dtype, edge_features_from.dtype, edge_features_to.dtype)
    features = features.astype(np.float32)
    edge_features = edge_features.astype(np.float32)
    # edge_len = edge_len.astype(np.int32)

    print('Saving node and edge features')
    # with open('features.npz', 'w') as f: 
    np.savez('features.npz', node_features=features, edge_features=edge_features, edge_len=edge_len, seq_times=seq_times)



def main():
    
    features, labels, edges, edge_from_id, edge_to_id, edge_features, edge_len, seq_times = create_features()
    create_dgl_graph(features, labels, edges, edge_from_id, edge_to_id, edge_features, edge_len, seq_times)    


if __name__ == '__main__':
    main()


# print('number of nodes:', g.number_of_nodes())
# print('number of edges (multi-direction):', g.number_of_edges())
# print('node features: ', features)
# print('node features shape: ', features.shape)
# print('edge features: ', g.edata['edge_features'])
# print('edge features shape: ', g.edata['edge_features'].shape)
# print('train_id:', train_id)
# print('val_id:', val_id)
# print('test_id:', test_id)
# print()
# ### Label
# print('labels:', labels)
# print('number of node classes:', num_classes)



