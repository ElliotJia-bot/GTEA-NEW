Dependencies(not include all)
------------
- torch v1.4:
- sklearn
- tensorflow (for tensorboard)
- dgl v0.4.3
```bash
$ pip install -r requirements.txt
```

```bash
$ pip install torch
$ pip install dgl
$ pip install tensorboardX
```

# Run experiment
- prepare the dataset (adj.txt, features,txt, labels.txt), the format for each of the file is listed below (Also see data/test_data for the example):

## adj.txt:
```
src_node_id, dst_node_id, timestamp, edge_features1, edge_features2, ...
```

## features.txt
```
node_id, feature1, feature2, ...
```

## label.txt
```
node_id, label_id
```

- preprose the dataset (adj.txt, features,txt, label.txt) and store in three different files: (Also see data/dynamic_event_loader):

## dynamic_dgl_graph.pkl
```
dgl graph
```

## labels.pkl
```
labels, train_mask, val_mask, test_mask
```

## features.npz
```
node_features: (num_nodes, feat_dim)
edge_features: with padding max length (num_edges, time_max_len, feat_dim)
edge_len: length(without padding) for each edge time sequence: (num_edges, )
seq_times: timestamp dimension features sequence, (num_edges, time_max_len) 
```

- models
```
GraphSAGE
EdgeLSTM: raw t + LSTM
EdgeTLSTM1: delta_t with TLSTM1
EdgeTLSTM2: delta_t with TLSTM2
ETLSTM: time2vec + LSTM
ETransformer: raw t + transformer
GTEA-LSTM+T2V: Graph Temporal Edge Attention with LSTM and Time2Vec
GTEA-Trans+T2V: Graph Temporal Edge Attention with Transformer and Time2Vec
```

- start the experiment using the following command
```bash
python main.py --data-dir ./data/dynamic_eth --model GraphSAGE
python main.py --data-dir ./data/test --model GraphSAGE
```

# params(not include all, see main.py for details)
## General parameter
- gpu (int): The gpu id. gpu should be set to -1 if you only want to use cpu
- epochs (int): The max number of epochs.
- lr (float): learning rate
- patience (int): patience for early stopping, if the loss stops decreasing for consecutive (patience) epochs, it will early stop.
- weight_decay (float): an L2 regularization. Higher value will result in stronger regularization
- model (str): The model name. (See below for details)

## Additional parameter for minibatch model
- batch_size (int): 
- test_batch_size (int):
- num_neighbors (int): number of samples used for neighbor sampling
- num_cpu: number of cpu used for minibatch sampling and neighbor sampling

# Experimental Results

## Datastes Reference

Ethereum Phishing Transaction Network: 
    https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network
IBM Transactions for Anti Money Laundering (AML):
    https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
Anti Money Laundering Transaction Data (SAML-D):
    https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

Please put datasets under ./data
Before using datasets pleases use preprocessing.py to process datasets

## IBM_HI_S Dataset
- Model: GTEA-LSTM+T2V
- Training Accuracy: ~0.997
- Validation Accuracy: ~0.999
- Test Accuracy: ~0.999
- Training Time per Epoch: ~359s
- Processing Speed: ~11.75 KTEPS
- Using preprocessing.py to preprocess the datasets

## SAML Dataset
- Model: GTEA-Trans+T2V
- Training Accuracy: ~0.999
- Validation Accuracy: 1.000
- Test Accuracy: 1.000
- Training Time per Epoch: ~265s
- Processing Speed: ~3.12 KTEPS
- Using preprocessing.py to preprocess the datasets

# Running Experiments
```bash
# Run GTEA-LSTM+T2V on IBM_HI_S dataset
python main.py --data-dir ./data/IBM_HI_S --model GTEA-LSTM+T2V

# Run GTEA-Trans+T2V on SAML dataset
python main.py --data-dir ./data/saml --model GTEA-Trans+T2V
```
