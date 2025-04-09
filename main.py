import argparse, time, math
import numpy as np
import sys
import logging
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from models import GTEATLSTM1Train, GTEATLSTM1Infer
from models import GTEATLSTM2Train, GTEATLSTM2Infer
from models import GTEATLSTM3Train, GTEATLSTM3Infer
from models import ETLSTMTrain, ETLSTMInfer
from models import ETransformerTrain, ETransformerInfer
from models import T2VTransformerTrain, T2VTransformerInfer
from models import ATEdgeLSTMTrain, ATEdgeLSTMInfer

from models import GraphSAGETrain, GraphSAGEInfer
from models import GATTrain, GATInfer
from models import MiniBatchGCNTrain, MiniBatchGCNInfer
from models import MiniBatchECConvTrain, MiniBatchECConvInfer
from models import MiniBatchGTEASTTrain, MiniBatchGTEASTInfer  

from models import TGATTrain, TGATInfer
from models import GATPlusTTrain, GATPlusTInfer
from models import GTEATransVTrain, GTEATransVInfer
from models import GTEALSTMVTrain, GTEALSTMVInfer
from models import MiniBatchEGNNTrain, MiniBatchEGNNInfer


from models import GTEATransT2V_VTrain, GTEATransT2V_VInfer
from models import GTEALSTMT2V_VTrain, GTEALSTMT2V_VInfer

from MiniBatchTrainer import MiniBatchTrainer

from data.dynamic_event_data_loader import Dataset


def main(args):

    log_path = os.path.join(args.log_dir, args.log_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(filename=os.path.join(log_path, 'log_file'),
                        filemode='w',
                        format='| %(asctime)s |\n%(message)s',
                        datefmt='%b %d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)

    if args.model in ['ECConv', 'GTEA-ST', 'EGNN']:
        data = Dataset(data_dir=args.data_dir, max_event=args.max_event, use_K=args.use_K, K=args.K, load_static_edge=True)

    else:

        data = Dataset(data_dir=args.data_dir, max_event=args.max_event, use_K=args.use_K, K=args.K, remove_node_features=args.remove_node_features)

    g = data.g

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    # num_edge_feats = data.num_edge_feats

    
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    train_id = np.nonzero(data.train_mask)[0].astype(np.int64)
    val_id = np.nonzero(data.val_mask)[0].astype(np.int64)
    test_id = np.nonzero(data.test_mask)[0].astype(np.int64)

    num_nodes = features.shape[0]
    node_in_dim = features.shape[1]

    num_edges = data.g.number_of_edges()
    edge_in_dim = data.edge_in_dim
    edge_timestep_len = data.edge_timestep_len
    num_classes = data.num_classes
    

    num_train_samples = train_mask.int().sum().item()
    num_val_samples = val_mask.int().sum().item()
    num_test_samples = test_mask.int().sum().item()

    logging.info("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Node_feat %d
      #Edge_feat %d
      #Edge_timestep %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (num_nodes, num_edges, 
           node_in_dim, edge_in_dim, edge_timestep_len,
              num_classes,
              num_train_samples,
              num_val_samples,
              num_test_samples))
    


    
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() and args.gpu >=0 else "cpu")
    infer_device = device if args.infer_gpu else torch.device('cpu')
    # create  model
    if args.model == 'GCN':
        model = MiniBatchGCNTrain(in_feats=node_in_dim, 
                                        n_hidden=args.node_hidden_dim, 
                                        n_classes=num_classes, 
                                        n_layers=args.num_layers, 
                                        activation=F.relu,
                                        dropout=args.dropout)

        model_infer = MiniBatchGCNInfer(in_feats=node_in_dim, 
                                        n_hidden=args.node_hidden_dim, 
                                        n_classes=num_classes, 
                                        n_layers=args.num_layers, 
                                        activation=F.relu)
    elif args.model == 'GraphSAGE':

        model = GraphSAGETrain(in_dim=node_in_dim, 
                                        hidden_dim=args.node_hidden_dim, 
                                        num_class=num_classes, 
                                        num_layers=args.num_layers, 
                                        dropout=args.dropout)

        model_infer = GraphSAGEInfer(in_dim=node_in_dim, 
                                        hidden_dim=args.node_hidden_dim, 
                                        num_class=num_classes, 
                                        num_layers=args.num_layers, 
                                        dropout=None)
    elif args.model == 'GAT':
        model = GATTrain(in_dim=node_in_dim, 
                        hidden_dim=args.node_hidden_dim, 
                        num_class=num_classes, 
                        num_layers=args.num_layers, 
                        dropout=args.dropout)

        model_infer = GATInfer(in_dim=node_in_dim, 
                                hidden_dim=args.node_hidden_dim, 
                                num_class=num_classes, 
                                num_layers=args.num_layers, 
                                dropout=None)

    elif args.model == 'ECConv':

        model = MiniBatchECConvTrain(node_in_dim=node_in_dim, 
                                     hidden_dim=args.node_hidden_dim,
                                     edge_in_dim=edge_in_dim,
                                     num_class=num_classes,
                                     num_layers=args.num_layers,
                                     dropout=args.dropout)
        model_infer = MiniBatchECConvInfer(node_in_dim=node_in_dim, 
                                     hidden_dim=args.node_hidden_dim,
                                     edge_in_dim=edge_in_dim,
                                     num_class=num_classes,
                                     num_layers=args.num_layers,
                                     dropout=None)
    elif args.model == 'GTEA-ST':

        model = MiniBatchGTEASTTrain(node_in_dim=node_in_dim, 
                                     node_hidden_dim=args.node_hidden_dim,
                                     edge_in_dim=edge_in_dim,
                                     num_class=num_classes,
                                     num_layers=args.num_layers,
                                     device=device,
                                     dropout=args.dropout)
        model_infer = MiniBatchGTEASTInfer(node_in_dim=node_in_dim, 
                                     node_hidden_dim=args.node_hidden_dim,
                                     edge_in_dim=edge_in_dim,
                                     num_class=num_classes,
                                     num_layers=args.num_layers,
                                     device=device,
                                     dropout=None)
    elif args.model == 'EGNN':

        model = MiniBatchEGNNTrain(node_in_dim=node_in_dim, 
                                     node_hidden_dim=args.node_hidden_dim,
                                     edge_in_dim=edge_in_dim,
                                     num_class=num_classes,
                                     num_layers=args.num_layers,
                                     device=device,
                                     dropout=args.dropout)
        model_infer = MiniBatchEGNNInfer(node_in_dim=node_in_dim, 
                                     node_hidden_dim=args.node_hidden_dim,
                                     edge_in_dim=edge_in_dim,
                                     num_class=num_classes,
                                     num_layers=args.num_layers,
                                     device=device,
                                     dropout=None)


    elif args.model == 'GAT+T':
        model = GATPlusTTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       time_hidden_dim=args.time_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GATPlusTInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       time_hidden_dim=args.time_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'TGAT':
        model = TGATTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       time_hidden_dim=args.time_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = TGATInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       time_hidden_dim=args.time_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       device=infer_device,
                                       dropout=None)

    elif args.model == 'GTEA-LSTM':
        model = ATEdgeLSTMTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = ATEdgeLSTMInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'GTEA-TLSTM1':
        model = GTEATLSTM1Train(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GTEATLSTM1Infer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'GTEA-TLSTM2':
        model = GTEATLSTM2Train(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GTEATLSTM2Infer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'GTEA-TLSTM3':
        model = GTEATLSTM3Train(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GTEATLSTM3Infer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'GTEA-LSTM+T2V':
        model = ETLSTMTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = ETLSTMInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'GTEA-Trans':
        model = ETransformerTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = ETransformerInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=infer_device, 
                                       dropout=None)
    elif args.model == 'GTEA-Trans+T2V':
        model = T2VTransformerTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = T2VTransformerInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=infer_device, 
                                       dropout=None)
    elif args.model == 'GTEA-Trans-V':
        model = GTEATransVTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GTEATransVInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=infer_device, 
                                       dropout=None)
    elif args.model == 'GTEA-LSTM-V':
        model = GTEALSTMVTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GTEALSTMVInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim, 
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    elif args.model == 'GTEA-Trans+T2V-V':
        model = GTEATransT2V_VTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=device, 
                                       dropout=args.dropout)

        model_infer =  GTEATransT2V_VInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_heads=args.num_heads,
                                       num_encoder_layers=args.num_lstm_layers, 
                                       device=infer_device, 
                                       dropout=None)
    elif args.model == 'GTEA-LSTM+T2V-V':
        model = GTEALSTMT2V_VTrain(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional,
                                       device=device, 
                                       dropout=args.dropout)

        model_infer = GTEALSTMT2V_VInfer(node_in_dim=node_in_dim, 
                                       node_hidden_dim=args.node_hidden_dim,
                                       edge_in_dim=edge_in_dim-1, 
                                       time_hidden_dim=args.time_hidden_dim,
                                       num_class=num_classes, 
                                       num_layers=args.num_layers, 
                                       num_lstm_layers=args.num_lstm_layers, 
                                       bidirectional=args.bidirectional, 
                                       device=infer_device,
                                       dropout=None)
    else:
        logging.info('The model \"{}\" is not implemented'.format(args.model))
        sys.exit(0)
    
    # send model to device
    model.to(device)
    model_infer.to(infer_device)

    # create optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn =  torch.nn.CrossEntropyLoss()


    # compute graph degrea 
    g.readonly()
    norm = 1. / g.in_degrees().float().unsqueeze(1)
    g.ndata['norm'] = norm
    # g.ndata['node_features'] = features

    degs = g.in_degrees().numpy()
    degs[degs > args.num_neighbors] = args.num_neighbors
    g.ndata['subg_norm'] = torch.FloatTensor(1./degs).unsqueeze(1)


    # create trainer
    checkpoint_path = os.path.join(log_path, str(args.model) + '_checkpoint.pt')
    trainer = MiniBatchTrainer(  g=g, 
                                 model=model, 
                                 model_infer=model_infer, 
                                 loss_fn=loss_fn, 
                                 optimizer=optimizer, 
                                 epochs=args.epochs, 
                                 features=features, 
                                 labels=labels, 
                                 train_id=train_id, 
                                 val_id=val_id, 
                                 test_id=test_id,
                                 patience=args.patience, 
                                 batch_size=args.batch_size,
                                 test_batch_size=args.test_batch_size,
                                 num_neighbors=args.num_neighbors, 
                                 num_layers=args.num_layers, 
                                 num_cpu=args.num_cpu, 
                                 device=device,
                                 infer_device=infer_device, 
                                 log_path=log_path,
                                 checkpoint_path=checkpoint_path)


    logging.info('Start training')
    best_val_result, test_result = trainer.train()

    # recording the result
    line = [datetime.datetime.now().__str__()] + [args.model] + ['K=' + str(args.use_K)] + \
    [str(x) for x in best_val_result] + [str(x) for x in test_result] + [str(args)]
    line = ','.join(line) + '\n'

    with open(os.path.join(args.log_dir, str(args.model) + '_result.csv'), 'a') as f:
        f.write(line)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GTEA training')
    parser.add_argument("--data_dir", type=str, default='data/dynamic_eth',
            help="dataset name")
    parser.add_argument("--model", type=str, default='GTEA-Trans+T2V',
            help="dataset name")    
    parser.add_argument("--use_K", type=int, default=None,
            help="select K-fold id, range from 0 to K-1")
    parser.add_argument("--K", type=int, default=5,
            help="Number of K in K-fold")
    parser.add_argument("--dropout", type=float, default=None,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--infer_gpu", action='store_false',
            help="infer device same as training device (default True)")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
            help="batch size")
    parser.add_argument("--max_event", type=int, default=20,
            help="max_event")
    parser.add_argument("--test_batch_size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num_neighbors", type=int, default=5,
            help="number of neighbors to be sampled")
    parser.add_argument("--node_hidden_dim", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--time_hidden_dim", type=int, default=32,
            help="time layer dim")
    parser.add_argument("--num_layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
            help="number of hidden lstm layers")
    parser.add_argument("--num_heads", type=int, default=1,
            help="number of head for transformer")
    parser.add_argument("--bidirectional", type=bool, default=False,
            help="bidirectional lstm layer")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight_decay", type=float, default=0,
            help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=100,
            help="patience")
    parser.add_argument("--num_cpu", type=int, default=2,
            help="num_cpu")
    parser.add_argument("--log_dir", type=str, default='./experiment',
            help="experiment directory")
    parser.add_argument("--log_name", type=str, default='test',
            help="log directory name for this run")
    parser.add_argument("--remove_node_features", action='store_true')
    args = parser.parse_args()

    # print(args)

    main(args)