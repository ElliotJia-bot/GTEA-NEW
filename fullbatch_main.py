import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import os
import argparse, time, math
import logging
import sys

from models import GCN, JKNet, APPNP, EGNN
from trainer import Trainer
from data.dynamic_event_data_loader import Dataset
import numpy as np




def main(args):

    # g, features, labels, train_mask, val_mask, test_mask = load_cora_data()


    # in_dim = features.shape[1]
    # hidden_dim = args.node_hidden_dim
    # out_dim = len(torch.unique(labels))
    # num_hops = args.num_hops

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

    
    data = Dataset(data_dir=args.data_dir, max_event=1, load_static_edge=True)


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


    # setting parameter
    in_dim = node_in_dim
    hidden_dim = args.node_hidden_dim
    out_dim = num_classes
    num_hops = args.num_layers
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() and args.gpu >=0 else "cpu")

    
    if args.model == 'GCN':
        model = GCN(in_dim=in_dim, 
                    hidden_dim=hidden_dim, 
                    out_dim=num_classes, 
                    num_hops=num_hops)
    elif args.model == 'JKNet':
        model = JKNet(in_dim=in_dim, 
                    hidden_dim=hidden_dim, 
                    out_dim=num_classes, 
                    num_hops=num_hops)
    elif args.model =='APPNP':
        model = APPNP(in_feats=in_dim,
                 hiddens=hidden_dim,
                 n_classes=num_classes,
                 activation=F.relu,
                 feat_drop=args.dropout,
                 edge_drop=args.dropout,
                 num_hops=num_hops,
                 alpha=0.1,
                 k=10)
    elif args.model == 'EGNN':
        model = EGNN(node_in_dim=in_dim, 
                    edge_in_dim=edge_in_dim,
                    hidden_dim=hidden_dim, 
                    out_dim=num_classes, 
                    num_hops=num_hops)
    else:
        print('Model {} not found.'.format(args.model))
        exit(0)

    loss_fn =  torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(device)
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)


    checkpoint_path = os.path.join(args.log_dir, str(args.model) + '_checkpoint.pt')

    trainer = Trainer(model=model,
                      loss_fn=loss_fn, 
                      optimizer=optimizer, 
                      g=g, 
                      features=features, 
                      labels=labels, 
                      train_mask=train_mask, 
                      val_mask=val_mask, 
                      test_mask=test_mask, 
                      batch_size=args.batch_size,
                      device=device,
                      epochs=args.epochs,
                      patience=args.patience,
                      log_path=log_path,
                      checkpoint_path=checkpoint_path)

    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='static graph training')
    parser.add_argument("--data_dir", type=str, default='data/dynamic_eth',
            help="dataset name")
    parser.add_argument("--model", type=str, default='GCN',
            help="dataset name")    
    parser.add_argument("--dropout", type=float, default=None,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--infer_gpu", type=bool, default=True,
            help="infer device same as training device (default True)")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
            help="batch size")
    parser.add_argument("--node_hidden_dim", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--num_layers", type=int, default=1,
            help="number of hidden gcn layers")
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

    args = parser.parse_args()

    

    main(args)
