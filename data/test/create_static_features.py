import numpy as np
import pandas as pd
import dgl
import os
from sklearn import preprocessing
import torch
import pickle
from datetime import datetime

money_dim = 0

def main():

    data = np.load('features.npz')
    edge_features, edge_len = data['edge_features'], data['edge_len']


    static_edge_features = []
    for i in range(len(edge_features)):

        feat = edge_features[i][:edge_len[i]]
        feat_ = feat[:, money_dim]
        static_edge_features.append([feat_.mean(0), feat_.std(0), feat_.min(0), feat_.max(0)])

    static_edge_features = np.stack(static_edge_features, 0)
    print(static_edge_features)

    np.save('static_edge_features.npy', static_edge_features)

if __name__ == '__main__':
    main()
