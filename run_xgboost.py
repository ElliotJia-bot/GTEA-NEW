import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse, time, math
import os
import pickle

from utils.metrics import accuracy, macro_f1

def load_data(data_dir):

    labels_pickle_path = os.path.join(data_dir, 'labels.pkl')
    feature_path = os.path.join(data_dir, 'features.npz')

    with open(labels_pickle_path, 'rb') as f:
        labels, train_mask, val_mask, test_mask = pickle.load(f)

    train_mask = train_mask.astype(np.bool)
    val_mask = val_mask.astype(np.bool)
    test_mask = test_mask.astype(np.bool)

    data = np.load(feature_path)
    node_features = data['node_features']



    x_train, x_val, x_test = node_features[train_mask], node_features[val_mask], node_features[test_mask]

    y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]

    return x_train, x_val, x_test, y_train, y_val, y_test

def main(args):

    data_dir = args.data_dir

    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir)

    print('data information', x_train.shape, x_val.shape, x_test.shape)


    eta_list = [0.01, 0.05, 0.1, 0.2]
    max_depth_list = [5, 7, 10]
    lambda_list = [0.1, 0.5, 1]
    model_save_path = os.path.join(args.log_dir, args.model_save_path)
    best_val_acc = 0
    best_val_f1 = 0
    print('Start training')

    for eta in eta_list:
        for max_depth in max_depth_list:
            for lambda_ in lambda_list:    

                model = XGBClassifier(gpu_id=args.gpu, n_jobs=args.num_cpu, eta=eta, max_depth=max_depth, reg_lambda =lambda_)
                model.fit(x_train, y_train)

                pred_y = model.predict(x_train)
                train_acc = accuracy(pred_y, y_train)
                train_f1 = macro_f1(pred_y, y_train)
                print('Train set: Accuracy: {:.4} F1: {:.4}'.format(train_acc, train_f1))

                pred_y = model.predict(x_val)
                val_acc = accuracy(pred_y, y_val)
                val_f1 = macro_f1(pred_y, y_val)
                print('Val set: Accuracy: {:.4} F1: {:.4}'.format(val_acc, val_f1))

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    pickle.dump(model, open(model_save_path, "wb"))

    print('Finishing training...')

    loaded_model = pickle.load(open(model_save_path, "rb"))

    pred_y = loaded_model.predict(x_test)
    test_acc = accuracy(pred_y, y_test)
    test_f1 = macro_f1(pred_y, y_test)
    print('Test set: Accuracy: {:.4} F1: {:.4}'.format(test_acc, test_f1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTEA training')
    parser.add_argument("--data_dir", type=str, default='data/dynamic_eth',
            help="dataset name")
    parser.add_argument("--num_cpu", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default='experiment')
    parser.add_argument("--model_save_path", type=str, default='xgboost.dat')

    args = parser.parse_args()
    main(args)


