import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import numpy as np
from utils.metrics import score
from utils.torch_utils import EarlyStopping
from collections import OrderedDict
class Trainer(object):
    def __init__(self, model, loss_fn, optimizer, g, features, labels, train_mask, val_mask, test_mask, batch_size, device, epochs, patience, log_path, checkpoint_path):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.g = g
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs 
        self.patience = patience   

        self.log_path = log_path    
        self.checkpoint_path = checkpoint_path 

        # initialize early stopping object
        self.early_stopping = EarlyStopping(patience=patience, log_dir=self.log_path, verbose=True)

    def train(self):       

        dur_time = []
        best_result = [0, 0, 0, 0]
        best_val_acc = 0

        print('Start training......')
        for e in range(self.epochs):

            start_t = time.time()

            logits = self.model(self.g, self.features)

            loss = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            end_t= time.time()
            dur_time.append(end_t - start_t)


            with torch.no_grad():
            # print(F.softmax(logits.detach(), 1)[:20])
                train_loss = loss.cpu().item()
                train_acc, train_prec, train_recall, train_f1 = score(logits[self.train_mask], self.labels[self.train_mask])
     
                val_loss = self.loss_fn(logits[self.val_mask], self.labels[self.val_mask])
                val_acc, val_prec, val_recall, val_f1 = score(logits[self.val_mask], self.labels[self.val_mask])

                logging.info("Epoch {:05d} | Time(s) {:.4f} | \n"
                "TrainLoss {:.4f} | TrainAcc {:.4f} | TrainPrecision {:.4f} | TrainRecall {:.4f} | TrainMacroF1 {:.4f}\n"
                "ValLoss {:.4f}   | ValAcc {:.4f}   | ValPrecision {:.4f}    | ValRecall {:.4f}   | ValMacroF1 {:.4f}\n"
                .format(e, dur_time[-1], 
                       train_loss, train_acc, train_prec, train_recall, train_f1, 
                       val_loss, val_acc, val_prec, val_recall, val_f1))

                if val_acc > best_val_acc:
                    best_result = [val_acc, val_prec, val_recall, val_f1]
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), self.checkpoint_path)

                self.early_stopping(val_acc, self.model)
                if self.early_stopping.early_stop:
                    logging.info("Early stopping")
                    break

        print('Finishing training. Average time per epoch: {:.4f}'.format(np.mean(dur_time)))
        print('Best result: ValAcc {:.4f}   | ValPrecision {:.4f}    | ValRecall {:.4f}   | ValMacroF1 {:.4f}'
            .format(best_result[0], best_result[1], best_result[2], best_result[3]))

        model_state_dict = {k:v.to(self.device) for k, v in torch.load(self.checkpoint_path).items()}
        model_state_dict = OrderedDict(model_state_dict)
        
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.g, self.features)
            test_loss = self.loss_fn(logits[self.test_mask], self.labels[self.test_mask])
            test_acc, test_prec, test_recall, test_f1 = score(logits[self.test_mask], self.labels[self.test_mask])

            logging.info("Epoch {:05d} | Time(s) {:.4f} | \n"
            "Test set: TestAcc {:.4f} | TestPrecision {:.4f} | TestRecall {:.4f} | TestMacroF1 {:.4f}\n"
            .format(e, dur_time[-1], 
                   test_acc, test_prec, test_recall, test_f1))

