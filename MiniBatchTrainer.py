import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import logging
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
from collections import OrderedDict
import matplotlib.pyplot as plt

from utils.metrics import torch_accuracy, accuracy, micro_f1, macro_f1, hamming_loss, micro_precision, micro_recall, macro_precision, macro_recall
from utils.torch_utils import EarlyStopping
from dgl.contrib.sampling import NeighborSampler
class MiniBatchTrainer(object):
    def __init__(self, 
                 g, 
                 model, 
                 model_infer, 
                 loss_fn, 
                 optimizer, 
                 epochs, 
                 features, 
                 labels, 
                 train_id, 
                 val_id, 
                 test_id,
                 patience, 
                 batch_size, 
                 test_batch_size, 
                 num_neighbors, 
                 num_layers, 
                 num_cpu, 
                 device, 
                 infer_device,
                 log_path,
                 checkpoint_path):

        self.g = g
        self.model = model
        self.model_infer = model_infer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_id = train_id
        self.val_id = val_id ####
        self.test_id = test_id####
        self.epochs = epochs
        self.features = features
        self.labels = labels
        # if use_tensorboardx:
        #     self.writer = SummaryWriter('/tmp/tensorboardx')
        self.patience = patience
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.num_cpu = num_cpu
        self.device = device
        self.infer_device = infer_device   
        self.log_path = log_path    
        self.checkpoint_path = checkpoint_path 

        # initialize early stopping object
        self.early_stopping = EarlyStopping(patience=patience, log_dir=self.log_path, verbose=True)


    def train(self):

        dur = []
        train_losses = []  # per mini-batch
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_val_acc = -1
        best_val_result = (0, 0, 0, 0)
        best_val_y = None

        for e in range(self.epochs):

            train_losses_temp = []
            train_accuracies_temp = []
            val_losses_temp = []
            val_accuracies_temp = []

            # minibatch train
            train_num_correct = 0  # number of correct prediction in validation set
            train_total_losses = 0  # total cross entropy loss
            if e >= 2:
                t0 = time.time()
            pred_temp = np.array([])
            label_temp = np.array([])


            for nf in NeighborSampler(self.g, 
                                        batch_size=self.batch_size,
                                        expand_factor=self.num_neighbors,
                                        neighbor_type='in',
                                        shuffle=True,
                                        num_hops=self.num_layers,
                                        add_self_loop=True,
                                        seed_nodes=self.train_id, 
                                        num_workers=self.num_cpu):

                # node_embed_names = [['node_features']]
                # edge_embed_names = [['edge_features_from', 'edge_features_to']]
                # for i in range(self.num_layers):
                #     node_embed_names.append(['norm', 'subg_norm'])
                # for i in range(1, self.num_layers):
                #     edge_embed_names.append(['edge_features_from', 'edge_features_to'])
                # nf.copy_from_parent(node_embed_names=node_embed_names, 
                #                     edge_embed_names=edge_embed_names,
                #                     ctx=self.device)
                nf.copy_from_parent(ctx=self.device)

                self.model.train()
                logits = self.model(nf)      

                batch_nids = nf.layer_parent_nid(-1).to(device=self.device, dtype=torch.long)
                batch_labels = self.labels[batch_nids].to(device=self.device)
                # print(logits.shape, batch_labels.shape)

                batch_size = len(batch_nids)

                # update step
                train_loss = self.loss_fn(logits, batch_labels)
                self.optimizer.zero_grad()
                train_loss.backward()

                nn.utils.clip_grad_norm(self.model.parameters(), 5)


                self.optimizer.step()

                mini_batch_accuracy = torch_accuracy(logits, batch_labels)
                train_num_correct += mini_batch_accuracy * batch_size                
                train_total_losses += (train_loss.item() * batch_size)

                _, indicies = torch.max(logits, dim=1)
                pred = indicies.cpu().detach().numpy()
                pred_temp = np.append(pred_temp, pred)
                label_temp = np.append(label_temp, batch_labels.cpu())

                # train_loss.append(loss.cpu().item())
                # train_num_correct += (logits.argmax(dim=1) == batch_labels).sum().cpu().item()


            # loss and accuracy of this epoch
            train_average_loss = train_total_losses / len(self.train_id)
            train_losses.append(train_average_loss)
            train_accuracy = train_num_correct / len(self.train_id)
            train_accuracies.append(train_accuracy)
            train_macro_precision = macro_precision(pred_temp, label_temp)
            train_macro_recall = macro_recall(pred_temp, label_temp)
            train_micro_f1 = micro_f1(pred_temp, label_temp)
            train_macro_f1 = macro_f1(pred_temp, label_temp)

            if e >= 2:
                dur.append(time.time() - t0)

            pred_temp = np.array([])
            label_temp = np.array([])


            # validation
            # if e % 5 != 0:
            #     continue

            val_num_correct = 0  # number of correct prediction in validation set
            val_total_losses = 0  # total cross entropy loss

            # infer_device = torch.device('cpu')
            infer_device = self.infer_device
            model_state_dict = {k:v.to(infer_device) for k, v in self.model.state_dict().items()}
            model_state_dict = OrderedDict(model_state_dict)
            self.model_infer = self.model_infer.to(infer_device)
            self.model_infer.load_state_dict(model_state_dict)

            for nf in NeighborSampler(self.g, 
                                        batch_size=self.batch_size,
                                        expand_factor=self.g.number_of_nodes(),
                                        neighbor_type='in',
                                        num_hops=self.num_layers,
                                        seed_nodes=self.val_id,
                                        add_self_loop=True, 
                                        num_workers=self.num_cpu):



                ## do not copy infer model to GPU
                # nf.copy_from_parent(ctx=self.device)
                # infer_device = torch.device('cpu')
                
                nf.copy_from_parent(ctx=infer_device)

                

                self.model_infer.eval()
                with torch.no_grad():
                    logits = self.model_infer(nf)

                    batch_nids = nf.layer_parent_nid(-1).to(device=infer_device, dtype=torch.long)
                    batch_labels = self.labels[batch_nids].to(device=infer_device)

                    batch_size = len(batch_nids)

                    mini_batch_val_loss = self.loss_fn(logits, batch_labels)

                    # print(mini_batch_val_loss.item(), self.batch_size)
                    mini_batch_accuracy = torch_accuracy(logits, batch_labels)
                    val_num_correct += mini_batch_accuracy * batch_size                    
                    val_total_losses += mini_batch_val_loss.cpu().item() * batch_size

                    _, indicies = torch.max(logits, dim=1)
                    pred = indicies.cpu().detach().numpy()
                    pred_temp = np.append(pred_temp, pred)
                    label_temp = np.append(label_temp, batch_labels.cpu())
                    # val_total_losses += 

            val_average_loss = val_total_losses / len(self.val_id)
            val_losses.append(val_average_loss)
            val_accuracy = val_num_correct / len(self.val_id)
            val_accuracies.append(val_accuracy)
            val_macro_precision = macro_precision(pred_temp, label_temp)
            val_macro_recall = macro_recall(pred_temp, label_temp)
            val_micro_f1 = micro_f1(pred_temp, label_temp)
            val_macro_f1 = macro_f1(pred_temp, label_temp)


            

            if val_accuracy > best_val_acc:
                best_val_result = (val_accuracy, val_macro_precision, val_macro_recall, val_macro_f1, val_micro_f1)
                best_val_acc = val_accuracy
                best_val_y = (pred_temp, label_temp)
                torch.save(self.model.state_dict(), self.checkpoint_path)

            logging.info("Epoch {:05d} | Time(s) {:.4f} | \n"
                "TrainLoss {:.4f} | TrainAcc {:.4f} | TrainPrecision {:.4f} | TrainRecall {:.4f} | TrainMacroF1 {:.4f}\n"
                "ValLoss {:.4f}   | ValAcc {:.4f}   | ValPrecision {:.4f}    | ValRecall {:.4f}   | ValMacroF1 {:.4f}\n"
                "ETputs(KTEPS) {:.2f}\n".
                format(e, np.mean(dur), 
                       train_average_loss, train_accuracy, train_macro_precision, train_macro_recall, train_macro_f1,  
                       val_average_loss, val_accuracy, val_macro_precision, val_macro_recall, val_macro_f1,  
                       self.g.number_of_edges() / np.mean(dur) / 1000))


            # early stopping
            self.early_stopping(val_accuracy, self.model_infer)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

            
        logging.info('Best val result: ValAcc {:.4f}   | ValPrecision {:.4f}    | ValRecall {:.4f}   | ValMacroF1 {:.4f}\n'
            .format(best_val_result[0], best_val_result[1], best_val_result[2], best_val_result[3]))
        
        logging.info(classification_report(best_val_y[1], best_val_y[0], digits=6))
        logging.info(confusion_matrix(best_val_y[1], best_val_y[0]))

        
        pred_temp = np.array([])
        label_temp = np.array([])


        # test

        test_num_correct = 0  # number of correct prediction in test set
        test_total_losses = 0  # total cross entropy loss

        model_state_dict = {k:v.to(infer_device) for k, v in torch.load(self.checkpoint_path).items()}
        model_state_dict = OrderedDict(model_state_dict)
        # self.model_infer = self.model_infer.to(infer_device)
        self.model_infer.load_state_dict(model_state_dict)
        


        for nf in NeighborSampler(self.g, 
                                    batch_size=self.batch_size,
                                    expand_factor=self.g.number_of_nodes(),
                                    neighbor_type='in',
                                    num_hops=self.num_layers,
                                    seed_nodes=self.test_id,
                                    add_self_loop=True, 
                                    num_workers=self.num_cpu):



            ## do not copy infer model to GPU
            # nf.copy_from_parent(ctx=self.device)
            # infer_device = torch.device('cpu')
            # infer_device = torch.device('cpu')
            nf.copy_from_parent(ctx=infer_device)

            
            self.model_infer.eval()
            with torch.no_grad():
                logits = self.model_infer(nf)

                batch_nids = nf.layer_parent_nid(-1).to(device=infer_device, dtype=torch.long)
                batch_labels = self.labels[batch_nids].to(device=infer_device)

                batch_size = len(batch_nids)

                mini_batch_test_loss = self.loss_fn(logits, batch_labels)

                
                mini_batch_accuracy = torch_accuracy(logits, batch_labels)
                test_num_correct += mini_batch_accuracy * batch_size                  
                test_total_losses += mini_batch_test_loss.cpu().item() * batch_size

                _, indicies = torch.max(logits, dim=1)
                pred = indicies.cpu().detach().numpy()
                pred_temp = np.append(pred_temp, pred)
                label_temp = np.append(label_temp, batch_labels.cpu())
                

        test_average_loss = test_total_losses / len(self.test_id)
        # test_losses.append(test_average_loss)
        test_accuracy = test_num_correct / len(self.test_id)
        test_macro_precision = macro_precision(pred_temp, label_temp)
        test_macro_recall = macro_recall(pred_temp, label_temp)
        test_macro_f1 = macro_f1(pred_temp, label_temp)

        test_result = (test_average_loss, test_accuracy, test_macro_precision, test_macro_recall, test_macro_f1)

        logging.info('Test set: TestAcc {:.4f}   | TestPrecision {:.4f}    | TestRecall {:.4f}   | TestMacroF1 {:.4f}\n'
            .format(test_accuracy, test_macro_precision, test_macro_recall, test_macro_f1))

        logging.info(classification_report(label_temp, pred_temp, digits=6))
        logging.info(confusion_matrix(label_temp, pred_temp))

        self.plot(train_losses, val_losses, train_accuracies, val_accuracies)

        return best_val_result, test_result


    def plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        #####################################################################
        ##################### PLOT ##########################################
        #####################################################################
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_losses)+1),np.log(train_losses), label='Training Loss')
        plt.plot(range(1,len(val_losses)+1),np.log(val_losses),label='Validation Loss')

        # find position of lowest validation loss
        # minposs = val_losses.index(min(val_losses))+1 
        # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('log cross entropy loss')
        plt.xlim(0, len(train_losses)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.log_path, 'loss_plot.png'), bbox_inches='tight')


        # accuracy plot
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_accuracies)+1),train_accuracies, label='Training accuracies')
        plt.plot(range(1,len(val_accuracies)+1),val_accuracies,label='Validation accuracies')

        # find position of lowest validation loss
        # minposs = val_losses.index(min(val_losses))+1 
        # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('accuracies')
        plt.xlim(0, len(train_accuracies)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.log_path, 'accuracies_plot.png'), bbox_inches='tight')




