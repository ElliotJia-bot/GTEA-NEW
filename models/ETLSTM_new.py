import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from .TimeModel import LSTM
from .TimeModel import SineActivation

class NodeUpdate(nn.Module):
    def __init__(self, in_dim, out_dim, test=False):
        super(NodeUpdate, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.test = test

        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, h, block):
        """
        前向传播函数
        
        参数：
            h: 输入特征张量
            block: DGL块，包含归一化系数等信息
            
        返回：
            torch.Tensor: 更新后的节点特征
        """
        self_h = block.dstdata['self_h']
        self_h_tmp = block.dstdata['self_h_tmp']
        
        # 根据是否为测试模式选择不同的归一化方式
        if self.test:
            h = (h - self_h_tmp) * block.dstdata['norm']
        else:
            h = (h - self_h_tmp) * block.dstdata['subg_norm']

        # 连接自身特征和聚合特征
        h = torch.cat((self_h, h), dim=1)
        
        # 通过线性层和ReLU激活函数
        h = F.relu(self.layer(h))
        return h


class ETLSTMTrain(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_lstm_layers, bidirectional, device, dropout=None):
        super(ETLSTMTrain, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_lstm_layers = num_lstm_layers
        self.device = device
        self.test = False  # 添加 test 属性，训练模式设为 False

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.time_layers = nn.ModuleList()
        self.time_layers.append(SineActivation(1, time_hidden_dim))

        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(LSTM(edge_in_dim + time_hidden_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))


        self.edge_out_layers = nn.ModuleList()
        self.edge_out_layers.append(nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim))


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim + node_hidden_dim, node_hidden_dim))

        for i in range(1, self.num_layers):
            self.time_layers.append(SineActivation(1, time_hidden_dim))
            self.edge_layers.append(LSTM(edge_in_dim + time_hidden_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))
            self.edge_out_layers.append(nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            self.node_layers.append(NodeUpdate(2 * node_hidden_dim, node_hidden_dim))


        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, blocks, features):
        """
        前向传播函数
        
        参数：
            blocks: DGL的块列表，包含每一层的计算图
            features: 输入节点的特征
            
        返回：
            torch.Tensor: 节点的预测类别概率
        """
        h = features

        for i, (time_layer, edge_layer, edge_out_layer, node_layer) in enumerate(zip(self.time_layers, self.edge_layers, self.edge_out_layers, self.node_layers)):
            block = blocks[i]
            
            # 获取边的特征
            e = block.edata['edge_features']
            e_len = block.edata['edge_len']
            e_times = block.edata['seq_times']
            
            # 应用dropout
            e = self.dropout_layer(e)
            h = self.dropout_layer(h)
            
            # 存储特征到块中
            block.srcdata['h'] = h
            block.edata['e'] = e
            block.edata['e_len'] = e_len
            block.edata['e_times'] = e_times

            # 获取目标节点的特征
            self_h = h[:block.num_dst_nodes()]
            
            # 处理自身特征
            tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            self_h_tmp = torch.cat((self_h, tmp), dim=1)
            self_h_tmp = edge_out_layer(self_h_tmp)

            # 存储节点相关信息
            block.dstdata['self_h'] = self_h
            block.dstdata['self_h_tmp'] = self_h_tmp
            if not self.test:
                block.dstdata['subg_norm'] = block.dstdata['subg_norm']

            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times'] 

                num_edges = e_times.shape[0]                

                # 时间编码
                e_times = time_layer(e_times.reshape(-1, 1))
                e_times = e_times.reshape(num_edges, -1, self.time_hidden_dim)  
                
                # 组合边特征和时间特征
                e = torch.cat((e, e_times), dim=-1)
                
                # 使用LSTM处理时序特征
                e = edge_layer(e, e_len)

                # 组合并转换特征
                h = edge_out_layer(torch.cat((h, e), dim=1))
                h = F.relu(h)
                return {'m': h}

            def reduce_func(nodes):
                m = nodes.mailbox['m']                
                h = torch.sum(m, dim=1)
                return {'h': h}

            # 执行消息传递
            block.update_all(message_func, reduce_func)
            
            # 获取更新后的特征
            h = block.dstdata.pop('h')
            
            # 通过节点更新层，传入特征和块
            h = node_layer(h, block)

        # 最终的分类层
        h = self.fc(h)
        return h

class ETLSTMInfer(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim, time_hidden_dim, num_class, num_layers, num_lstm_layers, bidirectional, device, dropout=None):
        super(ETLSTMInfer, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.edge_in_dim = edge_in_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_lstm_layers = num_lstm_layers
        self.device = device
        self.test = True  # 添加 test 属性，设置为 True 表示推理模式

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        self.time_layers = nn.ModuleList()
        self.time_layers.append(SineActivation(1, time_hidden_dim))

        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(LSTM(edge_in_dim + time_hidden_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))


        self.edge_out_layers = nn.ModuleList()
        self.edge_out_layers.append(nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim))


        self.node_layers =  nn.ModuleList()
        self.node_layers.append(NodeUpdate(node_in_dim + node_hidden_dim, node_hidden_dim, test=True))

        for i in range(1, self.num_layers):
            self.time_layers.append(SineActivation(1, time_hidden_dim))
            self.edge_layers.append(LSTM(edge_in_dim + time_hidden_dim, node_hidden_dim, num_lstm_layers, device, bidirectional=bidirectional))
            self.edge_out_layers.append(nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            self.node_layers.append(NodeUpdate(2 * node_hidden_dim, node_hidden_dim, test=True))


        self.fc = nn.Linear(node_hidden_dim, num_class)

    def forward(self, blocks, features):
        """
        前向传播函数
        
        参数：
            blocks: DGL的块列表，包含每一层的计算图
            features: 输入节点的特征
            
        返回：
            torch.Tensor: 节点的预测类别概率
        """
        h = features

        for i, (time_layer, edge_layer, edge_out_layer, node_layer) in enumerate(zip(self.time_layers, self.edge_layers, self.edge_out_layers, self.node_layers)):
            block = blocks[i]
            
            # 获取边的特征
            e = block.edata['edge_features']
            e_len = block.edata['edge_len']
            e_times = block.edata['seq_times']
            
            # 应用dropout
            e = self.dropout_layer(e)
            h = self.dropout_layer(h)
            
            # 存储特征到块中
            block.srcdata['h'] = h
            block.edata['e'] = e
            block.edata['e_len'] = e_len
            block.edata['e_times'] = e_times

            # 获取目标节点的特征
            self_h = h[:block.num_dst_nodes()]
            
            # 处理自身特征
            tmp = torch.zeros((self_h.shape[0], self.node_hidden_dim), device=self.device)
            self_h_tmp = torch.cat((self_h, tmp), dim=1)
            self_h_tmp = edge_out_layer(self_h_tmp)

            # 存储节点相关信息
            block.dstdata['self_h'] = self_h
            block.dstdata['self_h_tmp'] = self_h_tmp
            if self.test:
                block.dstdata['norm'] = block.dstdata['norm']

            def message_func(edges):
                h = edges.src['h']
                e = edges.data['e']    
                e_len = edges.data['e_len']    
                e_times = edges.data['e_times'] 

                num_edges = e_times.shape[0]
                
                # 时间编码
                e_times = time_layer(e_times.reshape(-1, 1)).reshape(num_edges, -1, self.time_hidden_dim)    
                
                # 组合边特征和时间特征
                e = torch.cat((e, e_times), dim=-1)
                e = edge_layer(e, e_len)

                # 组合并转换特征
                h = edge_out_layer(torch.cat((h, e), dim=1))
                h = F.relu(h)
                return {'m': h}

            def reduce_func(nodes):
                m = nodes.mailbox['m']                
                h = torch.sum(m, dim=1)
                return {'h': h}

            # 执行消息传递
            block.update_all(message_func, reduce_func)
            
            # 获取更新后的特征
            h = block.dstdata.pop('h')
            
            # 通过节点更新层，传入特征和块
            h = node_layer(h, block)

        # 最终的分类层
        h = self.fc(h)
        return h