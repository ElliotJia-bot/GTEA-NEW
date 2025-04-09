import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
import dgl  # 导入DGL库，用于图神经网络
import os  # 导入os库，用于操作系统相关功能
from sklearn import preprocessing  # 从sklearn导入预处理模块
import torch  # 导入PyTorch库，用于深度学习
import pickle  # 导入pickle库，用于序列化和反序列化对象
from datetime import datetime  # 导入datetime模块，用于处理日期和时间
import gc  # 导入gc库，用于垃圾回收

max_event = 60  # 定义最大事件数，表示在处理每条边时最多考虑的事件数量
num_edge_feats = 6  # 定义边特征的维度

# 定义时间范围的起始和结束时间
start_date = datetime(2001, 12, 26, 0, 0).timestamp()  # 起始时间戳
end_date = datetime(2025, 12, 3, 0, 0).timestamp()  # 结束时间戳

def read_lines(fname):
    # 读取文件中的所有行
    with open(fname) as f:  # 打开文件
        return f.readlines()  # 返回文件的所有行

def padding_tensor(sequences, max_len=None):
    """
    对序列进行填充，使其长度一致
    :param sequences: list of tensors，输入的张量列表
    :param max_len: 最大长度，如果未提供，则自动计算
    :return: 填充后的张量和掩码
    """
    sequences = torch.tensor(sequences, dtype=torch.float32)  # 将输入序列转换为张量
    num = len(sequences)  # 获取序列的数量
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])  # 如果未提供最大长度，则计算最大长度
    out_dims = (num, max_len)  # 输出张量的维度
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)  # 创建一个全零的输出张量
    mask = sequences[0].data.new(*out_dims).fill_(0)  # 创建一个全零的掩码
    for i, tensor in enumerate(sequences):  # 遍历输入序列
        length = tensor.size(0)  # 获取当前序列的长度
        out_tensor[i, :length] = tensor  # 将当前序列填充到输出张量中
        mask[i, :length] = 1  # 在掩码中标记有效长度
    return out_tensor, mask  # 返回填充后的张量和掩码

def padding_sequences(sequences, max_len):
    """
    对序列进行填充，使其长度为max_len
    :param sequences: numpy array，输入的序列
    :param max_len: 最大长度
    :return: 填充后的序列和原序列的长度
    """
    new_s = np.zeros((max_len, sequences.shape[1]))  # 创建一个全零的数组
    new_s[:len(sequences), :] = sequences  # 将原序列填充到新数组中
    return new_s, len(sequences)  # 返回填充后的数组和原序列长度

def agg_fun(x):
    """
    聚合函数，按时间排序并截取前max_event个事件
    :param x: 输入的DataFrame
    :return: 处理后的NumPy数组
    """
    x = x.sort_values(['time'])  # 按时间排序
    length = min(len(x), max_event)  # 取最小值，确保不超过max_event
    x = x.to_numpy()[:length, 2:]  # 转换为NumPy数组并截取
    return x  # 返回处理后的数组

def process_adj_in_batches(num_nodes, labels, batch_size=1000000):
    """
    使用批处理方式处理邻接矩阵
    """
    print('Start reading adj')
    # 使用chunksize参数分批读取
    adj_chunks = pd.read_csv('adj.csv', sep=',', chunksize=batch_size)
    
    # 初始化存储结构
    all_edge_features = []
    all_edge_from_id = []
    all_edge_to_id = []
    all_edge_len = []
    all_seq_times = []
    
    features_nodes = set(range(num_nodes))
    first_chunk = True
    scaler = None
    
    for chunk_idx, adj in enumerate(adj_chunks):
        print(f'Processing chunk {chunk_idx}')
        
        # 重命名列
        adj = adj.rename(columns={'timestamp':'time'})
        
        # 过滤节点
        cond1 = adj['srcId'].isin(features_nodes)
        cond2 = adj['dstId'].isin(features_nodes)
        adj = adj.loc[cond1 & cond2]
        
        if len(adj) == 0:
            continue
            
        # 第一个chunk用于拟合scaler
        if first_chunk:
            scaler = preprocessing.StandardScaler().fit(adj[adj.columns[2:]].values)
            first_chunk = False
            
        # 标准化特征
        adj[adj.columns[2:]] = scaler.transform(adj[adj.columns[2:]])
        
        # 创建反向边
        adj_reverse = adj.copy()
        adj_reverse[['srcId', 'dstId']] = adj_reverse[['dstId', 'srcId']]
        adj['direction'] = 0
        adj_reverse['direction'] = 1
        
        # 合并正向和反向边
        adj_ = pd.concat([adj, adj_reverse], ignore_index=True)
        del adj_reverse  # 释放内存
        
        # 按时间排序并分组
        adj_ = adj_.sort_values(['srcId', 'dstId', 'time'])
        
        # 使用numpy操作代替pandas groupby
        edge_groups = []
        current_group = []
        current_key = None
        
        # 使用numpy数组进行快速迭代
        data = adj_.values
        for row in data:
            key = (row[0], row[1])  # srcId, dstId
            
            if current_key != key:
                if current_group:
                    edge_groups.append((current_key, np.array(current_group)))
                current_group = [row[2:]]  # 特征从第3列开始
                current_key = key
            else:
                if len(current_group) < max_event:
                    current_group.append(row[2:])
        
        if current_group:
            edge_groups.append((current_key, np.array(current_group)))
        
        # 处理每组边
        for (src, dst), group_data in edge_groups:
            num_transactions = len(group_data)
            
            if num_transactions > 0:
                # 填充到固定长度
                padded_features = np.zeros((max_event, group_data.shape[1]))
                padded_features[:num_transactions] = group_data
                
                all_edge_from_id.append(src)
                all_edge_to_id.append(dst)
                all_edge_features.append(padded_features)
                all_edge_len.append(num_transactions)
                
                # 提取时间序列
                times = group_data[:, -1]  # 假设时间是最后一列
                padded_times = np.zeros(max_event)
                padded_times[:len(times)] = times
                all_seq_times.append(padded_times)
        
        # 清理内存
        del adj_, edge_groups
        gc.collect()
    
    # 合并所有批次的结果
    edge_features = np.array(all_edge_features)
    edge_len = np.array(all_edge_len, dtype=np.int32)
    seq_times = np.array(all_seq_times)
    edge_from_id = np.array(all_edge_from_id, dtype=np.int32)
    edge_to_id = np.array(all_edge_to_id, dtype=np.int32)
    
    print('Edge information:', edge_features.shape, len(edge_from_id), len(edge_to_id))
    
    return edge_from_id, edge_to_id, edge_features, edge_len, seq_times

def process_features():
    """
    处理节点特征
    :return: 处理后的节点特征
    """
    print('Loading node features')  # 打印信息，表示开始加载节点特征
    features = pd.read_csv('features.csv', sep=',')  # 读取节点特征的CSV文件
    print('Start standard node features')  # 打印信息，表示开始标准化节点特征
    features = features.to_numpy()  # 将DataFrame转换为NumPy数组

    scaler = preprocessing.StandardScaler().fit(features)  # 对节点特征进行标准化
    features = scaler.transform(features)  # 应用标准化
    print('Finish loading node features', features.shape)  # 打印加载完成的信息和特征形状
    return features  # 返回处理后的节点特征

def process_labels():
    """
    处理标签数据
    :return: 标签数据
    """
    labels = pd.read_csv('labels.csv', sep=',')  # 读取标签的CSV文件
    return labels  # 返回标签数据

def create_dgl_graph(features, labels, edge_from_id, edge_to_id, edge_features, edge_len, seq_times):
    """
    创建与phish_eth相同结构的多重有向图
    """
    number_of_nodes = features.shape[0]
    
    # 创建多重图
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(number_of_nodes)
    g.add_edges(u=edge_from_id, v=edge_to_id)
    
    print('number of nodes:', g.number_of_nodes())
    print('number of edges:', g.number_of_edges())
    print('node features shape:', features.shape)
    print('edge features shape:', edge_features.shape)
    
    # 保存图和特征
    print('Writing graph to pkl')
    with open('dynamic_dgl_graph.pkl', 'wb') as f:
        pickle.dump(g, f)
    
    # 处理标签和掩码
    train_index, val_index, test_index = np.split(
        labels.sample(frac=1, random_state=10), 
        [int(.6*len(labels)), int(.8*len(labels))]
    )
    
    # 创建掩码
    train_mask = np.zeros(number_of_nodes)
    val_mask = np.zeros(number_of_nodes)
    test_mask = np.zeros(number_of_nodes)
    
    train_mask[train_index['id']] = 1
    val_mask[val_index['id']] = 1
    test_mask[test_index['id']] = 1
    
    # 创建标签数组
    new_labels = np.zeros(number_of_nodes)
    for _, row in labels.iterrows():
        new_labels[row['id']] = row['label']
    
    # 保存标签和特征
    with open('labels.pkl', 'wb') as f:
        pickle.dump((new_labels, train_mask, val_mask, test_mask), f)
    
    np.savez('features.npz', 
             node_features=features.astype(np.float32), 
             edge_features=edge_features.astype(np.float32), 
             edge_len=edge_len,
             seq_times=seq_times)
    print('Finish creating dgl graph')

def main():
    # 加载特征和标签
    features = process_features()
    labels = process_labels()
    
    # 使用批处理方式处理边数据
    edge_from_id, edge_to_id, edge_features, edge_len, seq_times = process_adj_in_batches(
        features.shape[0], labels
    )
    
    # 创建和保存图
    create_dgl_graph(
        features, labels, edge_from_id, edge_to_id, 
        edge_features, edge_len, seq_times
    )

if __name__ == '__main__':
    main()  # 运行主函数


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



