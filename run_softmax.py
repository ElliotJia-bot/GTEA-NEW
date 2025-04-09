import torch
print(torch.cuda.is_available())

import os
# os.system('pip install scikit-learn==0.19.1')
# os.system('pip install numpy==1.18.4')
# os.system('pip install pandas==0.24.0')
# os.system('pip install pickleshare==0.7.5')
# os.system('pip install matplotlib==3.2.1')
# os.system('pip install torch==1.4.0')
# os.system('pip install dgl')


run_dynamic = True


# models = ['GCN', 'GraphSAGE', 'GAT', 'ECConv', 'GTEA-ST', 'GTEA-LSTM', "GTEA-TLSTM1", "GTEA-LSTM+T2V", "GTEA-Trans", "GTEA-Trans+T2V"]

test_time = 2
num_layers = [1]
num_neighbors = [10]
node_hidden_dim = [128]
num_heads = [4]

epochs = 250
patience = 30
num_lstm_layers = [1, 2]
gpu = 1
num_cpu = 16
batch_size = [32]
# data_dir = "/cephfs/group/ixin-pay-wechat-pay-coop-kdd-net/qiufang/data_gcn/hb_data/"
# log_dir = data_dir + "exp/exp1/"
log_dir = 'experiment'
data_dir = 'data/dynamic_eth'


models = ['GTEA-LSTM+T2V-V', 'GTEA-Trans+T2V-V']
leaning_rates = [1e-4]
# lstm layers and heads
if run_dynamic:
    for t in range(test_time):
        for lstm_l in num_lstm_layers:
            for m in models:
                    for l in num_layers:
                        for n in num_neighbors:
                            for d in node_hidden_dim:
                                for bs in batch_size:
                                    for h in num_heads:
                                        for lr in leaning_rates:
                                            log_name = "{model}_{num_neighbors}_{node_hidden_dim}_{num_layers}_{num_heads}_{test_time}".format(
                                                model=m, num_neighbors=n, node_hidden_dim=d, num_layers=l, num_heads=h,
                                                test_time=t)
                                            print(log_name)
                                            os.system(
                                                "python -u main.py --data_dir {data_dir} " 
                                                "--model {model} --gpu {gpu} --epochs {epochs} "
                                                "--num_neighbors {num_neighbors} --node_hidden_dim {node_hidden_dim} "
                                                "--num_layers {num_layers} --num_heads {num_heads} --num_cpu {num_cpu} "
                                                "--log_dir {log_dir} --log_name {log_name} --num_lstm_layers {num_lstm_layers} "
                                                "--batch_size {batch_size}  --lr {lr} --patience {patience} ".format(data_dir=data_dir, model=m, 
                                                                                                   gpu=gpu,epochs=epochs,num_neighbors=n,
                                                                                                   node_hidden_dim=d, num_layers=l, num_heads=h,
                                                                                                   num_cpu=num_cpu,log_dir=log_dir,log_name=log_name, 
                                                                                                   num_lstm_layers=lstm_l, batch_size=bs, patience=patience, lr=lr))
