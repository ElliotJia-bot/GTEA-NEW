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



models = ['APPNP']
test_time = 2
num_layers = [1]
node_hidden_dim = [64, 128]

epochs = 1000
gpu = -1
num_cpu = 16


# data_dir = "/cephfs/group/ixin-pay-wechat-pay-coop-kdd-net/qiufang/data_gcn/hb_data/"
# log_dir = data_dir + "exp/exp1/"
log_dir = 'experiment'
data_dir = 'data/test_data'

for t in range(test_time):
        for m in models:
                for l in num_layers:
                        for d in node_hidden_dim:
                            for bs in batch_size:
                                    log_name = "{model}_{node_hidden_dim}_{num_layers}_{test_time}".format(
                                        model=m, node_hidden_dim=d, num_layers=l,
                                        test_time=t)
                                    print(log_name)
                                    cmd = "python -u fullbatch_main.py --lr 0.01 --data_dir {data_dir} --patience 300"\
                                        "--model {model} --gpu {gpu} --epochs {epochs} "\
                                        "--node_hidden_dim {node_hidden_dim} "\
                                        "--num_layers {num_layers}  --num_cpu {num_cpu} "\
                                        "--log_dir {log_dir} --log_name {log_name} "\
                                        .format(data_dir=data_dir, model=m, 
                                               gpu=gpu,epochs=epochs,
                                               node_hidden_dim=d, num_layers=l, 
                                               num_cpu=num_cpu,log_dir=log_dir,log_name=log_name)
                                    print(cmd)
                                    os.system(cmd)
