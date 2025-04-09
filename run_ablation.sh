#!/bin/bash

# 设置基础参数
GPU=0
EPOCHS=50
BATCH_SIZE=32
NUM_LAYERS=2
NODE_HIDDEN_DIM=64
TIME_HIDDEN_DIM=32
NUM_LSTM_LAYERS=1
NUM_HEADS=4
WEIGHT_DECAY=0
MAX_EVENT=5
NUM_NEIGHBORS=10;


# 数据集列表
DATASETS=(
    "saml"
)

# 模型列表
MODELS=(
    # "GCN"
    # "GraphSAGE"
    # "GAT"
    #"ECConv"
    #"GTEA-ST"
    #"EGNN"
    #"GAT+T"
    #"TGAT"
    "GTEA-LSTM+T2V"
    "GTEA-Trans"
    #"GTEA-Trans+T2V"
    #"GTEA-LSTM"
    # "GTEA-Trans-V"
    # "GTEA-LSTM-V"
    # "GTEA-Trans+T2V-V"
    # "GTEA-LSTM+T2V-V"
)

# 运行每个数据集和模型的组合
for dataset in "${DATASETS[@]}"; do
    echo "Running on dataset: ${dataset}"
    echo "========================================"
    
    # 设置数据集特定的参数
    DATA_DIR="data/${dataset}"
    LOG_DIR="experiment/${dataset}"
    
    # 为每个数据集创建汇总报告
    echo "Model Performance Summary for ${dataset}" > ${LOG_DIR}/summary.txt
    echo "========================================" >> ${LOG_DIR}/summary.txt
    echo "" >> ${LOG_DIR}/summary.txt
    
    # 运行每个模型
    for model in "${MODELS[@]}"; do
        echo "Running model: ${model} on ${dataset}"
        
        # 创建模型特定的日志目录
        mkdir -p ${LOG_DIR}/${model}
        
        # 运行模型并捕获错误
        if python main.py \
            --data_dir ${DATA_DIR} \
            --model ${model} \
            --gpu ${GPU} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --num_layers ${NUM_LAYERS} \
            --node_hidden_dim ${NODE_HIDDEN_DIM} \
            --time_hidden_dim ${TIME_HIDDEN_DIM} \
            --num_lstm_layers ${NUM_LSTM_LAYERS} \
            --num_heads ${NUM_HEADS} \
            --weight_decay ${WEIGHT_DECAY} \
            --log_dir ${LOG_DIR} \
            --log_name ${model} \
            --max_event ${MAX_EVENT} \
            > ${LOG_DIR}/${model}/output.log 2>&1; then
            
            echo "Successfully finished running ${model} on ${dataset}"
            
            # 记录模型性能到汇总报告
            echo "Model: ${model}" >> ${LOG_DIR}/summary.txt
            echo "----------------------------------------" >> ${LOG_DIR}/summary.txt
            grep "ValAcc" ${LOG_DIR}/${model}/output.log | tail -n 1 >> ${LOG_DIR}/summary.txt
            echo "" >> ${LOG_DIR}/summary.txt
        else
            echo "Error running ${model} on ${dataset}"
            echo "Model: ${model} - Failed" >> ${LOG_DIR}/summary.txt
            echo "----------------------------------------" >> ${LOG_DIR}/summary.txt
            echo "Error occurred during execution" >> ${LOG_DIR}/summary.txt
            echo "" >> ${LOG_DIR}/summary.txt
        fi
        
        echo "----------------------------------------"
    done
    
    echo "Finished running all models on ${dataset}"
    echo "========================================"
done

# 生成总体汇总报告
echo "Generating overall summary report..."
echo "Overall Model Performance Summary" > experiment/overall_summary.txt
echo "========================================" >> experiment/overall_summary.txt
echo "" >> experiment/overall_summary.txt

for dataset in "${DATASETS[@]}"; do
    echo "Dataset: ${dataset}" >> experiment/overall_summary.txt
    echo "========================================" >> experiment/overall_summary.txt
    cat experiment/${dataset}/summary.txt >> experiment/overall_summary.txt
    echo "" >> experiment/overall_summary.txt
done

echo "All experiments have been completed."
echo "Check experiment/overall_summary.txt for complete results."