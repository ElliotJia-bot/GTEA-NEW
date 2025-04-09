import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 创建输出目录
output_dir = "datainfo"
os.makedirs(output_dir, exist_ok=True)

# 设置日志文件
log_file = os.path.join(output_dir, "analysis_log.txt")
sys.stdout = open(log_file, "w")

# 打印当前时间
print(f"Analysis started at: {datetime.now()}\n")

# 假设你加载了交易数据
try:
    df = pd.read_csv("adj.csv")  # 或 parquet, json 等
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# 创建账户对键
df['account_pair'] = df['srcId'].astype(str) + "→" + df['dstId'].astype(str)

# 统计每对账户对出现的次数
pair_counts = df['account_pair'].value_counts()

# 描述性统计
print("\nDescriptive Statistics:")
print(pair_counts.describe())

# 比例统计
print("\nInteraction Count Proportions:")
print(f">30次交互账户对占比: {(pair_counts > 30).sum() / len(pair_counts):.2%}")
print(f">20次交互账户对占比: {(pair_counts > 20).sum() / len(pair_counts):.2%}")
print(f"=1次交互账户对占比: {(pair_counts == 1).sum() / len(pair_counts):.2%}")

# 可视化分布（前100）
plt.figure(figsize=(10, 6))
pair_counts.hist(bins=50)
plt.title("Account Pair Interaction Count Distribution")
plt.xlabel("Number of interactions")
plt.ylabel("Number of account pairs")
plt.yscale("log")

# 保存图片
image_path = os.path.join(output_dir, "interaction_distribution.png")
plt.savefig(image_path)
print(f"\nPlot saved to: {image_path}")

# 关闭日志文件
sys.stdout.close()

# 恢复标准输出
sys.stdout = sys.__stdout__
print(f"Analysis completed. Results saved to {output_dir} directory.")
