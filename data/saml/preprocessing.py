import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 读取数据
df = pd.read_csv("SAML-D.csv")

# 首先对账户进行重新编号
all_accounts = pd.concat([
    df[["Sender_account"]].rename(columns={"Sender_account": "old_id"}),
    df[["Receiver_account"]].rename(columns={"Receiver_account": "old_id"})
]).drop_duplicates()

# 创建新的ID映射
all_accounts['new_id'] = range(len(all_accounts))
id_mapping = dict(zip(all_accounts['old_id'], all_accounts['new_id']))

# 保存ID映射关系
all_accounts.to_csv("account_id_mapping.csv", index=False)
print("✅ 账户ID映射关系已保存：account_id_mapping.csv")

# 更新原始数据中的账户ID
df['Sender_account'] = df['Sender_account'].map(id_mapping)
df['Receiver_account'] = df['Receiver_account'].map(id_mapping)

# 合并日期与时间为时间戳
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["timestamp"] = df["datetime"].view('int64')
df["hour"] = df["datetime"].dt.hour
df["date_only"] = df["datetime"].dt.date
df["weekday"] = df["datetime"].dt.weekday

# 是否跨境交易
df["is_cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)

# 是否跨币种交易
df["is_cross_currency"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)

# 是否大额交易（可自定义阈值，这里设为10000）
df["is_high_value"] = (df["Amount"] > 10000).astype(int)

# ---------------------------
# 构造节点特征（账户层面聚合）
# ---------------------------

# 所有账户（发送方 + 接收方）集合
accounts = pd.concat([
    df[["Sender_account"]].rename(columns={"Sender_account": "account_id"}),
    df[["Receiver_account"]].rename(columns={"Receiver_account": "account_id"})
]).drop_duplicates()

# 1. 发出交易统计
out_stats = df.groupby("Sender_account").agg(
    out_tx_count=("Amount", "count"),
    out_total_amount=("Amount", "sum"),
    out_avg_amount=("Amount", "mean"),
    out_std_amount=("Amount", "std"),
    out_unique_receiver_count=("Receiver_account", "nunique"),
    out_currency_diversity=("Payment_currency", "nunique"),
    out_payment_type_diversity=("Payment_type", "nunique"),
    out_cross_border_ratio=("is_cross_border", "mean"),
    out_cross_currency_ratio=("is_cross_currency", "mean"),
    out_high_value_ratio=("is_high_value", "mean"),
    active_days_sender=("date_only", "nunique"),
).reset_index().rename(columns={"Sender_account": "account_id"})

# 2. 接收交易统计
in_stats = df.groupby("Receiver_account").agg(
    in_tx_count=("Amount", "count"),
    in_total_amount=("Amount", "sum"),
    in_avg_amount=("Amount", "mean"),
    in_std_amount=("Amount", "std"),
    in_unique_sender_count=("Sender_account", "nunique"),
    in_currency_diversity=("Received_currency", "nunique"),
    in_payment_type_diversity=("Payment_type", "nunique"),
    in_cross_border_ratio=("is_cross_border", "mean"),
    in_cross_currency_ratio=("is_cross_currency", "mean"),
    in_high_value_ratio=("is_high_value", "mean"),
    active_days_receiver=("date_only", "nunique"),
).reset_index().rename(columns={"Receiver_account": "account_id"})

# 合并所有账户特征
node_features = accounts.merge(in_stats, on="account_id", how="left").merge(out_stats, on="account_id", how="left")

# 缺失值填充为0（比如某些账户只收或只发）
node_features.fillna(0, inplace=True)

# ---------------------------
# 构造交易边特征（每一笔交易）+ 转换 timestamp + Label Encoding
# ---------------------------

# 构造新的字段
df["currency_pair"] = df["Payment_currency"] + "->" + df["Received_currency"]
df["location_pair"] = df["Sender_bank_location"] + "->" + df["Receiver_bank_location"]

# 使用原始 datetime 转为 UNIX timestamp（秒）
df["timestamp_unix"] = df["datetime"].view("int64") // 10**9

# 创建边特征 DataFrame
edge_features = df[[
    "Sender_account", "Receiver_account", "timestamp_unix", "Amount",
    "Payment_type", "currency_pair", "location_pair",
    "is_cross_border", "is_cross_currency", "is_high_value", "Is_laundering"
]].rename(columns={
    "Sender_account": "srcId",
    "Receiver_account": "dstId",
    "timestamp_unix": "timestamp",
    "Amount": "amount",
    "Is_laundering": "label"
})

# Label Encoding
categorical_cols = ["Payment_type", "currency_pair", "location_pair"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    edge_features[col] = le.fit_transform(edge_features[col])  # 直接覆盖原列
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# 对amount进行归一化到[-1,1]
amount_scaler = MinMaxScaler(feature_range=(-1, 1))
edge_features['amount'] = amount_scaler.fit_transform(edge_features[['amount']])

# 重命名编码后的列（移除_id后缀）
for col in categorical_cols:
    edge_features.rename(columns={col: col.replace("_id", "")}, inplace=True)

# 保存处理后的 edge_features
edge_features.to_csv("adj.csv", index=False)
print("✅ 边特征中的分类变量已编码完成，amount已归一化到[-1,1]，已保存：adj.csv")



# ---------------------------
# 输出结果
# ---------------------------

# 选择需要归一化/标准化的数值列（排除 account_id）
numerical_cols = node_features.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'account_id']

# =====================
# 1. Min-Max 归一化到[-1,1]
# =====================
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
node_features_minmax = node_features.copy()
node_features_minmax[numerical_cols] = minmax_scaler.fit_transform(node_features[numerical_cols])
node_features_minmax.to_csv("feature.csv", index=False)
print("✅ 已保存归一化后的节点特征（范围[-1,1]）：feature.csv")



# ============================
# 9. 构造节点级标签 label.csv
# ============================

# 从交易中找出出现过的所有账户（发送方 + 接收方）
all_accounts = pd.concat([
    df[["Sender_account"]].rename(columns={"Sender_account": "id"}),
    df[["Receiver_account"]].rename(columns={"Receiver_account": "id"})
]).drop_duplicates()

# 找出参与过 laundering 的账户（无论是 sender 还是 receiver）
fraud_senders = df[df["Is_laundering"] == 1]["Sender_account"]
fraud_receivers = df[df["Is_laundering"] == 1]["Receiver_account"]
fraud_accounts = pd.concat([fraud_senders, fraud_receivers]).unique()

# 构造账户级标签
all_accounts["label"] = all_accounts["id"].apply(lambda x: 1 if x in fraud_accounts else 0)

# 保存为 label.csv
all_accounts.to_csv("labels.csv", index=False)
print("✅ 节点级 labels.csv 已保存（账户是否与欺诈交易有关）")