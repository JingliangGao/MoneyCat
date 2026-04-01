import csv
import torch

csv_file = 'ssq_data.csv'  # 你的 CSV 文件路径

raw_red_balls = []
raw_blue_balls = []

# # 读取 CSV 文件
# with open(csv_file, 'r', encoding='utf-8-sig') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         # 红球可能是逗号分隔的多个数字
#         reds = row['red'].split(',')
#         reds_list = [int(red.strip()) for red in reds]  # 去除可能的空格
#         raw_red_balls.append(reds_list)
#         raw_blue_balls.append([int(row['blue'])])
#         # print(f"红球: {reds_list}, 蓝球: {row['blue']}, 日期: {row['date']}")

# 读取 CSV 文件
with open(csv_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 红球可能是逗号分隔的多个数字
        reds = row['red'].split(',')
        reds_list = [(int(red.strip())/33 - 0.5) for red in reds]  # 去除可能的空格
        raw_red_balls.append(reds_list)
        raw_blue_balls.append([   int(row['blue'])/16 - 0.5  ])
        # print(f"红球: {reds_list}, 蓝球: {row['blue']}, 日期: {row['date']}")

# print(f"raw_blue_balls : {raw_blue_balls}")


# convert to tensors
red_tensor = torch.tensor(raw_red_balls, dtype=torch.float32)
blue_tensor = torch.tensor(raw_blue_balls, dtype=torch.float32)
# print(f"red_tensor shape: {red_tensor.shape}")
# print(f"blue_tensor shape: {blue_tensor.shape}")

time_series = torch.cat((red_tensor, blue_tensor), dim=1).T
# print(f"time_series shape: {time_series.shape}")

# use esn
import torch
import torch.nn as nn

# 模拟你的数据
data = time_series      # 原始数据 [7, 3431]
data = data.T                      # 转置为 [3431,7] -> 每行是一个样本

# 输入 X: 前 n-1 个时间步
X = data[:-1]                      # shape [3430, 7]
# 输出 Y: 下一个时间步（这里预测全部7个数字）
Y = data[1:]                        # shape [3430, 7]

# ESN 参数
input_size = 7
reservoir_size = 200
output_size = 7
spectral_radius = 0.9
leaky = 1.0

# 初始化 ESN 权重
torch.manual_seed(42)
Win = torch.randn(reservoir_size, input_size) * 0.1           # 输入权重
W = torch.randn(reservoir_size, reservoir_size)               # 内部权重
# 缩放 W 保持谱半径
eig = torch.linalg.eigvals(W).abs().max()
W = W * (spectral_radius / eig)
Wout = torch.zeros(output_size, reservoir_size + input_size)   # 输出权重

# 训练 ESN
states = []
x = torch.zeros(reservoir_size)  # 初始状态

for t in range(X.shape[0]):
    u = X[t]
    # 更新状态
    x = (1-leaky) * x + leaky * torch.tanh(torch.matmul(W, x) + torch.matmul(Win, u))
    states.append(torch.cat([x, u]))  # 状态 + 输入

# 将状态矩阵转为 tensor
states = torch.stack(states)       # shape [3430, reservoir_size+input_size]

# 线性回归求 Wout
# Wout * states.T ≈ Y.T
# 最小二乘解: Wout = Y.T @ states @ (states.T @ states + reg*I)^-1
reg = 1e-8
states_T = states.T
Wout = Y.T @ states @ torch.linalg.inv(states_T @ states + reg*torch.eye(states_T.shape[0]))

# 预测
pred = (Wout @ states_T).T         # shape [3430, 7]
#print("预测 shape:", pred.shape)


gloden_values = pred[:1] + 0.5
golden_red_balls = (gloden_values[:, :6] * 33).round().int().numpy().tolist()[0]
golden_blue_balls = (gloden_values[:, 6:] * 16).round().int().numpy().tolist()[0]
print("预测的红球:", golden_red_balls)
print("预测的蓝球:", golden_blue_balls)