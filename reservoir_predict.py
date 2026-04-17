import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import numpy as np

# data_array = 'data/time_series.npy' 
data_array = 'data/normal_time_series.npy' 


time_series_array = np.load(data_array)
time_series = torch.tensor(time_series_array, dtype=torch.float32)
print(f"time_series shape: {time_series.shape}")


# 转为 [T, input_dim]
data = time_series.T  # [3435, 7]

# 标准化（非常重要）
mean = data.mean(dim=0, keepdim=True)
std = data.std(dim=0, keepdim=True) + 1e-8
data = (data - mean) / std

T, input_dim = data.shape

# =========================
# 2. ESN 参数
# =========================
reservoir_size = 500
spectral_radius_target = 0.9
reg = 1e-4
washout = 200
output_dim = input_dim

# =========================
# 3. 初始化权重
# =========================
torch.manual_seed(68)

W_in = torch.randn(reservoir_size, input_dim) * 0.1

W = torch.randn(reservoir_size, reservoir_size)

# 谱半径归一化（关键）
eigvals = torch.linalg.eigvals(W)
spectral_radius = torch.max(torch.abs(eigvals))
W = W / spectral_radius * spectral_radius_target

# =========================
# 4. 运行 Reservoir
# =========================
def run_reservoir(data):
    x = torch.zeros(reservoir_size)
    states = []

    for t in range(data.shape[0]):
        u = data[t]
        x = torch.tanh(W_in @ u + W @ x)
        states.append(x)

    return torch.stack(states)  # [T, reservoir_size]

states = run_reservoir(data)

# =========================
# 5. 训练输出层（Ridge Regression）
# =========================
X = states[washout:-1]     # [T-washout-1, reservoir]
Y = data[washout+1:]       # [T-washout-1, input_dim]

I = torch.eye(reservoir_size)

W_out = torch.linalg.solve(
    X.T @ X + reg * I,
    X.T @ Y
)

# =========================
# 6. 训练集拟合效果
# =========================
train_pred = states @ W_out  # [T, input_dim]

# =========================
# 7. 未来预测（自回归）
# =========================
def predict(initial_input, steps):
    x = torch.zeros(reservoir_size)
    u = initial_input.clone()

    outputs = []

    for _ in range(steps):
        x = torch.tanh(W_in @ u + W @ x)
        y = x @ W_out
        outputs.append(y)
        u = y  # 自回归

    return torch.stack(outputs)

future_steps = 200
future_pred = predict(data[-1], future_steps)

# 反归一化
future_pred = future_pred * std + mean
train_pred = train_pred * std + mean
data_original = data * std + mean

# =========================
# 8. 可视化（只画第0维）
# =========================
plt.figure(figsize=(36, 4))

plt.plot(data_original[:, 0].numpy(), label="True")
plt.plot(train_pred[:, 0].detach().numpy(), label="Fitted", alpha=0.7)

# 未来预测接在后面
start = len(data_original)
plt.plot(
    range(start, start + future_steps),
    future_pred[:, 0].detach().numpy(),
    label="Future",
    linestyle="dashed"
)

plt.legend()
plt.title("Reservoir Computing Time Series Prediction")
plt.savefig("result/esn_prediction.png", dpi=300)
plt.show()