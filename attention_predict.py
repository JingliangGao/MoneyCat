import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 数据准备
# =========================
data_array = 'data/normal_time_series.npy' 
time_series_array = np.load(data_array)
time_series = torch.tensor(time_series_array, dtype=torch.float32)

# [7, T] -> [T, 7]
data = time_series.T

# 标准化
mean = data.mean(dim=0, keepdim=True)
std = data.std(dim=0, keepdim=True) + 1e-8
data = (data - mean) / std

# =========================
# 2. 构造数据集
# =========================
def create_dataset(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.stack(X), torch.stack(Y)

seq_len = 50
X, Y = create_dataset(data, seq_len)

# =========================
# 3. 划分数据
# =========================
train_size = int(0.8 * len(X))

X_train = X[:train_size]
Y_train = Y[:train_size]

X_test = X[train_size:]
Y_test = Y[train_size:]

# =========================
# 4. Attention 模型
# =========================
class AttentionModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Attention
        self.attn = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)   # [B, seq_len, hidden_dim]

        # attention权重
        attn_weights = torch.softmax(
            self.attn(lstm_out).squeeze(-1), dim=1
        )  # [B, seq_len]

        # 加权求和
        context = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1),
            dim=1
        )  # [B, hidden_dim]

        out = self.fc(context)  # [B, input_dim]
        return out

# =========================
# 5. 训练
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train = X_train.to(device)
Y_train = Y_train.to(device)

epochs = 30
batch_size = 64

for epoch in range(epochs):
    perm = torch.randperm(X_train.size(0))

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]

        xb = X_train[idx]
        yb = Y_train[idx]

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# =========================
# 6. 测试
# =========================
model.eval()

with torch.no_grad():
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    pred_test = model(X_test)
    test_loss = criterion(pred_test, Y_test)

print("Test Loss:", test_loss.item())

# =========================
# 7. 多步预测
# =========================
def predict_future(model, init_seq, steps):
    model.eval()

    seq = init_seq.clone().to(device)
    outputs = []

    for _ in range(steps):
        with torch.no_grad():
            pred = model(seq.unsqueeze(0))  # [1, 7]

        outputs.append(pred.squeeze(0).cpu())

        # 关键：保持二维
        seq = torch.cat([seq[1:], pred], dim=0)

    return torch.stack(outputs)

future_steps = 200
init_seq = data[-seq_len:]

future = predict_future(model, init_seq, future_steps)

# =========================
# 8. 反归一化
# =========================
future = future * std + mean
data_original = data * std + mean

# =========================
# 9. 可视化（第0维）
# =========================
num_dims = data_original.shape[1]

plt.figure(figsize=(16, 12))

for d in range(num_dims):
    plt.subplot(num_dims, 1, d + 1)

    # 真实数据
    plt.plot(
        data_original[train_size:, d].numpy(),
        label="Real"
    )

    # 预测数据
    start = len(data_original) - train_size
    plt.plot(
        range(start, start + future_steps),
        future[:, d].numpy(),
        linestyle="dashed",
        label="Future"
    )

    plt.title(f"Dimension {d}")
    plt.legend()

plt.tight_layout()
plt.savefig("data/convtrans_prediction.png", dpi=300)
plt.show()


