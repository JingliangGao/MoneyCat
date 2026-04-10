import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(86)

# =========================
# 1. 数据准备
# =========================
data_array = 'data/time_series.npy' 
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

seq_len = 64
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
# 4. 位置编码
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# =========================
# 5. Conv + Transformer 模型
# =========================
class ConvTransModel(nn.Module):
    def __init__(self, input_dim=7, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        # CNN 提取局部特征
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出层
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: [B, seq_len, input_dim]

        # → CNN: [B, input_dim, seq_len]
        x = x.permute(0, 2, 1)

        x = self.conv1(x)  # [B, d_model, seq_len]

        # → [B, seq_len, d_model]
        x = x.permute(0, 2, 1)

        # 加位置编码
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)  # [B, seq_len, d_model]

        # 取最后一个时间步
        out = x[:, -1, :]

        out = self.fc(out)  # [B, input_dim]

        return out

# =========================
# 6. 训练
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvTransModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train = X_train.to(device)
Y_train = Y_train.to(device)

epochs = 1000
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
# 7. 测试
# =========================
model.eval()

with torch.no_grad():
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    pred_test = model(X_test)
    test_loss = criterion(pred_test, Y_test)

print("Test Loss:", test_loss.item())

# =========================
# 8. 多步预测
# =========================
def predict_future(model, init_seq, steps):
    model.eval()

    seq = init_seq.clone().to(device)
    outputs = []

    for _ in range(steps):
        with torch.no_grad():
            pred = model(seq.unsqueeze(0))  # [1, 7]

        outputs.append(pred.squeeze(0).cpu())

        # 保持二维
        seq = torch.cat([seq[1:], pred], dim=0)

    return torch.stack(outputs)

future_steps = 200
init_seq = data[-seq_len:]

future = predict_future(model, init_seq, future_steps)

# =========================
# 9. 反归一化
# =========================
future = future * std + mean
data_original = data * std + mean

print("Future shape:", future.shape)
print("Future Predictions (first 5):")
print(future[:5])

# =========================
# 10. 可视化
# =========================
plt.figure(figsize=(16, 4))

plt.plot(data_original[:, 0].numpy(), label="True")

start = len(data_original)
plt.plot(
    range(start, start + future_steps),
    future[:, 0].numpy(),
    label="Future",
    linestyle="dashed"
)

plt.legend()
plt.title("Conv + Transformer Prediction")
plt.savefig("data/convtrans_prediction.png", dpi=300)
plt.show()