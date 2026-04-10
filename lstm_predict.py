import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据准备
# =========================
data_array = 'data/time_series.npy' 
time_series_array = np.load(data_array)
time_series = torch.tensor(time_series_array, dtype=torch.float32)

# 转换为 [T, feature]
data = time_series.T  # [3436, 7]

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
# 3. 划分训练/测试
# =========================
train_size = int(0.8 * len(X))

X_train = X[:train_size]
Y_train = Y[:train_size]

X_test = X[train_size:]
Y_test = Y[train_size:]

# =========================
# 4. LSTM模型
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# =========================
# 5. 训练
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train = X_train.to(device)
Y_train = Y_train.to(device)

epochs = 20
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
# 6. 测试集评估
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
    
    seq = init_seq.clone().to(device)   # [seq_len, 7]
    outputs = []
    
    for _ in range(steps):
        with torch.no_grad():
            pred = model(seq.unsqueeze(0))   # [1, 7]
        
        outputs.append(pred.squeeze(0).cpu())  # 这里只是存结果，可以 squeeze
        
        # ❗关键修复：不要 squeeze
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
plt.title("LSTM Time Series Prediction")
plt.savefig("lstm_prediction.png", dpi=300)
plt.show()