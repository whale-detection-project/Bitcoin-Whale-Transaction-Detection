import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# ✅ 데이터 로딩
df = pd.read_csv("BTC_5min_1year_data.csv")
features = df[['open', 'high', 'low', 'close', 'volume']].values

# ✅ 차분 정규화
differenced = features[1:] - features[:-1]
min_vals = differenced.min(axis=0)
max_vals = differenced.max(axis=0)
scaled = (differenced - min_vals) / (max_vals - min_vals + 1e-8)

# ✅ 정규화 파라미터 저장
np.save("min_vals_auto.npy", min_vals)
np.save("max_vals_auto.npy", max_vals)

# ✅ 시퀀스 생성 함수
WINDOW_SIZE = 24
def create_sequences(data, window):
    x = []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
    return np.array(x)

X_np = create_sequences(scaled, WINDOW_SIZE)
X = torch.tensor(X_np, dtype=torch.float32)

# ✅ 데이터셋 분리
dataset = TensorDataset(X)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# ✅ LSTM Autoencoder 정의
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (h_n, _) = self.encoder(x)
        z = self.latent(h_n[-1])
        d_input = self.decoder_input(z).unsqueeze(0)
        d_input = d_input.repeat(seq_len, 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(d_input)
        return out

model = LSTMAutoencoder().to(device)

# ✅ 학습 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 루프
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for batch_x, in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        batch_x = batch_x.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_x, in val_loader:
            val_x = val_x.to(device)
            val_output = model(val_x)
            val_loss += criterion(val_output, val_x).item() * val_x.size(0)

    print(f"✅ Epoch {epoch} | Train Loss: {train_loss/len(train_ds):.6f} | Val Loss: {val_loss/len(val_ds):.6f}")

# ✅ 모델 저장
torch.save(model.state_dict(), "lstm_autoencoder_ws24.pt")
print("✅ 모델 저장 완료: lstm_autoencoder_ws24.pt")
