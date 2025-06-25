"""
Oversampling + 4-layer MLP  (PyTorch / SciPy 만 사용)

•  로그+표준화(수작업) → SciPy K-Means 로 임시 라벨
•  클래스 불균형: numpy 로 간단 Random oversampling
•  PyTorch 로 256-128-64-32 MLP 학습, Early-Stopping(25 epoch patience)
•  τ(99.5-percentile) 컷오프 계산 → centers, τ 저장

필수 pip: pandas numpy scipy torch joblib
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from scipy.cluster.vq import kmeans2            # SciPy K-Means
from scipy.spatial.distance import cdist
import joblib, time, random, os

# ├─────────────────────────────────── 1. 데이터 로드
df       = pd.read_csv("dataset/1000btc_train.csv")
FEAT     = ["input_count","output_count","max_output_ratio","max_input_ratio"]
X_log    = np.log1p(df[FEAT].values)           # (N,4)

# ├─────────────────────────────────── 2. 수작업 StandardScaler
mean_, std_ = X_log.mean(0), X_log.std(0)
X_scaled    = (X_log - mean_) / std_

# ├─────────────────────────────────── 3. K-Means 라벨 (k=4)
centers, labels = kmeans2(X_scaled, k=4, minit="++", iter=100, seed=42)
y = labels.astype(np.int64)

# ├─────────────────────────────────── 4. Random Oversampling (간단 구현)
class_counts = np.bincount(y)
max_count    = class_counts.max()
idx_all      = np.hstack([
    np.random.choice(np.where(y==k)[0], size=max_count, replace=True)
    for k in range(4)
])
np.random.shuffle(idx_all)
X_bal, y_bal = X_scaled[idx_all], y[idx_all]

# ├─────────────────────────────────── 5. Train / Val split (80/20)
split = int(0.8 * len(X_bal))
X_tr, X_val = X_bal[:split], X_bal[split:]
y_tr, y_val = y_bal[:split], y_bal[split:]

# Torch Tensor
Xtr = torch.tensor(X_tr,  dtype=torch.float32)
ytr = torch.tensor(y_tr,  dtype=torch.long)
Xva = torch.tensor(X_val, dtype=torch.float32)
yva = torch.tensor(y_val, dtype=torch.long)

# ├─────────────────────────────────── 6. 4-Layer MLP 정의
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128,64),  nn.ReLU(),
            nn.Linear(64,32),   nn.ReLU(),
            nn.Linear(32,4)
        )
    def forward(self,x): return self.net(x)

model = MLP()
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit  = nn.CrossEntropyLoss()
BATCH = 256
PATIENCE=25
best_val, patience = 1e9, 0

# ├─────────────────────────────────── 7. 학습 loop + EarlyStopping
t0=time.time()
for epoch in range(400):
    # mini-batch shuffle
    idx = torch.randperm(len(Xtr))
    for s in range(0,len(idx),BATCH):
        xb = Xtr[idx[s:s+BATCH]]
        yb = ytr[idx[s:s+BATCH]]
        optim.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); optim.step()

    # validation
    with torch.no_grad():
        val_loss = crit(model(Xva), yva).item()
    if val_loss < best_val - 1e-4:
        best_val = val_loss; patience = 0
    else:
        patience += 1
    print(f"Epoch {epoch:3d}  train-loss {loss.item():.4f}  val {val_loss:.4f}")
    if patience >= PATIENCE: break
print("✅ finished in", time.time()-t0,"sec   best-val",best_val)

# ├─────────────────────────────────── 8. τ (99.5%) 치환곱 계산
d_min = cdist(X_scaled, centers).min(1)
tau   = np.percentile(d_min, 99.5)
print("τ =", tau)

# ├─────────────────────────────────── 9. 저장
mdl_dir = Path("model"); mdl_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), mdl_dir/"mlp4_torch.pt")
joblib.dump({"mean":mean_, "std":std_}, mdl_dir/"scaler_np.pkl")
joblib.dump({"centers":centers, "tau":tau}, mdl_dir/"km_tau.pkl")
print("📦 saved →", mdl_dir.resolve())
