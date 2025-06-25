import math, joblib, torch
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist
from .config import FEATURES

# ──────────────────────── MLP 모델 클래스 정의
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.net(x)

# ──────────────────────── 모델 로드 함수
def load_models():
    scaler_dict   = joblib.load("model/scaler_np.pkl")    # {'mean': ..., 'std': ...}
    centers_tau   = joblib.load("model/km_tau.pkl")       # {'centers': ..., 'tau': ...}
    pca           = joblib.load("model/pca4.pkl")          # Optional: 2D 시각화용

    model = MLP()
    model.load_state_dict(torch.load("model/mlp4_torch.pt", map_location="cpu"))
    model.eval()

    return scaler_dict, model, centers_tau, pca

# ──────────────────────── 예측 함수
def predict_cluster(data_dict, scaler_dict, model, centers_tau, pca):
    try:
        for f in FEATURES:
            if f not in data_dict or data_dict[f] is None or data_dict[f] < 0:
                raise ValueError(f"Invalid or missing feature: {f} = {data_dict.get(f)}")

        # ① 전처리 (로그 + 수작업 스케일링)
        X_input = torch.tensor(
            [[math.log1p(data_dict[f]) for f in FEATURES]],
            dtype=torch.float32
        )
        mean, std = torch.tensor(scaler_dict["mean"]), torch.tensor(scaler_dict["std"])
        X_scaled = (X_input - mean) / std

        # ② 거리 기반 Unknown 컷오프
        centers = centers_tau["centers"]
        tau     = centers_tau["tau"]
        d_min = cdist(X_scaled.numpy(), centers).min()

        if d_min > tau:
            label = 4  # Unknown
        else:
            logits = model(X_scaled)
            label = int(torch.argmax(logits, dim=1).item())

        # ③ PCA 임베딩
        embedding = pca.transform(X_scaled.numpy())[0].tolist()

        return {
            "predicted_cluster": label,
            "pca_embedding": embedding,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return {"error": str(e)}
