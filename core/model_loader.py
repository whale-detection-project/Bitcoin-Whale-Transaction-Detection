import joblib, math
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist      
from .config import FEATURES

# ───────────────────────────── 1. 모델 로드
def load_models():
    """
    스케일러 · MLP · PCA + (KMeans 중심, τ 컷오프) 로드
    """
    scaler     = joblib.load("model/scaler4.pkl")
    mlp_model  = joblib.load("model/mlp_model4.pkl")
    pca        = joblib.load("model/pca4.pkl")

    # τ 파일에는 (centers, tau) 튜플이 저장돼 있음
    centers, tau = joblib.load("model/km_tau.pkl")     
    return scaler, mlp_model, pca, centers, tau

# ───────────────────────────── 2. 예측 함수
def predict_cluster(data_dict, scaler, model, pca, centers, tau):
    """
    ① 입력 검증 ② 로그+스케일 ③ Mahalanobis/EUCLIDEAN 컷오프
    ④ MLP 예측 ⑤ PCA 임베딩
    """
    try:
        # ① 입력 검증
        for f in FEATURES:
            if f not in data_dict or data_dict[f] is None or data_dict[f] < 0:
                raise ValueError(f"Invalid or missing feature: {f} = {data_dict.get(f)}")

        # ② 전처리
        X_df     = pd.DataFrame({f: [math.log1p(data_dict[f])] for f in FEATURES})
        X_scaled = scaler.transform(X_df)

        # ③ 거리 컷오프 (전역 τ, 유클리드)
        d_min = cdist(X_scaled, centers, metric="euclidean").min()  # 가장 가까운 중심 거리
        if d_min > tau:           # τ = 99.5 백분위 값 등
            label = 4             # Unknown / 외부
        else:
            label = int(model.predict(X_scaled)[0])   # 0 – 3

        # ④ PCA 임베딩
        embedding = pca.transform(X_scaled)[0].tolist()

        # ⑤ 응답
        return {
            "predicted_cluster": label,
            "pca_embedding"   : embedding,
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return {"error": str(e)}
