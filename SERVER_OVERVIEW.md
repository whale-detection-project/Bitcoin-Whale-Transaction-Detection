# FastAPI 기반 실시간 고래 탐지 서버 개요

이 문서는 `xgb_whale_classifier_with_normal.joblib` 모델을 활용하여 비트코인 트랜잭션을 실시간으로 분류하는 서버 시스템의 기본 동작을 설명합니다.

## 동작 흐름
1. **데이터 수신**: 비트코인 트랜잭션을 WebSocket을 통해 수신합니다.
2. **전처리**: 필요한 경우 입력 데이터를 모델에 맞는 형태로 변환합니다.
3. **예측**: 학습된 XGBoost 모델을 사용해 각 거래가 어떤 고래 유형에 해당하는지 분류합니다.
4. **결과 반환**: 예측된 고래 유형을 서버 응답으로 돌려줍니다.

## FastAPI 서버 예시 코드
```python
import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket

app = FastAPI()
model = joblib.load("model/xgb_whale_classifier_with_normal.joblib")

LABELS = {
    0: "normal",
    1: "less_output_whale",
    2: "less_input_whale",
    3: "fast_transfer_whale",
}

def preprocess(tx: dict) -> pd.DataFrame:
    """트랜잭션 딕셔너리를 모델 입력용 데이터프레임으로 변환"""
    df = pd.DataFrame([tx])
    df["has_zero_output"] = df["has_zero_output"].astype(int)
    features = [
        "input_count", "output_count", "total_input_value",
        "max_input_value", "max_output_value", "max_output_ratio",
        "fee_per_max_ratio", "has_zero_output"
    ]
    return df[features]

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        X = preprocess(data)
        pred = model.predict(X)[0]
        label = LABELS.get(int(pred), f"type_{pred}")
        await ws.send_json({"whale_type": label})
```

이 예시는 시스템 구성의 출발점으로, 실제 운영 환경에서는 보안과 성능을 고려한 추가 구성이 필요합니다.
