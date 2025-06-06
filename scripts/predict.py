import pandas as pd
import joblib
import json

# 🐋 라벨 → 한글 설명
original_whale_label_map = {
    0: '0: 다수입력 단일출력형',
    1: '1: 단일입력 다수출력형',
    2: '2: 잔돈합치기형',
    3: '3: 급행전송형',
    4: '4: 은닉형',
    5: '5: 기타/미분류형'
}

# 📁 모델 및 라벨맵 로드
model = joblib.load("models/whale_classifier.joblib")
with open("models/index_to_label.json") as f:
    index_to_label = {int(k): int(v) for k, v in json.load(f).items()}

# 📁 예측 대상 데이터
df = pd.read_csv("data/whale_test.csv")

# 🧹 문자열 제거
drop_cols = ['tx_hash'] if 'tx_hash' in df.columns else []
drop_cols += df.select_dtypes(include='object').columns.tolist()
X = df.drop(columns=drop_cols)

# 🔍 예측 및 라벨 역매핑
y_pred_idx = model.predict(X)
y_pred = pd.Series(y_pred_idx).map(index_to_label)
df['predicted_whale_type'] = y_pred
df['predicted_whale_name'] = df['predicted_whale_type'].map(original_whale_label_map)

# 💾 결과 저장
df.to_csv("data/whale_test_with_predictions.csv", index=False)

# 📊 분포 출력
print(f"📦 총 분류된 트랜잭션 수: {len(df)}\n")
print("📊 고래 유형별 예측 분포:")
pred_dist = df['predicted_whale_type'].value_counts(normalize=True).sort_index()
for code, ratio in pred_dist.items():
    name = original_whale_label_map.get(code, "알 수 없음")
    print(f"• {code}: {name} → {ratio:.2%}")

print("\n✅ 예측 결과 저장 완료 → data/whale_test_with_predictions.csv")
