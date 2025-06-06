import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import json
import os

# 🐋 라벨 → 한글 이름 매핑 (표시용)
original_whale_label_map = {
    0: '0: 다수입력 단일출력형',
    1: '1: 단일입력 다수출력형',
    2: '2: 잔돈합치기형',
    3: '3: 급행전송형',
    4: '4: 은닉형',
    5: '5: 기타/미분류형'
}

# 📁 데이터 로딩
df = pd.read_csv("data/labeled_whales.csv")

# 🧹 문자열 제거
drop_cols = ['tx_hash'] if 'tx_hash' in df.columns else []
drop_cols += df.select_dtypes(include='object').columns.tolist()
X = df.drop(columns=['whale_type'] + drop_cols)
y_original = df['whale_type']

# 🔁 라벨 리맵핑 (XGBoost는 0부터 연속된 정수만 허용)
unique_labels = sorted(y_original.unique())
label_to_index = {label: i for i, label in enumerate(unique_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}
y = y_original.map(label_to_index)

# 📊 학습용 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧠 XGBoost 학습
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(unique_labels),
    eval_metric='mlogloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)

# 🔍 평가
y_pred_idx = model.predict(X_test)
y_pred = pd.Series(y_pred_idx).map(index_to_label)
y_test_original = y_test.map(index_to_label)
label_names = [original_whale_label_map[label] for label in unique_labels]

print("📋 테스트셋 성능 평가:")
print(classification_report(y_test_original, y_pred, target_names=label_names))

# 💾 모델 및 라벨 맵 저장
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/whale_classifier.joblib")

# numpy.int64 → int 로 변환
index_to_label_json = {int(k): int(v) for k, v in index_to_label.items()}

with open("models/index_to_label.json", "w") as f:
    json.dump(index_to_label_json, f)

print("✅ 모델과 라벨 매핑 저장 완료: models/")