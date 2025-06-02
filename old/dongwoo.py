import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # CSV 파일 경로
    file_path = 'dongwoo.csv'  # 실제 파일명으로 수정하세요
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 사용할 피처 목록
    features = [
        'output_count',
        'avg_output_value',
        'std_output_value',
        'fee_per_output',
        'largest_output_ratio',
        'counterparty_entropy',
        'unique_output_count',
        'fragmentation_ratio',
        'max_to_second_ratio'
    ]
    
    # 피처 데이터 준비
    X = df[features].copy()
    
    # 결측치 처리
    X['fragmentation_ratio'].fillna(0, inplace=True)
    X['max_to_second_ratio'].fillna(1, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest 모델 학습
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    model.fit(X_scaled)
    
    # 이상치 점수 및 예측 레이블 추가
    df['anomaly_score'] = model.decision_function(X_scaled)
    df['anomaly'] = model.predict(X_scaled)  # -1: 이상치, 1: 정상
    
    # 모델 및 스케일러 저장
    joblib.dump(model, 'mixing_if_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # 결과 저장
    df.to_csv('features_with_anomalies.csv', index=False)
    
    print("학습 완료. 모델과 결과를 저장했습니다:")
    print(" - 모델: mixing_if_model.pkl")
    print(" - 스케일러: scaler.pkl")
    print(" - 결과 CSV: features_with_anomalies.csv")
    print("이상치 레이블 분포:")
    print(df['anomaly'].value_counts())

if __name__ == "__main__":
    main()
