import pandas as pd

def show_anomalies(csv_path='features_with_anomalies.csv', output_path='anomalies_only.csv'):
    """
    지정된 CSV에서 anomaly == -1인 행만 필터링하여
    별도 CSV로 저장하고 DataFrame을 반환합니다.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{csv_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    
    # anomaly 컬럼이 -1인 이상치만 선택
    anomalies = df[df['anomaly'] == -1]
    
    # 이상치 목록을 별도 CSV로 저장
    anomalies.to_csv(output_path, index=False)
    print(f"이상치 목록을 '{output_path}'로 저장했습니다. 총 {len(anomalies)}건")
    
    return anomalies

if __name__ == "__main__":
    # 필요에 따라 파일 경로를 수정하세요
    anomalies_df = show_anomalies(
        csv_path="features_with_anomalies.csv",
        output_path="anomalies_only.csv"
    )
    # 이상치 DataFrame을 출력하거나, 추가 디버깅을 진행할 수 있습니다.
    print(anomalies_df.head())
