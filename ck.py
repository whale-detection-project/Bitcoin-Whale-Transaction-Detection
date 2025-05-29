import pandas as pd
import matplotlib.pyplot as plt

def main():
    # CSV 파일 경로 (예: features.csv)
    file_path = 'block_ages.csv'  # 실제 파일명에 맞게 수정하세요

    # 데이터 로드
    df = pd.read_csv(file_path)

    # 분석할 피처 목록
    age_cols = ['input_age_median', 'max_age']
    pct_cols = ['pct_age_over_30', 'pct_value_age_over_365']

    # 1) 수치형 통계 출력
    for col in age_cols + pct_cols:
        print(f"=== Statistics for {col} ===")
        print(df[col].describe())
        print()

    # 2) 히스토그램
    for col in age_cols + pct_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f'{col} Histogram')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # 3) 박스플롯 (input_age_median vs max_age)
    plt.figure()
    data = [df['input_age_median'].dropna(), df['max_age'].dropna()]
    plt.boxplot(data, vert=False, labels=['input_age_median', 'max_age'])
    plt.title('Boxplot of Age Metrics')
    plt.xlabel('Days')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
