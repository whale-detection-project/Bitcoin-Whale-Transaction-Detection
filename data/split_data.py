import pandas as pd
from sklearn.model_selection import train_test_split

# CSV 로드
file_path = "data/1000btc.csv"
df = pd.read_csv(file_path)

# 랜덤하게 절반으로 분할
df_train, df_test = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)

# 저장
train_path = "data/whale_train.csv"
test_path = "data/whale_test.csv"
df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

(train_path, test_path)
