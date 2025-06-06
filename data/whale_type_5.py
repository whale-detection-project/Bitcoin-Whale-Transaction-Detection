import pandas as pd

# 📁 CSV 파일 경로
input_path = "data/labeled_whales.csv"  # 필요 시 경로 수정
output_path = "data/misc_whales_only.csv"

# 📥 데이터 불러오기
df = pd.read_csv(input_path)

# 🐋 whale_type == 5인 행만 필터링
misc_whales = df[df['whale_type'] == 5]

# 💾 새 CSV 파일로 저장
misc_whales.to_csv(output_path, index=False)

print(f"✅ 기타/미분류형 고래만 저장 완료: {output_path}")
print(f"총 개수: {len(misc_whales)}개")
