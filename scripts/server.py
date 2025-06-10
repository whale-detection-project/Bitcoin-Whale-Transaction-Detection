from crawl4ai import crawl_and_extract
import pandas as pd

# 📡 크롤링할 페이지 URL 목록 생성
base_url = "https://bitinfocharts.com/top-100-richest-bitcoin-addresses"
urls = [base_url + ".html"] + [f"{base_url}-{i}.html" for i in range(2, 11)]

# 📥 데이터 크롤링
results = crawl_and_extract(
    urls=urls,
    extraction_type="table",
    table_index=0,
    include_raw_html=False,
)

# 🧹 정제 및 필터링
all_data = []
for page in results:
    if not page['tables']:
        continue
    df = page['tables'][0]
    df.columns = [
        "Rank", "Address", "Balance", "% of coins", "First In", "Last In",
        "Ins", "First Out", "Last Out", "Outs"
    ]
    df["Balance"] = df["Balance"].str.replace(r"[^\d.]", "", regex=True).astype(float)
    df = df[df["Balance"] >= 1000]
    all_data.append(df)

# 🔗 통합 및 저장
df_all = pd.concat(all_data, ignore_index=True)
df_all.to_csv("top_btc_wallets_over_1000btc.csv", index=False)
print("✅ 저장 완료: top_btc_wallets_over_1000btc.csv")
