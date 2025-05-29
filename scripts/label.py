#!/usr/bin/env python3
# label_whale_binary.py
# -------------------------------------------------
# data/BTC_2024.csv -> data/BTC_2024_binary.csv
# whale_any : 1   if max_input_value>=TH or max_output_value>=TH
#           : 0   otherwise
# -------------------------------------------------

import pandas as pd
from pathlib import Path

IN_FILE  = Path("data/BTC_2024.csv")
OUT_FILE = IN_FILE.with_name("BTC_2024_binary.csv")
THRESHOLD_BTC = 100                    # 고래 기준 (BTC)
TH_SATS      = THRESHOLD_BTC * 1e8     # 사토시 단위

def main():
    df = pd.read_csv(IN_FILE)

    # 필수 컬럼 확인
    for col in ("max_input_value", "max_output_value"):
        if col not in df.columns:
            raise KeyError(f"{col} 컬럼이 없습니다!")

    df["whale_any"] = (
        (df["max_input_value"]  >= TH_SATS) |
        (df["max_output_value"] >= TH_SATS)
    ).astype(int)

    df.to_csv(OUT_FILE, index=False)
    print(f"[√] binary 라벨 저장 → {OUT_FILE}")
    print(df["whale_any"].value_counts())

if __name__ == "__main__":
    main()
