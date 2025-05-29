#!/usr/bin/env python3
# label_whale_fixed.py
# -------------------------------------------------
# data/BTC_2024.csv → data/BTC_2024_labeled.csv
# 고래 기준 : 100 BTC 이상
#   • whale_recv 1 ← max_output_value ≥ 100 BTC
#   • whale_send 1 ← max_input_value  ≥ 100 BTC
#   • whale_cls  : 0(일반) 1(매수) 2(매매) 3(양방)
# -------------------------------------------------

import pandas as pd
from pathlib import Path

# ---------- 설정 ----------
IN_PATH  = Path("data/BTC_2024.csv")
OUT_PATH = IN_PATH.with_name("BTC_2024_labeled.csv")
THRESHOLD_BTC = 100
TH_SATOSHI    = THRESHOLD_BTC * 1e8
# ---------------------------

def main():
    # 1) CSV 로드
    df = pd.read_csv(IN_PATH)

    # 2) 필수 컬럼 확인
    for col in ("max_output_value", "max_input_value"):
        if col not in df.columns:
            raise KeyError(f"{col} 컬럼이 없습니다!")

    # 3) 라벨링
    df["whale_recv"] = (df["max_output_value"] >= TH_SATOSHI).astype(int)
    df["whale_send"] = (df["max_input_value"]  >= TH_SATOSHI).astype(int)
    df["whale_cls"]  = df["whale_recv"] + 2 * df["whale_send"]

    # 4) 저장
    df.to_csv(OUT_PATH, index=False)
    print(f"[√] 라벨링 완료 → {OUT_PATH}")
    print(df[["max_output_value", "max_input_value",
              "whale_recv", "whale_send", "whale_cls"]].head())

if __name__ == "__main__":
    main()
