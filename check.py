#!/usr/bin/env python3
# count_whales_binary.py
# ----------------------------------------------------------
#  data/BTC_2024_binary.csv  에서
#     • whale_any = 1  (고래)   / 0 (일반)
#  이진 라벨 분포·비율을 출력합니다.
# ----------------------------------------------------------

import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/BTC_2024_binary.csv")   # ← 이진 라벨링된 파일

def main():
    df = pd.read_csv(CSV_PATH)

    # 1) 필수 컬럼 확인
    if "whale_any" not in df.columns:
        raise KeyError("'whale_any' 컬럼이 없습니다. 라벨링 스크립트부터 실행해 주세요.")

    # 2) 분포 집계
    total   = len(df)
    pos_cnt = int(df["whale_any"].sum())          # 고래(1)
    neg_cnt = total - pos_cnt                     # 일반(0)

    pos_pct = pos_cnt / total * 100
    neg_pct = 100 - pos_pct
    ratio   = f"{neg_cnt/pos_cnt:.1f} : 1" if pos_cnt else "∞"

    # 3) 출력
    print(f"총 트랜잭션 수 : {total:,}")
    print(f"고래(1)        : {pos_cnt:,} 건  ({pos_pct:.4f} %)")
    print(f"일반(0)        : {neg_cnt:,} 건  ({neg_pct:.4f} %)")
    print(f"불균형 비율    : {ratio}  (일반 : 고래)")

if __name__ == "__main__":
    main()
