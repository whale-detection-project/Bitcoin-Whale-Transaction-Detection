#!/usr/bin/env python3
# count_whales.py
# ----------------------------------------------
# data/BTC_2024_labeled.csv 에서
#   • whale_recv = 1  (매수-고래)
#   • whale_send = 1  (매매-고래)
#   • whale_cls  = 0/1/2/3
# 를 집계해 출력합니다.
# ----------------------------------------------

import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/BTC_2024_labeled.csv")   # 라벨링된 파일

def main():
    df = pd.read_csv(CSV_PATH)

    # 1) 안전 체크
    need = {"whale_recv", "whale_send", "whale_cls"}
    if not need.issubset(df.columns):
        raise KeyError(f"필수 컬럼 누락: {need - set(df.columns)}")

    # 2) 고래 개수 집계
    total_rows   = len(df)
    recv_whales  = df["whale_recv"].sum()       # 매수-고래
    send_whales  = df["whale_send"].sum()       # 매매-고래
    both_whales  = ((df["whale_recv"] & df["whale_send"])).sum()
    cls_counts   = df["whale_cls"].value_counts().sort_index()

    # 3) 결과 출력
    print(f"총 행 수             : {total_rows:,}")
    print(f"매수-고래(whale_recv) : {recv_whales:,}")
    print(f"매매-고래(whale_send) : {send_whales:,}")
    print(f"양방-고래(둘 다 1)    : {both_whales:,}\n")

    print("whale_cls 분포 (0=일반, 1=매수, 2=매매, 3=양방):")
    for cls, cnt in cls_counts.items():
        print(f"  클래스 {cls}: {cnt:,} 건")

if __name__ == "__main__":
    main()
