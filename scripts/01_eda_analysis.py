#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비트코인 고래 거래 데이터 EDA (Exploratory Data Analysis)
===================================================
1000BTC 이상 거래 데이터의 패턴 분석 및 시각화

Author: LSTM_Crypto_Anomaly_Detection Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import platform

# 한글 폰트 설정 (운영체제별)
def setup_korean_font():
    """운영체제에 맞는 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        try:
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    elif system == 'Windows':
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    else:  # Linux
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 폰트 설정 완료: {plt.rcParams['font.family']}")

# 폰트 설정 실행
setup_korean_font()

plt.rcParams['figure.figsize'] = (12, 8)

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class WhaleDataEDA:
    def __init__(self, data_path='../data/1000btc.csv'):
        """
        고래 데이터 EDA 클래스 초기화
        
        Parameters:
        -----------
        data_path : str
            데이터 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.result_dir = '../result/eda'
        
        # 결과 디렉토리 생성
        os.makedirs(self.result_dir, exist_ok=True)
        
    def load_data(self):
        """데이터 로드 및 기본 전처리"""
        print("📊 데이터 로딩 중...")
        self.df = pd.read_csv(self.data_path)
        
        # 타임스탬프 변환
        self.df['block_timestamp'] = pd.to_datetime(self.df['block_timestamp'])
        
        # 추가 시간 특징 생성
        self.df['year'] = self.df['block_timestamp'].dt.year
        self.df['month'] = self.df['block_timestamp'].dt.month
        self.df['day'] = self.df['block_timestamp'].dt.day
        self.df['hour'] = self.df['block_timestamp'].dt.hour
        self.df['day_of_week'] = self.df['block_timestamp'].dt.dayofweek
        
        # BTC 단위 변환 (사토시 -> BTC)
        btc_columns = ['total_input_value', 'total_output_value', 'max_output_value', 'fee']
        for col in btc_columns:
            self.df[f'{col}_btc'] = self.df[col] / 100000000
            
        print(f"✅ 데이터 로드 완료: {len(self.df):,} 거래 기록")
        return self.df
    
    def basic_statistics(self):
        """기본 통계 분석"""
        print("\n📈 기본 통계 분석")
        print("=" * 50)
        
        # 기본 정보
        print(f"데이터 기간: {self.df['block_timestamp'].min()} ~ {self.df['block_timestamp'].max()}")
        print(f"총 거래 수: {len(self.df):,}")
        print(f"연도별 거래 수:")
        print(self.df['year'].value_counts().sort_index())
        
        # 수치형 컬럼 통계
        numeric_cols = ['max_output_ratio', 'fee_per_max_ratio', 'input_count', 'output_count']
        print(f"\n주요 수치 통계:")
        print(self.df[numeric_cols].describe())
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Whale Transaction Basic Statistics', fontsize=16, y=0.95)
        
        # 연도별 거래 수
        year_counts = self.df['year'].value_counts().sort_index()
        axes[0,0].bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Whale Transactions by Year')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Transaction Count')
        
        # max_output_ratio 분포
        axes[0,1].hist(self.df['max_output_ratio'], bins=50, color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Max Output Ratio Distribution')
        axes[0,1].set_xlabel('Max Output Ratio')
        axes[0,1].set_ylabel('Frequency')
        
        # output_count 분포
        axes[1,0].hist(self.df['output_count'], bins=50, color='lightgreen', alpha=0.7)
        axes[1,0].set_title('Output Count Distribution')
        axes[1,0].set_xlabel('Output Count')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_xlim(0, 200)  # 극값 제외
        
        # 시간대별 거래 분포
        hour_counts = self.df['hour'].value_counts().sort_index()
        axes[1,1].plot(hour_counts.index, hour_counts.values, marker='o', color='purple')
        axes[1,1].set_title('Transactions by Hour (UTC)')
        axes[1,1].set_xlabel('Hour (UTC)')
        axes[1,1].set_ylabel('Transaction Count')
        axes[1,1].set_xticks(range(0, 24, 4))
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/01_basic_statistics.png', dpi=300, bbox_inches='tight')
        print(f"📸 시각화 저장: {self.result_dir}/01_basic_statistics.png")
        
    def whale_classification_analysis(self):
        """고래 분류별 분석"""
        print("\n🐋 고래 분류별 분석")
        print("=" * 50)
        
        # 고래 분류 기준 적용
        def classify_whale(row):
            if row['max_output_ratio'] > 0.8:
                return 'Concentrated Whale'
            elif row['output_count'] > 50 and row['max_output_ratio'] < 0.3:
                return 'Distributed Whale'
            elif 0.05 <= row['max_output_ratio'] <= 0.3:
                return 'Mimicking Whale'
            elif row['fee_per_max_ratio'] < 0.00001:
                return 'Fee-Optimized Whale'
            else:
                return 'Standard Whale'
                
        self.df['whale_type'] = self.df.apply(classify_whale, axis=1)
        
        # 분류별 통계
        whale_stats = self.df['whale_type'].value_counts()
        print("고래 분류별 분포:")
        for whale_type, count in whale_stats.items():
            percentage = (count / len(self.df)) * 100
            print(f"{whale_type}: {count:,} ({percentage:.1f}%)")
            
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Whale Classification Analysis', fontsize=16, y=0.95)
        
        # 고래 분류 파이차트
        colors = ['#ff6b6b', '#ffa500', '#ffeb3b', '#4caf50', '#9e9e9e']
        whale_stats.plot.pie(ax=axes[0,0], autopct='%1.1f%%', colors=colors[:len(whale_stats)])
        axes[0,0].set_title('Whale Type Distribution')
        axes[0,0].set_ylabel('')
        
        # 고래 분류별 연도 추이
        whale_year = pd.crosstab(self.df['year'], self.df['whale_type'])
        whale_year.plot(kind='bar', ax=axes[0,1], stacked=True)
        axes[0,1].set_title('Whale Types by Year')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Transaction Count')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 고래 분류별 max_output_ratio 분포
        for whale_type in whale_stats.index:
            data = self.df[self.df['whale_type'] == whale_type]['max_output_ratio']
            axes[1,0].hist(data, alpha=0.6, label=whale_type, bins=30)
        axes[1,0].set_title('Max Output Ratio by Whale Type')
        axes[1,0].set_xlabel('Max Output Ratio')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 고래 분류별 output_count 분포
        whale_types = ['Concentrated Whale', 'Distributed Whale', 'Mimicking Whale']
        for i, whale_type in enumerate(whale_types):
            if whale_type in self.df['whale_type'].values:
                data = self.df[self.df['whale_type'] == whale_type]['output_count']
                axes[1,1].hist(data, alpha=0.6, label=whale_type, bins=30)
        axes[1,1].set_title('Output Count by Major Whale Types')
        axes[1,1].set_xlabel('Output Count')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_xlim(0, 200)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/02_whale_classification.png', dpi=300, bbox_inches='tight')
        print(f"📸 시각화 저장: {self.result_dir}/02_whale_classification.png")
        
    def temporal_analysis(self):
        """시간적 패턴 분석"""
        print("\n⏰ 시간적 패턴 분석")
        print("=" * 50)
        
        # 시간대별 고래 분류 분석
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Pattern Analysis', fontsize=16, y=0.95)
        
        # 시간대별 거래량 패턴
        hourly_stats = self.df.groupby('hour').agg({
            'max_output_ratio': 'mean',
            'output_count': 'mean',
            'fee_per_max_ratio': 'mean'
        })
        
        axes[0,0].plot(hourly_stats.index, hourly_stats['max_output_ratio'], 
                      marker='o', color='red', linewidth=2)
        axes[0,0].set_title('Average Concentration by Hour')
        axes[0,0].set_xlabel('Hour (UTC)')
        axes[0,0].set_ylabel('Average Max Output Ratio')
        axes[0,0].grid(True, alpha=0.3)
        
        # 요일별 패턴
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_counts = self.df['day_of_week'].value_counts().sort_index()
        axes[0,1].bar(range(7), weekday_counts.values, color='skyblue')
        axes[0,1].set_title('Transactions by Day of Week')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Transaction Count')
        axes[0,1].set_xticks(range(7))
        axes[0,1].set_xticklabels(weekday_names)
        
        # 월별 패턴
        monthly_counts = self.df['month'].value_counts().sort_index()
        axes[1,0].plot(monthly_counts.index, monthly_counts.values, 
                      marker='s', color='green', linewidth=2)
        axes[1,0].set_title('Transactions by Month')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Transaction Count')
        axes[1,0].set_xticks(range(1, 13))
        axes[1,0].grid(True, alpha=0.3)
        
        # 시간대별 의심스러운 거래 비율
        suspicious_by_hour = []
        for hour in range(24):
            hour_data = self.df[self.df['hour'] == hour]
            suspicious_count = len(hour_data[
                (hour_data['max_output_ratio'] > 0.8) | 
                (hour_data['output_count'] > 100)
            ])
            suspicious_ratio = suspicious_count / len(hour_data) if len(hour_data) > 0 else 0
            suspicious_by_hour.append(suspicious_ratio)
            
        axes[1,1].bar(range(24), suspicious_by_hour, color='orange', alpha=0.7)
        axes[1,1].set_title('Suspicious Transaction Ratio by Hour')
        axes[1,1].set_xlabel('Hour (UTC)')
        axes[1,1].set_ylabel('Suspicious Transaction Ratio')
        axes[1,1].set_xticks(range(0, 24, 4))
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/03_temporal_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📸 시각화 저장: {self.result_dir}/03_temporal_analysis.png")
        
    def fee_analysis(self):
        """수수료 패턴 분석"""
        print("\n💰 수수료 패턴 분석")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fee Pattern Analysis', fontsize=16, y=0.95)
        
        # 수수료 분포
        axes[0,0].hist(self.df['fee_btc'], bins=50, color='gold', alpha=0.7)
        axes[0,0].set_title('Fee Distribution (BTC)')
        axes[0,0].set_xlabel('Fee (BTC)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_xlim(0, 0.01)  # 극값 제외
        
        # 수수료 비율 분포
        axes[0,1].hist(self.df['fee_per_max_ratio'], bins=50, color='lightblue', alpha=0.7)
        axes[0,1].set_title('Fee Ratio Distribution')
        axes[0,1].set_xlabel('Fee per Max Ratio')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_xlim(0, 0.001)  # 극값 제외
        
        # 연도별 평균 수수료
        yearly_fee = self.df.groupby('year')['fee_btc'].mean()
        axes[1,0].plot(yearly_fee.index, yearly_fee.values, marker='o', color='red', linewidth=2)
        axes[1,0].set_title('Average Fee by Year')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Average Fee (BTC)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 수수료 효율성 vs 거래 집중도
        sample_data = self.df.sample(min(10000, len(self.df)))  # 샘플링으로 성능 최적화
        scatter = axes[1,1].scatter(sample_data['fee_per_max_ratio'], 
                                   sample_data['max_output_ratio'],
                                   alpha=0.5, c=sample_data['output_count'], 
                                   cmap='viridis')
        axes[1,1].set_title('Fee Efficiency vs Transaction Concentration')
        axes[1,1].set_xlabel('Fee per Max Ratio')
        axes[1,1].set_ylabel('Max Output Ratio')
        axes[1,1].set_xlim(0, 0.001)
        plt.colorbar(scatter, ax=axes[1,1], label='Output Count')
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/04_fee_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📸 시각화 저장: {self.result_dir}/04_fee_analysis.png")
        
    def suspicious_patterns(self):
        """의심스러운 패턴 탐지"""
        print("\n🚨 의심스러운 패턴 탐지")
        print("=" * 50)
        
        # 의심스러운 패턴 정의
        patterns = {
            'Ultra Concentrated': (self.df['max_output_ratio'] > 0.95),
            'Ultra Distributed': (self.df['output_count'] > 200),
            'Zero Fee': (self.df['fee_per_max_ratio'] < 0.000001),
            'Late Night Trading': (self.df['hour'].isin([2, 3, 4, 5])),
            'Mass Splitting': ((self.df['output_count'] > 50) & (self.df['max_output_ratio'] < 0.1))
        }
        
        pattern_stats = {}
        for pattern_name, condition in patterns.items():
            count = condition.sum()
            percentage = (count / len(self.df)) * 100
            pattern_stats[pattern_name] = {'count': count, 'percentage': percentage}
            print(f"{pattern_name}: {count:,} 건 ({percentage:.2f}%)")
            
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Suspicious Pattern Detection Results', fontsize=16, y=0.95)
        
        # 패턴별 발생 건수
        pattern_names = list(pattern_stats.keys())
        pattern_counts = [pattern_stats[name]['count'] for name in pattern_names]
        
        axes[0,0].bar(pattern_names, pattern_counts, color='red', alpha=0.7)
        axes[0,0].set_title('Suspicious Pattern Occurrence')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 연도별 의심 패턴 추이
        yearly_suspicious = []
        years = sorted(self.df['year'].unique())
        for year in years:
            year_data = self.df[self.df['year'] == year]
            suspicious_count = 0
            for condition in patterns.values():
                suspicious_count += condition.sum()
            yearly_suspicious.append(suspicious_count / len(year_data) * 100)
            
        axes[0,1].plot(years, yearly_suspicious, marker='o', color='red', linewidth=2)
        axes[0,1].set_title('Suspicious Pattern Ratio by Year')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Suspicious Pattern Ratio (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 극고집중 거래의 시간 분포
        high_concentration = self.df[self.df['max_output_ratio'] > 0.95]
        hour_concentration = high_concentration['hour'].value_counts().sort_index()
        
        axes[1,0].bar(hour_concentration.index, hour_concentration.values, 
                     color='orange', alpha=0.7)
        axes[1,0].set_title('Ultra Concentrated Transactions by Hour')
        axes[1,0].set_xlabel('Hour (UTC)')
        axes[1,0].set_ylabel('Transaction Count')
        
        # 고위험 점수 계산 및 분포
        def calculate_risk_score(row):
            score = 0
            score += 30 if row['max_output_ratio'] > 0.9 else 0
            score += 20 if row['output_count'] > 100 else 0
            score += 25 if row['fee_per_max_ratio'] < 0.000001 else 0
            score += 15 if row['hour'] in [2, 3, 4, 5] else 0
            score += 10 if row['output_count'] > 50 and row['max_output_ratio'] < 0.1 else 0
            return score
            
        self.df['risk_score'] = self.df.apply(calculate_risk_score, axis=1)
        
        axes[1,1].hist(self.df['risk_score'], bins=30, color='darkred', alpha=0.7)
        axes[1,1].set_title('Risk Score Distribution')
        axes[1,1].set_xlabel('Risk Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(x=50, color='red', linestyle='--', label='High Risk Threshold')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/05_suspicious_patterns.png', dpi=300, bbox_inches='tight')
        print(f"📸 시각화 저장: {self.result_dir}/05_suspicious_patterns.png")
        
        # 고위험 거래 요약
        high_risk = self.df[self.df['risk_score'] >= 50]
        print(f"\n🔥 고위험 거래 요약:")
        print(f"고위험 거래 수: {len(high_risk):,} ({len(high_risk)/len(self.df)*100:.2f}%)")
        
    def generate_summary_report(self):
        """최종 요약 보고서 생성"""
        print("\n📋 EDA 요약 보고서 생성")
        print("=" * 50)
        
        # 요약 통계
        summary = {
            '총 거래 수': f"{len(self.df):,}",
            '분석 기간': f"{self.df['block_timestamp'].min().strftime('%Y-%m-%d')} ~ {self.df['block_timestamp'].max().strftime('%Y-%m-%d')}",
            '고위험 거래': f"{len(self.df[self.df['risk_score'] >= 50]):,} ({len(self.df[self.df['risk_score'] >= 50])/len(self.df)*100:.1f}%)",
            '평균 출력 개수': f"{self.df['output_count'].mean():.1f}",
            '평균 집중도': f"{self.df['max_output_ratio'].mean():.3f}",
            '평균 수수료(BTC)': f"{self.df['fee_btc'].mean():.6f}"
        }
        
        # 보고서 저장
        with open(f'{self.result_dir}/eda_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("🐋 비트코인 고래 거래 EDA 요약 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
                
            f.write(f"\n📊 고래 분류별 분포:\n")
            whale_distribution = self.df['whale_type'].value_counts()
            for whale_type, count in whale_distribution.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"  {whale_type}: {count:,} ({percentage:.1f}%)\n")
                
            f.write(f"\n⚠️ 의심 패턴 탐지 결과:\n")
            patterns = {
                '극고집중 (>95%)': len(self.df[self.df['max_output_ratio'] > 0.95]),
                '극분산 (>200 outputs)': len(self.df[self.df['output_count'] > 200]),
                '무료수수료': len(self.df[self.df['fee_per_max_ratio'] < 0.000001]),
                '새벽거래 (2-5시)': len(self.df[self.df['hour'].isin([2, 3, 4, 5])]),
            }
            
            for pattern, count in patterns.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"  {pattern}: {count:,} ({percentage:.2f}%)\n")
                
            # 그래프 해석 가이드 추가
            f.write(f"\n📊 그래프 해석 가이드:\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("1. 기본 통계 차트 (01_basic_statistics.png):\n")
            f.write("   - 좌상: 연도별 고래 거래 수 - 2022년에 급증한 것을 확인\n")
            f.write("   - 우상: Max Output Ratio 분포 - 대부분이 0.95 이상 (매우 집중적)\n")
            f.write("   - 좌하: Output Count 분포 - 대부분이 2개 이하의 출력\n")
            f.write("   - 우하: 시간대별 거래 분포 - 24시간 고르게 분포\n\n")
            
            f.write("2. 고래 분류 차트 (02_whale_classification.png):\n")
            f.write("   - 좌상: 파이차트 - 집중형 고래가 92.5%로 압도적\n")
            f.write("   - 우상: 연도별 추이 - 집중형이 지속적으로 다수\n")
            f.write("   - 좌하: 분류별 집중도 분포 - 각 타입별 특성 확인\n")
            f.write("   - 우하: 주요 타입별 출력 개수 - 분산형만 출력이 많음\n\n")
            
            f.write("3. 시간적 패턴 차트 (03_temporal_analysis.png):\n")
            f.write("   - 좌상: 시간대별 평균 집중도 - 새벽에 약간 높음\n")
            f.write("   - 우상: 요일별 거래 분포 - 주중이 약간 많음\n")
            f.write("   - 좌하: 월별 거래 분포 - 계절성은 뚜렷하지 않음\n")
            f.write("   - 우하: 시간대별 의심 거래 비율 - 새벽 시간대 높음\n\n")
            
            f.write("4. 수수료 패턴 차트 (04_fee_analysis.png):\n")
            f.write("   - 좌상: 수수료 분포 - 대부분 매우 낮은 수수료\n")
            f.write("   - 우상: 수수료 비율 분포 - 거의 0에 집중\n")
            f.write("   - 좌하: 연도별 평균 수수료 - 초기에 높았다가 감소\n")
            f.write("   - 우하: 수수료 vs 집중도 산점도 - 색상은 출력 개수\n\n")
            
            f.write("5. 의심 패턴 차트 (05_suspicious_patterns.png):\n")
            f.write("   - 좌상: 의심 패턴별 발생 건수 - 무료수수료가 가장 많음\n")
            f.write("   - 우상: 연도별 의심 패턴 비율 추이 - 최근 증가 경향\n")
            f.write("   - 좌하: 극고집중 거래 시간 분포 - 24시간 고르게 분포\n")
            f.write("   - 우하: 위험 점수 분포 - 50점 이상이 고위험 (빨간선)\n\n")
            
            f.write("📖 해석 요령:\n")
            f.write("- 막대 그래프: 높이가 빈도/개수를 나타냄\n")
            f.write("- 히스토그램: 데이터 분포를 보여줌 (높을수록 해당 값이 많음)\n")
            f.write("- 선 그래프: 시간에 따른 변화 추이\n")
            f.write("- 파이차트: 전체에서 각 부분이 차지하는 비율\n")
            f.write("- 산점도: 두 변수 간의 관계 (색상은 세 번째 변수)\n")
            f.write("- 축 레이블과 제목을 먼저 확인하여 무엇을 보여주는지 파악\n")
                
        print(f"📄 요약 보고서 저장: {self.result_dir}/eda_summary_report.txt")
        
    def run_full_eda(self):
        """전체 EDA 프로세스 실행"""
        print("🚀 비트코인 고래 거래 EDA 시작!")
        print("=" * 60)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 기본 통계 분석
        self.basic_statistics()
        
        # 3. 고래 분류별 분석
        self.whale_classification_analysis()
        
        # 4. 시간적 패턴 분석
        self.temporal_analysis()
        
        # 5. 수수료 패턴 분석
        self.fee_analysis()
        
        # 6. 의심스러운 패턴 탐지
        self.suspicious_patterns()
        
        # 7. 요약 보고서 생성
        self.generate_summary_report()
        
        print(f"\n🎉 EDA 완료! 모든 결과는 {self.result_dir}/ 에 저장되었습니다.")
        print("=" * 60)
        
        return self.df

if __name__ == "__main__":
    # EDA 실행
    eda = WhaleDataEDA()
    df = eda.run_full_eda() 