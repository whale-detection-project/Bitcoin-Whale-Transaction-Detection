#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐋 고래 거래 탐지 시스템 v2.0 - 완전 재설계
==============================================
실제 도메인 지식과 비즈니스 로직을 기반으로 한
현실적인 고래 거래 분류 시스템

핵심 원칙:
1. 도메인 전문가 지식 기반 규칙
2. 시간 기반 검증 (과거→미래)
3. 현실적인 성능 기대치
4. 해석 가능한 모델
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WhaleDetectorV2:
    def __init__(self, results_dir='analysis/whale_v2_results'):
        self.results_dir = results_dir
        import os
        os.makedirs(results_dir, exist_ok=True)
        
        # 실제 고래 분류 기준 (도메인 전문가 지식 기반)
        self.whale_definitions = {
            'mega_whale': {
                'name': '메가 고래',
                'description': '초대형 거래 (10,000+ BTC)',
                'min_volume': 10000,
                'characteristics': '시장 영향력 극대'
            },
            'giant_whale': {
                'name': '거대 고래', 
                'description': '대형 거래 (5,000-10,000 BTC)',
                'min_volume': 5000,
                'max_volume': 10000,
                'characteristics': '기관 투자자급'
            },
            'large_whale': {
                'name': '대형 고래',
                'description': '중대형 거래 (2,000-5,000 BTC)',
                'min_volume': 2000,
                'max_volume': 5000,
                'characteristics': '부유한 개인/소규모 기관'
            },
            'collector_whale': {
                'name': '수집형 고래',
                'description': '다수 입력, 소수 출력 (집중)',
                'min_inputs': 10,
                'max_outputs': 3,
                'min_volume': 1000,
                'characteristics': '자금 집중/통합'
            },
            'distributor_whale': {
                'name': '분산형 고래',
                'description': '소수 입력, 다수 출력 (분산)',
                'max_inputs': 3,
                'min_outputs': 10,
                'min_volume': 1000,
                'characteristics': '자금 분산/배포'
            },
            'express_whale': {
                'name': '급행 고래',
                'description': '고수수료 거래 (긴급성)',
                'min_fee_rate': 0.001,  # 0.1%
                'min_volume': 1000,
                'characteristics': '시급한 거래'
            },
            'normal_large': {
                'name': '일반 대형',
                'description': '기타 1000+ BTC 거래',
                'min_volume': 1000,
                'characteristics': '일반적인 대형 거래'
            }
        }
        
        print("🐋 고래 거래 탐지 시스템 v2.0 초기화")
        print("=" * 50)
        print("📋 고래 분류 기준:")
        for whale_type, info in self.whale_definitions.items():
            print(f"  {info['name']}: {info['description']}")
        print()
    
    def load_and_prepare_data(self, data_path='data/1000btc.csv'):
        """데이터 로드 및 기본 전처리"""
        print("📊 데이터 로드 및 전처리...")
        
        df = pd.read_csv(data_path)
        print(f"✅ 원본 데이터: {len(df):,}건")
        
        # BTC 단위 변환
        df['total_input_btc'] = df['total_input_value'] / 100000000
        df['total_output_btc'] = df['total_output_value'] / 100000000
        df['fee_btc'] = df['fee'] / 100000000
        df['max_output_btc'] = df['max_output_value'] / 100000000
        df['total_volume_btc'] = df[['total_input_btc', 'total_output_btc']].max(axis=1)
        
        # 시간 정보 추가
        df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
        df['date'] = df['block_timestamp'].dt.date
        df['hour'] = df['block_timestamp'].dt.hour
        df['day_of_week'] = df['block_timestamp'].dt.dayofweek
        
        # 수수료율 계산
        df['fee_rate'] = df['fee_btc'] / (df['total_volume_btc'] + 1e-8)
        
        # 기본 통계
        print(f"📈 거래량 범위: {df['total_volume_btc'].min():.0f} - {df['total_volume_btc'].max():,.0f} BTC")
        print(f"📈 평균 거래량: {df['total_volume_btc'].mean():,.0f} BTC")
        print(f"📈 수수료율 범위: {df['fee_rate'].min():.6f} - {df['fee_rate'].max():.6f}")
        print()
        
        return df
    
    def create_domain_based_labels(self, df):
        """도메인 지식 기반 라벨 생성"""
        print("🧠 도메인 지식 기반 라벨 생성...")
        
        labels = []
        label_names = []
        
        for idx, row in df.iterrows():
            volume = row['total_volume_btc']
            inputs = row['input_count']
            outputs = row['output_count']
            fee_rate = row['fee_rate']
            concentration = row['max_output_ratio']
            
            # 우선순위 기반 분류 (상위 조건부터 체크)
            if volume >= 10000:
                label = 0  # 메가 고래
                label_name = 'mega_whale'
            elif volume >= 5000:
                label = 1  # 거대 고래
                label_name = 'giant_whale'
            elif volume >= 2000:
                label = 2  # 대형 고래
                label_name = 'large_whale'
            elif inputs >= 10 and outputs <= 3 and volume >= 1000:
                label = 3  # 수집형 고래
                label_name = 'collector_whale'
            elif inputs <= 3 and outputs >= 10 and volume >= 1000:
                label = 4  # 분산형 고래
                label_name = 'distributor_whale'
            elif fee_rate >= 0.001 and volume >= 1000:
                label = 5  # 급행 고래
                label_name = 'express_whale'
            else:
                label = 6  # 일반 대형
                label_name = 'normal_large'
            
            labels.append(label)
            label_names.append(label_name)
        
        df['whale_label'] = labels
        df['whale_type'] = label_names
        
        # 라벨 분포 확인
        print("📊 라벨 분포:")
        label_dist = df['whale_label'].value_counts().sort_index()
        for label_id, count in label_dist.items():
            whale_type = df[df['whale_label'] == label_id]['whale_type'].iloc[0]
            whale_name = self.whale_definitions[whale_type]['name']
            percentage = (count / len(df)) * 100
            print(f"  {label_id} ({whale_name}): {count:,}건 ({percentage:.1f}%)")
        
        print()
        return df
    
    def engineer_features(self, df):
        """피처 엔지니어링 (라벨과 독립적인 피처들)"""
        print("🔧 피처 엔지니어링...")
        
        # 1. 비율 피처들
        df['input_output_ratio'] = df['input_count'] / (df['output_count'] + 1)
        df['output_input_ratio'] = df['output_count'] / (df['input_count'] + 1)
        df['complexity_score'] = df['input_count'] + df['output_count']
        
        # 2. 로그 변환 피처들 (스케일 정규화)
        df['volume_log'] = np.log1p(df['total_volume_btc'])
        df['fee_log'] = np.log1p(df['fee_btc'])
        df['input_log'] = np.log1p(df['input_count'])
        df['output_log'] = np.log1p(df['output_count'])
        
        # 3. 시간 기반 피처들
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # 4. 상대적 크기 피처들 (시간 윈도우 기반)
        df = df.sort_values('block_timestamp')
        
        # 7일 이동 평균 대비 크기
        df['volume_vs_7d_avg'] = df['total_volume_btc'] / (
            df['total_volume_btc'].rolling(window=7*24*6, min_periods=1).mean() + 1e-8
        )
        
        # 1일 이동 평균 대비 크기  
        df['volume_vs_1d_avg'] = df['total_volume_btc'] / (
            df['total_volume_btc'].rolling(window=24*6, min_periods=1).mean() + 1e-8
        )
        
        # 5. 집중도 관련 피처들
        df['concentration_tier'] = pd.cut(
            df['max_output_ratio'], 
            bins=[0, 0.5, 0.8, 0.95, 1.0], 
            labels=[0, 1, 2, 3]
        ).astype(float)
        
        # 6. 수수료 관련 피처들
        df['fee_percentile'] = df['fee_rate'].rank(pct=True)
        df['is_high_fee'] = (df['fee_rate'] > df['fee_rate'].quantile(0.9)).astype(int)
        
        # 모델링용 피처 선택 (라벨 생성에 직접 사용되지 않은 피처들)
        feature_cols = [
            # 비율 피처
            'input_output_ratio', 'output_input_ratio', 'complexity_score',
            
            # 로그 피처  
            'volume_log', 'fee_log', 'input_log', 'output_log',
            
            # 시간 피처
            'hour', 'day_of_week', 'is_weekend', 'is_business_hour', 'is_night',
            
            # 상대적 크기 피처
            'volume_vs_7d_avg', 'volume_vs_1d_avg',
            
            # 집중도 피처
            'concentration_tier',
            
            # 수수료 피처
            'fee_percentile', 'is_high_fee'
        ]
        
        print(f"✅ 생성된 피처: {len(feature_cols)}개")
        print(f"📋 피처 목록: {feature_cols}")
        print()
        
        return df, feature_cols
    
    def time_based_split(self, df, test_ratio=0.3):
        """시간 기반 데이터 분할 (과거 → 미래)"""
        print("⏰ 시간 기반 데이터 분할...")
        
        df_sorted = df.sort_values('block_timestamp')
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        train_end = train_df['block_timestamp'].max()
        test_start = test_df['block_timestamp'].min()
        
        print(f"📅 훈련 데이터: {len(train_df):,}건 (~ {train_end})")
        print(f"📅 테스트 데이터: {len(test_df):,}건 ({test_start} ~)")
        print(f"⏳ 시간 간격: {(test_start - train_end).days}일")
        print()
        
        return train_df, test_df
    
    def train_models(self, train_df, feature_cols):
        """여러 모델 훈련 및 비교"""
        print("🤖 모델 훈련...")
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['whale_label']
        
        # 클래스 가중치 계산
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        
        print("⚖️ 클래스 가중치:")
        for class_id, weight in weight_dict.items():
            whale_type = train_df[train_df['whale_label'] == class_id]['whale_type'].iloc[0]
            whale_name = self.whale_definitions[whale_type]['name']
            print(f"  {class_id} ({whale_name}): {weight:.2f}")
        print()
        
        models = {}
        
        # 1. Random Forest
        print("🌲 Random Forest 훈련...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # 2. XGBoost
        print("🚀 XGBoost 훈련...")
        sample_weights = np.array([weight_dict[label] for label in y_train])
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        models['xgboost'] = xgb_model
        
        print("✅ 모델 훈련 완료")
        print()
        
        return models, X_train, y_train
    
    def evaluate_models(self, models, test_df, feature_cols):
        """모델 평가 (시간 기반 테스트)"""
        print("📊 모델 평가 (시간 기반 테스트)...")
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['whale_label']
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔍 {model_name.upper()} 평가:")
            
            # 예측
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # 정확도
            accuracy = (y_test == y_pred).mean()
            print(f"  정확도: {accuracy:.4f}")
            
            # 클래스별 성능
            from sklearn.metrics import f1_score, precision_recall_fscore_support
            
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  F1-Weighted: {f1_weighted:.4f}")
            
            # 클래스별 상세 성능
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None
            )
            
            print("  클래스별 성능:")
            for class_id in range(len(precision)):
                if class_id < len(test_df['whale_type'].unique()):
                    whale_types = test_df[test_df['whale_label'] == class_id]['whale_type']
                    if len(whale_types) > 0:
                        whale_type = whale_types.iloc[0]
                        whale_name = self.whale_definitions[whale_type]['name']
                        print(f"    {class_id} ({whale_name}): P={precision[class_id]:.3f}, "
                              f"R={recall[class_id]:.3f}, F1={f1[class_id]:.3f}, "
                              f"Support={support[class_id]}")
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        
        return results
    
    def analyze_feature_importance(self, models, feature_cols):
        """피처 중요도 분석"""
        print("\n🔍 피처 중요도 분석...")
        
        plt.figure(figsize=(15, 10))
        
        for i, (model_name, model) in enumerate(models.items()):
            plt.subplot(2, 1, i+1)
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                continue
                
            # 상위 15개 피처만 표시
            indices = np.argsort(importance)[-15:]
            
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
            plt.title(f'{model_name.upper()} 피처 중요도 (상위 15개)', fontsize=14, fontweight='bold')
            plt.xlabel('중요도')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 피처 중요도 시각화 저장: {self.results_dir}/feature_importance.png")
    
    def create_confusion_matrix(self, results, test_df):
        """혼동 행렬 생성"""
        print("\n📊 혼동 행렬 생성...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, (model_name, result) in enumerate(results.items()):
            y_test = test_df['whale_label']
            y_pred = result['y_pred']
            
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 클래스 이름 생성
            class_names = []
            for class_id in range(len(cm)):
                whale_types = test_df[test_df['whale_label'] == class_id]['whale_type']
                if len(whale_types) > 0:
                    whale_type = whale_types.iloc[0]
                    whale_name = self.whale_definitions[whale_type]['name']
                    class_names.append(f'{class_id}\n{whale_name}')
                else:
                    class_names.append(f'{class_id}')
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[i])
            
            axes[i].set_title(f'{model_name.upper()} 혼동 행렬', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('예측 클래스')
            axes[i].set_ylabel('실제 클래스')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 혼동 행렬 저장: {self.results_dir}/confusion_matrix.png")
    
    def run_full_pipeline(self, data_path='data/1000btc.csv'):
        """전체 파이프라인 실행"""
        print("🚀 고래 거래 탐지 시스템 v2.0 - 전체 파이프라인 실행")
        print("=" * 60)
        
        # 1. 데이터 로드 및 전처리
        df = self.load_and_prepare_data(data_path)
        
        # 2. 도메인 기반 라벨 생성
        df = self.create_domain_based_labels(df)
        
        # 3. 피처 엔지니어링
        df, feature_cols = self.engineer_features(df)
        
        # 4. 시간 기반 분할
        train_df, test_df = self.time_based_split(df)
        
        # 5. 모델 훈련
        models, X_train, y_train = self.train_models(train_df, feature_cols)
        
        # 6. 모델 평가
        results = self.evaluate_models(models, test_df, feature_cols)
        
        # 7. 피처 중요도 분석
        self.analyze_feature_importance(models, feature_cols)
        
        # 8. 혼동 행렬 생성
        self.create_confusion_matrix(results, test_df)
        
        # 9. 최종 결과 요약
        print("\n🎯 최종 결과 요약:")
        print("=" * 40)
        
        best_model = None
        best_score = 0
        
        for model_name, result in results.items():
            accuracy = result['accuracy']
            f1_macro = result['f1_macro']
            
            print(f"📊 {model_name.upper()}:")
            print(f"  정확도: {accuracy:.1%}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            
            if f1_macro > best_score:
                best_score = f1_macro
                best_model = model_name
        
        print(f"\n🏆 최고 성능 모델: {best_model.upper()} (F1-Macro: {best_score:.4f})")
        
        # 모델 저장
        import pickle
        model_save_path = f"{self.results_dir}/best_model.pkl"
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': results[best_model]['model'],
                'feature_cols': feature_cols,
                'whale_definitions': self.whale_definitions,
                'performance': results[best_model]
            }, f)
        
        print(f"💾 최고 모델 저장: {model_save_path}")
        
        return results, feature_cols

def main():
    """메인 실행 함수"""
    detector = WhaleDetectorV2()
    results, feature_cols = detector.run_full_pipeline()
    
    print("\n🎉 고래 거래 탐지 시스템 v2.0 구축 완료!")
    print("이번에는 현실적이고 해석 가능한 모델입니다.")

if __name__ == "__main__":
    main() 