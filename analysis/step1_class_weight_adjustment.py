"""
🌳 Step 1: Class Weight 조정을 통한 클래스 불균형 해결

목적: 1000 BTC 이상 대형 거래에서 실제 거래 행동 패턴 기반 고래 분류
- 데이터: data/1000btc.csv (888,943건의 대형 Bitcoin 거래)
- 방법: 실제 거래 행동 패턴으로 라벨링 후 Class Weight 조정
- 목표: Random Forest 기반 거래 행동 패턴 분류 성능 개선

거래 행동 패턴 분류:
1. 수집형 고래: Input > 상위 20% AND Output ≤ 2개 (~150,000건, 17%)
2. 분산형 고래: Output > 상위 15% AND 집중도 < 0.8 (~130,000건, 15%)
3. 급행형 고래: Fee > 상위 5% (44,448건, 5%)
4. 집중형 고래: 집중도 > 0.99 AND 거래량 > 중앙값 (~340,000건, 38%)
5. 거대형 고래: 거래량 > 상위 1% (8,890건, 1%)

주요 작업:
1. 실제 거래 행동 패턴 기반 라벨링
2. 자연스러운 클래스 불균형 상황 생성
3. Class Weight 조정 (4가지 전략) 비교
4. Random Forest 모델 성능 개선 분석
5. 최적 데이터셋 준비

예상 결과: Random Forest용 최적화된 거래 행동 패턴 분류 데이터셋
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WhaleDetectionAnalyzer:
    def __init__(self, results_dir='analysis/step1_results/class_weight_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 실제 거래 행동 패턴 기반 고래 분류 체계
        self.whale_types = {
            'collector': 0,    # 수집형 고래 - Input 많고 Output 적음
            'distributor': 1,  # 분산형 고래 - Output 많고 분산
            'express': 2,      # 급행형 고래 - 높은 수수료
            'focused': 3,      # 집중형 고래 - 높은 집중도
            'mega': 4          # 거대형 고래 - 극대 거래량
        }
        
        # 클래스 이름 매핑 (실제 거래 행동 패턴)
        self.class_names = {
            0: '수집형고래',    # Input > 상위 20% AND Output ≤ 2개
            1: '분산형고래',    # Output > 상위 15% AND 집중도 < 0.8  
            2: '급행형고래',    # Fee > 상위 5%
            3: '집중형고래',    # 집중도 > 0.99 AND 거래량 > 중앙값
            4: '거대형고래'     # 거래량 > 상위 1%
        }
        
        print("🐋 실제 거래 행동 패턴 기반 고래 분류 분석 도구가 초기화되었습니다.")
        print(f"📁 결과 저장 경로: {results_dir}")
        print("📊 분석 대상: 1000 BTC 이상 대형 거래의 행동 패턴 분류")
        print("🎯 목표: 수집형, 분산형, 급행형, 집중형, 거대형 고래 구분")
    
    def load_and_label_data(self, data_path='data/1000btc.csv'):
        """1000 BTC 이상 대형 거래 데이터 로드 및 실제 행동 패턴 기반 고래 분류"""
        print("📊 대형 거래 데이터 로딩 및 행동 패턴 기반 고래 분류 중...")
        
        # 원본 데이터 로드
        df = pd.read_csv(data_path)
        print(f"✅ 대형 거래 데이터 로드 완료: {len(df):,}건")
        
        # BTC 단위로 변환 (satoshi → BTC)
        df['total_output_btc'] = df['total_output_value'] / 100000000
        df['total_input_btc'] = df['total_input_value'] / 100000000  
        df['fee_btc'] = df['fee'] / 100000000
        df['max_output_btc'] = df['max_output_value'] / 100000000
        
        # 총 거래량 (Input과 Output 중 최대값)
        df['total_volume_btc'] = df[['total_input_btc', 'total_output_btc']].max(axis=1)
        
        # 집중도 계산 (이미 있는 max_output_ratio 활용)
        df['concentration'] = df['max_output_ratio']
        
        # 분류 기준 임계값 계산
        volume_p99 = df['total_volume_btc'].quantile(0.99)  # 상위 1%
        volume_median = df['total_volume_btc'].median()     # 중앙값
        input_p80 = df['input_count'].quantile(0.80)       # 상위 20%
        output_p85 = df['output_count'].quantile(0.85)     # 상위 15%
        fee_p95 = df['fee_btc'].quantile(0.95)             # 상위 5%
        
        print(f"📊 분류 기준 임계값:")
        print(f"  거래량 상위 1%: {volume_p99:.0f} BTC")
        print(f"  거래량 중앙값: {volume_median:.0f} BTC")
        print(f"  Input 상위 20%: {input_p80:.0f}개")
        print(f"  Output 상위 15%: {output_p85:.0f}개")
        print(f"  수수료 상위 5%: {fee_p95:.4f} BTC")
        
        # 실제 거래 행동 패턴 기반 분류
        def classify_whale_behavior(row):
            # 우선순위 기반 분류 (겹치는 경우 우선순위가 높은 것으로 분류)
            
            # 1. 거대형 고래 (최우선) - 거래량 > 상위 1%
            if row['total_volume_btc'] >= volume_p99:
                return 4  # 거대형 고래
            
            # 2. 급행형 고래 - Fee > 상위 5%
            elif row['fee_btc'] >= fee_p95:
                return 2  # 급행형 고래
            
            # 3. 수집형 고래 - Input > 상위 20% AND Output ≤ 2개
            elif row['input_count'] >= input_p80 and row['output_count'] <= 2:
                return 0  # 수집형 고래
            
            # 4. 분산형 고래 - Output > 상위 15% AND 집중도 < 0.8
            elif row['output_count'] >= output_p85 and row['concentration'] < 0.8:
                return 1  # 분산형 고래
            
            # 5. 집중형 고래 - 집중도 > 0.99 AND 거래량 > 중앙값
            elif row['concentration'] > 0.99 and row['total_volume_btc'] > volume_median:
                return 3  # 집중형 고래
            
            # 6. 기본값 - 집중형 고래 (나머지)
            else:
                return 3  # 집중형 고래 (기본)
        
        # 라벨링 적용
        df['whale_class'] = df.apply(classify_whale_behavior, axis=1)
        
        # 피처 선택 (행동 패턴 분석에 적합한 피처들)
        feature_cols = [
            'total_volume_btc',    # 총 거래량 (핵심 지표)
            'input_count',         # 입력 개수 (수집 패턴)
            'output_count',        # 출력 개수 (분산 패턴)  
            'concentration',       # 집중도 (max_output_ratio)
            'fee_btc'              # 수수료 (급행 패턴)
        ]
        
        # 데이터 정리
        X = df[feature_cols].copy()
        y = df['whale_class'].copy()
        
        # 결측값 처리
        X = X.fillna(0)
        
        # 극단값 처리 (IQR 방식)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[col] = X[col].clip(lower_bound, upper_bound)
        
        print(f"📊 선택된 피처: {feature_cols}")
        print(f"🏷️ 클래스 수: {len(y.unique())}개")
        
        # 실제 거래 행동 패턴 분포 출력
        class_dist = y.value_counts().sort_index()
        total = len(y)
        print("\n📊 실제 거래 행동 패턴 분포:")
        for cls, count in class_dist.items():
            percentage = (count / total) * 100
            print(f"  클래스 {cls} ({self.class_names[cls]}): {count:,}건 ({percentage:.1f}%)")
        
        # 각 클래스별 특성 확인
        print("\n💼 각 클래스별 실제 특성:")
        for cls in sorted(y.unique()):
            mask = y == cls
            cls_data = df[mask]
            avg_volume = cls_data['total_volume_btc'].mean()
            avg_input = cls_data['input_count'].mean()
            avg_output = cls_data['output_count'].mean()
            avg_concentration = cls_data['concentration'].mean()
            avg_fee = cls_data['fee_btc'].mean()
            
            print(f"  클래스 {cls} ({self.class_names[cls]}):")
            print(f"    평균 거래량: {avg_volume:.0f} BTC")
            print(f"    평균 Input: {avg_input:.1f}개")
            print(f"    평균 Output: {avg_output:.1f}개")
            print(f"    평균 집중도: {avg_concentration:.3f}")
            print(f"    평균 수수료: {avg_fee:.4f} BTC")
        
        # 클래스 불균형 비율 계산
        max_class = class_dist.max()
        min_class = class_dist.min()
        imbalance_ratio = max_class / min_class
        print(f"\n⚖️ 클래스 불균형 비율: {imbalance_ratio:.1f}:1")
        
        return X, y
    
    def calculate_class_weights(self, y):
        """실제 거래 행동 패턴 기반 고래 분류에 특화된 class weight 전략 계산"""
        print("\n⚖️ 거래 행동 패턴 기반 Class Weight 전략 계산 중...")
        
        classes = np.unique(y)
        
        # 1. Balanced (sklearn 기본)
        balanced_weights = compute_class_weight('balanced', classes=classes, y=y)
        balanced_dict = dict(zip(classes, balanced_weights))
        
        # 2. Behavior-Focused (행동 패턴 중심)
        # 특수한 행동 패턴(수집형, 분산형, 급행형, 거대형)에 높은 가중치
        behavior_dict = {}
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        for cls in classes:
            base_weight = total_samples / (len(classes) * class_counts[cls])
            if cls == 4:  # 거대형 고래 (극희귀)
                behavior_dict[cls] = base_weight * 50  # 최고 가중치
            elif cls == 2:  # 급행형 고래 (높은 수수료)
                behavior_dict[cls] = base_weight * 20  # 높은 가중치
            elif cls == 0:  # 수집형 고래 (특수 패턴)
                behavior_dict[cls] = base_weight * 10
            elif cls == 1:  # 분산형 고래 (분산 패턴)
                behavior_dict[cls] = base_weight * 5
            else:  # 집중형 고래 (일반적)
                behavior_dict[cls] = base_weight * 1
        
        # 3. Rarity-Based (희귀도 기반)
        # 클래스별 희귀도에 따른 가중치 조정
        rarity_dict = {}
        for cls in classes:
            count = class_counts[cls]
            percentage = (count / total_samples) * 100
            
            if percentage < 2:  # 2% 미만 - 극희귀
                rarity_dict[cls] = 30.0
            elif percentage < 5:  # 5% 미만 - 매우 희귀
                rarity_dict[cls] = 15.0
            elif percentage < 15:  # 15% 미만 - 희귀
                rarity_dict[cls] = 5.0
            elif percentage < 25:  # 25% 미만 - 적음
                rarity_dict[cls] = 2.0
            else:  # 25% 이상 - 일반적
                rarity_dict[cls] = 0.8
        
        # 4. Business-Priority (비즈니스 우선순위)
        # 실제 비즈니스 중요도에 따른 가중치
        business_dict = {}
        base_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        priority_multipliers = {
            4: 25.0,  # 거대형 고래 - 최고 우선순위 (시장 영향력)
            2: 15.0,  # 급행형 고래 - 높은 우선순위 (긴급성)
            0: 8.0,   # 수집형 고래 - 중간 우선순위 (축적 패턴)
            1: 5.0,   # 분산형 고래 - 중간 우선순위 (유통 패턴)
            3: 2.0    # 집중형 고래 - 낮은 우선순위 (일반적)
        }
        
        for i, cls in enumerate(classes):
            business_dict[cls] = base_weights[i] * priority_multipliers.get(cls, 1.0)
        
        strategies = {
            'balanced': balanced_dict,
            'behavior_focused': behavior_dict,
            'rarity_based': rarity_dict,
            'business_priority': business_dict
        }
        
        # 전략 출력
        for name, weights in strategies.items():
            print(f"\n🎯 {name.title().replace('_', ' ')} 전략:")
            for cls, weight in weights.items():
                print(f"  클래스 {cls} ({self.class_names[cls]}): {weight:.2f}")
        
        return strategies
    
    def evaluate_model_with_cv(self, X, y, class_weight=None, strategy_name="Baseline"):
        """현실적인 고래 거래 탐지 모델 평가 (진행률 표시 포함)"""
        print(f"\n🔄 {strategy_name} 전략 평가 시작...")
        
        # 표준화 (5% 완료)
        print(f"  ⏳ 데이터 표준화 중... (5%)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 고래 탐지에 적합한 모델 설정 (10% 완료)
        print(f"  ⏳ Random Forest 모델 설정 중... (10%)")
        model = RandomForestClassifier(
            n_estimators=100,       # 충분한 트리 수
            random_state=42,
            class_weight=class_weight,
            max_depth=8,           # 적절한 깊이
            min_samples_split=50,   # 과적합 방지
            min_samples_leaf=20,    # 리프 노드 최소 샘플
            max_features='sqrt',    # 피처 샘플링
            bootstrap=True,
            oob_score=True
        )
        
        # 5-Fold 계층화 교차 검증 (10% -> 70% 완료)
        print(f"  ⏳ 5-Fold 교차 검증 실행 중...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 교차 검증을 단계별로 실행하여 진행률 표시
        cv_scores = []
        fold_progress = [20, 30, 45, 60, 70]  # 각 fold별 진행률
        
        for i, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            print(f"    📊 Fold {i+1}/5 처리 중... ({fold_progress[i]}%)")
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # 모델 훈련 및 예측
            model_fold = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight=class_weight,
                max_depth=8, min_samples_split=50, min_samples_leaf=20,
                max_features='sqrt', bootstrap=True
            )
            model_fold.fit(X_train_fold, y_train_fold)
            y_pred_fold = model_fold.predict(X_val_fold)
            
            # F1-Score 계산
            fold_f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
            cv_scores.append(fold_f1)
        
        cv_scores = np.array(cv_scores)
        
        # 홀드아웃 테스트 세트로 최종 평가 (70% -> 90% 완료)
        print(f"  ⏳ 홀드아웃 테스트 평가 중... (80%)")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"  ⏳ 최종 모델 훈련 중... (85%)")
        model.fit(X_train, y_train)
        
        print(f"  ⏳ 최종 예측 및 평가 중... (90%)")
        y_pred = model.predict(X_test)
        
        # 다양한 평가 지표 계산 (90% -> 100% 완료)
        print(f"  ⏳ 평가 지표 계산 중... (95%)")
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 고래 클래스 전용 F1-Score (클래스 1,2,3,4)
        whale_mask = y_test > 0
        if whale_mask.sum() > 0:
            whale_f1 = f1_score(y_test[whale_mask], y_pred[whale_mask], average='macro')
        else:
            whale_f1 = 0.0
        
        # 상세 리포트
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"  ✅ {strategy_name} 평가 완료! (100%)")
        print(f"📊 {strategy_name}:")
        print(f"  교차검증 F1-Macro: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"  테스트 F1-Macro: {f1_macro:.4f}")
        print(f"  고래 전용 F1-Score: {whale_f1:.4f}")
        print(f"  OOB Score: {model.oob_score_:.4f}")
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_f1_macro': f1_macro,
            'test_f1_weighted': f1_weighted,
            'whale_f1': whale_f1,
            'oob_score': model.oob_score_,
            'cv_scores': cv_scores,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
    
    def run_analysis(self, data_path='data/1000btc.csv'):
        """전체 분석 실행 (진행률 표시 포함)"""
        print("🚀 실제 거래 행동 패턴 기반 고래 분류 분석을 시작합니다!")
        print("=" * 60)
        print("📊 분석 범위: 1000 BTC 이상 대형 거래의 실제 행동 패턴 분류")
        print("🎯 목표: 수집형, 분산형, 급행형, 집중형, 거대형 고래 구분")
        print("🌳 모델: Random Forest 기반 분류")
        print("⚖️ 목적: Class Weight 조정을 통한 최적 데이터셋 준비")
        
        # 데이터 준비 (0% -> 20%)
        print(f"\n⏳ 전체 진행률: 0% - 데이터 로딩 및 라벨링 시작...")
        X, y = self.load_and_label_data(data_path)
        print(f"✅ 전체 진행률: 20% - 데이터 준비 완료!")
        
        # Class Weight 전략 계산 (20% -> 25%)
        print(f"\n⏳ 전체 진행률: 20% - Class Weight 전략 계산 중...")
        strategies = self.calculate_class_weights(y)
        print(f"✅ 전체 진행률: 25% - Class Weight 전략 계산 완료!")
        
        # 결과 저장
        results = {}
        
        print(f"\n🌳 거래 행동 패턴 분류 모델 성능 비교 (교차 검증 포함):")
        print("=" * 50)
        
        total_strategies = 5  # baseline + 4개 전략
        current_strategy = 0
        
        # 1. Baseline (가중치 없음) (25% -> 40%)
        current_strategy += 1
        progress = 25 + (current_strategy / total_strategies) * 60  # 25%에서 85%까지
        print(f"\n🔄 전체 진행률: {progress:.0f}% - Baseline 전략 평가 중... ({current_strategy}/{total_strategies})")
        baseline_result = self.evaluate_model_with_cv(X, y, None, "Baseline")
        results['baseline'] = baseline_result
        print(f"✅ Baseline 전략 완료!")
        
        # 2. 각 전략별 평가 (40% -> 85%)
        for i, (strategy_name, class_weights) in enumerate(strategies.items()):
            current_strategy += 1
            progress = 25 + (current_strategy / total_strategies) * 60
            strategy_display_name = strategy_name.title().replace('_', ' ')
            print(f"\n🔄 전체 진행률: {progress:.0f}% - {strategy_display_name} 전략 평가 중... ({current_strategy}/{total_strategies})")
            result = self.evaluate_model_with_cv(X, y, class_weights, strategy_display_name)
            results[strategy_name] = result
            print(f"✅ {strategy_display_name} 전략 완료!")
        
        # 결과 저장 및 시각화 (85% -> 100%)
        print(f"\n⏳ 전체 진행률: 85% - 결과 저장 및 시각화 중...")
        print(f"  📄 상세 결과 저장 중... (90%)")
        self.save_results(results, strategies, X, y)
        print(f"  📊 시각화 생성 중... (95%)")
        self.create_visualizations(results)
        
        # 최적화된 데이터셋 저장 (95% -> 100%)
        best_strategy = max(results.keys(), key=lambda k: results[k]['test_f1_macro'])
        print(f"  💾 최적화된 데이터셋 저장 중... (98%)")
        dataset_files = self.save_optimized_dataset(X, y, best_strategy, strategies, results)
        
        print(f"✅ 전체 진행률: 100% - 모든 분석 완료! 🎉")
        
        # 최종 요약
        best_strategy = max(results.keys(), key=lambda k: results[k]['test_f1_macro'])
        best_f1 = results[best_strategy]['test_f1_macro']
        baseline_f1 = results['baseline']['test_f1_macro']
        improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
        
        print(f"\n🏆 최적 전략: {best_strategy.replace('_', ' ').title()}")
        print(f"📊 최고 F1-Score: {best_f1:.4f}")
        print(f"📈 개선율: {improvement:+.1f}%")
        print(f"📁 결과 저장 위치: {self.results_dir}/")
        
        # 저장된 데이터셋 파일 정보 출력
        print(f"\n💾 저장된 최적화 데이터셋:")
        print(f"  📊 전체 데이터셋: {dataset_files['dataset_file']}")
        print(f"  🎯 훈련 데이터셋: {dataset_files['train_file']}")
        print(f"  🧪 테스트 데이터셋: {dataset_files['test_file']}")
        print(f"  ⚙️ 최적 설정: {dataset_files['config_file']}")
        print(f"  📐 스케일러: {dataset_files['scaler_file']}")
        print(f"  📖 사용 가이드: {dataset_files['guide_file']}")
        
        return results
    
    def create_visualizations(self, results):
        """향상된 시각화 생성"""
        # 1. 성능 비교 차트 (2x3 레이아웃)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        
        strategies = list(results.keys())
        
        # 교차 검증 F1-Score 비교
        cv_means = [results[s]['cv_mean'] for s in strategies]
        cv_stds = [results[s]['cv_std'] for s in strategies]
        
        bars1 = ax1.bar(strategies, cv_means, yerr=cv_stds, capsize=5, 
                       color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax1.set_title('교차검증 F1-Macro 비교\n(에러바: ±2σ)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, mean, std in zip(bars1, cv_means, cv_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 테스트 F1-Macro 비교  
        test_f1s = [results[s]['test_f1_macro'] for s in strategies]
        bars2 = ax2.bar(strategies, test_f1s, 
                       color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax2.set_title('테스트 F1-Macro 비교', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, f1 in zip(bars2, test_f1s):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # OOB Score 비교
        oob_scores = [results[s]['oob_score'] for s in strategies]
        bars3 = ax3.bar(strategies, oob_scores, 
                       color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax3.set_title('OOB Score 비교\n(과적합 검증)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('OOB Score')
        ax3.set_ylim(0, 1.0)
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, score in zip(bars3, oob_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 교차 검증 표준편차 (안정성)
        ax4.bar(strategies, cv_stds, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax4.set_title('교차검증 표준편차\n(낮을수록 안정적)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('표준편차')
        ax4.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for i, (strategy, std) in enumerate(zip(strategies, cv_stds)):
            ax4.text(i, std + max(cv_stds)*0.02, f'{std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 성능 개선율
        baseline_f1 = results['baseline']['test_f1_macro']
        improvements = [(results[s]['test_f1_macro'] - baseline_f1) * 100 for s in strategies[1:]]
        strategy_names = [s.replace('_', ' ').title() for s in strategies[1:]]
        
        colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
        bars5 = ax5.bar(strategy_names, improvements, color=colors)
        ax5.set_title('Baseline 대비 개선율 (%)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('개선율 (%)')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, imp in zip(bars5, improvements):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            y_pos = height + max(abs(max(improvements)), abs(min(improvements))) * 0.05 if height >= 0 else height - max(abs(max(improvements)), abs(min(improvements))) * 0.05
            ax5.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{imp:+.1f}%', ha='center', va=va, fontweight='bold', fontsize=10)
        
        # 피처 중요도 (baseline 모델 기준)
        feature_importance = results['baseline']['feature_importance']
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        bars6 = ax6.barh(features, importances, color='skyblue')
        ax6.set_title('피처 중요도\n(Baseline 모델)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('중요도')
        
        # 값 표시
        for bar, imp in zip(bars6, importances):
            width = bar.get_width()
            ax6.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{imp:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 혼동 행렬 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (strategy, result) in enumerate(results.items()):
            if i >= 6:  # 최대 6개만 표시
                break
                
            cm = result['confusion_matrix']
            
            # 정규화된 혼동 행렬
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            im = axes[i].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[i].set_title(f'{strategy.title().replace("_", " ")}\n혼동 행렬', fontsize=12)
            
            # 클래스 라벨
            class_labels = [f'{j}\n({self.class_names[j]})' for j in range(len(cm))]
            axes[i].set_xticks(range(len(cm)))
            axes[i].set_yticks(range(len(cm)))
            axes[i].set_xticklabels(class_labels, rotation=45, ha='right')
            axes[i].set_yticklabels(class_labels)
            
            # 값 표시
            for j in range(len(cm)):
                for k in range(len(cm)):
                    text_color = 'white' if cm_normalized[j, k] > 0.5 else 'black'
                    axes[i].text(k, j, f'{cm_normalized[j, k]:.2f}',
                               ha='center', va='center', color=text_color, fontweight='bold')
            
            axes[i].set_ylabel('실제 클래스')
            axes[i].set_xlabel('예측 클래스')
        
        # 빈 subplot 숨기기
        for i in range(len(results), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 성능 비교 차트 저장: {self.results_dir}/performance_comparison.png")
        print(f"📊 혼동 행렬 저장: {self.results_dir}/confusion_matrices.png")

    def save_results(self, results, strategies, X, y):
        """결과를 텍스트 파일로 저장"""
        print("\n💾 상세 결과 저장 중...")
        
        results_file = f"{self.results_dir}/detailed_analysis_results.txt"
        
        # 데이터 분할 정보 계산
        total_samples = len(X)
        train_samples = int(total_samples * 0.7)
        test_samples = total_samples - train_samples
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("🌳 고래 거래 탐지 분석 - 상세 분석 결과\n")
            f.write("="*80 + "\n")
            f.write(f"📅 분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 총 샘플 수: {total_samples:,}건\n")
            f.write(f"🎯 훈련 데이터: {train_samples:,}건 (70%)\n")
            f.write(f"🎯 테스트 데이터: {test_samples:,}건 (30%)\n")
            f.write(f"📈 피처 수: {len(X.columns)}개\n")
            f.write(f"📈 사용 피처: {list(X.columns)}\n")
            
            # 클래스 분포
            class_counts = pd.Series(y).value_counts().sort_index()
            max_class = class_counts.max()
            min_class = class_counts.min()
            imbalance_ratio = max_class / min_class
            f.write(f"⚖️ 클래스 불균형 비율: {imbalance_ratio:.1f}:1\n\n")
            
            # 클래스 분포 상세
            f.write("📊 클래스 분포:\n")
            for class_id, count in class_counts.items():
                percentage = (count / len(X)) * 100
                class_name = self.class_names.get(class_id, f"클래스{class_id}")
                f.write(f"  클래스 {class_id} ({class_name}): {count:,}건 ({percentage:.1f}%)\n")
            
            # Class Weight 전략
            f.write("\n" + "="*50 + "\n")
            f.write("⚖️ 계산된 Class Weight 전략:\n")
            f.write("="*50 + "\n")
            
            for strategy, weights in strategies.items():
                f.write(f"\n🎯 {strategy.title().replace('_', ' ')} 전략:\n")
                for class_id, weight in weights.items():
                    f.write(f"  클래스 {class_id}: {weight:.2f}\n")
            
            # 모델 성능 비교
            f.write("\n" + "="*50 + "\n")
            f.write("🌳 모델 성능 비교:\n")
            f.write("="*50 + "\n")
            
            all_strategies = ['baseline'] + list(strategies.keys())
            all_names = ['Baseline (가중치 없음)'] + [s.title().replace('_', ' ') for s in strategies.keys()]
            
            for strategy, name in zip(all_strategies, all_names):
                result = results[strategy]
                f.write(f"\n📊 {name}:\n")
                f.write(f"  교차검증 F1-Macro: {result['cv_mean']:.4f} (±{result['cv_std']*2:.4f})\n")
                f.write(f"  테스트 F1-Macro: {result['test_f1_macro']:.4f}\n")
                f.write(f"  고래 전용 F1-Score: {result['whale_f1']:.4f}\n")
                f.write(f"  OOB Score: {result['oob_score']:.4f}\n")
                
                if strategy != 'baseline':
                    baseline_f1 = results['baseline']['test_f1_macro']
                    improvement = ((result['test_f1_macro'] - baseline_f1) / baseline_f1) * 100
                    f.write(f"  개선율: {improvement:+.1f}%\n")
            
            # 피처 중요도 분석
            f.write("\n" + "="*50 + "\n")
            f.write("📊 피처 중요도 분석 (Baseline 모델):\n")
            f.write("="*50 + "\n")
            
            feature_importance = results['baseline']['feature_importance']
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features:
                f.write(f"  {feature}: {importance:.4f}\n")
            
            # 클래스별 상세 성능 분석 (상위 3개 전략만)
            f.write("\n" + "="*50 + "\n")
            f.write("📈 클래스별 상세 성능 분석:\n")
            f.write("="*50 + "\n")
            
            # 상위 3개 전략 선택
            strategy_f1_scores = {s: results[s]['test_f1_macro'] for s in all_strategies}
            top_strategies = sorted(strategy_f1_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for strategy, _ in top_strategies:
                strategy_name = 'Baseline (가중치 없음)' if strategy == 'baseline' else strategy.title().replace('_', ' ')
                f.write(f"\n🎯 {strategy_name} 전략:\n")
                f.write("-" * 40 + "\n")
                
                report = results[strategy]['classification_report']
                
                for class_id in range(5):
                    class_name = self.class_names.get(class_id, f"클래스{class_id}")
                    
                    try:
                        class_report = report[str(class_id)]
                    except KeyError:
                        try:
                            class_report = report[class_id]
                        except KeyError:
                            continue
                    
                    precision = class_report['precision']
                    recall = class_report['recall']
                    f1_score = class_report['f1-score']
                    support = class_report['support']
                    
                    f.write(f"  클래스 {class_id} ({class_name}):\n")
                    f.write(f"    Precision: {precision:.4f}\n")
                    f.write(f"    Recall: {recall:.4f}\n")
                    f.write(f"    F1-Score: {f1_score:.4f}\n")
                    f.write(f"    Support: {support}\n")
            
            # 권장사항
            f.write("\n" + "="*50 + "\n")
            f.write("💡 분석 결과 및 권장사항:\n")
            f.write("="*50 + "\n")
            
            # 최적 전략 찾기
            best_strategy = max(strategy_f1_scores, key=strategy_f1_scores.get)
            best_f1 = strategy_f1_scores[best_strategy]
            baseline_f1 = strategy_f1_scores['baseline']
            overall_improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
            
            f.write(f"\n🏆 최적 전략: {best_strategy}\n")
            f.write(f"📊 최고 F1-Score: {best_f1:.4f}\n")
            f.write(f"📈 전체 개선율: {overall_improvement:+.1f}%\n")
            
            f.write(f"\n💼 프로덕션 권장사항:\n")
            if overall_improvement > 1:
                f.write(f"1. '{best_strategy}' 전략 사용 권장 - {overall_improvement:.1f}% 성능 개선\n")
                f.write(f"2. 소수 클래스 탐지율 개선 효과 확인됨\n")
            else:
                f.write(f"1. Class Weight 조정으로는 제한적 개선 ({overall_improvement:.1f}%)\n")
                f.write(f"2. 대안: SMOTE 오버샘플링, 피처 엔지니어링, 앙상블 고려\n")
            
            f.write(f"3. 과적합 방지를 위한 교차 검증 지속 모니터링\n")
            f.write(f"4. 비즈니스 요구사항에 따른 클래스별 가중치 조정 고려\n")
        
        print(f"📄 상세 결과 저장: {results_file}")
        
        # 빠른 요약 파일도 생성
        summary_file = f"{self.results_dir}/quick_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("🌳 고래 거래 탐지 분석 결과 요약\n")
            f.write("="*40 + "\n")
            f.write(f"📊 데이터: {len(X):,}건\n")
            f.write(f"⚖️ 클래스 불균형: {imbalance_ratio:.1f}:1\n")
            f.write(f"🏆 최적 전략: {best_strategy}\n")
            f.write(f"📈 F1-Score: {best_f1:.4f}\n")
            f.write(f"📊 개선율: {overall_improvement:+.1f}%\n")
            f.write(f"📈 사용 피처: {list(X.columns)}\n")
        
        print(f"📋 요약 저장: {summary_file}")

    def save_optimized_dataset(self, X, y, best_strategy, strategies, results):
        """최적 Class Weight 전략이 적용된 데이터셋 저장"""
        print(f"\n💾 최적화된 데이터셋 저장 중...")
        
        # 최적 Class Weight 가져오기
        if best_strategy == 'baseline':
            best_class_weights = None
            class_weights_dict = {}
        else:
            best_class_weights = strategies[best_strategy]
            class_weights_dict = best_class_weights
        
        # 1. 원본 데이터 + 라벨 + Class Weight 정보 저장
        dataset_df = X.copy()
        dataset_df['whale_class'] = y
        dataset_df['class_name'] = y.map(self.class_names)
        
        # Class Weight 정보 추가
        if class_weights_dict:
            dataset_df['class_weight'] = y.map(class_weights_dict)
        else:
            dataset_df['class_weight'] = 1.0  # Baseline은 모든 클래스 가중치 1.0
        
        # 데이터셋 저장
        dataset_file = f"{self.results_dir}/optimized_whale_dataset.csv"
        dataset_df.to_csv(dataset_file, index=False, encoding='utf-8')
        print(f"📊 최적화된 데이터셋 저장: {dataset_file}")
        
        # 2. 최적 Class Weight 설정 파일 저장
        config_file = f"{self.results_dir}/optimal_class_weights.json"
        import json
        
        config_data = {
            "optimal_strategy": best_strategy,
            "strategy_description": {
                "baseline": "가중치 없음 (모든 클래스 동일 처리)",
                "balanced": "sklearn 기본 balanced 가중치",
                "behavior_focused": "거래 행동 패턴 중심 가중치",
                "rarity_based": "클래스 희귀도 기반 가중치",
                "business_priority": "비즈니스 우선순위 기반 가중치"
            },
            "optimal_class_weights": {str(k): float(v) for k, v in class_weights_dict.items()},
            "performance_metrics": {
                "test_f1_macro": float(results[best_strategy]['test_f1_macro']),
                "cv_mean_f1": float(results[best_strategy]['cv_mean']),
                "cv_std_f1": float(results[best_strategy]['cv_std']),
                "oob_score": float(results[best_strategy]['oob_score'])
            },
            "class_mapping": {str(k): v for k, v in self.class_names.items()},
            "feature_columns": list(X.columns),
            "dataset_info": {
                "total_samples": int(len(X)),
                "feature_count": int(len(X.columns)),
                "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()}
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"⚙️ 최적 설정 저장: {config_file}")
        
        # 3. 훈련용/테스트용 분할된 데이터셋도 저장
        from sklearn.model_selection import train_test_split
        
        # 표준화
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 훈련 데이터셋 저장
        train_df = X_train.copy()
        train_df['whale_class'] = y_train
        train_df['class_name'] = y_train.map(self.class_names)
        if class_weights_dict:
            train_df['class_weight'] = y_train.map(class_weights_dict)
        else:
            train_df['class_weight'] = 1.0
        
        train_file = f"{self.results_dir}/train_dataset_optimized.csv"
        train_df.to_csv(train_file, index=False, encoding='utf-8')
        print(f"🎯 훈련 데이터셋 저장: {train_file}")
        
        # 테스트 데이터셋 저장  
        test_df = X_test.copy()
        test_df['whale_class'] = y_test
        test_df['class_name'] = y_test.map(self.class_names)
        if class_weights_dict:
            test_df['class_weight'] = y_test.map(class_weights_dict)
        else:
            test_df['class_weight'] = 1.0
            
        test_file = f"{self.results_dir}/test_dataset_optimized.csv"
        test_df.to_csv(test_file, index=False, encoding='utf-8')
        print(f"🧪 테스트 데이터셋 저장: {test_file}")
        
        # 4. 스케일러도 저장 (나중에 새 데이터 예측할 때 사용)
        import pickle
        scaler_file = f"{self.results_dir}/feature_scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"📐 피처 스케일러 저장: {scaler_file}")
        
        # 5. 사용법 가이드 저장
        guide_file = f"{self.results_dir}/dataset_usage_guide.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("# 🌳 최적화된 고래 분류 데이터셋 사용 가이드\n\n")
            f.write("## 📊 저장된 파일들\n\n")
            f.write("### 1. 데이터셋 파일\n")
            f.write("- `optimized_whale_dataset.csv`: 전체 최적화된 데이터셋\n")
            f.write("- `train_dataset_optimized.csv`: 훈련용 데이터셋 (70%, 표준화 적용)\n")
            f.write("- `test_dataset_optimized.csv`: 테스트용 데이터셋 (30%, 표준화 적용)\n\n")
            f.write("### 2. 설정 파일\n")
            f.write("- `optimal_class_weights.json`: 최적 Class Weight 설정\n")
            f.write("- `feature_scaler.pkl`: 피처 표준화 스케일러\n\n")
            f.write("### 3. 분석 결과\n")
            f.write("- `detailed_analysis_results.txt`: 상세 분석 결과\n")
            f.write("- `quick_summary.txt`: 빠른 요약\n")
            f.write("- `performance_comparison.png`: 성능 비교 차트\n")
            f.write("- `confusion_matrices.png`: 혼동 행렬\n\n")
            f.write("## 🚀 사용 방법\n\n")
            f.write("### Python에서 데이터셋 로드\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("import json\n")
            f.write("import pickle\n")
            f.write("from sklearn.ensemble import RandomForestClassifier\n\n")
            f.write("# 1. 훈련 데이터 로드\n")
            f.write("train_df = pd.read_csv('train_dataset_optimized.csv')\n")
            f.write("X_train = train_df.drop(['whale_class', 'class_name', 'class_weight'], axis=1)\n")
            f.write("y_train = train_df['whale_class']\n\n")
            f.write("# 2. 최적 Class Weight 로드\n")
            f.write("with open('optimal_class_weights.json', 'r', encoding='utf-8') as f:\n")
            f.write("    config = json.load(f)\n")
            f.write("optimal_weights = config['optimal_class_weights']\n\n")
            f.write("# 3. 모델 훈련\n")
            f.write("model = RandomForestClassifier(\n")
            f.write("    n_estimators=100,\n")
            f.write("    random_state=42,\n")
            f.write("    class_weight=optimal_weights,\n")
            f.write("    max_depth=8,\n")
            f.write("    min_samples_split=50,\n")
            f.write("    min_samples_leaf=20,\n")
            f.write("    max_features='sqrt'\n")
            f.write(")\n")
            f.write("model.fit(X_train, y_train)\n\n")
            f.write("# 4. 새 데이터 예측시 스케일러 사용\n")
            f.write("with open('feature_scaler.pkl', 'rb') as f:\n")
            f.write("    scaler = pickle.load(f)\n")
            f.write("# new_data_scaled = scaler.transform(new_data)\n")
            f.write("```\n\n")
            f.write(f"## 📈 최적 설정 정보\n\n")
            f.write(f"- **최적 전략**: {best_strategy}\n")
            f.write(f"- **F1-Score**: {results[best_strategy]['test_f1_macro']:.4f}\n")
            if best_strategy != 'baseline':
                baseline_f1 = results['baseline']['test_f1_macro']
                improvement = ((results[best_strategy]['test_f1_macro'] - baseline_f1) / baseline_f1) * 100
                f.write(f"- **성능 개선**: {improvement:+.1f}%\n")
            f.write(f"- **교차검증 안정성**: ±{results[best_strategy]['cv_std']*2:.4f}\n\n")
            f.write("## 🎯 클래스 정보\n\n")
            for cls_id, cls_name in self.class_names.items():
                count = y.value_counts().get(cls_id, 0)
                percentage = (count / len(y)) * 100
                weight = class_weights_dict.get(cls_id, 1.0) if class_weights_dict else 1.0
                f.write(f"- **클래스 {cls_id}** ({cls_name}): {count:,}건 ({percentage:.1f}%), 가중치: {weight:.2f}\n")
        
        print(f"📖 사용 가이드 저장: {guide_file}")
        
        return {
            'dataset_file': dataset_file,
            'config_file': config_file,
            'train_file': train_file,
            'test_file': test_file,
            'scaler_file': scaler_file,
            'guide_file': guide_file
        }

if __name__ == "__main__":
    # 고래 거래 탐지 분석 실행
    analyzer = WhaleDetectionAnalyzer()
    analyzer.run_analysis() 