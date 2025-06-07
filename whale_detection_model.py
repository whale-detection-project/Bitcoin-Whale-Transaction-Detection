"""
🐋 비트코인 고래 탐지 및 분류 모델 (Random Forest)
================================================================
최적화된 데이터셋을 사용한 프로덕션 레벨의 고래 탐지 시스템
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                           precision_score, recall_score, accuracy_score, roc_auc_score)
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WhaleDetectionModel:
    """
    🐋 비트코인 고래 탐지 및 분류 모델
    
    기능:
    - 5가지 고래 유형 분류 (수집형, 분산형, 급행형, 집중형, 거대형)
    - 최적화된 Class Weight 적용
    - 교차 검증 및 하이퍼파라미터 최적화
    - 모델 저장/로드 기능
    - 실시간 예측 API
    """
    
    def __init__(self, model_dir='models/whale_detection'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 고래 분류 체계
        self.class_names = {
            0: '수집형고래',    # Input 많고 Output 적음
            1: '분산형고래',    # Output 많고 분산
            2: '급행형고래',    # 높은 수수료
            3: '집중형고래',    # 높은 집중도
            4: '거대형고래'     # 극대 거래량
        }
        
        # 최적 Class Weight (분석 결과 기반)
        self.optimal_class_weights = {
            0: 5.0,   # 수집형고래
            1: 15.0,  # 분산형고래
            2: 15.0,  # 급행형고래
            3: 0.8,   # 집중형고래
            4: 30.0   # 거대형고래
        }
        
        # 모델 및 스케일러
        self.model = None
        self.scaler = None
        self.feature_columns = ['total_volume_btc', 'input_count', 'output_count', 'concentration', 'fee_btc']
        
        print("🐋 고래 탐지 및 분류 모델이 초기화되었습니다.")
        print(f"📁 모델 저장 경로: {model_dir}")
        print("🎯 분류 대상: 5가지 고래 유형")
        
    def load_data(self, data_path='analysis/step1_results/class_weight_results/optimized_whale_dataset.csv'):
        """최적화된 데이터셋 로드"""
        print("📊 최적화된 고래 데이터셋 로딩 중...")
        
        try:
            df = pd.read_csv(data_path)
            print(f"✅ 데이터 로드 완료: {len(df):,}건")
            
            # 피처와 레이블 분리
            X = df[self.feature_columns].copy()
            y = df['whale_class'].copy()
            
            print(f"📊 피처 수: {len(self.feature_columns)}개")
            print(f"🏷️ 클래스 수: {len(y.unique())}개")
            
            # 클래스 분포 확인
            class_dist = y.value_counts().sort_index()
            total = len(y)
            print("\n📊 클래스 분포:")
            for cls, count in class_dist.items():
                percentage = (count / total) * 100
                print(f"  클래스 {cls} ({self.class_names[cls]}): {count:,}건 ({percentage:.1f}%)")
            
            return X, y
            
        except FileNotFoundError:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
            print("💡 먼저 step1_class_weight_adjustment.py를 실행해주세요.")
            return None, None
        except Exception as e:
            print(f"❌ 데이터 로딩 오류: {e}")
            return None, None
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """데이터 전처리 및 분할"""
        print("\n🔧 데이터 전처리 및 분할 중...")
        
        # 결측값 확인
        missing_values = X.isnull().sum()
        if missing_values.any():
            print(f"⚠️ 결측값 발견: {missing_values[missing_values > 0].to_dict()}")
            X = X.fillna(0)
            print("✅ 결측값을 0으로 처리했습니다.")
        
        # 데이터 분할 (계층화)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"🎯 훈련 데이터: {len(X_train):,}건 ({(1-test_size)*100:.0f}%)")
        print(f"🧪 테스트 데이터: {len(X_test):,}건 ({test_size*100:.0f}%)")
        
        # 표준화
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataFrame으로 변환 (컬럼명 유지)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        print("✅ 데이터 표준화 완료")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, optimize_hyperparameters=True):
        """모델 훈련 (하이퍼파라미터 최적화 포함)"""
        print("\n🌳 Random Forest 모델 훈련 시작...")
        
        if optimize_hyperparameters:
            print("⚙️ 하이퍼파라미터 최적화 진행 중...")
            
            # 하이퍼파라미터 그리드
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10, None],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [10, 20, 30],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # 기본 모델
            base_model = RandomForestClassifier(
                random_state=42,
                class_weight=self.optimal_class_weights,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1
            )
            
            # Grid Search with 3-Fold CV
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3,  # 데이터가 크므로 3-fold로 축소
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            
            print("🔍 최적 하이퍼파라미터 탐색 중... (약 5-10분 소요)")
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            
            print("✅ 하이퍼파라미터 최적화 완료!")
            print(f"🏆 최적 파라미터: {grid_search.best_params_}")
            print(f"📊 최적 CV Score: {grid_search.best_score_:.4f}")
            
        else:
            print("⚙️ 기본 하이퍼파라미터로 훈련 중...")
            
            # 분석 결과 기반 최적 파라미터 사용
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42,
                class_weight=self.optimal_class_weights,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            print("✅ 모델 훈련 완료!")
        
        print(f"🌳 트리 개수: {self.model.n_estimators}")
        print(f"📊 OOB Score: {self.model.oob_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, show_detailed_report=True):
        """모델 성능 평가"""
        print("\n📊 모델 성능 평가 중...")
        
        if self.model is None:
            print("❌ 훈련된 모델이 없습니다. 먼저 train_model()을 실행해주세요.")
            return None
        
        # 예측
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # 기본 성능 지표
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        # 결과 딕셔너리
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'oob_score': self.model.oob_score_,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print("📈 전체 성능 지표:")
        print(f"  정확도 (Accuracy): {accuracy:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  정밀도 (Precision): {precision_macro:.4f}")
        print(f"  재현율 (Recall): {recall_macro:.4f}")
        print(f"  OOB Score: {self.model.oob_score_:.4f}")
        
        if show_detailed_report:
            print("\n📋 클래스별 상세 성능:")
            print("-" * 60)
            for class_id in sorted(y_test.unique()):
                class_report = results['classification_report'][str(class_id)]
                print(f"클래스 {class_id} ({self.class_names[class_id]}):")
                print(f"  Precision: {class_report['precision']:.4f}")
                print(f"  Recall: {class_report['recall']:.4f}")
                print(f"  F1-Score: {class_report['f1-score']:.4f}")
                print(f"  Support: {class_report['support']}")
                print()
        
        return results
    
    def cross_validate(self, X, y, cv_folds=5):
        """교차 검증 수행"""
        print(f"\n🔄 {cv_folds}-Fold 교차 검증 수행 중...")
        
        if self.model is None:
            print("❌ 훈련된 모델이 없습니다.")
            return None
        
        # 계층화 교차 검증
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # 여러 지표로 교차 검증
        scoring_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"  {metric}: {scores.mean():.4f} (±{scores.std()*2:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, plot=True):
        """피처 중요도 분석"""
        print("\n📊 피처 중요도 분석...")
        
        if self.model is None:
            print("❌ 훈련된 모델이 없습니다.")
            return None
        
        # 피처 중요도
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔍 피처 중요도 순위:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title('고래 분류 피처 중요도', fontsize=14, fontweight='bold')
            plt.xlabel('중요도')
            plt.ylabel('피처')
            
            # 값 표시
            for idx, (importance, feature) in enumerate(zip(importance_df['importance'], importance_df['feature'])):
                plt.text(importance + 0.01, idx, f'{importance:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig(f'{self.model_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"📊 피처 중요도 차트 저장: {self.model_dir}/feature_importance.png")
        
        return importance_df
    
    def create_visualizations(self, results):
        """성능 시각화"""
        print("\n📊 성능 시각화 생성 중...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 혼동 행렬
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax1,
                    xticklabels=[f'{i}\n({self.class_names[i]})' for i in range(len(cm))],
                    yticklabels=[f'{i}\n({self.class_names[i]})' for i in range(len(cm))])
        ax1.set_title('정규화된 혼동 행렬', fontsize=12, fontweight='bold')
        ax1.set_xlabel('예측 클래스')
        ax1.set_ylabel('실제 클래스')
        
        # 2. 클래스별 F1-Score
        class_f1_scores = []
        class_labels = []
        for class_id in sorted(results['y_test'].unique()):
            f1 = results['classification_report'][str(class_id)]['f1-score']
            class_f1_scores.append(f1)
            class_labels.append(f'{class_id}\n{self.class_names[class_id]}')
        
        bars = ax2.bar(class_labels, class_f1_scores, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax2.set_title('클래스별 F1-Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1.0)
        
        # 값 표시
        for bar, score in zip(bars, class_f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 클래스별 예측 분포
        y_test_counts = pd.Series(results['y_test']).value_counts().sort_index()
        y_pred_counts = pd.Series(results['y_pred']).value_counts().sort_index()
        
        x = np.arange(len(y_test_counts))
        width = 0.35
        
        ax3.bar(x - width/2, y_test_counts.values, width, label='실제', alpha=0.7)
        ax3.bar(x + width/2, y_pred_counts.values, width, label='예측', alpha=0.7)
        ax3.set_title('클래스별 실제 vs 예측 분포', fontsize=12, fontweight='bold')
        ax3.set_xlabel('클래스')
        ax3.set_ylabel('개수')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{i}\n{self.class_names[i]}' for i in y_test_counts.index])
        ax3.legend()
        
        # 4. 성능 지표 레이더 차트
        metrics = ['정확도', 'F1-Macro', 'F1-Weighted', '정밀도', '재현율']
        values = [
            results['accuracy'],
            results['f1_macro'], 
            results['f1_weighted'],
            results['precision_macro'],
            results['recall_macro']
        ]
        
        # 레이더 차트를 위한 각도 계산
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += [values[0]]  # 닫힌 도형을 위해
        angles += [angles[0]]
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax4.fill(angles, values, alpha=0.25, color='blue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('전체 성능 지표', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_dir}/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 성능 시각화 저장: {self.model_dir}/model_performance.png")
    
    def save_model(self):
        """모델 저장"""
        print("\n💾 모델 저장 중...")
        
        if self.model is None or self.scaler is None:
            print("❌ 저장할 모델 또는 스케일러가 없습니다.")
            return False
        
        try:
            # 모델 저장
            model_file = f'{self.model_dir}/whale_detection_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ 모델 저장: {model_file}")
            
            # 스케일러 저장
            scaler_file = f'{self.model_dir}/feature_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ 스케일러 저장: {scaler_file}")
            
            # 모델 설정 저장
            config = {
                'class_names': self.class_names,
                'optimal_class_weights': self.optimal_class_weights,
                'feature_columns': self.feature_columns,
                'model_type': 'RandomForestClassifier',
                'created_at': datetime.now().isoformat(),
                'model_parameters': self.model.get_params()
            }
            
            config_file = f'{self.model_dir}/model_config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✅ 설정 저장: {config_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            return False
    
    def load_model(self):
        """저장된 모델 로드"""
        print("\n📂 저장된 모델 로딩 중...")
        
        try:
            # 모델 로드
            model_file = f'{self.model_dir}/whale_detection_model.pkl'
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ 모델 로드: {model_file}")
            
            # 스케일러 로드
            scaler_file = f'{self.model_dir}/feature_scaler.pkl'
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ 스케일러 로드: {scaler_file}")
            
            # 설정 로드
            config_file = f'{self.model_dir}/model_config.json'
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # JSON에서 로드한 class_names의 키를 정수로 변환
            self.class_names = {int(k): v for k, v in config['class_names'].items()}
            # optimal_class_weights의 키도 정수로 변환
            self.optimal_class_weights = {int(k): v for k, v in config['optimal_class_weights'].items()}
            self.feature_columns = config['feature_columns']
            
            print(f"✅ 설정 로드: {config_file}")
            print(f"📅 모델 생성일: {config.get('created_at', 'Unknown')}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
            return False
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return False
    
    def predict(self, X, return_proba=False):
        """새로운 데이터 예측"""
        if self.model is None or self.scaler is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        # DataFrame이 아닌 경우 변환
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        
        # 피처 순서 확인
        X = X[self.feature_columns]
        
        # 표준화
        X_scaled = self.scaler.transform(X)
        
        # 예측
        predictions = self.model.predict(X_scaled)
        
        if return_proba:
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        
        return predictions
    
    def predict_single(self, total_volume_btc, input_count, output_count, concentration, fee_btc, show_details=True):
        """단일 거래 예측 (사용하기 쉬운 인터페이스)"""
        
        # 입력 데이터 생성
        data = pd.DataFrame({
            'total_volume_btc': [total_volume_btc],
            'input_count': [input_count],
            'output_count': [output_count],
            'concentration': [concentration],
            'fee_btc': [fee_btc]
        })
        
        # 예측
        prediction, probabilities = self.predict(data, return_proba=True)
        
        predicted_class = prediction[0]
        predicted_name = self.class_names[predicted_class]
        confidence = probabilities[0][predicted_class]
        
        if show_details:
            print(f"\n🔍 고래 분류 예측 결과:")
            print(f"📊 입력 데이터:")
            print(f"  총 거래량: {total_volume_btc:,.0f} BTC")
            print(f"  Input 개수: {input_count}")
            print(f"  Output 개수: {output_count}")
            print(f"  집중도: {concentration:.4f}")
            print(f"  수수료: {fee_btc:.6f} BTC")
            print(f"\n🎯 예측 결과:")
            print(f"  클래스: {predicted_class} ({predicted_name})")
            print(f"  신뢰도: {confidence:.4f} ({confidence*100:.1f}%)")
            print(f"\n📊 각 클래스별 확률:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  클래스 {i} ({self.class_names[i]}): {prob:.4f} ({prob*100:.1f}%)")
        
        return {
            'predicted_class': predicted_class,
            'predicted_name': predicted_name,
            'confidence': confidence,
            'probabilities': probabilities[0],
            'input_data': data.iloc[0].to_dict()
        }

def main():
    """메인 실행 함수"""
    print("🚀 비트코인 고래 탐지 및 분류 모델 개발 시작!")
    print("=" * 60)
    
    # 모델 초기화
    whale_model = WhaleDetectionModel()
    
    # 데이터 로드
    X, y = whale_model.load_data()
    if X is None:
        return
    
    # 데이터 준비
    X_train, X_test, y_train, y_test = whale_model.prepare_data(X, y)
    
    # 모델 훈련 (하이퍼파라미터 최적화 여부 선택)
    optimize = input("\n🔧 하이퍼파라미터 최적화를 수행하시겠습니까? (y/n, 기본값: n): ").lower().strip()
    optimize_hyperparameters = optimize == 'y'
    
    whale_model.train_model(X_train, y_train, optimize_hyperparameters=optimize_hyperparameters)
    
    # 모델 평가
    results = whale_model.evaluate_model(X_test, y_test)
    
    # 교차 검증
    cv_results = whale_model.cross_validate(X, y)
    
    # 피처 중요도 분석
    importance_df = whale_model.get_feature_importance(plot=True)
    
    # 시각화
    whale_model.create_visualizations(results)
    
    # 모델 저장
    whale_model.save_model()
    
    print("\n🎉 모델 개발 완료!")
    print(f"📊 최종 F1-Macro Score: {results['f1_macro']:.4f}")
    print(f"📊 최종 정확도: {results['accuracy']:.4f}")
    print(f"📁 모델 저장 위치: {whale_model.model_dir}/")
    
    # 예시 예측
    print("\n🔍 예시 예측 테스트:")
    whale_model.predict_single(
        total_volume_btc=5000,
        input_count=1,
        output_count=2,
        concentration=0.99,
        fee_btc=0.001
    )

if __name__ == "__main__":
    main() 

    