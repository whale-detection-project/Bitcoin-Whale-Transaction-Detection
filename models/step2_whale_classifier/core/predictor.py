"""
🐋 고래 거래 예측 엔진
Random Forest 기반 실시간 고래 패턴 분류
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging
from datetime import datetime

from ..config.settings import (
    MODEL_CONFIG, WHALE_CLASSES, FEATURES, 
    ANALYSIS_CONFIG, DATA_PATH, MODEL_PATH
)

class WhalePredictor:
    """🐋 고래 거래 패턴 예측기"""
    
    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = FEATURES['input_features']
        self.is_trained = False
        self.feature_importance = None
        self.model_metrics = {}
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 모델 저장 경로 생성
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Step 1에서 생성된 최적화 데이터 로드"""
        try:
            data_file = DATA_PATH / "optimized_whale_dataset.csv"
            self.logger.info(f"📊 데이터 로드 중: {data_file}")
            
            df = pd.read_csv(data_file)
            self.logger.info(f"✅ 데이터 로드 완료: {len(df):,}건")
            
            # 피처와 라벨 분리
            X = df[self.feature_names]
            y = df['whale_class']
            
            # 데이터 품질 검증
            self._validate_data(X, y)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            raise
    
    def _validate_data(self, X: pd.DataFrame, y: pd.Series):
        """데이터 품질 검증"""
        # 결측값 확인
        if X.isnull().any().any():
            self.logger.warning("⚠️ 피처에 결측값 발견")
        
        # 클래스 분포 확인
        class_dist = y.value_counts().sort_index()
        self.logger.info("📊 클래스 분포:")
        for cls, count in class_dist.items():
            whale_name = WHALE_CLASSES[cls]['name']
            percentage = count / len(y) * 100
            self.logger.info(f"   클래스 {cls} ({whale_name}): {count:,}건 ({percentage:.1f}%)")
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """모델 훈련 및 최적화"""
        try:
            self.logger.info("🚀 모델 훈련 시작...")
            
            # 데이터 정규화
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Class Weight 설정 (Step 1 최적 전략 사용)
            class_weights = {
                cls: WHALE_CLASSES[cls]['weight'] 
                for cls in WHALE_CLASSES.keys()
            }
            
            # Random Forest 모델 생성
            model_config = MODEL_CONFIG.copy()
            model_config.pop('class_weight_strategy', None)  # 제거
            model_config['oob_score'] = True  # OOB 점수 계산 활성화
            
            self.model = RandomForestClassifier(
                **model_config,
                class_weight=class_weights
            )
            
            # 교차 검증으로 성능 평가
            self.logger.info("📈 교차 검증 수행 중...")
            cv_scores = cross_val_score(
                self.model, X_scaled, y, 
                cv=5, scoring='f1_macro', n_jobs=-1
            )
            
            # 모델 훈련
            self.logger.info("🔧 최종 모델 훈련 중...")
            self.model.fit(X_scaled, y)
            
            # 피처 중요도 저장
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
            
            # 모델 메트릭 저장
            oob_score = getattr(self.model, 'oob_score_', 0.0)  # 안전한 접근
            self.model_metrics = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'oob_score': oob_score,
                'n_estimators': self.model.n_estimators,
                'training_samples': len(X),
                'training_time': datetime.now().isoformat()
            }
            
            self.is_trained = True
            
            self.logger.info("✅ 모델 훈련 완료!")
            self.logger.info(f"📊 교차검증 F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            self.logger.info(f"📊 OOB Score: {self.model.oob_score_:.4f}")
            
            return self.model_metrics
            
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 실패: {e}")
            raise
    
    def predict_whale_type(self, transaction_data: Dict) -> Dict:
        """단일 거래 데이터로 고래 유형 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 실행하세요.")
        
        try:
            # 입력 데이터 검증 및 변환
            features = self._prepare_features(transaction_data)
            
            # 예측 수행
            features_scaled = self.scaler.transform([features])
            
            # 클래스 확률 예측
            class_probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = self.model.predict(features_scaled)[0]
            confidence = class_probabilities[predicted_class]
            
            # 피처 기여도 계산 (트리 기반 설명)
            feature_contributions = self._calculate_feature_contributions(features_scaled[0])
            
            # 결과 구성
            result = {
                'predicted_class': int(predicted_class),
                'whale_info': WHALE_CLASSES[predicted_class],
                'confidence': float(confidence),
                'class_probabilities': {
                    cls: float(prob) for cls, prob in enumerate(class_probabilities)
                },
                'feature_contributions': feature_contributions,
                'input_features': dict(zip(self.feature_names, features)),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 예측 실패: {e}")
            raise
    
    def _prepare_features(self, transaction_data: Dict) -> List[float]:
        """입력 데이터를 모델 피처로 변환"""
        features = []
        
        for feature_name in self.feature_names:
            if feature_name not in transaction_data:
                raise ValueError(f"필수 피처 누락: {feature_name}")
            
            value = transaction_data[feature_name]
            
            # 데이터 타입 검증
            if not isinstance(value, (int, float)):
                raise ValueError(f"피처 {feature_name}는 숫자여야 합니다: {value}")
            
            # 범위 검증
            if feature_name == 'concentration' and not (0 <= value <= 1):
                raise ValueError(f"집중도는 0~1 사이여야 합니다: {value}")
            
            if feature_name in ['input_count', 'output_count'] and value < 1:
                raise ValueError(f"{feature_name}는 1 이상이어야 합니다: {value}")
            
            if feature_name in ['total_volume_btc', 'fee_btc'] and value < 0:
                raise ValueError(f"{feature_name}는 0 이상이어야 합니다: {value}")
            
            features.append(float(value))
        
        return features
    
    def _calculate_feature_contributions(self, features: np.ndarray) -> Dict:
        """피처별 예측 기여도 계산"""
        contributions = {}
        
        # 피처 중요도와 현재 값의 상대적 크기 고려
        for i, (feature_name, importance) in enumerate(self.feature_importance.items()):
            # 정규화된 피처 값과 중요도를 결합
            contribution_score = abs(features[i]) * importance
            contributions[feature_name] = {
                'importance': float(importance),
                'normalized_value': float(features[i]),
                'contribution_score': float(contribution_score)
            }
        
        return contributions
    
    def get_similar_transactions(self, transaction_data: Dict, n_similar: int = 5) -> List[Dict]:
        """유사한 거래 패턴 검색 (시뮬레이션)"""
        try:
            # 실제 구현에서는 데이터베이스나 벡터 검색 사용
            # 여기서는 간단한 시뮬레이션
            
            X, y = self.load_training_data()
            
            # 현재 거래와의 유사도 계산 (유클리드 거리 기반)
            current_features = self._prepare_features(transaction_data)
            X_scaled = self.scaler.transform(X)
            current_scaled = self.scaler.transform([current_features])
            
            # 거리 계산
            distances = np.linalg.norm(X_scaled - current_scaled, axis=1)
            similar_indices = np.argsort(distances)[:n_similar]
            
            similar_transactions = []
            for idx in similar_indices:
                similar_data = X.iloc[idx].to_dict()
                similar_class = y.iloc[idx]
                distance = distances[idx]
                
                similar_transactions.append({
                    'features': similar_data,
                    'whale_class': int(similar_class),
                    'whale_name': WHALE_CLASSES[similar_class]['name'],
                    'similarity_score': float(1 / (1 + distance))  # 0~1 점수
                })
            
            return similar_transactions
            
        except Exception as e:
            self.logger.warning(f"⚠️ 유사 거래 검색 실패: {e}")
            return []
    
    def save_model(self, filename: str = None):
        """훈련된 모델 저장"""
        if not self.is_trained:
            raise ValueError("저장할 훈련된 모델이 없습니다.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"whale_classifier_{timestamp}"
        
        try:
            # 모델과 스케일러 저장
            model_path = MODEL_PATH / f"{filename}_model.pkl"
            scaler_path = MODEL_PATH / f"{filename}_scaler.pkl"
            metrics_path = MODEL_PATH / f"{filename}_metrics.pkl"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump({
                'metrics': self.model_metrics,
                'feature_importance': self.feature_importance,
                'feature_names': self.feature_names
            }, metrics_path)
            
            self.logger.info(f"✅ 모델 저장 완료: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 저장 실패: {e}")
            raise
    
    def load_model(self, filename: str):
        """저장된 모델 로드"""
        try:
            model_path = MODEL_PATH / f"{filename}_model.pkl"
            scaler_path = MODEL_PATH / f"{filename}_scaler.pkl"
            metrics_path = MODEL_PATH / f"{filename}_metrics.pkl"
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            if metrics_path.exists():
                metrics_data = joblib.load(metrics_path)
                self.model_metrics = metrics_data['metrics']
                self.feature_importance = metrics_data['feature_importance']
                self.feature_names = metrics_data['feature_names']
            
            self.is_trained = True
            self.logger.info(f"✅ 모델 로드 완료: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """모델 정보 조회"""
        if not self.is_trained:
            return {"status": "모델이 훈련되지 않음"}
        
        return {
            "status": "훈련 완료",
            "metrics": self.model_metrics,
            "feature_importance": self.feature_importance,
            "model_config": MODEL_CONFIG,
            "feature_names": self.feature_names
        } 