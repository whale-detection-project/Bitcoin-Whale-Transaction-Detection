"""
🐋 고래 거래 분류 시스템 (Step 2)
실시간 비트코인 고래 거래 패턴 분석 및 분류
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 내부 모듈 import
from .core.predictor import WhalePredictor
from .core.insights import WhaleInsightGenerator
from .config.settings import (
    WHALE_CLASSES, FEATURES, MODEL_CONFIG, 
    ANALYSIS_CONFIG, UI_CONFIG
)

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class WhaleClassificationSystem:
    """🐋 통합 고래 분류 시스템"""
    
    def __init__(self, enable_logging: bool = True):
        """시스템 초기화"""
        # 로깅 설정
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        
        self.logger = logging.getLogger(__name__)
        self.colors = UI_CONFIG['colors']
        
        # 컴포넌트 초기화
        self.predictor = WhalePredictor()
        self.insight_generator = WhaleInsightGenerator()
        
        # 시스템 상태
        self.is_trained = False
        self.system_ready = False
        
        self.logger.info("🐋 고래 분류 시스템 초기화 완료")
    
    def setup_system(self, train_new_model: bool = True, model_filename: str = None) -> bool:
        """시스템 설정 및 모델 준비"""
        try:
            self._print_welcome_message()
            
            if train_new_model:
                self.logger.info("🚀 새 모델 훈련 시작...")
                success = self._train_new_model()
            else:
                if not model_filename:
                    raise ValueError("기존 모델 로드 시 파일명이 필요합니다.")
                self.logger.info(f"📁 기존 모델 로드: {model_filename}")
                success = self._load_existing_model(model_filename)
            
            if success:
                self.is_trained = True
                self.system_ready = True
                self._print_setup_complete()
                return True
            else:
                self.logger.error("❌ 시스템 설정 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 시스템 설정 중 오류: {e}")
            return False
    
    def analyze_transaction(self, transaction_data: Dict, include_similar: bool = True) -> Dict:
        """거래 데이터 종합 분석"""
        if not self.system_ready:
            raise RuntimeError("시스템이 준비되지 않았습니다. setup_system()을 먼저 실행하세요.")
        
        try:
            self.logger.info("📊 거래 분석 시작...")
            
            # 1. 고래 유형 예측
            prediction_result = self.predictor.predict_whale_type(transaction_data)
            
            # 2. 유사 거래 검색 (선택적)
            similar_transactions = []
            if include_similar:
                try:
                    similar_transactions = self.predictor.get_similar_transactions(
                        transaction_data, n_similar=5
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ 유사 거래 검색 실패: {e}")
            
            # 3. 전문가급 인사이트 생성
            expert_insights = self.insight_generator.generate_expert_insights(
                prediction_result, similar_transactions
            )
            
            # 4. 종합 결과 구성
            analysis_result = {
                'prediction': prediction_result,
                'similar_transactions': similar_transactions,
                'expert_insights': expert_insights,
                'system_info': self._get_system_info()
            }
            
            self.logger.info("✅ 거래 분석 완료")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ 거래 분석 실패: {e}")
            raise
    
    def analyze_and_display(self, transaction_data: Dict, show_insights: bool = True) -> Dict:
        """거래 분석 및 결과 출력"""
        # 분석 수행
        result = self.analyze_transaction(transaction_data)
        
        # 결과 출력
        if show_insights:
            print(result['expert_insights'])
        else:
            self._print_simple_result(result['prediction'])
        
        return result
    
    def batch_analyze(self, transaction_list: List[Dict], save_results: bool = False) -> List[Dict]:
        """배치 거래 분석"""
        if not self.system_ready:
            raise RuntimeError("시스템이 준비되지 않았습니다.")
        
        results = []
        self.logger.info(f"📊 배치 분석 시작: {len(transaction_list)}건")
        
        for i, transaction_data in enumerate(transaction_list, 1):
            try:
                self.logger.info(f"분석 진행: {i}/{len(transaction_list)}")
                result = self.analyze_transaction(transaction_data, include_similar=False)
                results.append(result)
            except Exception as e:
                self.logger.error(f"거래 {i} 분석 실패: {e}")
                continue
        
        if save_results:
            self._save_batch_results(results)
        
        self.logger.info(f"✅ 배치 분석 완료: {len(results)}건 성공")
        return results
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        return {
            'is_trained': self.is_trained,
            'system_ready': self.system_ready,
            'model_info': self.predictor.get_model_info() if self.is_trained else None,
            'supported_features': FEATURES['input_features'],
            'whale_classes': WHALE_CLASSES,
            'model_config': MODEL_CONFIG
        }
    
    def save_trained_model(self, filename: str = None) -> str:
        """훈련된 모델 저장"""
        if not self.is_trained:
            raise RuntimeError("저장할 훈련된 모델이 없습니다.")
        
        self.predictor.save_model(filename)
        self.logger.info(f"✅ 모델 저장 완료: {filename}")
        return filename
    
    def create_sample_transaction(self) -> Dict:
        """테스트용 샘플 거래 생성"""
        import random
        
        sample = {
            'total_volume_btc': random.uniform(1000, 5000),
            'input_count': random.randint(1, 10),
            'output_count': random.randint(1, 15),
            'concentration': random.uniform(0.3, 0.95),
            'fee_btc': random.uniform(0.0001, 0.01)
        }
        
        self.logger.info("🎲 샘플 거래 데이터 생성")
        return sample
    
    # === 내부 메서드들 ===
    
    def _train_new_model(self) -> bool:
        """새 모델 훈련"""
        try:
            # 데이터 로드
            X, y = self.predictor.load_training_data()
            
            # 모델 훈련
            metrics = self.predictor.train_model(X, y)
            
            self.logger.info("✅ 모델 훈련 완료")
            self.logger.info(f"📊 성능 메트릭: {metrics}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 실패: {e}")
            return False
    
    def _load_existing_model(self, filename: str) -> bool:
        """기존 모델 로드"""
        try:
            self.predictor.load_model(filename)
            self.logger.info("✅ 모델 로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            return False
    
    def _print_welcome_message(self):
        """환영 메시지 출력"""
        message = f"""
{self.colors['bold']}{'='*80}{self.colors['end']}
{self.colors['bold']}{self.colors['info']}🐋 WHALE TRANSACTION CLASSIFIER - STEP 2 🐋{self.colors['end']}
{self.colors['bold']}{'='*80}{self.colors['end']}
{self.colors['info']}🚀 AI 기반 실시간 고래 거래 패턴 분석 시스템{self.colors['end']}
📊 Random Forest Classifier (Step 1 최적화 결과 적용)
🎯 F1-Score: 0.5081 (16.9% 성능 개선)
🧠 Level 3 전문가급 인사이트 제공
{self.colors['bold']}{'='*80}{self.colors['end']}
"""
        print(message)
    
    def _print_setup_complete(self):
        """설정 완료 메시지"""
        model_info = self.predictor.get_model_info()
        
        message = f"""
{self.colors['success']}✅ 시스템 설정 완료!{self.colors['end']}
📊 모델 성능: F1-Score {model_info['metrics']['cv_f1_mean']:.4f}
🌊 지원 고래 유형: {len(WHALE_CLASSES)}개 클래스
🔧 피처 수: {len(FEATURES['input_features'])}개
🚀 실시간 분석 준비 완료!
"""
        print(message)
    
    def _print_simple_result(self, prediction: Dict):
        """간단한 결과 출력"""
        whale_info = prediction['whale_info']
        confidence = prediction['confidence']
        
        print(f"""
{self.colors['bold']}🐋 분석 결과{self.colors['end']}
{'─'*40}
{whale_info['emoji']} 고래 유형: {whale_info['name']}
🎯 신뢰도: {confidence:.1%}
📝 설명: {whale_info['description']}
""")
    
    def _get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            'version': '2.0.0',
            'step': 'Step 2 - Whale Classification',
            'model_type': 'Random Forest Classifier',
            'features_count': len(FEATURES['input_features']),
            'classes_count': len(WHALE_CLASSES),
            'is_ready': self.system_ready
        }
    
    def _save_batch_results(self, results: List[Dict]):
        """배치 결과 저장"""
        from datetime import datetime
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_analysis_{timestamp}.json"
        
        # 결과 요약 생성
        summary = {
            'total_analyzed': len(results),
            'timestamp': datetime.now().isoformat(),
            'results': [r['prediction'] for r in results]
        }
        
        # JSON 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📁 배치 결과 저장: {filename}")


# === 편의 함수들 ===

def quick_analyze(transaction_data: Dict, train_new: bool = True) -> Dict:
    """빠른 분석 (단일 거래)"""
    system = WhaleClassificationSystem(enable_logging=False)
    
    if not system.setup_system(train_new_model=train_new):
        raise RuntimeError("시스템 설정 실패")
    
    return system.analyze_and_display(transaction_data)

def demo_analysis():
    """데모 분석 실행"""
    system = WhaleClassificationSystem()
    
    print("🚀 고래 분류 시스템 데모 시작...")
    
    # 시스템 설정
    if not system.setup_system():
        print("❌ 시스템 설정 실패")
        return
    
    # 샘플 데이터 생성 및 분석
    sample_data = system.create_sample_transaction()
    
    print(f"\n{system.colors['info']}📊 샘플 거래 데이터:{system.colors['end']}")
    for key, value in sample_data.items():
        feature_desc = FEATURES['feature_descriptions'][key]
        print(f"   {feature_desc}: {value:,.4f}")
    
    print(f"\n{system.colors['bold']}🔍 분석 시작...{system.colors['end']}")
    result = system.analyze_and_display(sample_data)
    
    return result

if __name__ == "__main__":
    # 시스템 정보 출력
    system = WhaleClassificationSystem()
    print("🐋 고래 분류 시스템")
    print("사용법: python whale_classifier.py")
    print("데모 실행: demo_analysis()")
    
    # 데모 실행
    demo_analysis() 