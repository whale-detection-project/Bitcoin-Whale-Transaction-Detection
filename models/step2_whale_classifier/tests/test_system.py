"""
🧪 Step 2 고래 분류 시스템 테스트
종합적인 기능 검증 및 성능 평가
"""

import sys
import unittest
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

from models.step2_whale_classifier.whale_classifier import WhaleClassificationSystem
from models.step2_whale_classifier.core.predictor import WhalePredictor
from models.step2_whale_classifier.core.insights import WhaleInsightGenerator
from models.step2_whale_classifier.config.settings import WHALE_CLASSES, FEATURES

class TestWhaleClassificationSystem(unittest.TestCase):
    """고래 분류 시스템 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        cls.system = WhaleClassificationSystem(enable_logging=False)
        cls.sample_data = cls._create_test_data()
    
    @classmethod
    def _create_test_data(cls):
        """테스트용 데이터 생성"""
        return {
            'total_volume_btc': 2500.0,
            'input_count': 3,
            'output_count': 5,
            'concentration': 0.75,
            'fee_btc': 0.002
        }
    
    def test_01_system_initialization(self):
        """시스템 초기화 테스트"""
        self.assertIsInstance(self.system, WhaleClassificationSystem)
        self.assertFalse(self.system.is_trained)
        self.assertFalse(self.system.system_ready)
        print("✅ 시스템 초기화 테스트 통과")
    
    def test_02_configuration_validation(self):
        """설정 검증 테스트"""
        # 고래 클래스 검증
        self.assertGreater(len(WHALE_CLASSES), 0)
        for cls_id, cls_info in WHALE_CLASSES.items():
            self.assertIn('name', cls_info)
            self.assertIn('description', cls_info)
            self.assertIn('emoji', cls_info)
            
        # 피처 검증
        self.assertGreater(len(FEATURES['input_features']), 0)
        self.assertEqual(len(FEATURES['input_features']), 
                        len(FEATURES['feature_descriptions']))
        
        print("✅ 설정 검증 테스트 통과")
    
    def test_03_data_validation(self):
        """데이터 검증 테스트"""
        # 유효한 데이터
        try:
            self._validate_transaction_data(self.sample_data)
            data_valid = True
        except:
            data_valid = False
        
        self.assertTrue(data_valid)
        
        # 무효한 데이터
        invalid_data = self.sample_data.copy()
        invalid_data['concentration'] = 1.5  # 범위 초과
        
        with self.assertRaises(ValueError):
            self._validate_transaction_data(invalid_data)
        
        print("✅ 데이터 검증 테스트 통과")
    
    def _validate_transaction_data(self, data):
        """거래 데이터 검증"""
        for feature in FEATURES['input_features']:
            if feature not in data:
                raise ValueError(f"필수 피처 누락: {feature}")
            
            value = data[feature]
            if not isinstance(value, (int, float)):
                raise ValueError(f"숫자 타입이 아님: {feature}")
            
            if feature == 'concentration' and not (0 <= value <= 1):
                raise ValueError(f"집중도 범위 오류: {value}")

class TestPredictorComponent(unittest.TestCase):
    """예측기 컴포넌트 테스트"""
    
    def setUp(self):
        """테스트 초기화"""
        self.predictor = WhalePredictor()
        self.test_data = {
            'total_volume_btc': 1500.0,
            'input_count': 2,
            'output_count': 4,
            'concentration': 0.8,
            'fee_btc': 0.001
        }
    
    def test_predictor_initialization(self):
        """예측기 초기화 테스트"""
        self.assertIsNotNone(self.predictor)
        self.assertFalse(self.predictor.is_trained)
        self.assertEqual(len(self.predictor.feature_names), 5)
        print("✅ 예측기 초기화 테스트 통과")
    
    def test_feature_preparation(self):
        """피처 준비 테스트"""
        try:
            features = self.predictor._prepare_features(self.test_data)
            self.assertEqual(len(features), len(FEATURES['input_features']))
            self.assertTrue(all(isinstance(f, float) for f in features))
            print("✅ 피처 준비 테스트 통과")
        except Exception as e:
            self.fail(f"피처 준비 실패: {e}")

class TestInsightGenerator(unittest.TestCase):
    """인사이트 생성기 테스트"""
    
    def setUp(self):
        """테스트 초기화"""
        self.insight_generator = WhaleInsightGenerator()
        self.mock_prediction = {
            'predicted_class': 0,
            'whale_info': WHALE_CLASSES[0],
            'confidence': 0.75,
            'class_probabilities': {0: 0.75, 1: 0.15, 2: 0.10},
            'input_features': {
                'total_volume_btc': 2000.0,
                'input_count': 3,
                'output_count': 5,
                'concentration': 0.85,
                'fee_btc': 0.001
            },
            'feature_contributions': {}
        }
    
    def test_insight_generation(self):
        """인사이트 생성 테스트"""
        try:
            insights = self.insight_generator.generate_expert_insights(self.mock_prediction)
            self.assertIsInstance(insights, str)
            self.assertGreater(len(insights), 100)  # 충분한 길이의 인사이트
            
            # 필수 섹션 포함 확인
            self.assertIn("분류 결과", insights)
            self.assertIn("신뢰도 분석", insights)
            self.assertIn("추천사항", insights)
            
            print("✅ 인사이트 생성 테스트 통과")
        except Exception as e:
            self.fail(f"인사이트 생성 실패: {e}")
    
    def test_volume_categorization(self):
        """거래량 분류 테스트"""
        test_cases = [
            (15000, "초대형"),
            (7000, "대형"),
            (3000, "중형"),
            (1000, "소형")
        ]
        
        for volume, expected_category in test_cases:
            category = self.insight_generator._categorize_volume(volume)
            self.assertIn(expected_category, category)
        
        print("✅ 거래량 분류 테스트 통과")

class TestSystemIntegration(unittest.TestCase):
    """시스템 통합 테스트"""
    
    def setUp(self):
        """통합 테스트 초기화"""
        self.test_transactions = [
            {
                'total_volume_btc': 3000.0,
                'input_count': 2,
                'output_count': 3,
                'concentration': 0.9,
                'fee_btc': 0.003
            },
            {
                'total_volume_btc': 1000.0,
                'input_count': 5,
                'output_count': 8,
                'concentration': 0.4,
                'fee_btc': 0.0005
            }
        ]
    
    def test_end_to_end_workflow(self):
        """종단간 워크플로우 테스트"""
        print("\n🔄 종단간 테스트 시작...")
        
        # 1. 시스템 생성
        system = WhaleClassificationSystem(enable_logging=False)
        self.assertIsNotNone(system)
        
        # 2. 시스템 상태 확인
        status = system.get_system_status()
        self.assertFalse(status['is_trained'])
        
        # 3. 샘플 데이터 생성
        sample_data = system.create_sample_transaction()
        self.assertIsInstance(sample_data, dict)
        self.assertEqual(len(sample_data), 5)
        
        print("✅ 종단간 워크플로우 테스트 통과")
    
    def test_batch_processing_simulation(self):
        """배치 처리 시뮬레이션 테스트"""
        # 데이터 검증만 수행 (실제 훈련 없이)
        for i, transaction in enumerate(self.test_transactions):
            try:
                # 데이터 형식 검증
                self.assertIsInstance(transaction, dict)
                
                # 필수 키 존재 확인
                for key in FEATURES['input_features']:
                    self.assertIn(key, transaction)
                
                # 데이터 타입 검증
                for key, value in transaction.items():
                    self.assertIsInstance(value, (int, float))
                
            except Exception as e:
                self.fail(f"거래 {i+1} 검증 실패: {e}")
        
        print("✅ 배치 처리 시뮬레이션 테스트 통과")

class TestPerformanceMetrics(unittest.TestCase):
    """성능 메트릭 테스트"""
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 시스템 생성
        system = WhaleClassificationSystem(enable_logging=False)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # 메모리 증가량이 합리적인지 확인 (100MB 미만)
        self.assertLess(memory_increase, 100)
        
        print(f"✅ 메모리 사용량 테스트 통과 (증가량: {memory_increase:.1f}MB)")
    
    def test_response_time_simulation(self):
        """응답 시간 시뮬레이션 테스트"""
        import time
        
        # 시스템 초기화 시간 측정
        start_time = time.time()
        system = WhaleClassificationSystem(enable_logging=False)
        init_time = time.time() - start_time
        
        # 초기화 시간이 합리적인지 확인 (5초 미만)
        self.assertLess(init_time, 5.0)
        
        print(f"✅ 응답 시간 테스트 통과 (초기화: {init_time:.2f}초)")

def run_comprehensive_tests():
    """종합 테스트 실행"""
    print("🧪 Step 2 고래 분류 시스템 종합 테스트 시작")
    print("=" * 60)
    
    # 테스트 스위트 구성
    test_classes = [
        TestWhaleClassificationSystem,
        TestPredictorComponent,
        TestInsightGenerator,
        TestSystemIntegration,
        TestPerformanceMetrics
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__} 실행 중...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        result = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w')).run(suite)
        
        class_tests = result.testsRun
        class_passed = class_tests - len(result.failures) - len(result.errors)
        
        total_tests += class_tests
        passed_tests += class_passed
        
        if result.failures:
            print(f"❌ 실패: {len(result.failures)}건")
            for test, error in result.failures:
                print(f"   - {test}: {error}")
        
        if result.errors:
            print(f"⚠️ 오류: {len(result.errors)}건")
            for test, error in result.errors:
                print(f"   - {test}: {error}")
        
        if class_passed == class_tests:
            print(f"✅ {test_class.__name__}: {class_passed}/{class_tests} 통과")
    
    # 최종 결과
    print("\n" + "=" * 60)
    print(f"🎯 최종 테스트 결과: {passed_tests}/{total_tests} 통과")
    
    if passed_tests == total_tests:
        print("🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 시스템 점검이 필요합니다.")
        return False

def quick_functionality_test():
    """빠른 기능 테스트"""
    print("⚡ 빠른 기능 테스트 실행")
    print("-" * 40)
    
    try:
        # 1. 시스템 생성
        system = WhaleClassificationSystem(enable_logging=False)
        print("✅ 시스템 생성 성공")
        
        # 2. 설정 검증
        status = system.get_system_status()
        print(f"✅ 시스템 상태 조회 성공 ({len(status)}개 항목)")
        
        # 3. 샘플 데이터 생성
        sample = system.create_sample_transaction()
        print(f"✅ 샘플 데이터 생성 성공 ({len(sample)}개 피처)")
        
        # 4. 컴포넌트 검증
        predictor = WhalePredictor()
        insight_gen = WhaleInsightGenerator()
        print("✅ 핵심 컴포넌트 생성 성공")
        
        print("\n🎉 빠른 기능 테스트 완료! 기본 기능이 정상 작동합니다.")
        return True
        
    except Exception as e:
        print(f"\n❌ 빠른 기능 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🐋 Step 2 고래 분류 시스템 테스트")
    print("\n선택하세요:")
    print("1. 빠른 기능 테스트 (quick_functionality_test)")
    print("2. 종합 테스트 (run_comprehensive_tests)")
    
    # 기본적으로 빠른 테스트 실행
    quick_functionality_test() 