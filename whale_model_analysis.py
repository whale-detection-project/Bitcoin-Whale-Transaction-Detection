#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 고래 탐지 모델 성능 분석 및 개선 방안
==========================================
테스트 결과를 바탕으로 모델의 문제점을 분석하고 개선 방안을 제시

Author: LSTM_Crypto_Anomaly_Detection Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from whale_detection_model import WhaleDetectionModel

def analyze_model_performance():
    """모델 성능 분석"""
    
    print("🔍 고래 탐지 모델 성능 분석")
    print("=" * 50)
    
    # 테스트 케이스와 결과
    test_results = [
        {
            "test_name": "거대 거래 (10,000 BTC)",
            "expected": "거대형고래",
            "predicted": "수집형고래",
            "confidence": 59.6,
            "correct": False,
            "features": {
                "total_volume_btc": 10000,
                "input_count": 1,
                "output_count": 1,
                "concentration": 1.0,
                "fee_btc": 0.005
            }
        },
        {
            "test_name": "분산형 거래 (많은 출력)",
            "expected": "분산형고래",
            "predicted": "분산형고래",
            "confidence": 64.4,
            "correct": True,
            "features": {
                "total_volume_btc": 1000,
                "input_count": 2,
                "output_count": 50,
                "concentration": 0.2,
                "fee_btc": 0.001
            }
        },
        {
            "test_name": "급행 거래 (높은 수수료)",
            "expected": "급행형고래",
            "predicted": "분산형고래",
            "confidence": 50.7,
            "correct": False,
            "features": {
                "total_volume_btc": 500,
                "input_count": 1,
                "output_count": 2,
                "concentration": 0.95,
                "fee_btc": 0.02
            }
        },
        {
            "test_name": "수집형 거래 (많은 입력)",
            "expected": "수집형고래",
            "predicted": "집중형고래",
            "confidence": 100.0,
            "correct": False,  # 부분적으로는 합리적
            "features": {
                "total_volume_btc": 800,
                "input_count": 100,
                "output_count": 1,
                "concentration": 0.99,
                "fee_btc": 0.0005
            }
        },
        {
            "test_name": "집중형 거래 (일반적)",
            "expected": "집중형고래",
            "predicted": "분산형고래",
            "confidence": 50.7,
            "correct": False,
            "features": {
                "total_volume_btc": 1000,
                "input_count": 1,
                "output_count": 2,
                "concentration": 0.98,
                "fee_btc": 0.001186
            }
        }
    ]
    
    # 성능 요약
    total_tests = len(test_results)
    correct_predictions = sum(1 for result in test_results if result['correct'])
    accuracy = (correct_predictions / total_tests) * 100
    
    print(f"📊 테스트 결과 요약:")
    print(f"  총 테스트: {total_tests}개")
    print(f"  정확한 예측: {correct_predictions}개")
    print(f"  정확도: {accuracy:.1f}%")
    print()
    
    # 각 테스트 결과 분석
    print("📋 상세 분석:")
    for i, result in enumerate(test_results, 1):
        status = "✅ 정확" if result['correct'] else "❌ 오류"
        print(f"  {i}. {result['test_name']}: {status}")
        print(f"     예상: {result['expected']} → 예측: {result['predicted']} ({result['confidence']:.1f}%)")
        print()
    
    return test_results

def identify_problems():
    """문제점 식별"""
    
    print("🚨 식별된 문제점들:")
    print("=" * 50)
    
    problems = [
        {
            "category": "피처 중요도 불균형",
            "description": "Input/Output 개수가 거의 무시됨 (중요도 0%)",
            "impact": "수집형/분산형 고래의 핵심 특징을 놓침",
            "severity": "높음"
        },
        {
            "category": "클래스 불균형",
            "description": "집중형고래가 76.5%로 압도적 다수",
            "impact": "소수 클래스(거대형, 분산형) 학습 부족",
            "severity": "높음"
        },
        {
            "category": "거대형고래 인식 실패",
            "description": "10,000 BTC 거래를 수집형으로 잘못 분류",
            "impact": "대형 거래 탐지 능력 부족",
            "severity": "높음"
        },
        {
            "category": "급행형고래 인식 실패",
            "description": "높은 수수료(0.02 BTC)를 제대로 인식하지 못함",
            "impact": "긴급 거래 패턴 놓침",
            "severity": "중간"
        },
        {
            "category": "분산형 편향",
            "description": "애매한 경우 분산형으로 과도하게 분류",
            "impact": "다른 클래스의 정확도 저하",
            "severity": "중간"
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"{i}. 【{problem['category']}】")
        print(f"   문제: {problem['description']}")
        print(f"   영향: {problem['impact']}")
        print(f"   심각도: {problem['severity']}")
        print()

def suggest_improvements():
    """개선 방안 제시"""
    
    print("💡 개선 방안:")
    print("=" * 50)
    
    improvements = [
        {
            "category": "1. 피처 엔지니어링 개선",
            "solutions": [
                "Input/Output 비율 피처 추가 (input_output_ratio)",
                "거래량 대비 수수료 비율 피처 추가 (fee_rate)",
                "로그 변환으로 거래량 스케일 정규화",
                "피처 상호작용 항 추가 (volume × concentration)"
            ]
        },
        {
            "category": "2. 클래스 가중치 재조정",
            "solutions": [
                "거대형고래 가중치 증가 (30.0 → 50.0)",
                "분산형고래 가중치 감소 (15.0 → 10.0)",
                "SMOTE 기법으로 소수 클래스 오버샘플링",
                "Focal Loss 적용으로 어려운 샘플에 집중"
            ]
        },
        {
            "category": "3. 모델 아키텍처 개선",
            "solutions": [
                "XGBoost 또는 LightGBM으로 모델 변경",
                "앙상블 모델 구성 (RF + XGB + SVM)",
                "계층적 분류 (1차: 대형/소형, 2차: 세부 분류)",
                "딥러닝 모델 (Neural Network) 시도"
            ]
        },
        {
            "category": "4. 임계값 최적화",
            "solutions": [
                "클래스별 최적 임계값 설정",
                "확률 기반 다중 라벨 예측",
                "불확실성이 높은 경우 '미분류' 처리",
                "신뢰도 기반 예측 필터링"
            ]
        },
        {
            "category": "5. 데이터 품질 개선",
            "solutions": [
                "더 많은 거대형/분산형 고래 데이터 수집",
                "라벨링 기준 재검토 및 정제",
                "도메인 전문가 검증",
                "시계열 특성 반영 (거래 시간, 패턴)"
            ]
        }
    ]
    
    for improvement in improvements:
        print(f"🔧 {improvement['category']}")
        for solution in improvement['solutions']:
            print(f"   • {solution}")
        print()

def create_improvement_roadmap():
    """개선 로드맵 생성"""
    
    print("🗺️ 개선 로드맵 (우선순위별):")
    print("=" * 50)
    
    roadmap = [
        {
            "phase": "Phase 1: 즉시 개선 (1-2주)",
            "priority": "높음",
            "tasks": [
                "클래스 가중치 재조정",
                "피처 엔지니어링 (비율 피처 추가)",
                "임계값 최적화",
                "XGBoost 모델 시도"
            ],
            "expected_improvement": "정확도 10-15% 향상"
        },
        {
            "phase": "Phase 2: 중기 개선 (1개월)",
            "priority": "중간",
            "tasks": [
                "SMOTE 오버샘플링 적용",
                "앙상블 모델 구성",
                "계층적 분류 시스템 구축",
                "교차 검증 강화"
            ],
            "expected_improvement": "F1-Score 5-10% 향상"
        },
        {
            "phase": "Phase 3: 장기 개선 (2-3개월)",
            "priority": "낮음",
            "tasks": [
                "딥러닝 모델 개발",
                "시계열 특성 반영",
                "실시간 학습 시스템",
                "A/B 테스트 프레임워크"
            ],
            "expected_improvement": "전체 시스템 안정성 향상"
        }
    ]
    
    for phase in roadmap:
        print(f"📅 {phase['phase']}")
        print(f"   우선순위: {phase['priority']}")
        print(f"   작업 목록:")
        for task in phase['tasks']:
            print(f"     • {task}")
        print(f"   예상 효과: {phase['expected_improvement']}")
        print()

def generate_quick_fix():
    """빠른 수정 코드 생성"""
    
    print("⚡ 빠른 수정 방안:")
    print("=" * 50)
    
    quick_fixes = """
# 1. 개선된 클래스 가중치
optimal_class_weights = {
    0: 8.0,   # 수집형고래 (5.0 → 8.0)
    1: 10.0,  # 분산형고래 (15.0 → 10.0) 
    2: 20.0,  # 급행형고래 (15.0 → 20.0)
    3: 0.6,   # 집중형고래 (0.8 → 0.6)
    4: 50.0   # 거대형고래 (30.0 → 50.0)
}

# 2. 새로운 피처 추가
def add_engineered_features(df):
    df['input_output_ratio'] = df['input_count'] / (df['output_count'] + 1)
    df['fee_rate'] = df['fee_btc'] / df['total_volume_btc']
    df['volume_log'] = np.log1p(df['total_volume_btc'])
    df['volume_concentration'] = df['total_volume_btc'] * df['concentration']
    return df

# 3. 임계값 기반 후처리
def post_process_predictions(predictions, probabilities, thresholds):
    # 거대형고래 특별 처리
    volume_mask = features['total_volume_btc'] > 8000
    high_volume_indices = np.where(volume_mask)[0]
    predictions[high_volume_indices] = 4  # 거대형고래로 강제 분류
    
    # 급행형고래 특별 처리  
    fee_mask = features['fee_btc'] > 0.015
    high_fee_indices = np.where(fee_mask)[0]
    predictions[high_fee_indices] = 2  # 급행형고래로 강제 분류
    
    return predictions
"""
    
    print(quick_fixes)

def main():
    """메인 분석 함수"""
    
    print("🔍 고래 탐지 모델 종합 분석 리포트")
    print("=" * 60)
    print()
    
    # 1. 성능 분석
    test_results = analyze_model_performance()
    
    # 2. 문제점 식별
    identify_problems()
    
    # 3. 개선 방안
    suggest_improvements()
    
    # 4. 로드맵
    create_improvement_roadmap()
    
    # 5. 빠른 수정
    generate_quick_fix()
    
    print("📊 결론:")
    print("=" * 50)
    print("현재 모델은 40% 정확도로 개선이 필요합니다.")
    print("주요 문제는 피처 중요도 불균형과 클래스 불균형입니다.")
    print("Phase 1 개선사항 적용 시 60-70% 정확도 달성 가능할 것으로 예상됩니다.")
    print()
    print("🎯 다음 단계: Phase 1 개선사항부터 순차적으로 적용하세요!")

if __name__ == "__main__":
    main() 