#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐋 고래 탐지 모델 데모 
======================
저장된 모델을 로드하여 실시간 고래 분류 예측

Author: LSTM_Crypto_Anomaly_Detection Team
Date: 2024
"""

from whale_detection_model import WhaleDetectionModel
import pandas as pd
import numpy as np

def demo_predictions():
    """다양한 예시 거래에 대한 고래 분류 데모"""
    
    print("🐋 비트코인 고래 탐지 모델 데모")
    print("=" * 50)
    
    # 모델 로드
    whale_model = WhaleDetectionModel()
    
    # 저장된 모델 로드
    if not whale_model.load_model():
        print("❌ 모델 로드에 실패했습니다.")
        print("💡 먼저 whale_detection_model.py를 실행해주세요.")
        return
    
    print("\n✅ 모델 로드 완료!")
    print("\n🔍 다양한 거래 패턴에 대한 예측 테스트:\n")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "🐋 거대 거래 (10,000 BTC)",
            "total_volume_btc": 10000,
            "input_count": 1,
            "output_count": 1,
            "concentration": 1.0,
            "fee_btc": 0.005
        },
        {
            "name": "🔄 분산형 거래 (많은 출력)",
            "total_volume_btc": 1000,
            "input_count": 2,
            "output_count": 50,
            "concentration": 0.2,
            "fee_btc": 0.001
        },
        {
            "name": "⚡ 급행 거래 (높은 수수료)",
            "total_volume_btc": 500,
            "input_count": 1,
            "output_count": 2,
            "concentration": 0.95,
            "fee_btc": 0.02
        },
        {
            "name": "📦 수집형 거래 (많은 입력)",
            "total_volume_btc": 800,
            "input_count": 100,
            "output_count": 1,
            "concentration": 0.99,
            "fee_btc": 0.0005
        },
        {
            "name": "🎯 집중형 거래 (일반적)",
            "total_volume_btc": 1000,
            "input_count": 1,
            "output_count": 2,
            "concentration": 0.98,
            "fee_btc": 0.00118625
        }
    ]
    
    # 각 테스트 케이스 예측
    for i, case in enumerate(test_cases, 1):
        print(f"【 테스트 {i} 】 {case['name']}")
        print("-" * 60)
        
        result = whale_model.predict_single(
            total_volume_btc=case['total_volume_btc'],
            input_count=case['input_count'],
            output_count=case['output_count'],
            concentration=case['concentration'],
            fee_btc=case['fee_btc'],
            show_details=False
        )
        
        # 결과 출력
        print(f"📊 입력: {case['total_volume_btc']:,} BTC, "
              f"Input: {case['input_count']}, "
              f"Output: {case['output_count']}, "
              f"집중도: {case['concentration']:.3f}, "
              f"수수료: {case['fee_btc']:.6f}")
        
        print(f"🎯 예측: 클래스 {result['predicted_class']} - {result['predicted_name']}")
        print(f"🔍 신뢰도: {result['confidence']:.1%}")
        
        # 상위 3개 클래스 확률 표시
        probabilities = result['probabilities']
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        
        print("📈 상위 3개 클래스 확률:")
        for idx in top_3_indices:
            class_name = whale_model.class_names[idx]
            prob = probabilities[idx]
            print(f"   {idx}. {class_name}: {prob:.1%}")
        
        print("\n")

def interactive_prediction():
    """사용자 입력을 받아 실시간 예측"""
    
    print("\n🎮 대화형 고래 분류 예측")
    print("=" * 50)
    
    # 모델 로드
    whale_model = WhaleDetectionModel()
    if not whale_model.load_model():
        print("❌ 모델 로드에 실패했습니다.")
        return
    
    print("✅ 모델 로드 완료!")
    print("\n📝 거래 정보를 입력하세요 (Enter로 기본값 사용):")
    
    try:
        # 사용자 입력
        volume_str = input("총 거래량 (BTC, 기본값: 1000): ").strip()
        total_volume_btc = float(volume_str) if volume_str else 1000.0
        
        input_str = input("Input 개수 (기본값: 1): ").strip()
        input_count = int(input_str) if input_str else 1
        
        output_str = input("Output 개수 (기본값: 2): ").strip()
        output_count = int(output_str) if output_str else 2
        
        concentration_str = input("집중도 (0-1, 기본값: 0.98): ").strip()
        concentration = float(concentration_str) if concentration_str else 0.98
        
        fee_str = input("수수료 (BTC, 기본값: 0.001): ").strip()
        fee_btc = float(fee_str) if fee_str else 0.001
        
        # 예측 수행
        print(f"\n🔍 입력된 거래 분석 중...")
        result = whale_model.predict_single(
            total_volume_btc=total_volume_btc,
            input_count=input_count,
            output_count=output_count,
            concentration=concentration,
            fee_btc=fee_btc,
            show_details=True
        )
        
    except ValueError as e:
        print(f"❌ 입력 오류: {e}")
        print("💡 숫자 형식으로 입력해주세요.")
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")

def batch_prediction_demo():
    """배치 예측 데모 (여러 거래 동시 처리)"""
    
    print("\n📊 배치 예측 데모")
    print("=" * 50)
    
    # 모델 로드
    whale_model = WhaleDetectionModel()
    if not whale_model.load_model():
        print("❌ 모델 로드에 실패했습니다.")
        return
    
    # 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'total_volume_btc': [1000, 5000, 500, 2000, 15000],
        'input_count': [1, 1, 10, 50, 1],
        'output_count': [2, 1, 1, 1, 100],
        'concentration': [0.98, 1.0, 0.99, 0.95, 0.3],
        'fee_btc': [0.001, 0.005, 0.02, 0.0005, 0.001]
    })
    
    print("📋 샘플 거래 데이터:")
    print(sample_data.to_string(index=False))
    
    # 배치 예측
    predictions = whale_model.predict(sample_data)
    
    print(f"\n🎯 예측 결과:")
    for i, pred in enumerate(predictions):
        class_name = whale_model.class_names[pred]
        print(f"거래 {i+1}: 클래스 {pred} ({class_name})")

if __name__ == "__main__":
    # 데모 실행
    demo_predictions()
    
    # 대화형 예측 (사용자가 원할 경우)
    while True:
        choice = input("\n🔄 대화형 예측을 시도하시겠습니까? (y/n): ").lower().strip()
        if choice == 'y':
            interactive_prediction()
            break
        elif choice == 'n':
            break
        else:
            print("y 또는 n을 입력해주세요.")
    
    # 배치 예측 데모
    batch_prediction_demo()
    
    print(f"\n🎉 데모 완료!")
    print(f"💡 모델 파일 위치: models/whale_detection/")
    print(f"📊 시각화 파일: models/whale_detection/model_performance.png")
    print(f"📈 피처 중요도: models/whale_detection/feature_importance.png") 