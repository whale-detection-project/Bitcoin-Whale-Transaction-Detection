"""
🎭 Step 2 고래 분류 시스템 실전 시연
AI 기반 실시간 비트코인 고래 거래 분석 데모
"""

import sys
import time
import random
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from whale_classifier import WhaleClassificationSystem, demo_analysis
from config.settings import WHALE_CLASSES, FEATURES, UI_CONFIG

def print_header():
    """시연 헤더 출력"""
    colors = UI_CONFIG['colors']
    
    header = f"""
{colors['bold']}{'='*80}{colors['end']}
{colors['bold']}{colors['info']}🐋 STEP 2: 고래 거래 분류 시스템 실전 시연 🐋{colors['end']}
{colors['bold']}{'='*80}{colors['end']}
{colors['info']}🚀 AI 기반 실시간 비트코인 고래 거래 패턴 분석{colors['end']}
📊 Random Forest Classifier (F1-Score: 0.5081)
🧠 Level 3 전문가급 인사이트 제공
🎯 16.9% 성능 개선 (Step 1 최적화 적용)
{colors['bold']}{'='*80}{colors['end']}
"""
    print(header)

def create_realistic_scenarios():
    """실제와 유사한 시나리오 데이터 생성"""
    scenarios = [
        {
            'name': '🚀 급행형 고래 - 대량 고속 거래',
            'description': '높은 수수료를 지불하여 빠른 처리를 원하는 대량 거래',
            'data': {
                'total_volume_btc': 8500.0,
                'input_count': 1,
                'output_count': 2,
                'concentration': 0.95,
                'fee_btc': 0.008
            },
            'expected_class': '급행형고래'
        },
        {
            'name': '🐋 거대형 고래 - 초대형 거래',
            'description': '엄청난 양의 비트코인을 한 번에 이동하는 거래',
            'data': {
                'total_volume_btc': 15000.0,
                'input_count': 2,
                'output_count': 3,
                'concentration': 0.88,
                'fee_btc': 0.005
            },
            'expected_class': '거대형고래'
        },
        {
            'name': '🌊 분산형 고래 - 복잡한 분산 거래',
            'description': '자금을 여러 주소로 분산시키는 복잡한 거래 패턴',
            'data': {
                'total_volume_btc': 3200.0,
                'input_count': 8,
                'output_count': 15,
                'concentration': 0.35,
                'fee_btc': 0.002
            },
            'expected_class': '분산형고래'
        },
        {
            'name': '⚖️ 균형형 고래 - 표준적 거래',
            'description': '일반적인 고래 거래 패턴의 균형적 특성',
            'data': {
                'total_volume_btc': 2800.0,
                'input_count': 3,
                'output_count': 5,
                'concentration': 0.72,
                'fee_btc': 0.001
            },
            'expected_class': '균형형고래'
        }
    ]
    
    return scenarios

def demonstrate_scenario(system, scenario, scenario_num):
    """개별 시나리오 시연"""
    colors = UI_CONFIG['colors']
    
    print(f"\n{colors['bold']}{'─'*80}{colors['end']}")
    print(f"{colors['bold']}시나리오 {scenario_num}: {scenario['name']}{colors['end']}")
    print(f"{colors['info']}📝 상황: {scenario['description']}{colors['end']}")
    print(f"{'─'*80}")
    
    # 입력 데이터 표시
    print(f"\n{colors['warning']}📊 입력 거래 데이터:{colors['end']}")
    for key, value in scenario['data'].items():
        feature_desc = FEATURES['feature_descriptions'][key]
        if key == 'total_volume_btc':
            print(f"   💰 {feature_desc}: {value:,.0f} BTC")
        elif key in ['input_count', 'output_count']:
            print(f"   🔗 {feature_desc}: {value}개")
        elif key == 'concentration':
            print(f"   🎯 {feature_desc}: {value:.1%}")
        elif key == 'fee_btc':
            print(f"   💸 {feature_desc}: {value:.6f} BTC ({value/scenario['data']['total_volume_btc']*100:.4f}%)")
    
    # 분석 진행 시뮬레이션
    print(f"\n{colors['info']}🔍 AI 분석 진행 중...", end="")
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(f" 완료!{colors['end']}")
    
    # 실제 분석 수행
    try:
        result = system.analyze_transaction(scenario['data'])
        
        # 예측 결과 요약
        prediction = result['prediction']
        whale_info = prediction['whale_info']
        confidence = prediction['confidence']
        
        print(f"\n{colors['success']}🎯 분석 결과:{colors['end']}")
        print(f"   {whale_info['emoji']} 예측된 고래 유형: {colors['bold']}{whale_info['name']}{colors['end']}")
        print(f"   📊 신뢰도: {confidence:.1%}")
        
        # 예상과 비교
        if whale_info['name'] == scenario['expected_class']:
            print(f"   ✅ 예상 결과와 일치!")
        else:
            print(f"   📋 예상: {scenario['expected_class']}, 실제: {whale_info['name']}")
        
        # 상위 3개 확률 표시
        class_probs = prediction['class_probabilities']
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n   📈 상세 확률 분포:")
        for i, (cls, prob) in enumerate(sorted_probs[:3]):
            whale_name = WHALE_CLASSES[cls]['name']
            emoji = WHALE_CLASSES[cls]['emoji']
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"      {emoji} {whale_name}: {bar} {prob:.1%}")
        
        return result
        
    except Exception as e:
        print(f"\n{colors['error']}❌ 분석 실패: {e}{colors['end']}")
        return None

def show_expert_insights(result):
    """전문가급 인사이트 표시"""
    colors = UI_CONFIG['colors']
    
    print(f"\n{colors['bold']}🧠 전문가급 상세 분석 보고서{colors['end']}")
    print(f"{'─'*80}")
    
    # 전체 인사이트 출력
    print(result['expert_insights'])

def run_interactive_demo():
    """인터랙티브 데모 실행"""
    colors = UI_CONFIG['colors']
    
    print_header()
    
    print(f"{colors['info']}🚀 시스템 초기화 중...{colors['end']}")
    system = WhaleClassificationSystem(enable_logging=False)
    
    print(f"{colors['info']}📚 AI 모델 훈련 중... (실제 환경에서는 사전 훈련된 모델 로드){colors['end']}")
    if not system.setup_system():
        print(f"{colors['error']}❌ 시스템 설정 실패{colors['end']}")
        return
    
    print(f"\n{colors['success']}✅ 시스템 준비 완료!{colors['end']}")
    
    # 시나리오 실행
    scenarios = create_realistic_scenarios()
    
    for i, scenario in enumerate(scenarios, 1):
        input(f"\n{colors['warning']}📢 Enter를 눌러 시나리오 {i} 시작...{colors['end']}")
        result = demonstrate_scenario(system, scenario, i)
        
        if result:
            # 상세 분석 보기 옵션
            show_details = input(f"\n{colors['info']}🧠 상세 전문가 분석을 보시겠습니까? (y/n): {colors['end']}").lower()
            if show_details == 'y':
                show_expert_insights(result)
    
    # 배치 분석 데모
    print(f"\n{colors['bold']}{'='*80}{colors['end']}")
    print(f"{colors['bold']}🔄 배치 분석 데모{colors['end']}")
    print(f"{'='*80}")
    
    batch_demo = input(f"{colors['info']}📊 여러 거래를 한 번에 분석하는 배치 모드를 실행하시겠습니까? (y/n): {colors['end']}").lower()
    
    if batch_demo == 'y':
        print(f"\n{colors['info']}⚡ 5개 무작위 거래 배치 분석 중...{colors['end']}")
        
        # 무작위 거래 생성
        batch_transactions = []
        for i in range(5):
            transaction = system.create_sample_transaction()
            batch_transactions.append(transaction)
        
        # 배치 분석 실행
        batch_results = system.batch_analyze(batch_transactions, save_results=False)
        
        # 결과 요약
        print(f"\n{colors['success']}📈 배치 분석 결과 요약:{colors['end']}")
        
        class_counts = {}
        total_confidence = 0
        
        for result in batch_results:
            whale_name = result['prediction']['whale_info']['name']
            confidence = result['prediction']['confidence']
            
            class_counts[whale_name] = class_counts.get(whale_name, 0) + 1
            total_confidence += confidence
        
        print(f"   📊 총 분석 거래: {len(batch_results)}건")
        print(f"   🎯 평균 신뢰도: {total_confidence/len(batch_results):.1%}")
        print(f"   🐋 검출된 고래 유형:")
        
        for whale_type, count in class_counts.items():
            percentage = count / len(batch_results) * 100
            print(f"      - {whale_type}: {count}건 ({percentage:.1f}%)")

def run_quick_demo():
    """빠른 데모 실행"""
    colors = UI_CONFIG['colors']
    
    print_header()
    print(f"{colors['info']}⚡ 빠른 데모 모드 - 샘플 데이터로 즉시 시연{colors['end']}")
    
    # 기본 시연 실행
    result = demo_analysis()
    
    print(f"\n{colors['success']}🎉 빠른 데모 완료!{colors['end']}")
    return result

def show_system_capabilities():
    """시스템 기능 소개"""
    colors = UI_CONFIG['colors']
    
    capabilities = f"""
{colors['bold']}🎯 Step 2 고래 분류 시스템 주요 기능{colors['end']}
{'='*60}

{colors['info']}🧠 핵심 AI 기술:{colors['end']}
  📊 Random Forest Classifier (16.9% 성능 개선)
  🎯 F1-Score: 0.5081 (최적화된 성능)
  ⚡ 실시간 예측 및 분석

{colors['info']}🐋 고래 유형 분류:{colors['end']}
  🚀 급행형고래 - 높은 수수료의 빠른 거래
  🐋 거대형고래 - 초대형 볼륨의 거래
  🌊 분산형고래 - 복잡한 분산 패턴
  ⚖️ 균형형고래 - 표준적 거래 특성

{colors['info']}📊 분석 피처:{colors['end']}
  💰 거래량 (BTC)
  🔗 입출력 주소 수
  🎯 자금 집중도
  💸 거래 수수료

{colors['info']}🧠 인사이트 제공:{colors['end']}
  📈 상세 거래 특성 분석
  ⚡ 시장 영향도 예측
  🔍 유사 거래 패턴 검색
  🚨 이상치 탐지
  💼 비즈니스 추천사항

{colors['info']}⚡ 처리 능력:{colors['end']}
  🎯 단일 거래 실시간 분석
  📊 배치 거래 대량 처리
  💾 모델 저장/로드 기능
  📁 분석 결과 저장
"""
    
    print(capabilities)

def main():
    """메인 실행 함수"""
    colors = UI_CONFIG['colors']
    
    print(f"{colors['bold']}🐋 Step 2 고래 분류 시스템 시연 선택{colors['end']}")
    print("="*50)
    print("1. 📋 시스템 기능 소개")
    print("2. ⚡ 빠른 데모 (즉시 실행)")
    print("3. 🎭 인터랙티브 데모 (상세 시연)")
    print("4. 🧪 시스템 테스트")
    print("0. 종료")
    print("="*50)
    
    while True:
        try:
            choice = input(f"\n{colors['info']}선택하세요 (0-4): {colors['end']}").strip()
            
            if choice == '0':
                print(f"{colors['success']}👋 시연을 종료합니다. 감사합니다!{colors['end']}")
                break
            elif choice == '1':
                show_system_capabilities()
            elif choice == '2':
                run_quick_demo()
            elif choice == '3':
                run_interactive_demo()
            elif choice == '4':
                from tests.test_system import quick_functionality_test
                quick_functionality_test()
            else:
                print(f"{colors['error']}잘못된 선택입니다. 0-4 중에서 선택해주세요.{colors['end']}")
                
        except KeyboardInterrupt:
            print(f"\n{colors['warning']}💡 Ctrl+C로 종료되었습니다.{colors['end']}")
            break
        except Exception as e:
            print(f"{colors['error']}❌ 오류 발생: {e}{colors['end']}")

if __name__ == "__main__":
    main() 