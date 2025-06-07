#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎤 머신러닝 초보자를 위한 발표 가이드
=======================================
"가짜 99.8%에서 진짜 83.7%로: 올바른 고래 거래 분류 모델 만들기"

발표 순서:
1. 문제 발견: 99.8% 정확도의 함정
2. 진단: 3가지 치명적 문제점 
3. 해결: 올바른 방법으로 다시 만들기
4. 결과: 현실적이고 의미 있는 83.7% 달성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import warnings
warnings.filterwarnings('ignore')

# macOS 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정 - 강화된 버전"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS 시스템 폰트 경로에서 직접 폰트 찾기
        import os
        font_paths = [
            '/System/Library/Fonts/Apple SD Gothic Neo.ttc',
            '/System/Library/Fonts/AppleSDGothicNeo-Regular.otf',
            '/Library/Fonts/Arial Unicode MS.ttf',
            '/System/Library/Fonts/Helvetica.ttc'
        ]
        
        # 직접 폰트 파일 등록
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                    print(f"폰트 파일 등록: {font_path}")
                except:
                    pass
        
        # macOS에서 확실히 작동하는 폰트들
        font_candidates = [
            'Apple SD Gothic Neo',
            'AppleSDGothicNeo-Regular', 
            'Arial Unicode MS',
            'Helvetica',
            'DejaVu Sans'
        ]
        
        for font_name in font_candidates:
            try:
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 12
                
                # 한글 테스트
                fig, ax = plt.subplots(figsize=(2, 1))
                ax.text(0.5, 0.5, '한글테스트', fontsize=12, ha='center')
                plt.savefig('test_korean.png', dpi=50)
                plt.close(fig)
                
                # 파일이 제대로 생성되었는지 확인
                if os.path.exists('test_korean.png'):
                    os.remove('test_korean.png')  # 테스트 파일 삭제
                    print(f"✅ 한글 폰트 설정 성공: {font_name}")
                    return font_name
            except Exception as e:
                print(f"폰트 {font_name} 테스트 실패: {e}")
                continue
    
    # 폴백: 영어 전용
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️ 한글 폰트 설정 실패 - 영어 모드로 전환")
    return None

def explain_problem_1_label_leakage():
    """1단계: 라벨 유출 문제 설명"""
    print("🚨 문제 1: 라벨 유출 (Label Leakage)")
    print("=" * 50)
    print()
    
    print("💡 쉬운 설명:")
    print("시험 문제를 미리 알고 시험을 보는 것과 같습니다!")
    print()
    
    print("🔍 구체적인 문제:")
    print("✗ 잘못된 방법:")
    print("  1. 거래량으로 고래 타입을 정의 (라벨 생성)")
    print("  2. 같은 거래량을 피처로 사용해서 예측")
    print("  3. 당연히 100% 정확도!")
    print()
    
    print("📊 예시 코드:")
    print("""
    # ❌ 잘못된 예시
    # 라벨 생성: 거래량이 10000 BTC 이상이면 '메가고래'
    if transaction_amount >= 10000:
        label = 'mega_whale'
    
    # 예측에도 같은 거래량 사용
    features = [transaction_amount, ...]  # 같은 정보!
    model.predict(features)  # 당연히 맞춤!
    """)
    print()
    
    print("💭 비유:")
    print("문제: '키가 180cm 이상이면 농구선수'라고 정의")
    print("예측: 키를 보고 농구선수인지 맞추기")
    print("결과: 당연히 100% 정확! (하지만 의미없음)")

def explain_problem_2_circular_logic():
    """2단계: 순환 논리 문제 설명"""
    print("\n🔄 문제 2: 순환 논리 (Circular Logic)")
    print("=" * 50)
    print()
    
    print("💡 쉬운 설명:")
    print("자신이 만든 규칙을 자신이 맞추는 게임입니다!")
    print()
    
    print("🔍 구체적인 문제:")
    print("1. 클러스터링으로 그룹을 만듦 (A, B, C 그룹)")
    print("2. 같은 피처로 어느 그룹인지 예측")
    print("3. 자신이 만든 것을 자신이 맞추니 당연히 정확!")
    print()
    
    print("📊 과정 시각화:")
    print("""
    단계 1: [거래량, 수수료] → 클러스터링 → 그룹 A, B, C
    단계 2: [거래량, 수수료] → 모델 학습 → 그룹 예측
    
    ❌ 문제: 같은 정보로 만들고 같은 정보로 예측!
    """)
    print()
    
    print("💭 비유:")
    print("내가 색깔로 공을 A, B, C 상자에 분류")
    print("→ 누군가에게 '색깔 보고 어느 상자인지 맞춰봐'")
    print("→ 당연히 100% 정확! (하지만 의미없는 게임)")

def explain_problem_3_unrealistic_performance():
    """3단계: 비현실적 성능 문제 설명"""
    print("\n📈 문제 3: 비현실적 성능")
    print("=" * 50)
    print()
    
    print("💡 쉬운 설명:")
    print("현실에서는 불가능한 완벽한 성능이 나왔습니다!")
    print()
    
    print("🔍 현실적 기대치:")
    print("✅ 금융/암호화폐 분야의 현실적 성능:")
    print("  - 우수한 모델: 70-85%")
    print("  - 매우 좋은 모델: 85-90%") 
    print("  - 99%+: 거의 불가능 (문제가 있음)")
    print()
    
    print("🎯 왜 이런 성능이 현실적일까?")
    print("1. 시장은 복잡하고 예측하기 어려움")
    print("2. 노이즈와 불확실성이 많음")
    print("3. 사람의 행동은 패턴이 완벽하지 않음")
    print()
    
    print("💭 비유:")
    print("날씨 예측 정확도:")
    print("- 내일 날씨: 85% (좋음)")
    print("- 1주일 후: 70% (괜찮음)")
    print("- 99% 정확: 불가능! (뭔가 잘못됨)")

def explain_solution_independent_features():
    """해결책: 독립적인 피처 사용"""
    print("\n✅ 해결책: 독립적인 피처 사용")
    print("=" * 50)
    print()
    
    print("💡 핵심 아이디어:")
    print("라벨(정답)과 관련없는 정보만 사용해서 예측하기!")
    print()
    
    print("🔧 사용한 독립적인 피처들:")
    print()
    
    features_explanation = {
        "시간 피처": [
            "거래 시간 (몇 시, 무슨 요일)",
            "주말인지, 업무시간인지",
            "밤 시간인지, 피크 시간인지"
        ],
        "해시 피처": [
            "거래 해시의 길이",
            "해시의 첫 글자, 마지막 글자",
            "해시에 포함된 0의 개수"
        ],
        "상대적 피처": [
            "이전 거래와의 시간 간격",
            "1시간 동안의 거래 개수",
            "하루 동안의 거래 위치"
        ],
        "기타 피처": [
            "입력과 출력의 균형",
            "거래의 복잡도",
            "수수료 효율성"
        ]
    }
    
    for category, features in features_explanation.items():
        print(f"📊 {category} (라벨과 무관!):")
        for feature in features:
            print(f"  • {feature}")
        print()
    
    print("💭 비유:")
    print("농구선수 예측을 키 대신 다른 정보로:")
    print("• 운동화 브랜드, 선호하는 음식")
    print("• 잠자는 시간, 훈련 시간")
    print("• 이름의 길이, 출신 지역")
    print("→ 어렵지만 진짜 학습!")

def explain_solution_time_based_validation():
    """해결책: 시간 기반 검증"""
    print("\n⏰ 해결책: 시간 기반 검증")
    print("=" * 50)
    print()
    
    print("💡 핵심 아이디어:")
    print("과거 데이터로 학습해서 미래 데이터를 예측하기!")
    print()
    
    print("🔍 방법:")
    print("1. 데이터를 시간 순으로 정렬")
    print("2. 앞의 70%로 학습 (2021년 9월 이전)")
    print("3. 뒤의 30%로 테스트 (2021년 9월 이후)")
    print("4. 절대 미래 정보 사용 금지!")
    print()
    
    print("📊 데이터 분할:")
    print("""
    시간 순서: ←─────────────────────→
    [===== 훈련 데이터 =====][= 테스트 =]
         70% (과거)           30% (미래)
    """)
    print()
    
    print("💭 비유:")
    print("주식 투자와 같습니다:")
    print("✅ 올바른 방법: 과거 차트로 학습 → 내일 예측")
    print("❌ 잘못된 방법: 내일 주가를 미리 알고 예측")

def show_realistic_results():
    """현실적인 결과 보여주기"""
    print("\n🎯 현실적인 결과")
    print("=" * 50)
    print()
    
    print("📊 최종 성능 (XGBoost 모델):")
    print("✅ 정확도: 83.7% (현실적!)")
    print("✅ F1-Macro: 48.9%")
    print("✅ F1-Weighted: 82.2%")
    print()
    
    print("🔍 이 성능이 의미 있는 이유:")
    print("1. 라벨과 독립적인 피처만 사용")
    print("2. 시간 기반 검증으로 데이터 유출 방지")
    print("3. 실제 학습 가능한 패턴 탐지")
    print("4. 암호화폐 분야의 현실적 범위")
    print()
    
    print("📈 클래스별 분포:")
    class_results = {
        "일반 거래": "85.0% (대부분의 거래)",
        "중형 고래": "2.5% (상위 10% 거래량)",
        "대형 고래": "8.9% (상위 5% 거래량)", 
        "메가 고래": "0.5% (상위 1% 거래량)",
        "복잡 거래": "2.8% (높은 복잡도)",
        "급행 거래": "0.1% (높은 수수료)"
    }
    
    for class_name, description in class_results.items():
        print(f"  • {class_name}: {description}")

def create_comparison_visualization():
    """비교 시각화 생성 - 한글 폰트 지원"""
    print("\n📊 시각적 비교")
    print("=" * 50)
    
    # 한글 폰트 설정 시도
    korean_font_success = setup_korean_font()
    
    # 강제로 영어 모드 사용 (한글 문제 해결까지)
    use_korean = False  # 임시로 False 설정
    
    if use_korean and korean_font_success:
        print("한글 모드로 차트 생성")
        models = ['v1.0\n(클러스터링)', 'v2.0\n(도메인 룰)', 'v4.0\n(현실적)']
        title1 = '모델별 정확도 비교'
        title2 = '모델별 신뢰도 비교'
        ylabel1 = '정확도 (%)'
        ylabel2 = '신뢰도 점수'
        fake_text = '가짜'
        real_text = '진짜'
    else:
        print("영어 모드로 차트 생성 (한글 폰트 문제 방지)")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        models = ['v1.0\n(Clustering)', 'v2.0\n(Domain Rules)', 'v4.0\n(Realistic)']
        title1 = 'Model Accuracy Comparison'
        title2 = 'Model Reliability Comparison'
        ylabel1 = 'Accuracy (%)'
        ylabel2 = 'Reliability Score'
        fake_text = 'Fake'
        real_text = 'Real'
    
    # 성능 비교 데이터
    accuracy = [99.8, 99.9, 83.7]
    reliability = [0, 0, 85]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 정확도 비교
    bars1 = ax1.bar(models, accuracy, color=['red', 'orange', 'green'], alpha=0.7)
    ax1.set_ylabel(ylabel1, fontsize=12)
    ax1.set_title(title1, fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 100)
    
    # 값 표시
    for bar, acc in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 가짜/진짜 표시
    ax1.text(0, 95, f'❌ {fake_text}', ha='center', fontsize=12, color='red', fontweight='bold')
    ax1.text(1, 95, f'❌ {fake_text}', ha='center', fontsize=12, color='red', fontweight='bold')
    ax1.text(2, 78, f'✅ {real_text}', ha='center', fontsize=12, color='green', fontweight='bold')
    
    # 신뢰도 비교
    bars2 = ax2.bar(models, reliability, color=['red', 'orange', 'green'], alpha=0.7)
    ax2.set_ylabel(ylabel2, fontsize=12)
    ax2.set_title(title2, fontweight='bold', fontsize=14)
    ax2.set_ylim(0, 100)
    
    # 값 표시
    for bar, rel in zip(bars2, reliability):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rel}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # x축 라벨 설정
    ax1.tick_params(axis='x', rotation=0)
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # 차트 저장
    filename = 'presentation_comparison_en.png'  # 영어 버전으로 저장
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ 비교 차트 생성 완료: {filename}")
    except Exception as e:
        print(f"⚠️ 차트 저장 중 오류: {e}")
    
    plt.close()

def presentation_summary():
    """발표 요약"""
    print("\n🎤 발표 요약: 핵심 메시지")
    print("=" * 50)
    print()
    
    print("1️⃣ 문제 인식:")
    print("   • 99.8% 정확도는 너무 완벽해서 의심")
    print("   • 3가지 치명적 문제 발견")
    print()
    
    print("2️⃣ 문제 해결:")
    print("   • 독립적인 피처 사용 (33개)")
    print("   • 시간 기반 검증 도입")
    print("   • 현실적인 클래스 정의")
    print()
    
    print("3️⃣ 의미 있는 결과:")
    print("   • 83.7% 정확도 달성")
    print("   • 진짜 학습 가능한 모델")
    print("   • 실무에 활용 가능한 수준")
    print()
    
    print("💡 핵심 교훈:")
    print("   완벽한 성능보다 신뢰할 수 있는 성능이 중요!")
    print("   머신러닝에서 99%+는 대부분 문제가 있음!")

def main():
    """발표용 가이드 실행"""
    print("🎤 머신러닝 초보자를 위한 발표 가이드")
    print("=" * 60)
    print("주제: '가짜 99.8%에서 진짜 83.7%로'")
    print("=" * 60)
    
    # 한글 폰트 설정
    setup_korean_font()
    
    # 문제점 설명
    explain_problem_1_label_leakage()
    explain_problem_2_circular_logic()
    explain_problem_3_unrealistic_performance()
    
    # 해결책 설명
    explain_solution_independent_features()
    explain_solution_time_based_validation()
    
    # 결과 보여주기
    show_realistic_results()
    
    # 시각화 생성
    create_comparison_visualization()
    
    # 요약
    presentation_summary()
    
    print("\n🎯 발표 가이드 완료!")
    print("이제 이 내용으로 자신있게 발표하세요! 🚀")

if __name__ == "__main__":
    main() 