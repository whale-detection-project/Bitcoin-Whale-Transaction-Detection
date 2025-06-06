"""
🧠 고래 거래 인사이트 생성기
레벨 3 전문가급 분석 및 해석 제공
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import statistics
from ..config.settings import (
    WHALE_CLASSES, FEATURES, ANALYSIS_CONFIG, 
    OUTPUT_CONFIG, UI_CONFIG
)

class WhaleInsightGenerator:
    """🧠 고래 거래 인사이트 분석기"""
    
    def __init__(self):
        self.colors = UI_CONFIG['colors']
    
    def generate_expert_insights(self, prediction_result: Dict, similar_transactions: List[Dict] = None) -> str:
        """레벨 3 전문가급 인사이트 생성"""
        
        insights = []
        
        # 헤더
        insights.append(self._create_header())
        
        # 기본 분류 결과
        insights.append(self._create_classification_summary(prediction_result))
        
        # 신뢰도 분석
        insights.append(self._create_confidence_analysis(prediction_result))
        
        # 피처 기여도 분석
        insights.append(self._create_feature_importance_analysis(prediction_result))
        
        # 거래 특성 분석
        insights.append(self._create_transaction_characteristics(prediction_result))
        
        # 시장 영향도 분석
        insights.append(self._create_market_impact_analysis(prediction_result))
        
        # 유사 거래 분석
        if similar_transactions:
            insights.append(self._create_similar_transactions_analysis(similar_transactions))
        
        # 이상치 탐지
        insights.append(self._create_anomaly_detection(prediction_result))
        
        # 비즈니스 추천사항
        insights.append(self._create_business_recommendations(prediction_result))
        
        # 푸터
        insights.append(self._create_footer())
        
        return "\n".join(insights)
    
    def _create_header(self) -> str:
        """분석 헤더 생성"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"""
{self.colors['bold']}{'='*80}{self.colors['end']}
{self.colors['bold']}{self.colors['info']}🐋 WHALE TRANSACTION ANALYSIS REPORT 🐋{self.colors['end']}
{self.colors['bold']}{'='*80}{self.colors['end']}
📅 분석 시간: {current_time}
🎯 분석 레벨: 전문가 (Level 3)
🧠 엔진: Random Forest Classifier (Step 1 최적화)
"""
        return header
    
    def _create_classification_summary(self, result: Dict) -> str:
        """분류 결과 요약"""
        whale_info = result['whale_info']
        confidence = result['confidence']
        
        # 신뢰도에 따른 색상 결정
        if confidence >= 0.8:
            confidence_color = self.colors['success']
        elif confidence >= 0.6:
            confidence_color = self.colors['warning']
        else:
            confidence_color = self.colors['error']
        
        summary = f"""
{self.colors['bold']}📊 분류 결과{self.colors['end']}
{'─'*40}
{whale_info['emoji']} 고래 유형: {self.colors['bold']}{whale_info['name']}{self.colors['end']}
🎯 신뢰도: {confidence_color}{confidence:.1%}{self.colors['end']}
📝 설명: {whale_info['description']}
🔄 행동 패턴: {whale_info['behavior']}
"""
        return summary
    
    def _create_confidence_analysis(self, result: Dict) -> str:
        """신뢰도 분석"""
        class_probs = result['class_probabilities']
        predicted_class = result['predicted_class']
        
        # 상위 3개 클래스 확률 분석
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        
        confidence_analysis = f"""
{self.colors['bold']}🎯 신뢰도 분석{self.colors['end']}
{'─'*40}
"""
        
        for i, (cls, prob) in enumerate(sorted_probs[:3]):
            whale_name = WHALE_CLASSES[cls]['name']
            emoji = WHALE_CLASSES[cls]['emoji']
            
            if i == 0:  # 최고 확률
                confidence_analysis += f"🥇 {emoji} {whale_name}: {prob:.1%} ← 예측 결과\n"
            elif i == 1:  # 2순위
                confidence_analysis += f"🥈 {emoji} {whale_name}: {prob:.1%}\n"
            else:  # 3순위
                confidence_analysis += f"🥉 {emoji} {whale_name}: {prob:.1%}\n"
        
        # 분류 확실성 평가
        entropy = -sum(p * np.log2(p + 1e-10) for p in class_probs.values())
        max_entropy = np.log2(len(class_probs))
        certainty = 1 - (entropy / max_entropy)
        
        confidence_analysis += f"\n📐 분류 확실성: {certainty:.1%} (엔트로피 기반)"
        
        return confidence_analysis
    
    def _create_feature_importance_analysis(self, result: Dict) -> str:
        """피처 기여도 분석"""
        contributions = result['feature_contributions']
        input_features = result['input_features']
        
        # 기여도 순으로 정렬
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: x[1]['contribution_score'], 
            reverse=True
        )
        
        analysis = f"""
{self.colors['bold']}🔍 피처 기여도 분석{self.colors['end']}
{'─'*40}
"""
        
        for feature_name, contrib_data in sorted_contributions:
            importance = contrib_data['importance']
            contribution = contrib_data['contribution_score']
            actual_value = input_features[feature_name]
            
            # 기여도에 따른 시각적 표시
            bar_length = int(contribution * 20)  # 0~20 길이 바
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            feature_desc = FEATURES['feature_descriptions'][feature_name]
            
            analysis += f"""
🔹 {feature_desc}
   값: {actual_value:,.4f} | 중요도: {importance:.1%} | 기여도: {contribution:.3f}
   {bar} {contribution:.1%}
"""
        
        return analysis
    
    def _create_transaction_characteristics(self, result: Dict) -> str:
        """거래 특성 분석"""
        features = result['input_features']
        whale_info = result['whale_info']
        
        # 거래량 분석
        volume_btc = features['total_volume_btc']
        volume_category = self._categorize_volume(volume_btc)
        
        # 복잡도 분석
        complexity = self._calculate_complexity(features)
        
        # 수수료 분석
        fee_analysis = self._analyze_fee(features)
        
        characteristics = f"""
{self.colors['bold']}📈 거래 특성 분석{self.colors['end']}
{'─'*40}
💰 거래량: {volume_btc:,.2f} BTC ({volume_category})
🔗 복잡도: {complexity['level']} ({complexity['score']:.2f}/10)
   ├─ 입력: {features['input_count']}개 주소
   ├─ 출력: {features['output_count']}개 주소  
   └─ 집중도: {features['concentration']:.1%}

💸 수수료 분석:
   ├─ 금액: {features['fee_btc']:.6f} BTC
   ├─ 비율: {fee_analysis['fee_rate']:.4%}
   └─ 수준: {fee_analysis['level']}

🎯 집중도 해석:
   {self._interpret_concentration(features['concentration'])}
"""
        
        return characteristics
    
    def _create_market_impact_analysis(self, result: Dict) -> str:
        """시장 영향도 분석"""
        whale_info = result['whale_info']
        features = result['input_features']
        
        # 시장 영향도 점수 계산
        impact_score = self._calculate_market_impact_score(features)
        
        # 영향 요인 분석
        impact_factors = self._analyze_impact_factors(features)
        
        analysis = f"""
{self.colors['bold']}⚡ 시장 영향도 분석{self.colors['end']}
{'─'*40}
📊 종합 영향도: {impact_score['total']:.1f}/10 ({impact_score['level']})
🔍 기본 영향: {whale_info['market_impact']}

📈 영향 요인 분해:
   ├─ 거래량 영향: {impact_factors['volume_impact']:.1f}/10
   ├─ 복잡도 영향: {impact_factors['complexity_impact']:.1f}/10
   ├─ 속도 영향: {impact_factors['speed_impact']:.1f}/10
   └─ 집중도 영향: {impact_factors['concentration_impact']:.1f}/10

💡 예상 시장 반응:
   {self._predict_market_reaction(impact_score, whale_info)}
"""
        
        return analysis
    
    def _create_similar_transactions_analysis(self, similar_transactions: List[Dict]) -> str:
        """유사 거래 분석"""
        if not similar_transactions:
            return ""
        
        analysis = f"""
{self.colors['bold']}🔍 유사 거래 패턴 분석{self.colors['end']}
{'─'*40}
📊 발견된 유사 거래: {len(similar_transactions)}건
"""
        
        for i, similar in enumerate(similar_transactions[:3], 1):
            similarity = similar['similarity_score']
            whale_name = similar['whale_name']
            features = similar['features']
            
            analysis += f"""
{i}. 유사도 {similarity:.1%} - {whale_name}
   거래량: {features['total_volume_btc']:.1f} BTC | 집중도: {features['concentration']:.1%}
"""
        
        # 패턴 분석
        pattern_analysis = self._analyze_transaction_patterns(similar_transactions)
        analysis += f"\n🧠 패턴 인사이트:\n   {pattern_analysis}"
        
        return analysis
    
    def _create_anomaly_detection(self, result: Dict) -> str:
        """이상치 탐지"""
        features = result['input_features']
        anomalies = self._detect_anomalies(features)
        
        if not anomalies:
            return f"""
{self.colors['bold']}🚨 이상치 탐지{self.colors['end']}
{'─'*40}
✅ 정상 범위 내 거래 - 이상 징후 없음
"""
        
        analysis = f"""
{self.colors['bold']}🚨 이상치 탐지{self.colors['end']}
{'─'*40}
⚠️ 다음 피처에서 이상치 발견:
"""
        
        for anomaly in anomalies:
            analysis += f"""
🔸 {anomaly['feature']}: {anomaly['value']:.4f}
   (정상 범위: {anomaly['normal_range']}, Z-Score: {anomaly['z_score']:.2f})
"""
        
        return analysis
    
    def _create_business_recommendations(self, result: Dict) -> str:
        """비즈니스 추천사항"""
        whale_info = result['whale_info']
        confidence = result['confidence']
        features = result['input_features']
        
        # 추천사항 생성
        recommendations = self._generate_recommendations(whale_info, confidence, features)
        
        rec_text = f"""
{self.colors['bold']}💼 비즈니스 추천사항{self.colors['end']}
{'─'*40}
"""
        
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec['icon']} {rec['title']}\n   {rec['description']}\n\n"
        
        return rec_text
    
    def _create_footer(self) -> str:
        """분석 푸터"""
        return f"""
{self.colors['bold']}{'='*80}{self.colors['end']}
{self.colors['info']}🎯 Step 2 고래 거래 분류 시스템 - AI 기반 실시간 분석{self.colors['end']}
📧 문의: Random Forest Classifier (F1-Score: 0.5081, 16.9% 성능 개선)
{self.colors['bold']}{'='*80}{self.colors['end']}
"""
    
    # === 헬퍼 메서드들 ===
    
    def _categorize_volume(self, volume_btc: float) -> str:
        """거래량 카테고리 분류"""
        if volume_btc >= 10000:
            return "초대형 (Mega Whale)"
        elif volume_btc >= 5000:
            return "대형 (Large Whale)"
        elif volume_btc >= 2000:
            return "중형 (Medium Whale)"
        else:
            return "소형 (Small Whale)"
    
    def _calculate_complexity(self, features: Dict) -> Dict:
        """거래 복잡도 계산"""
        input_count = features['input_count']
        output_count = features['output_count']
        concentration = features['concentration']
        
        # 복잡도 점수 계산 (0~10)
        complexity_score = min(10, (
            np.log10(input_count + 1) * 2 +
            np.log10(output_count + 1) * 2 +
            (1 - concentration) * 6
        ))
        
        if complexity_score >= 7:
            level = "매우 복잡"
        elif complexity_score >= 5:
            level = "복잡"
        elif complexity_score >= 3:
            level = "보통"
        else:
            level = "단순"
        
        return {"score": complexity_score, "level": level}
    
    def _analyze_fee(self, features: Dict) -> Dict:
        """수수료 분석"""
        fee_btc = features['fee_btc']
        volume_btc = features['total_volume_btc']
        
        fee_rate = (fee_btc / volume_btc) * 100 if volume_btc > 0 else 0
        
        if fee_rate >= 0.01:  # 0.01% 이상
            level = "매우 높음"
        elif fee_rate >= 0.005:  # 0.005% 이상
            level = "높음"
        elif fee_rate >= 0.001:  # 0.001% 이상
            level = "보통"
        else:
            level = "낮음"
        
        return {"fee_rate": fee_rate, "level": level}
    
    def _interpret_concentration(self, concentration: float) -> str:
        """집중도 해석"""
        if concentration >= 0.95:
            return "매우 높은 집중도 - 대부분 자금이 한 곳으로"
        elif concentration >= 0.8:
            return "높은 집중도 - 주요 수취인이 명확함"
        elif concentration >= 0.5:
            return "중간 집중도 - 적당한 분산"
        else:
            return "낮은 집중도 - 자금이 여러 곳으로 분산"
    
    def _calculate_market_impact_score(self, features: Dict) -> Dict:
        """시장 영향도 점수 계산"""
        volume_btc = features['total_volume_btc']
        concentration = features['concentration']
        fee_btc = features['fee_btc']
        
        # 각 요소별 영향도 (0~10)
        volume_impact = min(10, np.log10(volume_btc / 1000 + 1) * 3)
        concentration_impact = concentration * 5
        speed_impact = min(10, fee_btc * 10000)  # 높은 수수료 = 빠른 처리 = 높은 영향
        
        total_impact = (volume_impact + concentration_impact + speed_impact) / 3
        
        if total_impact >= 7:
            level = "매우 높음"
        elif total_impact >= 5:
            level = "높음"
        elif total_impact >= 3:
            level = "보통"
        else:
            level = "낮음"
        
        return {"total": total_impact, "level": level}
    
    def _analyze_impact_factors(self, features: Dict) -> Dict:
        """영향 요인 분석"""
        volume_btc = features['total_volume_btc']
        input_count = features['input_count']
        output_count = features['output_count']
        concentration = features['concentration']
        fee_btc = features['fee_btc']
        
        return {
            "volume_impact": min(10, np.log10(volume_btc / 1000 + 1) * 3),
            "complexity_impact": min(10, (input_count + output_count) / 5),
            "speed_impact": min(10, fee_btc * 10000),
            "concentration_impact": concentration * 10
        }
    
    def _predict_market_reaction(self, impact_score: Dict, whale_info: Dict) -> str:
        """시장 반응 예측"""
        total_impact = impact_score['total']
        
        if total_impact >= 7:
            return "단기 가격 변동성 증가 예상, 거래량 급증 가능성"
        elif total_impact >= 5:
            return "중간 정도의 시장 관심, 후속 거래 유발 가능성"
        elif total_impact >= 3:
            return "제한적 시장 영향, 정상적인 범위 내 거래"
        else:
            return "최소한의 시장 영향, 일반적인 거래 수준"
    
    def _analyze_transaction_patterns(self, similar_transactions: List[Dict]) -> str:
        """거래 패턴 분석"""
        if not similar_transactions:
            return "유사 거래 없음"
        
        # 클래스 분포 분석
        class_counts = {}
        for tx in similar_transactions:
            whale_name = tx['whale_name']
            class_counts[whale_name] = class_counts.get(whale_name, 0) + 1
        
        most_common = max(class_counts.items(), key=lambda x: x[1])
        
        return f"유사한 {most_common[1]}건이 모두 '{most_common[0]}' 패턴 - 일관된 행동 양상"
    
    def _detect_anomalies(self, features: Dict) -> List[Dict]:
        """이상치 탐지"""
        anomalies = []
        
        # 임계값 설정 (실제로는 훈련 데이터 통계 사용)
        thresholds = {
            'total_volume_btc': {'mean': 2000, 'std': 1000},
            'input_count': {'mean': 2, 'std': 3},
            'output_count': {'mean': 3, 'std': 4},
            'concentration': {'mean': 0.8, 'std': 0.2},
            'fee_btc': {'mean': 0.001, 'std': 0.002}
        }
        
        for feature_name, value in features.items():
            if feature_name in thresholds:
                threshold = thresholds[feature_name]
                z_score = abs(value - threshold['mean']) / threshold['std']
                
                if z_score > ANALYSIS_CONFIG['anomaly_z_score']:
                    anomalies.append({
                        'feature': FEATURES['feature_descriptions'][feature_name],
                        'value': value,
                        'z_score': z_score,
                        'normal_range': f"{threshold['mean'] - 2*threshold['std']:.3f} ~ {threshold['mean'] + 2*threshold['std']:.3f}"
                    })
        
        return anomalies
    
    def _generate_recommendations(self, whale_info: Dict, confidence: float, features: Dict) -> List[Dict]:
        """추천사항 생성"""
        recommendations = []
        
        # 신뢰도 기반 추천
        if confidence < 0.7:
            recommendations.append({
                'icon': '⚠️',
                'title': '추가 검증 필요',
                'description': '신뢰도가 낮으므로 추가 데이터 수집 후 재분석 권장'
            })
        
        # 고래 유형별 추천
        whale_type = whale_info['name']
        
        if whale_type == '급행형고래':
            recommendations.append({
                'icon': '⚡',
                'title': '시장 모니터링 강화',
                'description': '급행 거래는 중요한 시장 이벤트의 신호일 수 있음'
            })
        
        elif whale_type == '거대형고래':
            recommendations.append({
                'icon': '🚨',
                'title': '리스크 관리 준비',
                'description': '대량 거래로 인한 시장 변동성 대비 필요'
            })
        
        elif whale_type == '분산형고래':
            recommendations.append({
                'icon': '📊',
                'title': 'OTC 거래 가능성',
                'description': '분산 거래는 장외거래 준비 신호일 수 있음'
            })
        
        # 일반적 추천
        recommendations.append({
            'icon': '📈',
            'title': '거래 패턴 추적',
            'description': '동일 주소의 과거/미래 거래 패턴 모니터링 권장'
        })
        
        return recommendations 