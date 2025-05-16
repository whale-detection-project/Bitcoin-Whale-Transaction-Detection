import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드 시도
load_dotenv()

# 스트림릿 설정
st.set_page_config(
    page_title="바이낸스 실시간 차트 (업비트 스타일)",
    page_icon="📈",
    layout="wide"
)

# 토스+업비트 스타일 디자인 색상
DESIGN_COLORS = {
    "background": "#FFFFFF",
    "header_bg": "#093687",  # 업비트 블루
    "header_text": "#FFFFFF",
    "text": "#333333",
    "light_text": "#666666",
    "grid": "#E5E5E5",
    "up": "#EB5757",         # 좀더 선명한 상승색
    "down": "#1261C4",       # 업비트 하락색
    "volume_up": "#FFEEEE",  # 더 연한 볼륨 색상
    "volume_down": "#EDF4FF",
    "ma5": "#F2994A",        # 주황색
    "ma10": "#27AE60",       # 초록색
    "ma20": "#2F80ED",       # 파란색
    "ma60": "#9B51E0",       # 보라색
    "ma120": "#E51D93",      # 분홍색
    "button_bg": "#093687",  # 업비트 블루 (현재는 toss_blue 사용)
    "button_text": "#FFFFFF",
    "input_bg": "#F5F5F7",
    "border": "#E5E5E7",
    "card_bg": "#FFFFFF",
    "panel_bg": "#F9F9FB", # 이전 #F9F9FB
    "shadow": "rgba(0, 0, 0, 0.05)",
    "toss_blue": "#3182F6",  # 토스 파란색
    "toss_gray": "#F2F4F6",  # 토스 배경 회색
}

# CSS 스타일 적용
def apply_toss_upbit_style():
    st.markdown(f"""
    <style>
    /* 기본 폰트 및 안티앨리어싱 적용 */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* 전체 기본 스타일 */
    * {{
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        box-sizing: border-box;
    }}

    /* 전체 배경 및 글자색 */
    .stApp {{
        background-color: {DESIGN_COLORS["background"]};
        color: {DESIGN_COLORS["text"]};
    }}

    /* 컨테이너 블러 방지 및 성능 향상 */
    .stApp > header, .stApp > div[data-testid="stToolbar"], .stApp > footer, .main > div {{
        transform: translateZ(0);
        -webkit-transform: translateZ(0);
        will-change: transform; /* 하드웨어 가속 힌트 */
    }}

    /* Streamlit 기본 헤더 스타일 (필요시 주석 해제) */
    /*
    .stApp > header {{
        background-color: {DESIGN_COLORS["header_bg"]} !important;
        box-shadow: 0 2px 8px {DESIGN_COLORS["shadow"]} !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
    }}
    div[data-testid="stToolbar"] {{
        background-color: {DESIGN_COLORS["header_bg"]} !important;
    }}
    */

    /* 버튼 스타일 - 토스 스타일 */
    .stButton>button {{
        background-color: {DESIGN_COLORS["toss_blue"]};
        color: {DESIGN_COLORS["button_text"]};
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px {DESIGN_COLORS["shadow"]};
        /* width: 100%; /* use_container_width=True 와 유사하게 */
    }}
    .stButton>button:hover {{
        background-color: #2b74df;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px {DESIGN_COLORS["shadow"]};
    }}
    .stButton>button:active {{
        transform: translateY(0px);
        box-shadow: 0 2px 4px {DESIGN_COLORS["shadow"]};
    }}

    /* 라디오 버튼 스타일 - 토스 스타일 (선택지 버튼형) */
    div[data-baseweb="radio-group"] {{ /* Streamlit >= 1.13.0 */
        display: flex;
        gap: 8px;
        background-color: {DESIGN_COLORS["toss_gray"]};
        padding: 6px; /* 패딩 약간 줄임 */
        border-radius: 10px;
    }}
    div[data-baseweb="radio-group"] label {{
        flex: 1;
        background-color: {DESIGN_COLORS["background"]} !important;
        border: 1px solid {DESIGN_COLORS["border"]} !important;
        border-radius: 8px !important;
        padding: 8px 10px !important; /* 패딩 약간 줄임 */
        margin-right: 0 !important;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 14px !important;
        text-align: center;
        color: {DESIGN_COLORS["text"]} !important;
        box-shadow: 0 1px 2px {DESIGN_COLORS["shadow"]};
    }}
    div[data-baseweb="radio-group"] label:hover {{
        border-color: {DESIGN_COLORS["toss_blue"]} !important;
        box-shadow: 0 0 0 1px {DESIGN_COLORS["toss_blue"]}, 0 2px 4px {DESIGN_COLORS["shadow"]} !important;
    }}
    /* 선택된 라디오 버튼 스타일 */
    div[data-baseweb="radio-group"] input[type="radio"]:checked + div {{
        background-color: {DESIGN_COLORS["toss_blue"]} !important;
        border-color: {DESIGN_COLORS["toss_blue"]} !important;
        color: {DESIGN_COLORS["button_text"]} !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(49, 130, 246, 0.4) !important;
    }}
    /* 라디오 버튼의 실제 동그라미 숨기기 */
    div[data-baseweb="radio-group"] input[type="radio"] {{
        opacity: 0;
        width: 0;
        height: 0;
        position: absolute;
    }}

    /* 체크박스 스타일 - 토스 스타일 */
    .stCheckbox {{
        background-color: transparent;
        padding: 8px 0; /* 상하 패딩 추가 */
        margin-bottom: 0;
    }}
    .stCheckbox label {{
        display: flex;
        align-items: center;
        cursor: pointer;
        font-size: 15px;
        color: {DESIGN_COLORS["text"]};
        gap: 10px; /* 아이콘과 텍스트 간격 */
    }}
    /* 기본 체크박스 숨기기 */
    .stCheckbox input[type="checkbox"] {{
        opacity: 0;
        width: 0;
        height: 0;
        position: absolute;
    }}
    /* 커스텀 체크박스 아이콘 */
    .stCheckbox label div[data-baseweb="checkbox"] > div:first-child {{
        width: 20px;
        height: 20px;
        border: 2px solid {DESIGN_COLORS["border"]};
        border-radius: 6px;
        background-color: {DESIGN_COLORS["background"]};
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        flex-shrink: 0; /* 크기 고정 */
    }}
    /* 체크됐을 때 아이콘 스타일 */
    .stCheckbox input[type="checkbox"]:checked + div > div:first-child {{
        background-color: {DESIGN_COLORS["toss_blue"]};
        border-color: {DESIGN_COLORS["toss_blue"]};
    }}
    /* 체크 아이콘 (SVG) - Base64 인코딩된 SVG 사용 */
    .stCheckbox input[type="checkbox"]:checked + div > div:first-child::after {{
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white' width='14px' height='14px'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z'/%3E%3C/svg%3E");
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .stCheckbox label:hover div[data-baseweb="checkbox"] > div:first-child {{
        border-color: {DESIGN_COLORS["toss_blue"]};
    }}

    /* 입력 필드 스타일 */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {{
        background-color: {DESIGN_COLORS["input_bg"]};
        border-radius: 8px;
        border: 1px solid {DESIGN_COLORS["border"]};
        padding: 12px 16px;
        font-size: 15px;
        color: {DESIGN_COLORS["text"]};
        transition: all 0.2s ease;
    }}
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {{
        border-color: {DESIGN_COLORS["toss_blue"]};
        box-shadow: 0 0 0 2px rgba(49, 130, 246, 0.2);
    }}

    /* 선택 위젯 (Selectbox) 스타일 */
    .stSelectbox > div > div {{
        background-color: {DESIGN_COLORS["input_bg"]};
        border: 1px solid {DESIGN_COLORS["border"]};
        border-radius: 8px;
        font-size: 15px;
        color: {DESIGN_COLORS["text"]};
    }}
     .stSelectbox > div > div > div {{
        padding: 10px 14px;
     }}

    /* 멀티셀렉트 스타일 */
    .stMultiSelect > div[data-baseweb="select"] > div:first-child {{
        background-color: {DESIGN_COLORS["input_bg"]};
        border-radius: 8px;
        border: 1px solid {DESIGN_COLORS["border"]};
        padding: 6px 10px; /* 패딩 조정 */
    }}
    .stMultiSelect span[data-baseweb="tag"] {{
        background-color: {DESIGN_COLORS["toss_blue"]} !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        font-size: 13px !important;
        margin: 2px !important; /* 태그간 간격 */
    }}

    /* 배지 스타일 (MA 값 표시용) */
    .badge {{
        display: inline-block;
        padding: 5px 10px;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 6px;
        background-color: {DESIGN_COLORS["toss_gray"]};
        font-weight: 500;
        font-size: 13px;
        border: 1px solid {DESIGN_COLORS["border"]};
        /* color는 ma_badge 함수에서 style로 직접 설정 */
    }}

    /* 프로그레스 바 스타일 - 토스 스타일 */
    div[data-testid="stProgress"] {{
        height: 10px !important;
        border-radius: 10px !important;
        background-color: {DESIGN_COLORS["toss_gray"]} !important;
        overflow: hidden;
    }}
    div[data-testid="stProgress"] > div {{
        background-color: {DESIGN_COLORS["toss_blue"]} !important;
        background-image: none !important;
        border-radius: 10px !important;
    }}

    /* 라벨 스타일 (컨트롤 위젯용) */
    .control-label {{
        font-size: 13px;
        font-weight: 500;
        color: {DESIGN_COLORS["light_text"]};
        margin-bottom: 6px;
        display: block;
    }}

    /* 카드 스타일 */
    .card {{ /* 일반 카드 컨테이너 */
        background-color: {DESIGN_COLORS["card_bg"]};
        border-radius: 12px;
        border: 1px solid {DESIGN_COLORS["border"]};
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px {DESIGN_COLORS["shadow"]};
        overflow: hidden;
        transition: all 0.2s ease;
    }}
    .card:hover {{
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }}

    /* 설정 패널 (Expander 내부 등) */
    div[data-testid="stExpander"] {{
        border: none;
        box-shadow: none;
        background-color: transparent;
        margin-bottom: 16px;
        border-radius: 12px; /* Expander 자체에 radius */
        overflow: hidden; /* 내용물이 radius를 넘지 않도록 */
    }}
    div[data-testid="stExpander"] summary {{
        padding: 16px 20px;
        border-bottom: 1px solid {DESIGN_COLORS["border"]};
        font-size: 16px;
        font-weight: 600;
        background-color: {DESIGN_COLORS["panel_bg"]}; /* Expander 헤더 배경 */
        border-radius: 12px 12px 0 0; /* 상단 모서리만 radius */
    }}
    div[data-testid="stExpander"] summary:hover {{
        color: {DESIGN_COLORS["toss_blue"]};
    }}
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {{
        padding: 20px;
        background-color: {DESIGN_COLORS["panel_bg"]}; /* Expander 내용 배경 */
        border: 1px solid {DESIGN_COLORS["border"]};
        border-top: none; /* summary의 border-bottom과 중복 방지 */
        box-shadow: 0 2px 8px {DESIGN_COLORS["shadow"]};
        border-radius: 0 0 12px 12px; /* 하단 모서리만 radius */
    }}


    /* 헤더 메뉴 스타일 (커스텀 헤더용) */
    .menu-container {{
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: {DESIGN_COLORS["header_bg"]};
    }}

    /* 탭 스타일 */
    .tab-container {{
        display: flex;
        border-bottom: 2px solid {DESIGN_COLORS["border"]};
        margin-bottom: 20px;
        padding: 0 16px;
    }}
    .tab-item {{
        padding: 14px 22px;
        font-weight: 500;
        font-size: 15px;
        color: {DESIGN_COLORS["light_text"]};
        cursor: pointer;
        position: relative;
        transition: all 0.2s ease;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
    }}
    .tab-item:hover {{
        color: {DESIGN_COLORS["toss_blue"]};
    }}
    .tab-item.active {{
        color: {DESIGN_COLORS["toss_blue"]};
        font-weight: 600;
        border-bottom-color: {DESIGN_COLORS["toss_blue"]};
    }}

    /* 시세 정보 카드 스타일 (Price Card) */
    .price-card {{
        background-color: {DESIGN_COLORS["card_bg"]};
        border-radius: 12px;
        border: 1px solid {DESIGN_COLORS["border"]};
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px {DESIGN_COLORS["shadow"]};
    }}

    /* 가격 텍스트 스타일 */
    .price-value {{
        font-size: 30px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }}
    .price-change {{
        font-size: 17px;
        font-weight: 500;
        margin: 0;
    }}

    /* 시간 스타일 */
    .time-label {{
        font-size: 13px;
        color: {DESIGN_COLORS["light_text"]};
        margin-top: 8px;
        display: block;
        text-align: right; /* 우측 정렬 */
    }}

    /* 컨트롤 영역 스타일 */
    .control-area {{
        background-color: {DESIGN_COLORS["toss_gray"]};
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 24px; /* 간격 증가 */
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        border: 1px solid {DESIGN_COLORS["border"]};
    }}
    /* 컨트롤 영역 내 개별 아이템 스타일링 */
    .control-area > div {{ /* 컬럼 내부의 st.selectbox, st.checkbox 등이 담긴 div */
        flex-grow: 1;
    }}


    /* 데이터 테이블 스타일 */
    .stDataFrame {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {DESIGN_COLORS["border"]};
        box-shadow: 0 2px 6px {DESIGN_COLORS["shadow"]};
    }}
    /* .dataframe 클래스는 st.dataframe() 사용 시 자동으로 생성되지 않음. stDataFrame 내부 table 스타일링 */
    .stDataFrame table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    .stDataFrame th {{
        background-color: {DESIGN_COLORS["toss_gray"]};
        padding: 12px 14px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid {DESIGN_COLORS["border"]};
        color: {DESIGN_COLORS["text"]};
    }}
    .stDataFrame td {{
        padding: 12px 14px;
        border-bottom: 1px solid {DESIGN_COLORS["border"]};
        color: {DESIGN_COLORS["text"]};
    }}
    .stDataFrame tr:hover td {{
        background-color: #f0f4f9;
    }}
    .stDataFrame div[data-testid="stVirtualDataGrid"] {{
        border-radius: 0 0 10px 10px;
    }}

    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: {DESIGN_COLORS["toss_gray"]};
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #cccccc;
        border-radius: 5px;
        border: 2px solid {DESIGN_COLORS["toss_gray"]};
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #aaaaaa;
    }}

    /* 메시지 알림 스타일 (st.info, st.success 등) */
    div[data-testid="stNotification"] {{
        background-color: {DESIGN_COLORS["toss_gray"]};
        border-left: 5px solid {DESIGN_COLORS["toss_blue"]};
        border-radius: 8px;
        box-shadow: 0 2px 8px {DESIGN_COLORS["shadow"]};
        padding: 16px;
        margin-bottom: 16px;
        color: {DESIGN_COLORS["text"]};
    }}
    div[data-testid="stNotification"] p {{
        margin: 0;
        font-size: 14px;
    }}
    div[data-testid="stNotification"][data-message-type="success"] {{ /* Streamlit 1.22+ */
        border-left-color: {DESIGN_COLORS["ma10"]};
    }}
    div[data-testid="stNotification"][data-message-type="error"] {{
        border-left-color: {DESIGN_COLORS["up"]};
    }}
    div[data-testid="stNotification"][data-message-type="warning"] {{
        border-left-color: {DESIGN_COLORS["ma5"]};
    }}


    /* 지표 박스 스타일 (Metric Box) */
    .metric-container {{
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }}
    .metric-box {{
        flex: 1;
        background: {DESIGN_COLORS["card_bg"]};
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px {DESIGN_COLORS["shadow"]};
        display: flex;
        flex-direction: column;
        border: 1px solid {DESIGN_COLORS["border"]};
    }}
    .metric-label {{
        font-size: 14px;
        color: {DESIGN_COLORS["light_text"]};
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 600;
        color: {DESIGN_COLORS["text"]};
    }}

    /* Slider 스타일 */
    div[data-testid="stSlider"] {{
        padding: 8px 0;
    }}
    div[data-testid="stSlider"] div[role="slider"] {{ /* 슬라이더 핸들 */
        background-color: {DESIGN_COLORS["toss_blue"]} !important;
        border: 2px solid {DESIGN_COLORS["toss_blue"]} !important;
        box-shadow: 0 0 0 3px rgba(49, 130, 246, 0.2) !important; /* 포커스 효과 */
    }}
     div[data-testid="stSlider"] div[data-testid="stTickBar"] > div:nth-child(2) > div {{ /* 슬라이더 트랙 (채워진 부분) */
         background-color: {DESIGN_COLORS["toss_blue"]} !important;
    }}
    div[data-testid="stSlider"] div[data-testid="stTickBar"] > div:first-child > div {{ /* 슬라이더 트랙 (전체) */
         background-color: {DESIGN_COLORS["toss_gray"]} !important;
    }}


    /* 전반적인 여백과 섹션 구분 개선 */
    .main > div {{
        padding-top: 1rem;
    }}

    h1, h2, h3 {{ /* 기본 헤딩 스타일 */
        color: {DESIGN_COLORS["text"]};
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 22px; }}
    h3 {{ font-size: 18px; }}

    /* 차트 타이틀과 탭 사이 간격 */
    /* div.menu-container + div.tab-container 는 현재 구조에서 menu-container가 st.markdown 안에 있어서 직접적인 + 선택자 적용 어려움 */
    /* 대신, 차트 타이틀을 markdown으로 생성 시 margin-bottom 조정 */

    </style>
    """, unsafe_allow_html=True)

apply_toss_upbit_style()

# 업비트 스타일 헤더
def render_header():
    header_html = f"""
    <div class="menu-container" style="position: sticky; top: 0; z-index: 999;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="color: {DESIGN_COLORS['header_text']}; font-weight: 600; font-size: 18px;">
                <img src="https://cdn-icons-png.flaticon.com/512/5968/5968770.png" width="24" style="margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);"/>
                바이낸스 차트
            </div>
        </div>
        <div>
            <span style="color: {DESIGN_COLORS['header_text']}; font-weight: 500; font-size: 14px;">실시간 시세</span>
        </div>
    </div>
    """
    # Streamlit의 앱 헤더 영역에 직접 삽입 시도 (상단 고정 효과)
    # st.markdown(header_html, unsafe_allow_html=True) # 이 방식은 헤더 영역에 직접 삽입되지 않음
    # 대신, body 최상단에 위치하도록 st.container() 등을 사용하거나, CSS로 기존 헤더를 덮어쓰는 방식 고려
    # 여기서는 페이지 최상단에 고정된 형태로 렌더링
    st.markdown(f"<div style='position: sticky; top: 0; z-index: 1001; width: 100%;'>{header_html}</div>", unsafe_allow_html=True)


# 탭 UI 렌더링
def render_tabs(active_tab):
    tabs_data = [
        {"id": "시세", "label": "시세"},
        {"id": "정보", "label": "정보"}
    ]
    
    tab_items_html = ""
    for tab in tabs_data:
        is_active = 'active' if active_tab == tab["id"] else ''
        # Note: For interactive tabs, you'd typically use st.radio or st.tabs,
        # or implement callbacks if using pure HTML/JS.
        # This function is purely for display based on active_tab argument.
        tab_items_html += f'<div class="tab-item {is_active}">{tab["label"]}</div>'
        
    tab_html = f"""
    <div class="tab-container">
        {tab_items_html}
    </div>
    """
    st.markdown(tab_html, unsafe_allow_html=True)

# 카드 컴포넌트 (이제는 CSS로 스타일링되므로, div.card로 충분)
def card_wrapper(content_html, key=None): # 이름 변경하여 st.card와 구분
    # st.container().markdown(...) 방식은 내부에서 markdown을 또 호출하는 구조.
    # 직접 HTML을 반환하도록 하여 st.markdown으로 한번에 처리하는 것이 나을 수 있음.
    return f'<div class="card">{content_html}</div>'


# 가격 정보 컴포넌트
def price_card_html(price, change, change_pct, color): # 이름 변경
    change_symbol = "+" if change >= 0 else ""
    formatted_price = f"{price:,.2f}" if isinstance(price, (int, float)) else price
    formatted_change = f"{change:,.2f}" if isinstance(change, (int, float)) else change
    formatted_change_pct = f"{change_pct:.2f}" if isinstance(change_pct, (int, float)) else change_pct

    return f"""
    <div class="price-card">
        <h2 class="price-value" style="color: {color};">{formatted_price} USDT</h2>
        <p class="price-change" style="color: {color};">
            {change_symbol}{formatted_change} USDT ({change_symbol}{formatted_change_pct}%)
        </p>
    </div>
    """

# 메트릭 컨테이너 직접 렌더링 함수 - 고가/저가 렌더링용 
def render_high_low_metrics(high_value, low_value, high_color, low_color):
    # 값 포맷팅
    high_formatted = f"{high_value:,.2f} USDT" if isinstance(high_value, (int, float)) else high_value
    low_formatted = f"{low_value:,.2f} USDT" if isinstance(low_value, (int, float)) else low_value
    
    # 전체 HTML을 한 번에 생성하여 렌더링
    metrics_html = f"""
    <div class="metric-container">
        <div class="metric-box">
            <span class="metric-label">고가</span>
            <span class="metric-value" style="color: {high_color};">{high_formatted}</span>
        </div>
        <div class="metric-box">
            <span class="metric-label">저가</span>
            <span class="metric-value" style="color: {low_color};">{low_formatted}</span>
        </div>
    </div>
    """
    return st.markdown(metrics_html, unsafe_allow_html=True)

# 컨트롤 레이블 추가 함수
def labeled_control(label_text, control_function, key, **kwargs):
    # label_visibility="collapsed"를 사용하므로, st.markdown으로 레이블을 별도 추가
    st.markdown(f'<label class="control-label" for="{key}">{label_text}</label>', unsafe_allow_html=True)
    return control_function(key=key, label=label_text, label_visibility="collapsed", **kwargs)

# 멌더링 헤더 (페이지 최상단에 고정되도록 수정)
render_header()

# 암호화폐 목록
BINANCE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT",
    "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT", "LTC/USDT",
    "BNB/USDT", "AVAX/USDT", "SHIB/USDT", "TRX/USDT", "UNI/USDT"
]

# 차트 간격 설정
INTERVAL_MAPPING = {
    "1분": "1m", "3분": "3m", "5분": "5m", "15분": "15m", "30분": "30m",
    "1시간": "1h", "4시간": "4h", "1일": "1d", "1주": "1w", "1월": "1M",
}

# 상단 컨트롤 영역
st.markdown('<div class="control-area">', unsafe_allow_html=True)
# 각 컨트롤에 고유한 key를 할당해야 합니다.
# st.columns의 비율을 조정하여 컨트롤들이 한 줄에 잘 보이도록 합니다.
col1, col2, col3, col4, col5_check, col5_slider = st.columns([2.5, 1.2, 2.5, 1.5, 0.5, 1.5])


with col1:
    symbol = labeled_control("암호화폐", st.selectbox, key="symbol_select",
                          options=BINANCE_SYMBOLS,
                          index=0)
    chart_symbol = symbol.replace("/", "")

with col2:
    intervals = list(INTERVAL_MAPPING.keys())
    interval = labeled_control("차트 간격", st.selectbox, key="interval_select",
                            options=intervals,
                            index=intervals.index("5분"))
with col3:
    show_ma = labeled_control("이동평균선", st.multiselect, key="ma_multiselect",
                           options=["MA5", "MA10", "MA20", "MA60", "MA120"],
                           default=["MA5", "MA20", "MA60"])
with col4:
    date_ranges = ["전체", "최근 30일", "최근 7일", "최근 24시간", "최근 4시간"]
    date_range = labeled_control("조회기간", st.selectbox, key="daterange_select",
                              options=date_ranges,
                              index=0)
with col5_check:
    st.markdown('<label class="control-label" style="opacity:0;">자동</label>', unsafe_allow_html=True) # 높이 맞추기용
    auto_update = st.checkbox("자동", value=True, key="auto_update_checkbox", label_visibility="collapsed")

if auto_update:
    with col5_slider:
        update_seconds = labeled_control("업데이트 주기(초)", st.slider, key="update_slider",
                                        min_value=5,
                                        max_value=60,
                                        value=10)
else:
    with col5_slider: # 공간 유지
        st.empty()


st.markdown('</div>', unsafe_allow_html=True)


# 메인 컨텐츠 레이아웃 (차트 + 우측 정보 패널)
main_col1, main_col2 = st.columns([7, 3])


with main_col2:
    with st.expander("차트 설정", expanded=False): # CSS로 스타일링됨
        if 'show_bollinger' not in st.session_state:
            st.session_state.show_bollinger = False
        
        # CSS로 스타일링된 체크박스를 사용하기 위해 label_visibility 처리
        st.markdown('<label class="control-label" for="bollinger_checkbox_exp">볼린저 밴드</label>', unsafe_allow_html=True)
        show_bollinger_new = st.checkbox("표시", value=st.session_state.show_bollinger, key="bollinger_checkbox_exp", label_visibility="collapsed")
        if show_bollinger_new != st.session_state.show_bollinger:
            st.session_state.show_bollinger = show_bollinger_new
            st.rerun()
        
        # CSS로 스타일링된 라디오 버튼
        st.markdown('<label class="control-label" for="chart_style_radio_exp">차트 스타일</label>', unsafe_allow_html=True)
        chart_style = st.radio("", options=["캔들스틱", "라인", "막대"], index=0, horizontal=True, key="chart_style_radio_exp", label_visibility="collapsed")
    
    manual_update_btn_placeholder = st.empty() # 버튼 플레이스홀더


# 세션 상태 초기화
if 'price_data' not in st.session_state:
    st.session_state.price_data = None
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'update_counter' not in st.session_state:
    st.session_state.update_counter = 0


# 데이터 가져오기 함수 - 바이낸스
def get_binance_data(api_symbol, time_interval, limit=200): # limit 증가
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            # 'options': {'defaultType': 'future'} # 선물 데이터 필요시 주석 해제. 현물은 보통 이게 필요 없음.
        })
            
        ohlcv = exchange.fetch_ohlcv(
            symbol=api_symbol,
            timeframe=time_interval,
            limit=limit
        )
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        df['ma120'] = df['close'].rolling(window=120).mean()
        
        return df
    except Exception as e:
        # 에러를 main_col1에 표시하려면 이 함수가 main_col1 컨텍스트 내에서 호출되어야 함
        # 지금은 전역 함수이므로 st.error를 사용
        st.error(f"바이낸스 데이터 가져오기 오류: {e}")
        return create_sample_data(api_symbol, time_interval, limit)

# 샘플 데이터 생성 함수 (API 접근 실패 시 사용)
def create_sample_data(api_symbol, time_interval, limit=200):
    st.warning(f"바이낸스 API 접근에 실패하여 {api_symbol}에 대한 샘플 데이터를 표시합니다.")
    
    end_time = pd.Timestamp.now(tz='UTC') # 시간대 명시
    
    time_delta_map = {
        "1m": pd.Timedelta(minutes=1), "3m": pd.Timedelta(minutes=3), "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(weeks=1),
        "1M": pd.Timedelta(days=30) # 근사치
    }
    delta = time_delta_map.get(time_interval, pd.Timedelta(minutes=5))
        
    timestamps = [end_time - i * delta for i in range(limit)]
    timestamps.reverse()
    
    base_price = 30000 if 'BTC' in api_symbol else (2000 if 'ETH' in api_symbol else 100)
    np.random.seed(int(datetime.now().timestamp())) # 시드 변경으로 매번 다른 샘플
    
    opens = base_price + np.random.normal(0, base_price * 0.02, size=limit)
    closes = opens + np.random.normal(0, base_price * 0.03, size=limit)
    
    data = {
        'timestamp': timestamps,
        'open': opens,
        'high': np.maximum(opens, closes) + np.abs(np.random.normal(0, base_price * 0.01, size=limit)),
        'low': np.minimum(opens, closes) - np.abs(np.random.normal(0, base_price * 0.01, size=limit)),
        'close': closes,
        'volume': np.random.uniform(10, 100, size=limit) * (base_price / 1000)
    }
    df = pd.DataFrame(data)
    
    # low가 0보다 작아지는 것 방지
    df['low'] = df['low'].clip(lower=0.01)
    df['close'] = df['close'].clip(lower=0.01)
    df['open'] = df['open'].clip(lower=0.01)
    df['high'] = df['high'].clip(lower=0.01)


    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['ma120'] = df['close'].rolling(window=120).mean()
    
    return df

# Plotly를 사용한 차트 그리기 함수
def plot_binance_chart(df_orig, current_chart_style):
    if df_orig is None or df_orig.empty:
        st.warning("차트를 표시할 데이터가 없습니다.")
        return None

    df = df_orig.copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25]
    )

    if current_chart_style == "캔들스틱":
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                increasing_line_color=DESIGN_COLORS["up"], decreasing_line_color=DESIGN_COLORS["down"],
                increasing_fillcolor=DESIGN_COLORS["up"], decreasing_fillcolor=DESIGN_COLORS["down"],
                name='캔들'
            ), row=1, col=1
        )
    elif current_chart_style == "라인":
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='종가', line=dict(color=DESIGN_COLORS["toss_blue"], width=2)),
            row=1, col=1
        )
    elif current_chart_style == "막대":
        bar_colors = [DESIGN_COLORS["up"] if c >= o else DESIGN_COLORS["down"] for o, c in zip(df['open'], df['close'])]
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['close'], name='종가 (막대)', marker_color=bar_colors),
            row=1, col=1
        )

    ma_colors = {"MA5": DESIGN_COLORS["ma5"], "MA10": DESIGN_COLORS["ma10"], "MA20": DESIGN_COLORS["ma20"], "MA60": DESIGN_COLORS["ma60"], "MA120": DESIGN_COLORS["ma120"]}
    for ma in show_ma:
        ma_col = ma.lower()
        if ma_col in df.columns and not df[ma_col].isnull().all():
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[ma_col], line=dict(color=ma_colors[ma], width=1.5), name=ma),
                row=1, col=1
            )

    volume_colors = [DESIGN_COLORS["volume_up"] if i > 0 and df['close'].iloc[i] > df['close'].iloc[i-1] else DESIGN_COLORS["volume_down"] for i in range(len(df))]
    if not volume_colors and len(df) > 0 : volume_colors.append(DESIGN_COLORS["volume_up"])

    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'], marker_color=volume_colors, marker_line_width=0, name='거래량'),
        row=2, col=1
    )

    if 'show_bollinger' in st.session_state and st.session_state.show_bollinger:
        df['bollinger_mid'] = df['close'].rolling(window=20).mean()
        df['bollinger_std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_mid'] + 2 * df['bollinger_std']
        df['bollinger_lower'] = df['bollinger_mid'] - 2 * df['bollinger_std']

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bollinger_upper'], line=dict(color='rgba(250, 128, 114, 0.4)', width=1, dash='dot'), name='BB상단'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bollinger_lower'], line=dict(color='rgba(135, 206, 250, 0.4)', width=1, dash='dot'), name='BB하단', fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)'), row=1, col=1)

    fig.update_layout(
        title=None, xaxis_title=None, yaxis_title=None,
        font=dict(color=DESIGN_COLORS["text"], family="Pretendard, sans-serif"),
        plot_bgcolor=DESIGN_COLORS["background"], paper_bgcolor=DESIGN_COLORS["background"],
        xaxis_rangeslider_visible=False, height=550,
        margin=dict(l=10, r=40, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        modebar=dict( # Corrected modebar configuration
            orientation='h',
            bgcolor='rgba(255, 255, 255, 0.1)', # Slightly more visible background for modebar
            color='#999999',
            activecolor=DESIGN_COLORS["toss_blue"]
            # Removed x, y, xanchor, yanchor as they are invalid here
        ),
    )

    fig.update_xaxes(gridcolor=DESIGN_COLORS["grid"], zeroline=False, showgrid=True, tickfont=dict(size=10), row=1, col=1)
    fig.update_yaxes(gridcolor=DESIGN_COLORS["grid"], zeroline=False, showgrid=True, tickfont=dict(size=10), tickformat=',.0f', row=1, col=1, side='right', title_text="가격(USDT)", title_font_size=10, title_standoff=10)

    fig.update_xaxes(gridcolor=DESIGN_COLORS["grid"], zeroline=False, showgrid=True, tickfont=dict(size=10), row=2, col=1, title_text="시간", title_font_size=10, title_standoff=5)
    fig.update_yaxes(gridcolor=DESIGN_COLORS["grid"], zeroline=False, showgrid=True, tickfont=dict(size=10), row=2, col=1, side='right', title_text="거래량", title_font_size=10, title_standoff=10)

    hover_label_common = dict(bgcolor="white", font_size=12, font_family="Pretendard, sans-serif", bordercolor=DESIGN_COLORS["border"])

    if current_chart_style == "캔들스틱":
        fig.update_traces(
            selector={'type': 'candlestick'},
            hoverlabel=hover_label_common,
            hoverinfo="x+y+text",
            text=[f"시가: {o:,.2f}<br>고가: {h:,.2f}<br>저가: {l:,.2f}<br>종가: {c:,.2f}"
                for o, h, l, c in zip(df['open'], df['high'], df['low'], df['close'])]
        )
    elif current_chart_style == "라인":
         fig.update_traces(
            selector={'type': 'scatter', 'mode':'lines'},
            hoverlabel=hover_label_common,
            hovertemplate='<b>시간</b>: %{x}<br><b>가격</b>: %{y:,.2f} USDT<extra></extra>'
        )
    elif current_chart_style == "막대":
        fig.update_traces(
            selector={'type': 'bar'},
            notname='거래량',
            hoverlabel=hover_label_common,
            hovertemplate='<b>시간</b>: %{x}<br><b>가격</b>: %{y:,.2f} USDT<extra></extra>'
        )

    fig.update_traces(
        selector={'name': '거래량'},
        hoverlabel=hover_label_common,
        hovertemplate='<b>시간</b>: %{x}<br><b>거래량</b>: %{y:,.2f}<extra></extra>'
    )
    for ma_name in show_ma:
        fig.update_traces(
            selector={'name': ma_name},
            hoverlabel=hover_label_common,
            hovertemplate=f'<b>시간</b>: %{{x}}<br><b>{ma_name}</b>: %{{y:,.2f}} USDT<extra></extra>'
        )
    return fig

# 데이터 업데이트 함수
def update_data():
    df = get_binance_data(chart_symbol, INTERVAL_MAPPING[interval]) # chart_symbol 사용
    
    if df is not None and not df.empty:
        # 날짜 필터링
        if date_range != "전체":
            now = pd.Timestamp.now(tz='UTC') # 시간대 인식
            if date_range == "최근 30일": start_date = now - pd.Timedelta(days=30)
            elif date_range == "최근 7일": start_date = now - pd.Timedelta(days=7)
            elif date_range == "최근 24시간": start_date = now - pd.Timedelta(hours=24)
            elif date_range == "최근 4시간": start_date = now - pd.Timedelta(hours=4)
            
            # df['timestamp']도 시간대 인식이 되도록 확인 또는 변환
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            
            df = df[df['timestamp'] >= start_date].copy() # 필터 후 복사본 사용
    
    st.session_state.chart_data = df
    
    if df is not None and not df.empty:
        last_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
        price_change = last_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # 24시간 고가/저가 계산 (데이터가 충분한 경우 최근 24시간, 아니면 전체 기간)
        # 간격(interval)에 따라 하루치 데이터 개수 계산
        interval_str = INTERVAL_MAPPING[interval]
        if 'm' in interval_str: periods_in_day = 24 * 60 / int(interval_str.replace('m',''))
        elif 'h' in interval_str: periods_in_day = 24 / int(interval_str.replace('h',''))
        elif 'd' in interval_str: periods_in_day = 1
        else: periods_in_day = 1 # 주, 월 단위는 근사치로 1일치만 봄 (또는 더 많은 기간)
        
        periods_in_day = int(periods_in_day) if periods_in_day > 0 else 1

        day_high_df = df['high'].tail(periods_in_day)
        day_low_df = df['low'].tail(periods_in_day)

        price_data = {
            'last_price': last_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'day_high': day_high_df.max() if not day_high_df.empty else 'N/A',
            'day_low': day_low_df.min() if not day_low_df.empty else 'N/A',
            'recent_data': df.tail(5).copy(),
            'color': DESIGN_COLORS["up"] if price_change >= 0 else DESIGN_COLORS["down"]
        }
        st.session_state.price_data = price_data
    else:
        st.session_state.price_data = None # 데이터 없으면 None으로 설정

    st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.update_counter += 1
    return df

# 이동평균선 값 배지 스타일
def ma_badge(value, ma_type):
    ma_colors_map = {"MA5": DESIGN_COLORS["ma5"], "MA10": DESIGN_COLORS["ma10"], "MA20": DESIGN_COLORS["ma20"], "MA60": DESIGN_COLORS["ma60"], "MA120": DESIGN_COLORS["ma120"]}
    color = ma_colors_map.get(ma_type, DESIGN_COLORS["toss_blue"])
    return f'<span class="badge" style="color: {color}; border-color: {color}33; background-color: {color}1A;">{ma_type}: {value:,.2f}</span>' # 소수점 두자리

# 배지 렌더링 함수
def render_badges(badges_html_list): # HTML 문자열 리스트를 받음
    joined_html = "".join(badges_html_list)
    st.markdown(f'<div style="margin: 10px 0; line-height: 2.5;">{joined_html}</div>', unsafe_allow_html=True)

# 차트 및 데이터 표시 함수
def display_chart_and_data():
    with main_col1:
        st.markdown(f'<h2 style="font-size: 20px; color: #000; margin: 0 0 10px 0;">{symbol} - {interval} 차트</h2>', unsafe_allow_html=True)
        render_tabs("시세")
        
        chart_placeholder = st.empty() # 차트 플레이스홀더

        if st.session_state.chart_data is not None and not st.session_state.chart_data.empty:
            fig = plot_binance_chart(st.session_state.chart_data, chart_style) # chart_style 전달
            if fig:
                chart_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
                
                df = st.session_state.chart_data
                last_row = df.iloc[-1]
                ma_badges_html_list = []
                for ma in show_ma:
                    ma_col = ma.lower()
                    if ma_col in last_row and pd.notna(last_row[ma_col]):
                        ma_badges_html_list.append(ma_badge(last_row[ma_col], ma))
                if ma_badges_html_list:
                    render_badges(ma_badges_html_list)
            else:
                chart_placeholder.warning("차트를 생성할 수 없습니다.")
        else:
            chart_placeholder.warning(f"{symbol}에 대한 차트 데이터를 불러올 수 없습니다.")

        if st.session_state.price_data and not st.session_state.price_data['recent_data'].empty:
            st.markdown("<h3 style='font-size: 16px; margin-top: 24px; margin-bottom: 10px; color: #000;'>최근 거래 데이터</h3>", unsafe_allow_html=True)
            recent_df = st.session_state.price_data['recent_data']
            recent_df['time'] = recent_df['timestamp'].dt.strftime('%m-%d %H:%M') # 날짜 형식 변경
            display_df = recent_df[['time', 'open', 'high', 'low', 'close', 'volume']].rename(
                columns={'time': '시간', 'open': '시가', 'high': '고가', 'low': '저가', 'close': '종가', 'volume': '거래량'}
            )
            display_df = display_df.sort_values('시간', ascending=False).reset_index(drop=True)
            
            # 스타일 적용을 위해 Styler 객체 사용
            def format_df(df_to_style):
                return df_to_style.style.format({
                    '시가': '{:,.2f}', '고가': '{:,.2f}', '저가': '{:,.2f}',
                    '종가': '{:,.2f}', '거래량': '{:,.2f}' # 거래량 소수점 조정
                }).set_table_attributes('class="dataframe"') # CSS 적용 위한 클래스
            
            st.dataframe(format_df(display_df), height=210, use_container_width=True) # 높이 조정
    
    with main_col2:
        # 수동 업데이트 버튼 (여기서 정의)
        if manual_update_btn_placeholder.button('차트 수동 업데이트', key='manual_update_button_main', use_container_width=True):
            with st.spinner("데이터 업데이트 중..."):
                update_data()
            st.rerun() # 수동 업데이트 후 즉시 반영

        if st.session_state.price_data:
            data = st.session_state.price_data
            st.markdown("<h3 style='font-size: 18px; margin-top:0; margin-bottom:10px; color: #000;'>현재 시세</h3>", unsafe_allow_html=True)
            st.markdown(price_card_html(
                data['last_price'], data['price_change'], data['price_change_pct'], data['color']
            ), unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size: 16px; margin-top: 24px; margin-bottom:10px; color: #000;'>24시간 변동</h3>", unsafe_allow_html=True)
            
            # 메트릭 박스를 직접 렌더링하는 새 함수 사용
            render_high_low_metrics(
                data['day_high'], 
                data['day_low'],
                DESIGN_COLORS["up"],
                DESIGN_COLORS["down"]
            )
            
            if st.session_state.last_update_time:
                st.markdown(
                    f"<div class='time-label'>마지막 업데이트: {st.session_state.last_update_time} (카운트: {st.session_state.update_counter})</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("시세 정보를 불러오는 중이거나 데이터를 가져올 수 없습니다.")


# 초기 데이터 로드 또는 수동 업데이트 버튼이 display_chart_and_data 외부에 있어야 함
# manual_update 변수는 이제 display_chart_and_data 내부 버튼으로 대체됨
if st.session_state.chart_data is None: # 최초 실행 시 데이터 로드
    with st.spinner("초기 데이터 로드 중..."):
        update_data()

display_chart_and_data() # 항상 호출하여 UI를 그림


# 자동 업데이트 처리
if auto_update:
    # 자동 업데이트 알림은 한번만 표시하거나, 특정 위치에 고정
    # st.sidebar.info(f"자동 업데이트: {update_seconds}초마다") # 사이드바로 옮기거나
    # 또는 메인 영역 하단에 표시
    # 이 부분은 매번 rerun 시 마다 그려지므로, 위치를 잘 선정해야 함.
    # 여기서는 메인 컨텐츠 아래에 유지.

    if 'last_auto_update' not in st.session_state:
        st.session_state.last_auto_update = datetime.now()
    
    current_time = datetime.now()
    # update_seconds가 정의되지 않은 경우(auto_update가 False였다가 True로 바뀐 직후) 대비
    current_update_seconds = update_seconds if 'update_seconds' in locals() else (st.session_state.get('update_seconds_val', 10))
    if 'update_seconds' in locals() : st.session_state.update_seconds_val = update_seconds


    time_diff = (current_time - st.session_state.last_auto_update).total_seconds()
    
    if time_diff >= current_update_seconds:
        st.session_state.last_auto_update = current_time
        with st.spinner(f"{current_update_seconds}초 자동 업데이트 중..."):
            update_data()
        st.rerun()
    
    # 자동 업데이트 상태를 표시할 placeholder (루프 바깥에 두어 깜빡임 최소화)
    # 이 부분은 매번 rerun 시 표시되므로, 필요한 경우에만 표시하도록 조건 추가 가능
    st.markdown(
        f"""
        <div style='position: fixed; bottom: 10px; right: 10px; padding: 8px 12px; 
                    border-radius: 8px; background-color: rgba(9, 54, 135, 0.9); 
                    color: white; font-size: 13px; z-index:1000;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);'>
        자동 업데이트 활성 ({current_update_seconds}초). 다음 업데이트까지: {max(0, int(current_update_seconds - time_diff))}초
        </div>
        """, unsafe_allow_html=True
    )