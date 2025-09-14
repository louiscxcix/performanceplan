import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import re
import random
from calendar import monthrange
from PIL import Image

# --- 1. 앱 기본 설정 및 페이지 구성 ---
try:
    # 사용자 지정 아이콘을 로드합니다.
    # 중요: 'icon.png' 파일이 이 스크립트와 동일한 폴더에 있어야 합니다.
    icon = Image.open("icon.png")
except FileNotFoundError:
    # 파일을 찾지 못하면 기본 이모지를 사용합니다.
    icon = "🤖"

st.set_page_config(
    page_title="Peak Performance Planner (AI)",
    page_icon=icon,
    layout="wide"
)

# --- NEW UI STYLES ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica:wght@400;700&display=swap');

    /* General body styling */
    .stApp {
        background: #F1F2F5;
        font-family: 'Helvetica', sans-serif;
    }

    /* Responsive main container with improved padding */
    .block-container {
        max-width: 550px;
        margin: 0 auto; /* Ensure content is always centered */
        padding: 2rem 1.5rem 5rem 1.5rem !important; /* Increased side padding for better spacing */
    }

    /* Form and Input styling */
    .stForm {
        border: none;
        padding: 0;
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }

    /* Apply styles to all input widgets including date picker */
    .stTextInput > div, .stTextArea > div, .stDateInput > div {
        background-color: rgba(13, 125, 163, 0.04);
        border: 1px solid rgba(13, 125, 163, 0.04);
        border-radius: 12px;
        overflow: hidden; 
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stDateInput > div > div > input {
        background-color: transparent !important;
        border: none !important;
        color: #0D1628;
        line-height: 1.5;
    }
    
    /* Focus effect */
    .stTextInput > div:focus-within,
    .stTextArea > div:focus-within,
    .stDateInput > div:focus-within {
        border: 1px solid #2BA7D1;
        box-shadow: none;
    }

    /* Label styling - FIXED SPACING */
    .stTextInput > label,
    .stTextArea > label,
    .stDateInput > label {
        color: #86929A !important;
        font-size: 12px !important;
        font-family: 'Helvetica', sans-serif;
        padding: 10px 12px 2px 12px !important; /* Added bottom padding for spacing */
    }
    
    .stTextArea > div > textarea {
        min-height: 120px;
    }
    .stTextArea small {
        color: #86929A;
        font-size: 10px;
        font-family: 'Helvetica', sans-serif;
        font-weight: 400;
        padding-left: 5px;
        padding-top: 12px;
    }

    /* Submit Button Styling */
    .stButton > button {
        width: 100%;
        padding: 14px 36px;
        background: linear-gradient(135deg, rgba(98, 120, 246, 0.20) 0%, rgba(29, 48, 78, 0) 100%), #2BA7D1;
        box-shadow: 0px 5px 10px rgba(26, 26, 26, 0.10);
        border-radius: 12px;
        color: white;
        font-size: 14px;
        font-family: 'Helvetica', sans-serif;
        font-weight: 400;
        border: none;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(98, 120, 246, 0.20) 0%, rgba(29, 48, 78, 0) 100%), #2490b4;
        color: white;
        border: none;
    }
    
    /* Hide the default Streamlit header/footer */
    header, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Gemini API 키 설정 (Streamlit Secrets 활용) ---
GEMINI_API_KEY = None 
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    # This will be handled gracefully when the form is submitted
    pass

# --- 3. Gemini 분석 함수 (7단계 강도 시스템 적용) ---
def analyze_training_request_with_gemini(user_text, goal):
    """
    Gemini API를 사용하여 사용자의 텍스트를 분석하고,
    훈련 목록을 7단계 강도 레벨과 함께 JSON으로 반환
    """
    if not GEMINI_API_KEY:
        st.error("API 키가 설정되지 않았습니다. Streamlit Cloud의 'Settings > Secrets'에서 API 키를 설정해주세요.")
        return None
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    당신은 엘리트 선수들을 코칭하는 세계적인 스포츠 과학 전문가입니다. 사용자가 입력한 목표와 훈련 설명을 분석하여, 최적의 성과를 위한 종합 훈련 프로그램을 구성해주세요.

    **분석 및 구성 가이드라인:**
    1.  **사용자 요청 분석:** 사용자가 명시적으로 요청한 훈련 활동들을 모두 추출합니다.
    2.  **전문가적 판단으로 훈련 추가:** 사용자의 목표('{goal}')와 종목 특성을 고려할 때, 필수적인 보조 훈련들을 **반드시 추가**해주세요. (예: 마라톤 준비 시 '코어 운동', '스트레칭' 추가)
    3.  **7단계 강도 분류:** 모든 훈련 활동을 아래의 1부터 7까지의 강도 레벨 중 하나로 정확히 분류합니다.
        - **Level 1 (완전 휴식):** 수면, 명상 등 완전한 휴식.
        - **Level 2 (가벼운 회복):** 가벼운 산책, 회복 스트레칭.
        - **Level 3 (기술 훈련):** 심박수 부담이 적은 기술 연습, 폼 롤링.
        - **Level 4 (지구력 훈련):** 편안하게 대화 가능한 수준의 유산소 운동, 장거리 달리기.
        - **Level 5 (템포 훈련):** 약간 숨이 차는 강도의 지속적인 훈련, 역치 훈련.
        - **Level 6 (고강도 인터벌):** 최대 심박수에 근접하는 인터벌, 고중량 근력 운동.
        - **Level 7 (최대 강도):** 시합 또는 개인 최고 기록(PR)에 도전하는 수준의 최대 노력.
    4.  **JSON 형식으로 최종 출력:** 결과를 반드시 아래의 JSON 형식에 맞춰 다른 설명 없이 JSON 코드만 반환해주세요.

    **사용자 정보:**
    - **목표:** {goal}
    - **훈련 설명:** {user_text}

    **출력 JSON 형식:**
    {{
      "trainings": [
        {{"name": "훈련명1", "intensity_level": 레벨(숫자)}},
        {{"name": "훈련명2", "intensity_level": 레벨(숫자)}}
      ]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_text = re.sub(r'```json\n|```', '', response.text).strip()
        parsed_json = json.loads(cleaned_text)
        return parsed_json.get("trainings", [])
    except Exception as e:
        st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
        return None

# --- 4. 훈련 계획 생성 로직 (7단계 강도 시스템 적용) ---

def get_trainings_by_level(training_list):
    """훈련 목록을 1-7 레벨별로 분류하는 함수"""
    trainings = {level: [] for level in range(1, 8)}
    for t in training_list:
        level = t.get('intensity_level')
        if level in trainings:
            trainings[level].append(t['name'])
    
    level_defaults = {1: "완전 휴식", 2: "가벼운 회복", 3: "기술 훈련", 4: "지구력 훈련", 5: "템포 훈련", 6: "고강도 인터벌", 7: "최대 강도"}
    for level, default_name in level_defaults.items():
        if not trainings[level]:
            trainings[level] = [default_name]
    return trainings

def get_detailed_guide(workout_name):
    """훈련 종류에 따라 상세하고 다양한 가이드를 반환"""
    guide_book = {
        "인터벌": ["심박수가 최대치에 가깝게 유지되도록 집중하세요.", "휴식 시간을 정확히 지켜 효과를 극대화하세요.", "마지막 세트까지 자세가 무너지지 않도록 주의하세요."],
        "지속주": ["일정한 페이스를 유지하는 것이 핵심입니다.", "호흡이 너무 가빠지지 않는 선에서 속도를 조절하세요.", "마치 시합의 일부를 미리 달려보는 것처럼 집중해보세요."],
        "근력 운동": ["정확한 자세가 부상 방지와 효과의 핵심입니다.", "목표 부위의 근육 자극을 느끼며 천천히 수행하세요.", "세트 사이 휴식은 1~2분 이내로 조절하세요."],
        "회복 조깅": ["옆 사람과 편안히 대화할 수 있을 정도의 속도를 유지하세요.", "몸의 소리에 귀 기울이며 굳은 근육을 풀어주는 느낌으로 달리세요.", "시간이나 거리에 얽매이지 말고 편안하게 수행하세요."],
        "휴식": ["충분한 수면(7-8시간)은 최고의 회복입니다.", "가벼운 산책이나 스트레칭으로 혈액순환을 도우세요.", "훈련에 대한 생각은 잠시 잊고 편안한 마음을 가지세요."],
        "스트레칭": ["근육의 이완을 느끼며 15초 이상 유지하세요.", "호흡을 멈추지 말고, 길게 내쉬면서 스트레칭하세요.", "훈련 전에는 동적, 훈련 후에는 정적 스트레칭이 효과적입니다."],
        "코어": ["배에 힘을 주고 허리가 구부러지지 않도록 유지하세요.", "동작은 천천히, 자극에 집중하며 수행하세요.", "강력한 코어는 모든 움직임의 시작입니다."]
    }
    for key, guides in guide_book.items():
        if key in workout_name:
            return random.choice(guides)
    return "자신의 몸 상태에 맞춰 무리하지 마세요."

def generate_dynamic_plan(total_days, date_range, trainings):
    fitness = 50.0
    fatigue = 50.0
    
    level_load_map = {
        1: {'ts': 0, 'af': 0}, 2: {'ts': 5, 'af': 0.5}, 3: {'ts': 10, 'af': 0.7},
        4: {'ts': 18, 'af': 1.0}, 5: {'ts': 25, 'af': 1.2}, 6: {'ts': 35, 'af': 1.5},
        7: {'ts': 45, 'af': 1.8}
    }
    
    fatigue_decay = 0.4
    fitness_decay = 0.98

    plan = []
    consecutive_training_days = 0

    for i, day in enumerate(date_range):
        progress = i / total_days
        remaining_days = total_days - i

        workout_level = 1
        if remaining_days <= 10:
            phase = "테이퍼링"
            if remaining_days == 1: workout_level = 1
            elif remaining_days in [2, 4]: workout_level = 2
            elif remaining_days == 3: workout_level = 3
            elif remaining_days == 5: workout_level = 6
            else: workout_level = random.choice([2, 3])
            consecutive_training_days = 0
        else:
            phase = "준비기" if progress < 0.6 else "시합기"
            if consecutive_training_days < random.choice([2, 3]):
                consecutive_training_days += 1
                workout_level = random.choice([4, 4, 5, 3]) if phase == "준비기" else random.choice([6, 5, 4])
            else:
                workout_level = random.choice([2, 2, 3])
                consecutive_training_days = 0

        fitness *= fitness_decay
        fatigue *= fatigue_decay

        load = level_load_map[workout_level]
        training_stress = load['ts']
        adaptation_factor = load['af']
        
        if phase == "테이퍼링" and workout_level > 2:
            training_stress *= 0.6

        fatigue += training_stress
        fitness += training_stress * adaptation_factor * 0.1
        performance = fitness - fatigue
        
        workout_name = random.choice(trainings[workout_level])
        plan.append({
            "날짜": day.strftime("%Y-%m-%d"), "요일": day.strftime("%a"), "단계": phase,
            "훈련 내용": workout_name, "훈련 강도 레벨": workout_level,
            "예상 퍼포먼스": round(performance, 1), "상세 가이드": get_detailed_guide(workout_name)
        })
    return pd.DataFrame(plan)

# --- 5. 시각화 함수 (스크롤 기능 추가) ---

def create_performance_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['날짜'], y=df['예상 퍼포먼스'], name='예상 퍼포먼스',
        line=dict(color='#2BA7D1', width=3), fill='tozeroy',
        fillcolor='rgba(43, 167, 209, 0.1)', mode='lines',
        hovertemplate='<span style="font-size:12px;">%{x|%m월 %d일}</span><br><span style="color:#2BA7D1; font-size:14px;">■</span><span style="font-size:14px;"> <b>%{y}</b></span><extra></extra>'
    ))
    fig.update_layout(
        height=350, # 그래프 높이 조절
        title=None, xaxis_title=None, yaxis_title="레벨", plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family="Helvetica, sans-serif", size=12, color="#86929A"),
        showlegend=False, margin=dict(l=40, r=20, t=5, b=20),
        xaxis=dict(showgrid=False, showline=True, linecolor='#E8E8E8', tickformat='%m/%d',
                   rangeslider_visible=False), # 레인지 슬라이더 비활성화
        yaxis=dict(showgrid=True, gridcolor='#E8E8E8'),
        hoverlabel=dict(bgcolor="#0D1628", font_size=14, font_color="white", bordercolor="rgba(0,0,0,0)", font_family="Helvetica, sans-serif"),
        hovermode='x unified'
    )
    # 초기 줌 레벨 설정 (전체 기간이 7일 이상일 경우)
    if len(df) > 7:
        fig.update_xaxes(range=[df['날짜'].iloc[0], df['날짜'].iloc[6]])
    return fig

def create_intensity_chart(df, level_map):
    df['강도 설명'] = df['훈련 강도 레벨'].map(level_map)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['날짜'], y=df['훈련 강도 레벨'], name='훈련 강도',
        marker=dict(color='#EE7D8D', cornerradius=16),
        customdata=df['강도 설명'],
        hovertemplate='<span style="font-size:12px;">%{x|%m월 %d일}</span><br><span style="color:#EE7D8D; font-size:14px;">■</span><span style="font-size:14px;"> <b>%{customdata} (Lvl:%{y})</b></span><extra></extra>'
    ))
    fig.update_layout(
        height=350, # 그래프 높이 조절
        title=None, xaxis_title=None, yaxis_title=None, plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family="Helvetica, sans-serif", size=11, color="#86929A"),
        showlegend=False, margin=dict(l=25, r=20, t=5, b=20),
        xaxis=dict(showgrid=False, showline=True, linecolor='#E8E8E8', tickformat='%m/%d', tickfont=dict(size=11),
                   rangeslider_visible=False), # 레인지 슬라이더 비활성화
        yaxis=dict(showgrid=False, showticklabels=True, tickmode='array', tickvals=list(range(0, 8)), ticktext=[str(i) for i in range(0, 8)],
                   range=[0, 7.5], zeroline=False, tickfont=dict(size=9)),
        hoverlabel=dict(bgcolor="#0D1628", font_size=12, font_color="white", bordercolor="rgba(0,0,0,0)", font_family="Helvetica, sans-serif"),
        hovermode='x unified', bargap=0.4 # Adjust bargap to control spacing
    )
    # 초기 줌 레벨 설정 (전체 기간이 7일 이상일 경우)
    if len(df) > 7:
        fig.update_xaxes(range=[df['날짜'].iloc[0], df['날짜'].iloc[6]])
    return fig

def get_intuitive_df(df, level_map):
    df_display = df.copy()
    df_display["강도 수준"] = df_display["훈련 강도 레벨"].map(level_map)
    min_perf = df_display["예상 퍼포먼스"].min()
    max_perf = df_display["예상 퍼포먼스"].max()
    def map_performance(perf):
        normalized_perf = (perf - min_perf) / (max_perf - min_perf) * 100 if (max_perf - min_perf) > 0 else 50
        blocks = int(normalized_perf / 10)
        return "■" * blocks + "□" * (10 - blocks)
    df_display["퍼포먼스 레벨"] = df_display["예상 퍼포먼스"].apply(map_performance)
    return df_display[["날짜", "요일", "단계", "훈련 내용", "강도 수준", "퍼포먼스 레벨", "상세 가이드"]]

# --- 6. 메인 UI 구성 (디자인 레퍼런스 적용) ---
st.markdown("""
<div style="align-self: stretch; flex-direction: column; justify-content: flex-start; align-items: flex-start; gap: 12px; display: flex; margin-bottom: 40px;">
  <div style="padding: 8px; background: rgba(13, 125, 163, 0.04); border-radius: 48px; display: inline-flex; align-items: center; justify-content: center;">
      <div style="width: 52px; height: 52px; font-size: 40px; text-align: center; line-height: 52px;">🤖</div>
  </div>
  <div style="flex-direction: column; justify-content: flex-start; align-items: flex-start; gap: 8px; display: flex">
    <div style="color: #0D1628; font-size: 20px; font-family: Helvetica; font-weight: 700; line-height: 32px; word-wrap: break-word">AI 시합 계획 플래너</div>
    <div style="color: #86929A; font-size: 13px; font-family: Helvetica; font-weight: 400; line-height: 20px; word-wrap: break-word">당신의 훈련 목표를 자연어로 설명해주세요.<br/>AI가 주기화 이론에 맞춰 최적의 계획을 생성해 드립니다.</div>
  </div>
</div>
""", unsafe_allow_html=True)


with st.form("main_form"):
    with st.container():
        goal_name = st.text_input("훈련 목표 이름", placeholder="예: 2025 마라톤 대회 준비")
        
        col1, col2 = st.columns(2)
        with col1:
            start_day = st.date_input("시작일", date.today())
        with col2:
            d_day = st.date_input("종료일", date.today() + timedelta(days=90))

        user_description = st.text_area(
            "훈련 목표 계획을 설명해 주세요",
            placeholder="예: 마라톤 풀코스 준비를 위해 주 4회 훈련합니다. 인터벌, 지속주, 회복 조깅을 포함하고 싶습니다."
        )
    
    submitted = st.form_submit_button("다 음")

# --- 7. 계획 생성 및 상태 저장 로직 ---
if submitted:
    # Clear previous plan if it exists
    if 'plan_generated' in st.session_state:
        del st.session_state['plan_generated']

    if not user_description or user_description == "예: 마라톤 풀코스 준비를 위해 주 4회 훈련합니다. 인터벌, 지속주, 회복 조깅을 포함하고 싶습니다.":
        st.warning("훈련 계획 설명을 입력해주세요.")
    elif start_day >= d_day:
        st.error("오류: 훈련 시작일은 목표일보다 이전이어야 합니다.")
    elif not GEMINI_API_KEY:
        st.error("API 키가 설정되지 않았습니다. 이 앱을 배포하는 경우 Streamlit Cloud의 'Settings > Secrets'에 API 키를 추가해주세요.")
    else:
        with st.spinner('AI가 당신의 계획을 분석하고 최적의 스케줄을 생성 중입니다...'):
            training_list = analyze_training_request_with_gemini(user_description, goal_name)
            
            if training_list:
                st.success("✅ AI 분석 완료! 훈련 계획을 생성합니다.")
                
                # 생성된 계획을 세션 상태에 저장
                st.session_state.plan_generated = True
                st.session_state.goal_name = goal_name
                
                level_map = {
                    1: "Lvl 1: 완전 휴식 🟢", 2: "Lvl 2: 가벼운 회복 🔵", 3: "Lvl 3: 기술 훈련 🟡",
                    4: "Lvl 4: 지구력 훈련 🟠", 5: "Lvl 5: 템포 훈련 🔴", 6: "Lvl 6: 고강도 인터벌 🟣",
                    7: "Lvl 7: 최대 강도 🔥"
                }
                st.session_state.level_map = level_map

                total_days = (d_day - start_day).days + 1
                date_range = pd.to_datetime(pd.date_range(start=start_day, end=d_day))
                
                trainings = get_trainings_by_level(training_list)
                plan_df = generate_dynamic_plan(total_days, date_range, trainings)
                display_df = get_intuitive_df(plan_df, level_map)
                
                st.session_state.plan_df = plan_df
                st.session_state.display_df = display_df
            else:
                st.session_state.plan_generated = False

# --- 8. 결과 출력 (상태 확인) ---
if 'plan_generated' in st.session_state and st.session_state.plan_generated:
    # 세션 상태에서 데이터 로드
    goal_name = st.session_state.goal_name
    plan_df = st.session_state.plan_df
    display_df = st.session_state.display_df
    level_map = st.session_state.level_map

    st.markdown('<div id="capture-area" style="background-color: white; padding: 30px 20px 20px 20px; border-radius: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)
    st.header(f"🎯 '{goal_name}' 최종 훈련 계획")
    
    st.subheader("📊 주기화 그래프")
    st.markdown("""
    <style>
        div.stRadio > div { 
            display: flex; 
            flex-direction: row; 
            background-color: rgba(12, 124, 162, 0.04); 
            padding: 4px; 
            border-radius: 12px; 
            justify-content: center; 
            outline: 1px solid rgba(12, 124, 162, 0.04);
        }
        div.stRadio > div > label { 
            flex: 1; 
            text-align: center; 
            padding: 10px 4px; 
            border-radius: 8px; 
            margin: 0 !important; 
            -webkit-user-select: none; 
            -ms-user-select: none; 
            user-select: none; 
            transition: all 0.2s ease-in-out;
        }
        div.stRadio > div > label > div { 
            display: inline; 
            font-size: 12px;
            font-family: 'Helvetica', sans-serif;
            font-weight: 400;
        }
        div.stRadio input[type="radio"] { 
            display: none; 
        }
        div.stRadio div:has(input[type="radio"]:checked) > label { 
            background: white; 
            box-shadow: 0px 2px 2px rgba(0, 0, 0, 0.02); 
            color: #0D1628; 
            font-weight: 600; 
            border: 0.5px solid #F7F7F7;
        }
        div.stRadio div:has(input[type="radio"]:not(:checked)) > label { 
            background: transparent; 
            color: #86929A; 
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 라디오 버튼은 고유한 key를 가져야 합니다.
    chart_choice = st.radio("그래프 선택", options=['예상 퍼포먼스', '훈련 강도'], horizontal=True, label_visibility='collapsed', key='chart_selector')

    if chart_choice == '예상 퍼포먼스':
        st.plotly_chart(create_performance_chart(plan_df), use_container_width=True)
    else:
        st.plotly_chart(create_intensity_chart(plan_df, level_map), use_container_width=True)

    st.subheader("📅 상세 훈련 캘린더")
    st.dataframe(display_df, use_container_width=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        csv = display_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 CSV 파일로 다운로드", data=csv, file_name=f"{goal_name}_plan.csv", mime="text/csv", use_container_width=True)
    with col2:
        file_name_for_image = f"{goal_name.replace(' ', '_')}_plan.png"
        save_image_html = f"""
            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
            <script>
            function captureAndDownload() {{
                const el = document.getElementById("capture-area");
                const btn = document.getElementById("save-img-btn");
                btn.innerHTML = "저장 중..."; btn.disabled = true;
                setTimeout(() => {{
                    html2canvas(el, {{ scale: 2, backgroundColor: '#ffffff', useCORS: true }}).then(canvas => {{
                        const image = canvas.toDataURL("image.png");
                        const link = document.createElement("a");
                        link.href = image; link.download = "{file_name_for_image}";
                        document.body.appendChild(link); link.click(); document.body.removeChild(link);
                        btn.innerHTML = "📸 이미지로 저장"; btn.disabled = false;
                    }}).catch(err => {{
                        console.error("Image capture failed:", err);
                        btn.innerHTML = "오류 발생! 다시 시도하세요."; btn.disabled = false;
                    }});
                }}, 500);
            }}
            </script>
            <button id="save-img-btn" onclick="captureAndDownload()" style="width:100%; padding:12px; font-size:16px; font-weight:bold; color:white; background-color:#28a745; border:none; border-radius:5px; cursor:pointer;">📸 이미지로 저장</button>
        """
        components.html(save_image_html, height=50)

