import streamlit as st
import pandas as pd
from datetime import date, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import re
import random

# --- 1. 앱 기본 설정 및 페이지 구성 ---
st.set_page_config(
    page_title="Peak Performance Planner (AI)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Peak Performance Planner")
st.write("당신의 훈련 목표를 자연어로 설명해주세요. AI가 주기화 이론에 맞춰 최적의 계획을 생성해 드립니다.")

# --- 2. Gemini API 키 설정 (Streamlit Secrets 활용) ---
GEMINI_API_KEY = None # API 키 변수 초기화
try:
    # Streamlit Cloud에 배포된 경우 secrets.toml에서 키를 가져옴
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    st.sidebar.success("API 키가 성공적으로 연결되었습니다!", icon="✅")
except (KeyError, FileNotFoundError):
    # 로컬에서 실행하거나 secrets 설정이 안 된 경우
    st.sidebar.error("API 키를 찾을 수 없습니다.", icon="🚨")
    st.sidebar.info("이 앱을 배포하려면 Streamlit Cloud의 'Settings > Secrets'에 아래 내용을 추가해야 합니다.")
    st.sidebar.code("GEMINI_API_KEY = 'YOUR_GOOGLE_AI_API_KEY'")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Gemini")


# --- 3. Gemini 분석 함수 (수정됨) ---
def analyze_training_request_with_gemini(user_text, goal):
    """
    Gemini API를 사용하여 사용자의 텍스트를 분석하고,
    필요한 훈련을 추가하여 훈련 목록을 JSON으로 반환
    """
    if not GEMINI_API_KEY:
        st.error("API 키가 설정되지 않아 AI 분석을 수행할 수 없습니다.")
        return None
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    당신은 엘리트 선수들을 코칭하는 세계적인 스포츠 과학 전문가입니다. 사용자가 입력한 목표와 훈련 설명을 분석하여, 최적의 성과를 위한 종합 훈련 프로그램을 구성해주세요.

    **분석 및 구성 가이드라인:**
    1.  **사용자 요청 분석:** 사용자가 명시적으로 요청한 훈련 활동들을 모두 추출합니다.
    2.  **전문가적 판단으로 훈련 추가:** 사용자의 목표('{goal}')와 종목 특성을 고려할 때, 명시적으로 언급되지 않았지만 필수적인 보조 훈련들을 **반드시 추가**해주세요. (예: 마라톤 준비 시 '코어 근력 운동'이나 '유연성 스트레칭' 추가, 근력 운동 시 '유산소 운동' 추가 등)
    3.  **강도 분류:** 추출하고 추가한 모든 훈련 활동의 성격을 '고강도', '중강도', '저강도', '휴식' 중 하나로 정확히 분류합니다.
        - '고강도': 최대 심박수에 근접하는 활동, 인터벌, 고중량 웨이트, 전력 질주.
        - '중강도': 대화는 가능하지만 노래는 힘든 수준의 활동, 템포 런, 장거리 달리기.
        - '저강도': 심박수가 편안한 수준의 활동, 회복 조깅, 기술 훈련, 스트레칭.
        - '휴식': 완전 휴식, 명상, 가벼운 산책.
    4.  **JSON 형식으로 최종 출력:** 결과를 반드시 아래의 JSON 형식에 맞춰 다른 설명 없이 JSON 코드만 반환해주세요.

    **사용자 정보:**
    - **목표:** {goal}
    - **훈련 설명:** {user_text}

    **출력 JSON 형식:**
    {{
      "trainings": [
        {{"name": "훈련명1", "intensity": "강도"}},
        {{"name": "훈련명2", "intensity": "강도"}}
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

# --- 4. 과부하-초과회복 모델 기반 계획 생성 로직 (전면 수정) ---

def get_trainings_by_intensity(training_list):
    """Helper to categorize trainings."""
    trainings = {
        '고강도': [t['name'] for t in training_list if t['intensity'] == '고강도'],
        '중강도': [t['name'] for t in training_list if t['intensity'] == '중강도'],
        '저강도': [t['name'] for t in training_list if t['intensity'] == '저강도'],
        '휴식': [t['name'] for t in training_list if t['intensity'] == '휴식']
    }
    for key in ['고강도', '중강도', '저강도', '휴식']:
        if not trainings[key]:
            trainings[key] = [f'{key} 훈련']
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
    """
    Fitness-Fatigue 모델을 적용하여 우상향하는 퍼포먼스 곡선을 생성하고,
    시합일에 피킹하는 동적 계획 생성 함수
    """
    # 모델 파라미터
    fitness = 50.0  # 장기적인 체력 수준
    fatigue = 50.0  # 단기적인 피로 수준
    
    # 강도별 훈련 부하(Training Load) 정의
    # ts: Training Stress (피로도 기여), af: Adaptation Factor (체력 기여)
    load_map = {
        '고강도': {'ts': 30, 'af': 1.5},
        '중강도': {'ts': 18, 'af': 1.0},
        '저강도': {'ts': 5,  'af': 0.5},
        '휴식':   {'ts': 0,  'af': 0}
    }
    
    # 피로와 체력의 감소율 (매일 자연 감소)
    fatigue_decay = 0.4 # 피로는 빠르게 사라짐
    fitness_decay = 0.98 # 체력은 천천히 사라짐

    plan = []
    consecutive_training_days = 0

    for i, day in enumerate(date_range):
        progress = i / total_days
        remaining_days = total_days - i

        # --- 1. 훈련 유형 결정 (주기화 및 피킹 전략) ---
        workout_type = '휴식' # 기본값
        
        # 테이퍼링 기간 (마지막 10일)
        if remaining_days <= 10:
            phase = "테이퍼링"
            if remaining_days in [1, 2]: workout_type = '휴식'
            elif remaining_days == 3: workout_type = '저강도' # 컨디션 조절
            elif remaining_days == 5: workout_type = '고강도' # 마지막 고강도 자극
            else: workout_type = '저강도' if random.random() > 0.4 else '휴식'
            consecutive_training_days = 0
        
        # 일반 기간
        else:
            if progress < 0.6: phase = "준비기"
            else: phase = "시합기"
            
            # 2-3일 훈련 후 1일 휴식 패턴
            if consecutive_training_days < random.choice([2, 3]):
                consecutive_training_days += 1
                if phase == "준비기":
                    workout_type = random.choice(['중강도', '중강도', '고강도'])
                else: # 시합기
                    workout_type = random.choice(['고강도', '고강도', '중강도'])
            else:
                workout_type = '저강도' if random.random() > 0.5 else '휴식'
                consecutive_training_days = 0

        # --- 2. Fitness-Fatigue 모델 계산 ---
        
        # 매일 체력과 피로는 자연 감소
        fitness *= fitness_decay
        fatigue *= fatigue_decay

        # 훈련 부하 적용
        load = load_map[workout_type]
        training_stress = load['ts']
        adaptation_factor = load['af']
        
        # 테이퍼링 기간에는 훈련 부하 감소
        if phase == "테이퍼링" and workout_type != '휴식':
            training_stress *= 0.6

        fatigue += training_stress
        fitness += training_stress * adaptation_factor * 0.1 # 체력은 훈련 부하에 비례해 천천히 증가

        # 퍼포먼스(경기력) = 체력 - 피로
        performance = fitness - fatigue
        
        # --- 3. 계획 저장 ---
        workout_name = random.choice(trainings[workout_type])
        plan.append({
            "날짜": day.strftime("%Y-%m-%d"),
            "요일": day.strftime("%a"),
            "단계": phase,
            "훈련 내용": workout_name,
            "훈련 강도": training_stress, # 훈련 강도를 Training Stress로 사용
            "예상 퍼포먼스": round(performance, 1),
            "상세 가이드": get_detailed_guide(workout_name)
        })

    return pd.DataFrame(plan)


def get_intuitive_df(df):
    """데이터프레임을 직관적으로 표시하기 위해 변환"""
    df_display = df.copy()
    
    def map_intensity(intensity):
        if intensity > 25: return "매우 높음 🔴"
        if intensity > 15: return "높음 🟠"
        if intensity > 0: return "보통 🟡"
        return "회복 🟢"
    df_display["강도 수준"] = df_display["훈련 강도"].apply(map_intensity)

    # 퍼포먼스 레벨을 0-100 사이로 정규화하여 표시
    min_perf = df_display["예상 퍼포먼스"].min()
    max_perf = df_display["예상 퍼포먼스"].max()
    def map_performance(perf):
        normalized_perf = (perf - min_perf) / (max_perf - min_perf) * 100 if (max_perf - min_perf) > 0 else 50
        blocks = int(normalized_perf / 10)
        return "■" * blocks + "□" * (10 - blocks)
    df_display["퍼포먼스 레벨"] = df_display["예상 퍼포먼스"].apply(map_performance)
    
    return df_display[["날짜", "요일", "단계", "훈련 내용", "강도 수준", "퍼포먼스 레벨", "상세 가이드"]]


def plot_performance_graph(df):
    """이중 Y축을 사용하여 그래프 가독성 개선"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 예상 퍼포먼스 레벨 (Line Chart, 왼쪽 Y축)
    fig.add_trace(
        go.Scatter(
            x=df['날짜'], y=df['예상 퍼포먼스'],
            name='예상 퍼포먼스 레벨',
            line=dict(color='royalblue', width=4),
            fill='tozeroy'
        ),
        secondary_y=False,
    )

    # 훈련 강도 (Bar Chart, 오른쪽 Y축)
    fig.add_trace(
        go.Bar(
            x=df['날짜'], y=df['훈련 강도'],
            name='훈련 강도 (피로도)',
            marker_color='crimson',
            opacity=0.6
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text='예상 퍼포먼스와 훈련 강도 변화 (Fitness-Fatigue 모델)',
        legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.6)')
    )
    # Y축 제목 설정
    fig.update_yaxes(title_text="<b>퍼포먼스 레벨</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>훈련 강도</b>", secondary_y=True)
    
    return fig


# --- 5. 메인 UI 구성 ---
with st.form("main_form"):
    st.subheader("1. 기본 정보 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        goal_name = st.text_input("훈련 목표 이름", "2025 마라톤 대회 준비")
    with col2:
        start_day = st.date_input("훈련 시작일", date.today())
    with col3:
        d_day = st.date_input("목표일 (D-Day)", date.today() + timedelta(days=90))

    st.subheader("2. 훈련 계획 설명")
    user_description = st.text_area(
        "어떤 훈련을 계획하고 계신가요? AI가 분석할 수 있도록 자유롭게 설명해주세요.",
        height=150,
        placeholder="예시: 마라톤 풀코스를 준비하고 있습니다. 주 2회 인터벌 훈련과 주 1회 장거리 지속주를 기본으로 하고 싶어요. 중간에 하체 근력 운동도 넣고 싶고, 회복을 위한 조깅과 완전 휴식도 필요합니다."
    )
    
    submitted = st.form_submit_button("🚀 AI 훈련 계획 생성하기", type="primary", use_container_width=True)

# --- 6. 결과 출력 ---
if submitted:
    if not GEMINI_API_KEY:
        st.error("앱을 사용하려면 먼저 사이드바에서 API 키 설정을 확인해주세요.")
    elif not user_description:
        st.warning("훈련 계획 설명을 입력해주세요.")
    elif start_day >= d_day:
        st.error("오류: 훈련 시작일은 목표일보다 이전이어야 합니다.")
    else:
        with st.spinner('AI가 당신의 계획을 분석하고 최적의 스케줄을 생성 중입니다...'):
            training_list = analyze_training_request_with_gemini(user_description, goal_name)
            
            if training_list:
                st.success("✅ AI 분석 완료! 훈련 계획을 생성합니다.")
                total_days = (d_day - start_day).days + 1
                date_range = pd.to_datetime(pd.date_range(start=start_day, end=d_day))
                
                trainings = get_trainings_by_intensity(training_list)
                plan_df = generate_dynamic_plan(total_days, date_range, trainings)
                display_df = get_intuitive_df(plan_df)

                st.markdown('<div id="capture-area" style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)
                
                st.header(f"🎯 '{goal_name}' 최종 훈련 계획")
                
                st.subheader("📊 주기화 그래프")
                st.plotly_chart(plot_performance_graph(plan_df), use_container_width=True)
                
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
                            html2canvas(el, {{ scale: 2, backgroundColor: '#ffffff', useCORS: true }}).then(canvas => {{
                                const image = canvas.toDataURL("image/png");
                                const link = document.createElement("a");
                                link.href = image;
                                link.download = "{file_name_for_image}";
                                link.click();
                                btn.innerHTML = "📸 이미지로 저장"; btn.disabled = false;
                            }});
                        }}
                        </script>
                        <button id="save-img-btn" onclick="captureAndDownload()" style="width:100%; padding:12px; font-size:16px; font-weight:bold; color:white; background-color:#28a745; border:none; border-radius:5px; cursor:pointer;">📸 이미지로 저장</button>
                    """
                    components.html(save_image_html, height=50)