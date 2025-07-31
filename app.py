import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import re

# --- 1. 앱 기본 설정 및 페이지 구성 ---
st.set_page_config(
    page_title="Peak Performance Planner (AI)",
    page_icon="�",
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


# --- 3. Gemini 분석 함수 ---
def analyze_training_request_with_gemini(user_text):
    """Gemini API를 사용하여 사용자의 텍스트를 분석하고 훈련 목록을 JSON으로 반환"""
    if not GEMINI_API_KEY:
        st.error("API 키가 설정되지 않아 AI 분석을 수행할 수 없습니다.")
        return None
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    당신은 스포츠 과학 및 훈련 계획 전문가입니다. 사용자가 입력한 훈련 계획 설명을 분석하여, 훈련 종류와 해당 훈련의 강도를 JSON 형식으로 추출해주세요.

    **분석 가이드라인:**
    1.  사용자의 텍스트에서 핵심적인 훈련 활동들을 모두 찾아냅니다. (예: 인터벌, 지속주, 조깅, 근력 운동, 휴식 등)
    2.  각 훈련 활동의 성격을 파악하여 '고강도', '중강도', '저강도', '휴식' 중 하나로 분류합니다.
        - '고강도': 단거리 전력 질주, 인터벌, 고중량 근력 운동, 시합 페이스 훈련 등
        - '중강도': 지속주, 템포 런, 장거리 달리기, 일반적인 근력 운동 등
        - '저강도': 회복 조깅, 가벼운 스트레칭, 기술 훈련 등
        - '휴식': 완전 휴식, 수면 등
    3.  결과를 반드시 아래의 JSON 형식에 맞춰 다른 설명 없이 JSON 코드만 반환해주세요.

    **사용자 입력:**
    "{user_text}"

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
        # 마크다운 코드 블록 제거
        cleaned_text = re.sub(r'```json\n|```', '', response.text).strip()
        parsed_json = json.loads(cleaned_text)
        return parsed_json.get("trainings", [])
    except Exception as e:
        st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
        return None

# --- 4. 기존 헬퍼 및 계획 생성 함수들 (변경 없음) ---
def get_trainings_by_intensity(training_list):
    trainings = {
        '고강도': [t['name'] for t in training_list if t['intensity'] == '고강도'],
        '중강도': [t['name'] for t in training_list if t['intensity'] == '중강도'],
        '저강도': [t['name'] for t in training_list if t['intensity'] == '저강도'],
        '휴식': [t['name'] for t in training_list if t['intensity'] == '휴식']
    }
    if not trainings['고강도']: trainings['고강도'] = ['고강도 훈련']
    if not trainings['중강도']: trainings['중강도'] = ['중강도 훈련']
    if not trainings['저강도']: trainings['저강도'] = ['저강도/회복']
    if not trainings['휴식']: trainings['휴식'] = ['휴식']
    return trainings

def generate_plan(total_days, date_range, trainings):
    if total_days > 42:
        return generate_long_term_plan(total_days, date_range, trainings)
    elif 14 < total_days <= 42:
        return generate_mid_term_plan(total_days, date_range, trainings)
    else:
        return generate_short_term_plan(total_days, date_range, trainings)

def generate_long_term_plan(total_days, date_range, trainings):
    plan = []
    prep_days = int(total_days * 0.6)
    comp_days = int(total_days * 0.3)
    for i, day in enumerate(date_range):
        current_day_index = i + 1
        phase, workout, guide = "", "", ""
        if current_day_index <= prep_days:
            phase = "준비기"
            intensity_val, volume_val = np.linspace(40, 70, prep_days)[i], np.linspace(70, 100, prep_days)[i]
            cycle = i % 7
            if cycle in [0, 1, 4]: workout = np.random.choice(trainings['중강도'])
            elif cycle in [2, 5]: workout = np.random.choice(trainings['저강도'])
            elif cycle == 3: workout = np.random.choice(trainings['고강도'])
            else: workout = np.random.choice(trainings['휴식'])
        elif current_day_index <= prep_days + comp_days:
            phase = "시합기"
            phase_day_index = i - prep_days
            intensity_val, volume_val = np.linspace(70, 100, comp_days)[phase_day_index], np.linspace(100, 60, comp_days)[phase_day_index]
            cycle = i % 7
            if cycle in [0, 4]: workout = np.random.choice(trainings['고강도'])
            elif cycle == 2: workout = np.random.choice(trainings['중강도'])
            elif cycle in [1, 3, 5]: workout = np.random.choice(trainings['저강도'])
            else: workout = np.random.choice(trainings['휴식'])
            guide = "시합 페이스에 맞춰 수행"
        else:
            phase = "전환기"
            intensity_val, volume_val = 30, 30
            workout = np.random.choice(trainings['저강도'] + trainings['휴식'])
            guide = "가볍게 회복에 집중"
        plan.append({"날짜": day.strftime("%Y-%m-%d"), "요일": day.strftime("%a"), "단계": phase, "훈련 내용": workout, "강도": intensity_val, "볼륨": volume_val, "수행 가이드": guide})
    return pd.DataFrame(plan)

def generate_mid_term_plan(total_days, date_range, trainings):
    plan = []
    first_half_days = int(total_days * 0.5)
    for i, day in enumerate(date_range):
        current_day_index = i + 1
        if current_day_index <= first_half_days:
            phase = "초반부 (볼륨 집중)"
            intensity_val, volume_val = np.linspace(50, 80, first_half_days)[i], np.linspace(80, 100, first_half_days)[i]
            guide = "점진적으로 훈련량을 늘리세요"
        else:
            phase = "후반부 (강도 집중)"
            phase_day_index = i - first_half_days
            second_half_days = total_days - first_half_days
            intensity_val, volume_val = np.linspace(80, 100, second_half_days)[phase_day_index], np.linspace(100, 50, second_half_days)[phase_day_index]
            guide = "강도를 높여 시합에 대비하세요"
        cycle = i % 7
        if cycle in [0, 4]: workout = np.random.choice(trainings['중강도'] if phase.startswith("초반부") else trainings['고강도'])
        elif cycle == 2: workout = np.random.choice(trainings['고강도'] if phase.startswith("초반부") else trainings['중강도'])
        elif cycle in [1, 3, 5]: workout = np.random.choice(trainings['저강도'])
        else: workout = np.random.choice(trainings['휴식'])
        plan.append({"날짜": day.strftime("%Y-%m-%d"), "요일": day.strftime("%a"), "단계": phase, "훈련 내용": workout, "강도": intensity_val, "볼륨": volume_val, "수행 가이드": guide})
    return pd.DataFrame(plan)

def generate_short_term_plan(total_days, date_range, trainings):
    plan = []
    intensity_values, volume_values = np.linspace(90, 50, total_days), np.linspace(60, 20, total_days)
    for i, day in enumerate(date_range):
        remaining_days = total_days - i
        if remaining_days <= 2: workout, guide = np.random.choice(trainings['휴식']), "최상의 컨디션을 위한 완전 휴식"
        elif remaining_days <= 4: workout, guide = np.random.choice(trainings['저강도']), "가벼운 활동으로 컨디션 조절"
        else:
            cycle = i % 4
            if cycle == 0: workout, guide = np.random.choice(trainings['고강도']), "감각 유지를 위한 짧고 강한 훈련"
            elif cycle == 2: workout, guide = np.random.choice(trainings['중강도']), "평소보다 짧게 수행"
            else: workout, guide = np.random.choice(trainings['저강도']), "피로 회복에 집중"
        plan.append({"날짜": day.strftime("%Y-%m-%d"), "요일": day.strftime("%a"), "단계": "피킹 및 테이퍼링", "훈련 내용": workout, "강도": intensity_values[i], "볼륨": volume_values[i], "수행 가이드": guide})
    return pd.DataFrame(plan)

def plot_periodization_graph(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['날짜'], y=df['강도'], mode='lines+markers', name='훈련 강도(Intensity)', line=dict(color='crimson', width=3)))
    fig.add_trace(go.Scatter(x=df['날짜'], y=df['볼륨'], mode='lines+markers', name='훈련량(Volume)', line=dict(color='royalblue', width=3, dash='dash')))
    fig.update_layout(title='전체 기간 훈련 강도 및 볼륨 변화', xaxis_title='날짜', yaxis_title='수준 (0-100)', legend=dict(x=0.01, y=0.98), yaxis_range=[0, 110])
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
            training_list = analyze_training_request_with_gemini(user_description)
            
            if training_list:
                st.success("✅ AI 분석 완료! 훈련 계획을 생성합니다.")
                total_days = (d_day - start_day).days + 1
                date_range = pd.to_datetime(pd.date_range(start=start_day, end=d_day))
                
                trainings = get_trainings_by_intensity(training_list)
                plan_df = generate_plan(total_days, date_range, trainings)

                st.markdown('<div id="capture-area" style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)
                st.header(f"🎯 '{goal_name}' 최종 훈련 계획")
                
                tab1, tab2 = st.tabs(["📊 주기화 그래프", "📅 상세 훈련 캘린더"])
                with tab1:
                    st.plotly_chart(plot_periodization_graph(plan_df), use_container_width=True)
                with tab2:
                    st.dataframe(plan_df, use_container_width=True, height=500)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.write("")
                
                col1, col2 = st.columns(2)
                with col1:
                    csv = plan_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(label="📥 CSV 파일로 다운로드", data=csv, file_name=f"{goal_name}_plan.csv", mime="text/csv", use_container_width=True)
                
                file_name_for_image = f"{goal_name.replace(' ', '_')}_plan.png"
                save_image_html = f"""
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                    <script>
                    function captureAndDownload() {{
                        const el = document.getElementById("capture-area");
                        const btn = document.getElementById("save-img-btn");
                        btn.innerHTML = "저장 중..."; btn.disabled = true;
                        html2canvas(el, {{ scale: 2, backgroundColor: '#ffffff' }}).then(canvas => {{
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
                with col2:
                    components.html(save_image_html, height=50)
