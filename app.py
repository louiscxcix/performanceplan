import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components # 이미지 저장을 위해 추가

# --- 1. 앱 기본 설정 및 페이지 구성 ---
st.set_page_config(
    page_title="Peak Performance Planner",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Peak Performance Planner")
st.write("AI가 주기화 이론에 기반하여 당신의 훈련 계획을 최적화해 드립니다.")
st.write("---")


# --- 2. 헬퍼 함수 (로직 분리) ---

def get_trainings_by_intensity(training_list):
    """사용자가 입력한 훈련 목록을 강도별로 분류하는 함수"""
    trainings = {
        '고강도': [t['name'] for t in training_list if t['intensity'] == '고강도'],
        '중강도': [t['name'] for t in training_list if t['intensity'] == '중강도'],
        '저강도': [t['name'] for t in training_list if t['intensity'] == '저강도'],
        '휴식': [t['name'] for t in training_list if t['intensity'] == '휴식']
    }
    # 각 강도에 훈련이 없을 경우를 대비해 기본값 설정
    if not trainings['고강도']: trainings['고강도'] = ['고강도 훈련']
    if not trainings['중강도']: trainings['중강도'] = ['중강도 훈련']
    if not trainings['저강도']: trainings['저강도'] = ['저강도/회복']
    if not trainings['휴식']: trainings['휴식'] = ['휴식']
    return trainings

def generate_plan(total_days, date_range, trainings):
    """기간에 따라 적절한 계획 생성 함수를 호출하는 메인 로직 함수"""
    if total_days > 42: # 6주 이상
        return generate_long_term_plan(total_days, date_range, trainings)
    elif 14 < total_days <= 42: # 2주 초과 ~ 6주 이하
        return generate_mid_term_plan(total_days, date_range, trainings)
    else: # 2주 이하
        return generate_short_term_plan(total_days, date_range, trainings)

# --- 3. 기간별 계획 생성 알고리즘 ---

def generate_long_term_plan(total_days, date_range, trainings):
    """CASE 1: 장기 계획 (6주 이상) 생성 함수"""
    plan = []
    # 기간 분할
    prep_days = int(total_days * 0.6)
    comp_days = int(total_days * 0.3)
    trans_days = total_days - prep_days - comp_days

    for i, day in enumerate(date_range):
        current_day_index = i + 1
        phase = ""
        workout = ""
        intensity_val = 0
        volume_val = 0
        guide = ""

        # 주기 설정
        if current_day_index <= prep_days:
            phase = "준비기"
            intensity_val = np.linspace(40, 70, prep_days)[i]
            volume_val = np.linspace(70, 100, prep_days)[i]
            # 스케줄링: 중강도 위주 (중-중-저-고-중-저-휴)
            cycle = i % 7
            if cycle == 0 or cycle == 1 or cycle == 4:
                workout = np.random.choice(trainings['중강도'])
            elif cycle == 2 or cycle == 5:
                workout = np.random.choice(trainings['저강도'])
            elif cycle == 3:
                workout = np.random.choice(trainings['고강도'])
            else:
                workout = np.random.choice(trainings['휴식'])

        elif current_day_index <= prep_days + comp_days:
            phase = "시합기"
            phase_day_index = i - prep_days
            intensity_val = np.linspace(70, 100, comp_days)[phase_day_index]
            volume_val = np.linspace(100, 60, comp_days)[phase_day_index]
            # 스케줄링: 고강도 비중 증가 (고-저-중-저-고-저-휴)
            cycle = i % 7
            if cycle == 0 or cycle == 4:
                workout = np.random.choice(trainings['고강도'])
            elif cycle == 2:
                 workout = np.random.choice(trainings['중강도'])
            elif cycle == 1 or cycle == 3 or cycle == 5:
                workout = np.random.choice(trainings['저강도'])
            else:
                workout = np.random.choice(trainings['휴식'])
            guide = "시합 페이스에 맞춰 수행"

        else:
            phase = "전환기"
            intensity_val = 30
            volume_val = 30
            # 스케줄링: 저강도/휴식 위주
            workout = np.random.choice(trainings['저강도'] + trainings['휴식'])
            guide = "가볍게 회복에 집중"

        plan.append({
            "날짜": day.strftime("%Y-%m-%d"),
            "요일": day.strftime("%a"),
            "단계": phase,
            "훈련 내용": workout,
            "강도": intensity_val,
            "볼륨": volume_val,
            "수행 가이드": guide
        })
    return pd.DataFrame(plan)


def generate_mid_term_plan(total_days, date_range, trainings):
    """CASE 2: 중기 계획 (2-6주) 생성 함수"""
    plan = []
    # 기간 분할
    first_half_days = int(total_days * 0.5)

    for i, day in enumerate(date_range):
        current_day_index = i + 1
        phase = ""
        workout = ""
        intensity_val = 0
        volume_val = 0
        guide = ""

        if current_day_index <= first_half_days:
            phase = "초반부 (볼륨 집중)"
            intensity_val = np.linspace(50, 80, first_half_days)[i]
            volume_val = np.linspace(80, 100, first_half_days)[i]
            # 스케줄링: 중강도 중심
            cycle = i % 7
            if cycle == 0 or cycle == 4:
                workout = np.random.choice(trainings['중강도'])
            elif cycle == 2:
                workout = np.random.choice(trainings['고강도'])
            elif cycle == 1 or cycle == 3 or cycle == 5:
                workout = np.random.choice(trainings['저강도'])
            else:
                workout = np.random.choice(trainings['휴식'])
            guide = "점진적으로 훈련량을 늘리세요"

        else:
            phase = "후반부 (강도 집중)"
            phase_day_index = i - first_half_days
            second_half_days = total_days - first_half_days
            intensity_val = np.linspace(80, 100, second_half_days)[phase_day_index]
            volume_val = np.linspace(100, 50, second_half_days)[phase_day_index]
            # 스케줄링: 고강도 중심
            cycle = i % 7
            if cycle == 0 or cycle == 4:
                workout = np.random.choice(trainings['고강도'])
            elif cycle == 2:
                workout = np.random.choice(trainings['중강도'])
            elif cycle == 1 or cycle == 3 or cycle == 5:
                workout = np.random.choice(trainings['저강도'])
            else:
                workout = np.random.choice(trainings['휴식'])
            guide = "강도를 높여 시합에 대비하세요"

        plan.append({
            "날짜": day.strftime("%Y-%m-%d"),
            "요일": day.strftime("%a"),
            "단계": phase,
            "훈련 내용": workout,
            "강도": intensity_val,
            "볼륨": volume_val,
            "수행 가이드": guide
        })
    return pd.DataFrame(plan)


def generate_short_term_plan(total_days, date_range, trainings):
    """CASE 3: 단기 계획 (2주 이하) 생성 함수"""
    plan = []
    # 테이퍼링 로직
    intensity_values = np.linspace(90, 50, total_days)
    volume_values = np.linspace(60, 20, total_days)

    for i, day in enumerate(date_range):
        remaining_days = total_days - i
        phase = "피킹 및 테이퍼링"
        workout = ""
        guide = ""

        # D-Day가 가까워질수록 휴식 비중 증가
        if remaining_days <= 2:
            workout = np.random.choice(trainings['휴식'])
            guide = "최상의 컨디션을 위한 완전 휴식"
        elif remaining_days <= 4:
             workout = np.random.choice(trainings['저강도'])
             guide = "가벼운 활동으로 컨디션 조절"
        else:
            # 주기적 고강도로 감각 유지
            cycle = i % 4
            if cycle == 0:
                workout = np.random.choice(trainings['고강도'])
                guide = "감각 유지를 위한 짧고 강한 훈련"
            elif cycle == 2:
                workout = np.random.choice(trainings['중강도'])
                guide = "평소보다 짧게 수행"
            else:
                workout = np.random.choice(trainings['저강도'])
                guide = "피로 회복에 집중"

        plan.append({
            "날짜": day.strftime("%Y-%m-%d"),
            "요일": day.strftime("%a"),
            "단계": phase,
            "훈련 내용": workout,
            "강도": intensity_values[i],
            "볼륨": volume_values[i],
            "수행 가이드": guide
        })
    return pd.DataFrame(plan)


def plot_periodization_graph(df):
    """주기화 그래프를 생성하는 함수"""
    fig = go.Figure()
    # 강도 곡선
    fig.add_trace(go.Scatter(
        x=df['날짜'], y=df['강도'],
        mode='lines+markers', name='훈련 강도(Intensity)',
        line=dict(color='crimson', width=3),
        marker=dict(size=5)
    ))
    # 볼륨 곡선
    fig.add_trace(go.Scatter(
        x=df['날짜'], y=df['볼륨'],
        mode='lines+markers', name='훈련량(Volume)',
        line=dict(color='royalblue', width=3, dash='dash'),
        marker=dict(size=5)
    ))
    fig.update_layout(
        title='전체 기간 훈련 강도 및 볼륨 변화',
        xaxis_title='날짜',
        yaxis_title='수준 (0-100)',
        legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.6)'),
        yaxis_range=[0, 110]
    )
    return fig

# --- 4. 사용자 입력 UI (사이드바) ---

with st.sidebar:
    st.header("1. 기본 정보 입력")
    goal_name = st.text_input("훈련 목표 이름", "2025 전국체전 마라톤")
    
    # 날짜 입력
    today = date.today()
    d_day = st.date_input("목표일 (D-Day)", today + timedelta(days=90))
    start_day = st.date_input("훈련 시작일", today)

    st.header("2. 나의 훈련 목록")
    st.caption("수행할 훈련과 강도를 직접 추가하세요.")

    # 세션 상태 초기화
    if 'training_list' not in st.session_state:
        st.session_state.training_list = [
            {'name': '인터벌 트레이닝', 'intensity': '고강도'},
            {'name': '지속주', 'intensity': '중강도'},
            {'name': '회복 조깅', 'intensity': '저강도'},
            {'name': '하체 근력 운동', 'intensity': '고강도'},
            {'name': '완전 휴식', 'intensity': '휴식'}
        ]

    # 훈련 목록 표시
    for i, training in enumerate(st.session_state.training_list):
        st.text(f"  {i+1}. {training['name']} ({training['intensity']})")

    # 훈련 추가/삭제 UI
    with st.expander("훈련 추가/삭제하기"):
        new_training_name = st.text_input("훈련 이름")
        new_training_intensity = st.selectbox("강도", ['고강도', '중강도', '저강도', '휴식'])
        
        col1, col2 = st.columns(2)
        if col1.button("추가", use_container_width=True):
            if new_training_name:
                st.session_state.training_list.append({
                    'name': new_training_name,
                    'intensity': new_training_intensity
                })
                st.rerun()

        if col2.button("마지막 항목 삭제", use_container_width=True):
            if st.session_state.training_list:
                st.session_state.training_list.pop()
                st.rerun()

    # 계획 생성 버튼
    st.write("---")
    generate_button = st.button("🚀 훈련 계획 생성하기", type="primary", use_container_width=True)


# --- 5. 메인 화면 출력 ---

if generate_button:
    # 입력값 검증
    if start_day >= d_day:
        st.error("오류: 훈련 시작일은 목표일보다 이전이어야 합니다.")
    elif not st.session_state.training_list:
        st.error("오류: 하나 이상의 훈련을 추가해야 합니다.")
    else:
        with st.spinner('AI가 주기화 이론에 맞춰 계획을 생성 중입니다...'):
            total_days = (d_day - start_day).days + 1
            date_range = pd.to_datetime(pd.date_range(start=start_day, end=d_day))
            
            # 강도별 훈련 목록 분류
            trainings = get_trainings_by_intensity(st.session_state.training_list)

            # 기간에 맞는 계획 생성
            plan_df = generate_plan(total_days, date_range, trainings)

            # 결과 출력
            st.success(f"**{goal_name}** 목표를 위한 **{total_days}일** 맞춤형 훈련 계획이 생성되었습니다!")
            
            # --- 이미지 캡처를 위한 영역 정의 ---
            st.markdown('<div id="capture-area" style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            # 탭으로 결과 분리
            tab1, tab2 = st.tabs(["📊 주기화 그래프", "📅 상세 훈련 캘린더"])

            with tab1:
                st.plotly_chart(plot_periodization_graph(plan_df), use_container_width=True)

            with tab2:
                st.dataframe(plan_df, use_container_width=True, height=500)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("") # 여백 추가
            
            # --- 다운로드 버튼들 ---
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV 다운로드 기능
                csv = plan_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 CSV 파일로 다운로드",
                    data=csv,
                    file_name=f"{goal_name}_plan.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # 이미지 저장을 위한 HTML 컴포넌트
            file_name_for_image = f"{goal_name.replace(' ', '_')}_plan.png"
            
            save_image_html = f"""
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                <script>
                function captureAndDownload() {{
                    const captureElement = document.getElementById("capture-area");
                    const saveButton = document.getElementById("save-img-btn");
                    
                    const originalButtonText = saveButton.innerHTML;
                    saveButton.innerHTML = "저장 중...";
                    saveButton.disabled = true;

                    html2canvas(captureElement, {{
                        useCORS: true,
                        scale: 2, // 고해상도 이미지
                        backgroundColor: '#ffffff'
                    }}).then(canvas => {{
                        const image = canvas.toDataURL("image/png");
                        const link = document.createElement("a");
                        link.href = image;
                        link.download = "{file_name_for_image}";
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        
                        saveButton.innerHTML = originalButtonText;
                        saveButton.disabled = false;
                    }});
                }}
                </script>
                <button id="save-img-btn" onclick="captureAndDownload()" style="
                    display: block;
                    width: 100%;
                    padding: 12px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji';
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    background-color: #28a745;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    text-align: center;
                ">📸 이미지로 저장</button>
            """
            with col2:
                components.html(save_image_html, height=50)


else:
    st.info("좌측 사이드바에서 정보를 입력하고 '훈련 계획 생성하기' 버튼을 눌러주세요.")
    st.image("https://images.unsplash.com/photo-1552674605-db6ffd402907?q=80&w=1974&auto=format&fit=crop",
             caption="Image by sporlab on Unsplash")
