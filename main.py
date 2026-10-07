
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

# =========================================================
# 디자인
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* 제목 */
.title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #172033;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.05rem;
    color: #667085;
    margin-bottom: 1.5rem;
}

/* 단계 */
.step {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 1rem;
    text-align: center;
    height: 100%;
}

.step-number {
    font-size: 1.5rem;
    font-weight: 800;
    color: #3b82f6;
}

.step-title {
    font-weight: 700;
    color: #172033;
    margin-top: 0.2rem;
}

.step-text {
    color: #667085;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* 카드 */
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.2rem;
    height: 100%;
}

.card-label {
    color: #667085;
    font-size: 0.85rem;
}

.card-value {
    color: #172033;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 0.2rem;
}

.card-help {
    color: #98a2b3;
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* 모델 카드 */
.model {
    background: white;
    border: 2px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.3rem;
    text-align: center;
    height: 100%;
}

.model-best {
    border-color: #3b82f6;
    background: #f0f7ff;
}

.model-name {
    font-size: 1.2rem;
    font-weight: 800;
    color: #172033;
}

.model-description {
    color: #667085;
    font-size: 0.85rem;
    margin: 0.3rem 0 0.8rem;
}

.model-number {
    font-size: 2rem;
    font-weight: 800;
    color: #172033;
}

.model-unit {
    font-size: 0.85rem;
    color: #667085;
}

.badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* 설명 */
.look-box {
    background: #eef6ff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0 1.2rem;
    color: #29415c;
}

.warning-box {
    background: #fff8e6;
    border: 1px solid #f0d27a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #624d0b;
    margin-top: 1rem;
}

/* 섹션 */
.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #172033;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}

/* footer */
.footer {
    text-align: center;
    color: #98a2b3;
    border-top: 1px solid #e5e7eb;
    margin-top: 3rem;
    padding-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 데이터
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data(show_spinner="서울 기온 데이터를 가져오는 중...")
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    annual = (
        df.dropna(
            subset=["연도", "평균기온"]
        )
        .groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    annual["연도"] = annual["연도"].astype(int)

    # 1년 중 300일 이상 관측된 연도만 사용
    annual = annual[
        annual["관측일수"] >= 300
    ].copy()

    return annual.sort_values(
        "연도"
    ).reset_index(drop=True)


try:
    annual = load_data()

except Exception:

    st.error(
        "데이터를 가져오지 못했습니다. "
        "인터넷 연결을 확인해 주세요."
    )

    st.stop()


# =========================================================
# 학습 / 테스트
# =========================================================

train = annual[
    annual["연도"] < 2005
].copy()

test = annual[
    annual["연도"] >= 2005
].copy()


if len(train) < 10 or len(test) < 3:

    st.error(
        "분석에 필요한 연도별 데이터가 충분하지 않습니다."
    )

    st.stop()


# =========================================================
# 제목
# =========================================================

st.markdown(
    '<div class="title">🌡️ 기온 예측기</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '직선과 곡선을 만들어 보고, '
    '어떤 모델이 새로운 데이터를 더 잘 맞히는지 알아봅시다.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 사용 방법
# =========================================================

st.markdown(
    '<div class="section-title">👋 이렇게 해보세요</div>',
    unsafe_allow_html=True
)

steps = st.columns(4)

step_data = [
    ("①", "데이터 보기", "서울 기온의 전체 흐름을 봅니다."),
    ("②", "모델 만들기", "1차·3차·9차 곡선을 비교합니다."),
    ("③", "점수 확인", "테스트 데이터로 모델을 평가합니다."),
    ("④", "생각하기", "왜 모델의 성능이 다른지 생각합니다.")
]

for col, (number, title, text) in zip(
    steps,
    step_data
):

    with col:

        st.markdown(
            f"""
            <div class="step">

                <div class="step-number">
                    {number}
                </div>

                <div class="step-title">
                    {title}
                </div>

                <div class="step-text">
                    {text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 데이터 요약
# =========================================================

st.markdown(
    '<div class="section-title">📊 먼저 데이터를 확인해 볼까요?</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="look-box">

    <strong>쉽게 말하면</strong><br>
    모델에게 과거 데이터를 보여 주어 공부시키고,
    공부할 때 보지 않았던 데이터를 이용해 실력을 시험합니다.

    </div>
    """,
    unsafe_allow_html=True
)

a, b, c = st.columns(3)

with a:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">📚 공부에 사용할 연도</div>
            <div class="card-value">{len(train)}개</div>
            <div class="card-help">
                {train["연도"].min()} ~ {train["연도"].max()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with b:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">📝 시험에 사용할 연도</div>
            <div class="card-value">{len(test)}개</div>
            <div class="card-help">
                {test["연도"].min()} ~ {test["연도"].max()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">📅 전체 연도</div>
            <div class="card-value">{len(annual)}개</div>
            <div class="card-help">
                관측일수가 300일 이상인 연도
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 전체 기온 그래프
# =========================================================

st.markdown(
    '<div class="section-title">📈 1단계 · 서울 기온의 흐름 보기</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="look-box">

    👀 <strong>이 그래프에서 볼 것</strong><br>
    시간이 지나면서 연평균 기온이 어떤 방향으로 움직이는지 살펴보세요.
    그리고 2005년을 기준으로 공부할 데이터와 시험할 데이터를 나눕니다.

    </div>
    """,
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="lines+markers",
        name="연평균 기온",
        line=dict(width=2),
        marker=dict(size=5)
    )
)

# 학습 영역
fig.add_vrect(
    x0=train["연도"].min(),
    x1=train["연도"].max(),
    fillcolor="rgba(59,130,246,0.08)",
    line_width=0
)

# 테스트 영역
fig.add_vrect(
    x0=test["연도"].min(),
    x1=test["연도"].max(),
    fillcolor="rgba(245,158,11,0.10)",
    line_width=0
)

fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년부터 시험",
    annotation_position="top"
)

fig.update_layout(
    height=470,
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=35,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 모델
# =========================================================

BASE_YEAR = 1950


def transform_year(year):

    return (
        np.asarray(year) - BASE_YEAR
    ) / 100


X_train = transform_year(
    train["연도"]
)

y_train = train["평균기온"].values

X_test = transform_year(
    test["연도"]
)

y_test = test["평균기온"].values


def make_model(degree):

    coefficients = np.polyfit(
        X_train,
        y_train,
        degree
    )

    train_pred = np.polyval(
        coefficients,
        X_train
    )

    test_pred = np.polyval(
        coefficients,
        X_test
    )

    prediction_2050 = np.polyval(
        coefficients,
        transform_year(2050)
    )

    return {
        "coefficients": coefficients,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_mae": mean_absolute_error(
            y_train,
            train_pred
        ),
        "test_mae": mean_absolute_error(
            y_test,
            test_pred
        ),
        "test_mse": mean_squared_error(
            y_test,
            test_pred
        ),
        "test_r2": r2_score(
            y_test,
            test_pred
        ),
        "prediction_2050": prediction_2050
    }


models = {
    1: make_model(1),
    3: make_model(3),
    9: make_model(9)
}


# =========================================================
# 모델 비교
# =========================================================

best_degree = min(
    models,
    key=lambda degree:
        models[degree]["test_mae"]
)


st.markdown(
    '<div class="section-title">📐 2단계 · 세 가지 모델 비교하기</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="look-box">

    <strong>차수란?</strong><br>
    차수가 높아질수록 더 복잡한 모양의 곡선을 만들 수 있습니다.

    <br><br>

    🔵 <strong>1차</strong> = 직선<br>
    🟢 <strong>3차</strong> = 조금 더 복잡한 곡선<br>
    🔴 <strong>9차</strong> = 매우 복잡한 곡선

    </div>
    """,
    unsafe_allow_html=True
)


model_columns = st.columns(3)

model_names = {
    1: "직선",
    3: "3차 곡선",
    9: "9차 곡선"
}

model_icons = {
    1: "🔵",
    3: "🟢",
    9: "🔴"
}


for col, degree in zip(
    model_columns,
    [1, 3, 9]
):

    model = models[degree]

    is_best = degree == best_degree

    badge = (
        '<div class="badge">🏆 테스트 점수 1위</div>'
        if is_best else ""
    )

    with col:

        st.markdown(
            f"""
            <div class="model {'model-best' if is_best else ''}">

                {badge}

                <div class="model-name">
                    {model_icons[degree]}
                    {degree}차 · {model_names[degree]}
                </div>

                <div class="model-description">
                    테스트에서 평균
                </div>

                <div class="model-number">
                    {model["test_mae"]:.2f}
                    <span class="model-unit">℃</span>
                </div>

                <div class="model-description">
                    만큼 빗나감
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 회귀 그래프
# =========================================================

st.markdown(
    '<div class="section-title">👀 세 곡선을 직접 비교해 봅시다</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="look-box">

    <strong>그래프를 클릭하거나 마우스를 올려 보세요.</strong><br>
    실제 기온과 각 모델의 곡선이 얼마나 가까운지 비교해 보세요.

    </div>
    """,
    unsafe_allow_html=True
)


curve_years = np.linspace(
    annual["연도"].min(),
    2050,
    800
)

curve_x = transform_year(
    curve_years
)

curve_fig = go.Figure()

curve_fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="공부 데이터",
        marker=dict(size=6)
    )
)

curve_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="시험 데이터",
        marker=dict(
            size=7,
            symbol="diamond"
        )
    )
)

for degree in [1, 3, 9]:

    curve_y = np.polyval(
        models[degree]["coefficients"],
        curve_x
    )

    curve_fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_y,
            mode="lines",
            name=f"{degree}차"
        )
    )

curve_fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="시험 시작"
)

curve_fig.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050"
)

curve_fig.update_layout(
    height=600,
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    curve_fig,
    use_container_width=True
)


# =========================================================
# 점수
# =========================================================

st.markdown(
    '<div class="section-title">🏆 3단계 · 모델의 시험 점수 확인</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="look-box">

    <strong>MAE란?</strong><br>
    모델의 예측이 실제 기온에서 평균적으로 몇 ℃ 떨어져 있는지 나타내는 값입니다.<br>
    <strong>작을수록 더 잘 맞힌 것입니다.</strong>

    </div>
    """,
    unsafe_allow_html=True
)


result_rows = []

for degree in [1, 3, 9]:

    m = models[degree]

    result_rows.append({
        "모델": f"{degree}차",
        "훈련 MAE (℃)": m["train_mae"],
        "테스트 MAE (℃)": m["test_mae"],
        "테스트 MSE": m["test_mse"],
        "테스트 R²": m["test_r2"]
    })


results = pd.DataFrame(
    result_rows
)


error_fig = go.Figure()

error_fig.add_trace(
    go.Bar(
        x=results["모델"],
        y=results["훈련 MAE (℃)"],
        name="공부할 때 오차",
        text=[
            f"{x:.2f}℃"
            for x in results["훈련 MAE (℃)"]
        ],
        textposition="auto"
    )
)

error_fig.add_trace(
    go.Bar(
        x=results["모델"],
        y=results["테스트 MAE (℃)"],
        name="시험에서 오차",
        text=[
            f"{x:.2f}℃"
            for x in results["테스트 MAE (℃)"]
        ],
        textposition="auto"
    )
)

error_fig.update_layout(
    barmode="group",
    height=460,
    xaxis_title="모델",
    yaxis_title="평균 오차 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    error_fig,
    use_container_width=True
)


st.success(
    f"🎉 이번 테스트에서는 **{best_degree}차 모델**이 "
    f"가장 작은 테스트 오차를 보였습니다. "
    f"평균적으로 약 **{models[best_degree]['test_mae']:.2f}℃** "
    f"차이가 났습니다."
)


# =========================================================
# 결과표
# =========================================================

with st.expander("📋 숫자로 자세히 보기"):

    display_results = results.copy()

    display_results["훈련 MAE (℃)"] = \
        display_results["훈련 MAE (℃)"].round(3)

    display_results["테스트 MAE (℃)"] = \
        display_results["테스트 MAE (℃)"].round(3)

    display_results["테스트 MSE"] = \
        display_results["테스트 MSE"].round(3)

    display_results["테스트 R²"] = \
        display_results["테스트 R²"].round(3)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 2050
# =========================================================

st.markdown(
    '<div class="section-title">🔮 4단계 · 2050년 값을 계산해 보기</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="warning-box">

    ⚠️ <strong>중요!</strong><br><br>

    2050년은 모델이 공부한 범위보다 훨씬 뒤에 있습니다.
    따라서 아래 숫자는 실제 기후 예보가 아닙니다.

    <br><br>

    <strong>
    "지금 만든 수학식을 2050년까지 그대로 이어 붙이면
    어떤 숫자가 나오는가?"
    </strong>

    를 알아보는 활동입니다.

    </div>
    """,
    unsafe_allow_html=True
)


future_columns = st.columns(3)

for col, degree in zip(
    future_columns,
    [1, 3, 9]
):

    with col:

        st.markdown(
            f"""
            <div class="model">

                <div class="model-name">
                    {model_icons[degree]}
                    {degree}차
                </div>

                <div class="model-number">
                    {models[degree]["prediction_2050"]:.1f}
                    <span class="model-unit">℃</span>
                </div>

                <div class="model-description">
                    2050년 모델 계산값
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 탐구 질문
# =========================================================

st.markdown(
    '<div class="section-title">🧠 마지막으로 생각해 보기</div>',
    unsafe_allow_html=True
)


with st.expander("💭 질문 1 · 어떤 곡선이 공부 데이터에 가장 가까웠나요?"):

    train_best = min(
        models,
        key=lambda d:
            models[d]["train_mae"]
    )

    st.write(
        f"이번 데이터에서는 **{train_best}차 모델**의 "
        f"훈련 MAE가 가장 작습니다."
    )


with st.expander("💭 질문 2 · 공부를 잘한 모델이 시험도 잘했나요?"):

    st.write(
        "훈련 데이터에 가장 잘 맞는 모델과 "
        "테스트 데이터에서 가장 좋은 모델이 "
        "다를 수 있습니다."
    )


with st.expander("💭 질문 3 · 이런 현상을 무엇이라고 할까요?"):

    st.write(
        "**과대적합(overfitting)**이라고 합니다. "
        "모델이 공부 데이터에 지나치게 맞춰져 "
        "새로운 데이터에서는 성능이 떨어지는 현상입니다."
    )


with st.expander("💭 질문 4 · 2050년 값을 실제 기온 예보로 사용할 수 있을까요?"):

    st.write(
        "아니요. 이번 2050년 값은 학습 범위 밖에서 "
        "계산한 **외삽 결과**입니다. "
        "실제 기후 예보와는 다릅니다."
    )


# =========================================================
# 데이터 보기
# =========================================================

with st.expander("📋 연도별 데이터 직접 보기"):

    table = annual.copy()

    table.columns = [
        "연도",
        "연평균 기온 (℃)",
        "관측일수"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 마무리
# =========================================================

st.markdown(
    """
    <div class="look-box">

    🎯 <strong>오늘의 한 줄 정리</strong><br><br>

    복잡한 곡선이 데이터를 더 잘 따라갈 수는 있지만,
    <strong>새로운 데이터를 잘 예측하는지는 따로 확인해야 합니다.</strong>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        🌡️ 기온 예측기 · 데이터 과학 심화 탐구
        <br>
        송탄고등학교 · 회귀 모델과 과대적합
    </div>
    """,
    unsafe_allow_html=True
)
