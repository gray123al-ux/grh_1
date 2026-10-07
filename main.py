import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기 | 다항회귀 탐구",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- 전체 ---------- */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f8fbff 0%,
                #f4f7fb 100%
            );
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #102a43 0%,
                #1d4e89 55%,
                #3b82c4 100%
            );
        border-radius: 24px;
        padding: 2.4rem 2.6rem;
        margin-bottom: 1.6rem;
        color: white;
        box-shadow: 0 12px 35px rgba(16, 42, 67, 0.18);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        line-height: 1.7;
        max-width: 850px;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    /* ---------- Section ---------- */

    .section {
        font-size: 1.55rem;
        font-weight: 800;
        color: #172b4d;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }

    .section-note {
        color: #667085;
        margin-bottom: 1rem;
    }

    /* ---------- Cards ---------- */

    .card {
        background: white;
        border: 1px solid #e4eaf2;
        border-radius: 18px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 4px 16px rgba(16, 42, 67, 0.045);
    }

    .card-label {
        color: #667085;
        font-size: 0.88rem;
        font-weight: 650;
        margin-bottom: 0.45rem;
    }

    .card-value {
        color: #102a43;
        font-size: 1.85rem;
        font-weight: 850;
    }

    .card-sub {
        color: #98a2b3;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    /* ---------- Model cards ---------- */

    .model-card {
        background: white;
        border-radius: 18px;
        padding: 1.4rem;
        border: 1px solid #e4eaf2;
        min-height: 185px;
        box-shadow: 0 4px 16px rgba(16,42,67,0.045);
    }

    .model-card.best {
        border: 2px solid #3b82c4;
        box-shadow: 0 8px 24px rgba(59,130,196,0.13);
    }

    .model-tag {
        font-size: 0.82rem;
        font-weight: 750;
        color: #667085;
    }

    .model-title {
        font-size: 1.35rem;
        font-weight: 850;
        color: #102a43;
        margin: 0.35rem 0 0.8rem 0;
    }

    .model-number {
        font-size: 2rem;
        font-weight: 850;
        color: #172b4d;
    }

    .model-unit {
        font-size: 0.9rem;
        color: #667085;
    }

    .best-label {
        display: inline-block;
        background: #e8f3ff;
        color: #1769aa;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    /* ---------- Insight ---------- */

    .insight {
        background: #eef6ff;
        border-left: 5px solid #3b82c4;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        color: #23415f;
        line-height: 1.65;
    }

    .warning {
        background: #fff8e7;
        border-left: 5px solid #e5a900;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        color: #684d00;
        line-height: 1.65;
    }

    .success {
        background: #edf9f2;
        border-left: 5px solid #35a36d;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        color: #205c3d;
        line-height: 1.65;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #98a2b3;
        border-top: 1px solid #e4eaf2;
        margin-top: 3rem;
        padding-top: 1.5rem;
        line-height: 1.7;
    }

    /* ---------- Mobile ---------- */

    @media (max-width: 800px) {

        .hero-title {
            font-size: 2.1rem;
        }

        .hero {
            padding: 1.7rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# 데이터 주소
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# =========================================================
# 데이터 로딩
# =========================================================

@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중...")
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

    # 관측일수가 300일 이상인 연도만 사용
    annual = annual[
        annual["관측일수"] >= 300
    ].copy()

    annual = annual.sort_values(
        "연도"
    ).reset_index(drop=True)

    return annual


try:
    annual = load_data()

except Exception as e:

    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결 또는 데이터 주소를 확인해 주세요."
    )

    st.stop()


# =========================================================
# 데이터 조건 확인
# =========================================================

train = annual[
    annual["연도"] < 2005
].copy()

test = annual[
    annual["연도"] >= 2005
].copy()


if len(train) < 10 or len(test) < 3:

    st.error(
        "훈련용 또는 테스트용 데이터가 충분하지 않습니다."
    )

    st.stop()


# =========================================================
# Hero
# =========================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-badge">
            DATA SCIENCE · REGRESSION · MODEL EVALUATION
        </div>

        <div class="hero-title">
            🌡️ 기온 예측기
        </div>

        <div class="hero-subtitle">
            직선 하나로 설명하던 기온의 흐름을
            3차와 9차 곡선으로도 표현해 봅니다.
            <br>
            그리고 <strong>학습에 사용하지 않은 테스트 데이터</strong>로
            어떤 모델이 더 잘 예측하는지 비교합니다.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ 분석 설정")

    st.markdown("### 데이터 분할")

    st.info(
        "📚 훈련용\n"
        "2005년 이전\n\n"
        "🧪 테스트용\n"
        "2005년부터"
    )

    st.markdown("### 비교 모델")

    st.write("🔵 **1차** — 직선")
    st.write("🟢 **3차** — 3차 곡선")
    st.write("🔴 **9차** — 고차 곡선")

    st.divider()

    st.markdown("### 평가 방법")

    st.write(
        "**MAE**\n"
        "평균적으로 몇 ℃ 빗나갔는지 나타냅니다."
    )

    st.write(
        "**MSE**\n"
        "큰 오차에 더 큰 벌점을 줍니다."
    )

    st.write(
        "**R²**\n"
        "모델이 데이터의 변화를 얼마나 설명하는지 나타냅니다."
    )

    st.divider()

    st.caption(
        "고차 회귀 계산의 안정성을 위해 "
        "연도는 1950년을 기준으로 변환합니다."
    )


# =========================================================
# KPI
# =========================================================

st.markdown(
    '<div class="section">📊 실험 데이터</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">📚 훈련 데이터</div>
            <div class="card-value">{len(train)}년</div>
            <div class="card-sub">
                {train["연도"].min()} ~ {train["연도"].max()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">🧪 테스트 데이터</div>
            <div class="card-value">{len(test)}년</div>
            <div class="card-sub">
                {test["연도"].min()} ~ {test["연도"].max()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">📅 사용 기간</div>
            <div class="card-value">
                {annual["연도"].min()}~{annual["연도"].max()}
            </div>
            <div class="card-sub">
                연평균 기온
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:

    st.markdown(
        """
        <div class="card">
            <div class="card-label">🎯 평가 기준</div>
            <div class="card-value">MAE</div>
            <div class="card-sub">
                작을수록 좋은 예측
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 전체 데이터 그래프
# =========================================================

st.markdown(
    '<div class="section">📈 서울 연평균 기온의 흐름</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-note">'
    "파란 영역은 모델이 학습하는 구간, "
    "주황 영역은 학습하지 않고 평가하는 구간입니다."
    "</div>",
    unsafe_allow_html=True
)

fig = go.Figure()

# 전체 데이터
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

# 훈련 영역
fig.add_vrect(
    x0=train["연도"].min(),
    x1=train["연도"].max(),
    fillcolor="rgba(59,130,196,0.08)",
    line_width=0
)

# 테스트 영역
fig.add_vrect(
    x0=test["연도"].min(),
    x1=test["연도"].max(),
    fillcolor="rgba(245,158,11,0.08)",
    line_width=0
)

fig.add_vline(
    x=2005,
    line_dash="dash",
    line_width=2,
    annotation_text="2005 · 학습 → 테스트",
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
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 모델 계산
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


def build_model(degree):

    coefficients = np.polyfit(
        X_train,
        y_train,
        degree
    )

    train_prediction = np.polyval(
        coefficients,
        X_train
    )

    test_prediction = np.polyval(
        coefficients,
        X_test
    )

    prediction_2050 = np.polyval(
        coefficients,
        transform_year(2050)
    )

    return {
        "coefficients": coefficients,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "train_mae": mean_absolute_error(
            y_train,
            train_prediction
        ),
        "test_mae": mean_absolute_error(
            y_test,
            test_prediction
        ),
        "test_mse": mean_squared_error(
            y_test,
            test_prediction
        ),
        "test_r2": r2_score(
            y_test,
            test_prediction
        ),
        "prediction_2050": prediction_2050
    }


models = {}

for degree in [1, 3, 9]:

    models[degree] = build_model(
        degree
    )


# =========================================================
# 결과
# =========================================================

rows = []

for degree in [1, 3, 9]:

    model = models[degree]

    rows.append({
        "모델": f"{degree}차",
        "훈련 MAE (℃)": model["train_mae"],
        "테스트 MAE (℃)": model["test_mae"],
        "테스트 MSE": model["test_mse"],
        "테스트 R²": model["test_r2"],
        "2050년 예측 (℃)": model["prediction_2050"]
    })


results = pd.DataFrame(rows)

best_degree = min(
    [1, 3, 9],
    key=lambda degree:
        models[degree]["test_mae"]
)

best_model = models[best_degree]


# =========================================================
# 모델 카드
# =========================================================

st.markdown(
    '<div class="section">🏆 세 모델의 성능</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-note">'
    "테스트 MAE가 작을수록 처음 보는 데이터를 더 정확하게 예측한 것입니다."
    "</div>",
    unsafe_allow_html=True
)

model_columns = st.columns(3)

model_info = {
    1: ("🔵", "직선", "#3b82c4"),
    3: ("🟢", "3차 곡선", "#35a36d"),
    9: ("🔴", "9차 곡선", "#d95c5c")
}

for column, degree in zip(
    model_columns,
    [1, 3, 9]
):

    icon, name, color = model_info[degree]
    model = models[degree]

    is_best = degree == best_degree

    best_html = (
        '<div class="best-label">🏆 테스트 성능 1위</div>'
        if is_best else ""
    )

    with column:

        st.markdown(
            f"""
            <div class="model-card {'best' if is_best else ''}">

                {best_html}

                <div class="model-tag">
                    {icon} {degree}차
                </div>

                <div class="model-title">
                    {name}
                </div>

                <div class="card-label">
                    테스트 MAE
                </div>

                <div class="model-number">
                    {model["test_mae"]:.3f}
                    <span class="model-unit">℃</span>
                </div>

                <div class="card-sub">
                    훈련 MAE {model["train_mae"]:.3f}℃
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 핵심 해석
# =========================================================

st.markdown(
    f"""
    <div class="success">
        <strong>🏆 현재 데이터에서 가장 작은 테스트 MAE:</strong>
        {best_degree}차 모델
        ({best_model["test_mae"]:.3f}℃)
        <br><br>
        단, 이것은 <strong>이번 테스트 데이터에서의 결과</strong>입니다.
        차수가 높다고 항상 더 좋은 모델이라고 일반화할 수는 없습니다.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 그래프 탭
# =========================================================

tab_curve, tab_error, tab_future = st.tabs(
    [
        "📈 회귀곡선",
        "📉 오차 비교",
        "🔮 2050년"
    ]
)


# =========================================================
# 회귀곡선
# =========================================================

with tab_curve:

    st.subheader("1차 · 3차 · 9차 회귀곡선")

    curve_years = np.linspace(
        annual["연도"].min(),
        2050,
        800
    )

    curve_x = transform_year(
        curve_years
    )

    curve_fig = go.Figure()

    # 훈련 데이터
    curve_fig.add_trace(
        go.Scatter(
            x=train["연도"],
            y=train["평균기온"],
            mode="markers",
            name="훈련 데이터",
            marker=dict(
                size=6
            )
        )
    )

    # 테스트 데이터
    curve_fig.add_trace(
        go.Scatter(
            x=test["연도"],
            y=test["평균기온"],
            mode="markers",
            name="테스트 데이터",
            marker=dict(
                size=7,
                symbol="diamond"
            )
        )
    )

    # 모델 곡선
    for degree in [1, 3, 9]:

        y_curve = np.polyval(
            models[degree]["coefficients"],
            curve_x
        )

        curve_fig.add_trace(
            go.Scatter(
                x=curve_years,
                y=y_curve,
                mode="lines",
                name=f"{degree}차 모델",
                line=dict(width=3)
            )
        )

    # 학습 / 테스트 경계
    curve_fig.add_vline(
        x=2005,
        line_dash="dash",
        line_width=2,
        annotation_text="테스트 시작"
    )

    # 2050
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

    st.markdown(
        """
        <div class="insight">
            <strong>💡 그래프에서 찾아보기</strong><br>
            ① 세 곡선은 훈련 데이터에서 얼마나 비슷한가?<br>
            ② 테스트 구간에서는 어떤 곡선이 실제 데이터에 가까운가?<br>
            ③ 2050년으로 갈수록 세 곡선은 어떻게 달라지는가?
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 오차 비교
# =========================================================

with tab_error:

    st.subheader("훈련 데이터와 테스트 데이터의 오차")

    error_fig = go.Figure()

    error_fig.add_trace(
        go.Bar(
            x=results["모델"],
            y=results["훈련 MAE (℃)"],
            name="훈련 MAE",
            text=[
                f"{x:.3f}"
                for x in results["훈련 MAE (℃)"]
            ],
            textposition="auto"
        )
    )

    error_fig.add_trace(
        go.Bar(
            x=results["모델"],
            y=results["테스트 MAE (℃)"],
            name="테스트 MAE",
            text=[
                f"{x:.3f}"
                for x in results["테스트 MAE (℃)"]
            ],
            textposition="auto"
        )
    )

    error_fig.update_layout(
        barmode="group",
        height=500,
        xaxis_title="모델",
        yaxis_title="MAE (℃)",
        hovermode="x unified"
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="insight">
            <strong>🧠 핵심 개념 — 과대적합</strong><br><br>
            훈련 데이터에 지나치게 맞추어진 모델은
            새로운 테스트 데이터에서 오히려 성능이 떨어질 수 있습니다.
            이를 <strong>과대적합(overfitting)</strong>이라고 합니다.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("상세 평가표")

    formatted_results = results.copy()

    formatted_results["훈련 MAE (℃)"] = \
        formatted_results["훈련 MAE (℃)"].round(3)

    formatted_results["테스트 MAE (℃)"] = \
        formatted_results["테스트 MAE (℃)"].round(3)

    formatted_results["테스트 MSE"] = \
        formatted_results["테스트 MSE"].round(3)

    formatted_results["테스트 R²"] = \
        formatted_results["테스트 R²"].round(3)

    formatted_results["2050년 예측 (℃)"] = \
        formatted_results["2050년 예측 (℃)"].round(1)

    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 2050
# =========================================================

with tab_future:

    st.subheader("2050년 예측 비교")

    future_columns = st.columns(3)

    for column, degree in zip(
        future_columns,
        [1, 3, 9]
    ):

        prediction = \
            models[degree]["prediction_2050"]

        icon, name, _ = model_info[degree]

        with column:

            st.markdown(
                f"""
                <div class="model-card">

                    <div class="model-tag">
                        {icon} {degree}차 모델
                    </div>

                    <div class="model-title">
                        {name}
                    </div>

                    <div class="model-number">
                        {prediction:.1f}
                        <span class="model-unit">℃</span>
                    </div>

                    <div class="card-sub">
                        2050년 모델 계산값
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="warning">

            <strong>⚠️ 2050년 예측을 실제 기후 예보로 해석하면 안 됩니다.</strong>

            <br><br>

            이번 모델은 2005년 이전의 데이터만 사용해 학습했습니다.
            따라서 2050년 값은 모델이 학습한 범위를 넘어 계산한
            <strong>외삽(extrapolation)</strong>입니다.

            <br><br>

            특히 고차 다항식은 학습 범위 밖에서
            곡선이 크게 변할 수 있기 때문에 주의해야 합니다.

        </div>
        """,
        unsafe_allow_html=True
    )

    future_df = pd.DataFrame({
        "모델": [
            "1차",
            "3차",
            "9차"
        ],
        "2050년 예측": [
            models[1]["prediction_2050"],
            models[3]["prediction_2050"],
            models[9]["prediction_2050"]
        ]
    })

    future_fig = go.Figure()

    future_fig.add_trace(
        go.Bar(
            x=future_df["모델"],
            y=future_df["2050년 예측"],
            text=[
                f"{x:.1f}℃"
                for x in future_df["2050년 예측"]
            ],
            textposition="auto"
        )
    )

    future_fig.update_layout(
        height=420,
        xaxis_title="모델",
        yaxis_title="2050년 예측 기온 (℃)"
    )

    st.plotly_chart(
        future_fig,
        use_container_width=True
    )


# =========================================================
# 탐구 질문
# =========================================================

st.markdown(
    '<div class="section">🧠 탐구 활동</div>',
    unsafe_allow_html=True
)

q1, q2 = st.columns(2)

with q1:

    with st.expander(
        "① 훈련 데이터에 가장 잘 맞는 모델은?"
    ):

        train_best = min(
            [1, 3, 9],
            key=lambda d:
                models[d]["train_mae"]
        )

        st.write(
            f"현재 데이터에서는 **{train_best}차 모델**의 "
            f"훈련 MAE가 가장 작습니다."
        )

with q2:

    with st.expander(
        "② 테스트 데이터를 가장 잘 예측한 모델은?"
    ):

        st.write(
            f"현재 데이터에서는 **{best_degree}차 모델**의 "
            f"테스트 MAE가 가장 작습니다."
        )


q3, q4 = st.columns(2)

with q3:

    with st.expander(
        "③ 왜 훈련 성능과 테스트 성능이 다를까?"
    ):

        st.write(
            "훈련 데이터에 지나치게 맞추어진 모델은 "
            "새로운 데이터에서 오차가 커질 수 있습니다. "
            "이 현상을 과대적합이라고 합니다."
        )

with q4:

    with st.expander(
        "④ 2050년 값을 실제 예보라고 할 수 있을까?"
    ):

        st.write(
            "아닙니다. 2050년은 학습 범위 밖이므로 "
            "모델의 외삽 결과로 보아야 합니다."
        )


# =========================================================
# 데이터 보기
# =========================================================

with st.expander("📋 연도별 원본 분석 데이터 보기"):

    display_df = annual.copy()

    display_df.columns = [
        "연도",
        "연평균 기온 (℃)",
        "관측일수"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 최종 요약
# =========================================================

st.markdown(
    '<div class="section">🎯 오늘의 핵심</div>',
    unsafe_allow_html=True
)

s1, s2, s3 = st.columns(3)

with s1:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                01 · 모델 복잡성
            </div>

            <div class="card-value">
                1 → 3 → 9
            </div>

            <div class="card-sub">
                차수가 높아질수록 더 복잡한
                곡선을 표현할 수 있습니다.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with s2:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                02 · 공정한 평가
            </div>

            <div class="card-value">
                Test
            </div>

            <div class="card-sub">
                학습에 사용하지 않은 데이터로
                모델을 평가해야 합니다.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with s3:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                03 · 미래 예측
            </div>

            <div class="card-value">
                외삽
            </div>

            <div class="card-sub">
                학습 범위 밖의 예측은
                특히 주의해서 해석해야 합니다.
            </div>

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
        훈련 데이터에 잘 맞는 모델과
        새로운 데이터를 잘 예측하는 모델은 같을까요?
    </div>
    """,
    unsafe_allow_html=True
)
