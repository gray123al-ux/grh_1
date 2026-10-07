
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "2005년 이전의 연평균기온으로 1차·3차·9차 곡선을 학습하고, "
    "학습에 사용하지 않은 2005년 이후 데이터로 성능을 비교합니다."
)

# ---------------------------------------------------------
# 데이터 주소
# ---------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ---------------------------------------------------------
# 연도별 평균기온
# ---------------------------------------------------------
annual = (
    df.groupby("연도")["평균기온"]
    .agg(["mean", "count"])
    .reset_index()
)

annual.columns = ["연도", "연평균기온", "관측일수"]

# 1년의 관측값이 충분한 경우만 사용
annual = annual[annual["관측일수"] >= 300].copy()

annual = annual.sort_values("연도").reset_index(drop=True)

# ---------------------------------------------------------
# 학습 / 테스트 분리
# ---------------------------------------------------------
# 2005년 이전 → 학습
# 2005년부터 → 테스트
train = annual[annual["연도"] < 2005].copy()
test = annual[annual["연도"] >= 2005].copy()

# ---------------------------------------------------------
# 연도를 작은 숫자로 변환
# ---------------------------------------------------------
# 원래 연도(1908, 1909, ...)를 그대로 고차식에 넣으면
# x^9 등이 지나치게 커질 수 있으므로 기준 연도에서 뺀다.
BASE_YEAR = 1908

train_x = train["연도"].to_numpy() - BASE_YEAR
train_y = train["연평균기온"].to_numpy()

test_x = test["연도"].to_numpy() - BASE_YEAR
test_y = test["연평균기온"].to_numpy()

# ---------------------------------------------------------
# 다항식 학습
# ---------------------------------------------------------
degrees = [1, 3, 9]

models = {}

for degree in degrees:
    coefficients = np.polyfit(
        train_x,
        train_y,
        degree
    )

    models[degree] = np.poly1d(coefficients)


# ---------------------------------------------------------
# 테스트 데이터 평가
# ---------------------------------------------------------
results = []

for degree in degrees:
    model = models[degree]

    predictions = model(test_x)

    # 평균 절대 오차(MAE)
    mae = np.mean(
        np.abs(test_y - predictions)
    )

    # 2050년 예측
    x_2050 = 2050 - BASE_YEAR
    prediction_2050 = float(model(x_2050))

    results.append({
        "차수": f"{degree}차",
        "테스트 평균 오차": mae,
        "2050년 예측": prediction_2050
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 데이터 개수 표시
# ---------------------------------------------------------
st.subheader("📊 학습용 / 테스트용 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "학습용 연도 수",
        f"{len(train)}개",
        help="2005년 이전의 연도입니다."
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개",
        help="2005년부터의 연도입니다. 모델 학습에는 사용하지 않았습니다."
    )

st.info(
    "🔒 테스트용 데이터는 모델을 학습할 때 사용하지 않았습니다. "
    "세 모델 모두 2005년 이전 데이터만 보고 학습한 뒤, "
    "2005년 이후 실제 기온으로 채점합니다."
)

# ---------------------------------------------------------
# 결과 표
# ---------------------------------------------------------
st.subheader("🏆 곡선별 테스트 성능과 2050년 예측")

display_df = results_df.copy()

display_df["테스트 평균 오차"] = (
    display_df["테스트 평균 오차"].map(
        lambda x: f"{x:.2f} °C"
    )
)

display_df["2050년 예측"] = (
    display_df["2050년 예측"].map(
        lambda x: f"{x:.2f} °C"
    )
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "테스트 평균 오차는 평균 절대 오차(MAE)입니다. "
    "숫자가 작을수록 테스트 기간의 실제 기온을 더 잘 맞춘 것입니다."
)

# ---------------------------------------------------------
# 가장 좋은 모델
# ---------------------------------------------------------
best_index = results_df["테스트 평균 오차"].idxmin()

best_degree = results_df.loc[best_index, "차수"]
best_mae = results_df.loc[best_index, "테스트 평균 오차"]
best_2050 = results_df.loc[best_index, "2050년 예측"]

st.success(
    f"테스트 데이터에서 가장 오차가 작은 모델은 "
    f"**{best_degree}**입니다. "
    f"평균 오차는 **{best_mae:.2f}°C**, "
    f"2050년 예측은 **{best_2050:.2f}°C**입니다."
)

# ---------------------------------------------------------
# 그래프
# ---------------------------------------------------------
st.subheader("📈 학습 데이터와 테스트 데이터, 그리고 곡선")

fig = go.Figure()

# 실제 학습 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["연평균기온"],
        mode="markers",
        name="학습용 실제 기온",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 실제 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트용 실제 기온",
        marker=dict(
            size=8,
            symbol="diamond"
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "실제 기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 곡선을 부드럽게 그리기 위한 x
min_year = int(annual["연도"].min())
max_year = 2050

curve_years = np.linspace(
    min_year,
    max_year,
    800
)

curve_x = curve_years - BASE_YEAR

# 각 차수 곡선
for degree in degrees:

    model = models[degree]

    curve_y = model(curve_x)

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_y,
            mode="lines",
            name=f"{degree}차 곡선",
            hovertemplate=(
                "연도: %{x:.0f}<br>"
                "예측 기온: %{y:.2f}°C"
                "<extra></extra>"
            )
        )
    )

# 2005년 경계선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 → 테스트",
    annotation_position="top"
)

# 2050년 경계선
fig.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050년",
    annotation_position="top"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    height=600,
    legend_title="데이터 / 모델"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# 테스트 구간 실제값 vs 예측값
# ---------------------------------------------------------
st.subheader("🔎 테스트 기간에서 실제값과 예측값 비교")

comparison = test[["연도", "연평균기온"]].copy()
comparison = comparison.rename(
    columns={"연평균기온": "실제 기온"}
)

for degree in degrees:
    comparison[f"{degree}차 예측"] = models[degree](test_x)

st.dataframe(
    comparison.style.format(
        {
            "실제 기온": "{:.2f}",
            "1차 예측": "{:.2f}",
            "3차 예측": "{:.2f}",
            "9차 예측": "{:.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# 모델 설명
# ---------------------------------------------------------
st.subheader("💡 어떻게 비교했을까?")

st.markdown(
    """
- **1차 곡선**: 직선입니다.
- **3차 곡선**: 한 번 휘어지는 정도를 넘어 더 복잡한 추세를 표현할 수 있습니다.
- **9차 곡선**: 훨씬 복잡한 모양을 표현할 수 있습니다.
- 세 모델 모두 **2005년 이전 데이터만 사용해서 학습**했습니다.
- **2005년 이후 데이터는 학습 과정에서 사용하지 않고**, 마지막에 실제값과 비교해서 점수를 계산했습니다.
- 따라서 테스트 평균 오차는 모델이 **처음 보는 기간을 얼마나 잘 예측했는지**를 보여 줍니다.
- 고차식 계산에서는 `연도 - 1908`을 사용해 연도 숫자의 크기를 줄였습니다.
"""
)

# ---------------------------------------------------------
# 데이터 기준
# ---------------------------------------------------------
st.caption(
    f"분석 기준: 연평균기온은 관측일수가 300일 이상인 연도만 사용 | "
    f"학습: {train['연도'].min()}~{train['연도'].max()} | "
    f"테스트: {test['연도'].min()}~{test['연도'].max()} | "
    f"2050년은 학습 범위 밖의 미래 예측"
)

