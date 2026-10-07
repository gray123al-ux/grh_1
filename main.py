
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기 - 선형회귀 평가",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 선형회귀 모델")
st.write(
    "최근 50년과 최근 100년의 과거 기온으로 회귀선을 학습하고, "
    "공통된 2006~2025년 데이터로 미래 예측 성능을 비교합니다."
)

# =========================================================
# 데이터
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수와 평균기온
    annual = (
        df.groupby("연도")["평균기온"]
        .agg(["mean", "count"])
        .reset_index()
    )

    annual.columns = [
        "연도",
        "연평균기온",
        "관측일수"
    ]

    # 1년에 300일 이상 관측된 연도만 사용
    annual = annual[
        annual["관측일수"] >= 300
    ].copy()

    annual = annual.sort_values(
        "연도"
    ).reset_index(drop=True)

    return annual


annual = load_data()

# =========================================================
# 분석 기간
# =========================================================
TEST_START = 2006
TEST_END = 2025

# 테스트 데이터
test = annual[
    (annual["연도"] >= TEST_START)
    & (annual["연도"] <= TEST_END)
].copy()

# 최근 50년 학습
train_50 = annual[
    (annual["연도"] >= 1956)
    & (annual["연도"] <= 2005)
].copy()

# 최근 100년 학습
train_100 = annual[
    (annual["연도"] >= 1906)
    & (annual["연도"] <= 2005)
].copy()

# =========================================================
# 실제 데이터 범위 확인
# =========================================================
if len(test) == 0:
    st.error("2006~2025년 테스트 데이터를 찾을 수 없습니다.")
    st.stop()

if len(train_50) == 0:
    st.error("1956~2005년 학습 데이터를 찾을 수 없습니다.")
    st.stop()

if len(train_100) == 0:
    st.error("1906~2005년 학습 데이터를 찾을 수 없습니다.")
    st.stop()

# =========================================================
# 연도 숫자 안정화
# =========================================================
# 1900, 1956 등의 큰 숫자를 그대로 사용하지 않고
# 기준연도를 빼서 계산
BASE_YEAR = 1900

def make_x(years):
    return (
        np.asarray(years) - BASE_YEAR
    ).reshape(-1, 1)


# =========================================================
# 회귀 모델 학습
# =========================================================

# 최근 50년
X_train_50 = make_x(train_50["연도"])
y_train_50 = train_50["연평균기온"]

model_50 = LinearRegression()
model_50.fit(
    X_train_50,
    y_train_50
)

# 최근 100년
X_train_100 = make_x(train_100["연도"])
y_train_100 = train_100["연평균기온"]

model_100 = LinearRegression()
model_100.fit(
    X_train_100,
    y_train_100
)

# =========================================================
# 테스트 예측
# =========================================================
X_test = make_x(test["연도"])
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)

# =========================================================
# 평가 함수
# =========================================================
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    pred_50
)

mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    pred_100
)

# =========================================================
# 전체 데이터에 대한 기존 방식 평가
# =========================================================
# 전체 연평균기온에 직선을 맞춘 뒤
# 같은 전체 데이터에서 평가
# → 참고용이며 테스트 평가가 아님
X_all = make_x(annual["연도"])
y_all = annual["연평균기온"]

model_all = LinearRegression()

model_all.fit(
    X_all,
    y_all
)

pred_all = model_all.predict(
    X_all
)

mae_all = mean_absolute_error(
    y_all,
    pred_all
)

mse_all = mean_squared_error(
    y_all,
    pred_all
)

r2_all = r2_score(
    y_all,
    pred_all
)

# =========================================================
# 기울기
# =========================================================
slope_50 = float(
    model_50.coef_[0]
)

slope_100 = float(
    model_100.coef_[0]
)

slope_all = float(
    model_all.coef_[0]
)

# 100년당 변화량
slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100
slope_all_100 = slope_all * 100

# =========================================================
# 데이터 개수
# =========================================================
st.subheader("📊 학습 데이터와 테스트 데이터")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}개 연도"
    )
    st.caption(
        f"{train_50['연도'].min()}~"
        f"{train_50['연도'].max()}"
    )

with c2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}개 연도"
    )
    st.caption(
        f"{train_100['연도'].min()}~"
        f"{train_100['연도'].max()}"
    )

with c3:
    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )
    st.caption(
        f"{test['연도'].min()}~"
        f"{test['연도'].max()}"
    )

st.info(
    "🔒 2006~2025년은 두 모델의 학습 과정에서 사용하지 않았습니다. "
    "두 회귀선은 각각 과거 데이터만으로 만든 뒤, "
    "동일한 2006~2025년 실제 기온으로 채점했습니다."
)

# =========================================================
# 핵심 결과
# =========================================================
st.subheader("🏆 최근 50년 vs 최근 100년 테스트 성능")

result_df = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "학습 연도 수": [
        len(train_50),
        len(train_100)
    ],
    "기울기 (°C/년)": [
        slope_50,
        slope_100
    ],
    "100년당 변화 (°C)": [
        slope_50_100,
        slope_100_100
    ],
    "테스트 MAE (°C)": [
        mae_50,
        mae_100
    ],
    "테스트 MSE (°C²)": [
        mse_50,
        mse_100
    ],
    "테스트 R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    result_df.style.format({
        "기울기 (°C/년)": "{:.5f}",
        "100년당 변화 (°C)": "{:+.2f}",
        "테스트 MAE (°C)": "{:.3f}",
        "테스트 MSE (°C²)": "{:.3f}",
        "테스트 R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 기울기 크게 보여주기
# =========================================================
st.subheader("📈 회귀선의 기울기 비교")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "최근 50년 회귀선",
        f"{slope_50_100:+.2f} °C / 100년"
    )
    st.caption(
        "1956~2005년으로 학습"
    )

with c2:
    st.metric(
        "최근 100년 회귀선",
        f"{slope_100_100:+.2f} °C / 100년"
    )
    st.caption(
        "1906~2005년 범위로 학습"
    )

# =========================================================
# 가장 좋은 모델
# =========================================================
if mae_50 < mae_100:
    best_model = "최근 50년 모델"
    best_reason = "테스트 MAE가 더 작습니다."
elif mae_100 < mae_50:
    best_model = "최근 100년 모델"
    best_reason = "테스트 MAE가 더 작습니다."
else:
    best_model = "두 모델 동일"
    best_reason = "테스트 MAE가 같습니다."

st.success(
    f"테스트 기간에서 MAE 기준으로 가장 잘 맞은 모델은 "
    f"**{best_model}**입니다. {best_reason}"
)

# =========================================================
# 전체 데이터 평가
# =========================================================
st.subheader("📌 참고: 전체 데이터에 직선을 맞춘 경우")

st.warning(
    "아래 결과는 전체 데이터를 학습한 뒤 같은 전체 데이터로 평가한 "
    "결과입니다. 따라서 2006~2025년을 따로 떼어 놓은 위의 "
    "테스트 성능과는 의미가 다릅니다."
)

whole_df = pd.DataFrame({
    "평가 방식": [
        "전체 기간"
    ],
    "기간": [
        f"{annual['연도'].min()}~{annual['연도'].max()}"
    ],
    "연도 수": [
        len(annual)
    ],
    "기울기 (°C/년)": [
        slope_all
    ],
    "100년당 변화 (°C)": [
        slope_all_100
    ],
    "MAE (°C)": [
        mae_all
    ],
    "MSE (°C²)": [
        mse_all
    ],
    "R²": [
        r2_all
    ]
})

st.dataframe(
    whole_df.style.format({
        "기울기 (°C/년)": "{:.5f}",
        "100년당 변화 (°C)": "{:+.2f}",
        "MAE (°C)": "{:.3f}",
        "MSE (°C²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 회귀선 그래프
# =========================================================
st.subheader("📈 실제 연평균기온과 두 회귀선")

fig = go.Figure()

# 실제 데이터
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 회귀선용 연도
line_years = np.linspace(
    annual["연도"].min(),
    TEST_END,
    600
)

X_line = make_x(line_years)

line_50 = model_50.predict(X_line)
line_100 = model_100.predict(X_line)

# 최근 50년 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}<br>"
            "예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 최근 100년 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}<br>"
            "예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 학습 / 테스트 경계
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 종료"
)

fig.add_vline(
    x=2006,
    line_dash="dot",
    annotation_text="2006년: 테스트 시작"
)

fig.update_layout(
    height=600,
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    legend_title="데이터 / 회귀선"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 테스트 데이터에서 실제값과 예측값 비교
# =========================================================
st.subheader("🔎 2006~2025년 실제값과 두 모델의 예측")

comparison = pd.DataFrame({
    "연도": test["연도"].values,
    "실제 기온": y_test,
    "50년 모델 예측": pred_50,
    "100년 모델 예측": pred_100,
    "50년 모델 오차": y_test - pred_50,
    "100년 모델 오차": y_test - pred_100
})

st.dataframe(
    comparison.style.format({
        "실제 기온": "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "100년 모델 예측": "{:.2f}",
        "50년 모델 오차": "{:+.2f}",
        "100년 모델 오차": "{:+.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 테스트 기간 예측 그래프
# =========================================================
st.subheader("🎯 테스트 데이터에서 얼마나 잘 맞았나?")

fig_test = go.Figure()

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=y_test,
        mode="lines+markers",
        name="실제 기온",
        line=dict(width=3),
        marker=dict(size=7)
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="50년 학습 모델",
        line=dict(dash="dash", width=3)
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="100년 학습 모델",
        line=dict(dash="dot", width=3)
    )
)

fig_test.update_layout(
    height=500,
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)

# =========================================================
# 해석
# =========================================================
st.subheader("💡 결과를 어떻게 해석할까?")

st.markdown(
    f"""
### 1. MAE
**MAE가 작을수록 좋습니다.**

2006~2025년 동안 실제 기온과 예측 기온의 차이를
절댓값으로 계산해서 평균낸 값입니다.

- 최근 50년 모델: **{mae_50:.3f}°C**
- 최근 100년 모델: **{mae_100:.3f}°C**

### 2. MSE
**MSE가 작을수록 좋습니다.**

큰 오차에 더 큰 벌점을 주기 때문에,
특정 연도에 회귀선이 크게 빗나갔는지도 확인할 수 있습니다.

- 최근 50년 모델: **{mse_50:.3f}°C²**
- 최근 100년 모델: **{mse_100:.3f}°C²**

### 3. R²
**일반적으로 1에 가까울수록 좋습니다.**

다만 테스트 데이터에서 R²가 0보다 작다면
단순히 테스트 기간의 평균 기온을 사용하는 것보다도
회귀선의 예측이 좋지 않았다는 뜻이 될 수 있습니다.

- 최근 50년 모델: **{r2_50:.3f}**
- 최근 100년 모델: **{r2_100:.3f}**

### 4. 기울기

- 최근 50년: **{slope_50_100:+.2f}°C / 100년**
- 최근 100년: **{slope_100_100:+.2f}°C / 100년**

즉, 학습에 사용하는 과거 기간을 50년에서 100년으로
넓혔을 때 장기적인 온도 상승 추세를 나타내는
회귀선의 기울기가 얼마나 달라지는지를 직접 비교할 수 있습니다.

### 가장 중요한 점

**회귀선의 기울기가 크다고 해서 테스트 예측을 잘한다는 뜻은 아닙니다.**

이번 분석에서는

> 과거 데이터 → 회귀선 학습 → 2006~2025년을 처음 보는 데이터로 테스트

순서로 평가했기 때문에,

**기울기와 실제 미래 예측 성능을 따로 비교할 수 있습니다.**
"""
)

# =========================================================
# 데이터 사용 안내
# =========================================================
st.caption(
    f"데이터: 서울 기온 일자료 | "
    f"연평균기온은 연간 관측일수 300일 이상인 연도만 사용 | "
    f"전체 사용 연도: {annual['연도'].min()}~{annual['연도'].max()} | "
    f"테스트: 2006~2025"
)

st.caption(
    "※ 원본 자료가 1907년 10월부터 시작하므로 1906년 자료는 존재하지 않습니다. "
    "또한 1907년은 1년 전체가 아니므로 300일 이상 조건을 적용하면 "
    "100년 학습 데이터에는 포함되지 않습니다."
)

