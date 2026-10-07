import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 예측기")
st.caption("선형회귀와 다항회귀를 비교하고, 테스트 데이터로 모델을 평가합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    df["연도"] = df["날짜"].dt.year

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    return df


df = load_data()

# =========================================================
# 연도별 데이터 만들기
# =========================================================

annual = (
    df.dropna(subset=["연도", "평균기온"])
      .groupby("연도")
      .agg(
          평균기온=("평균기온", "mean"),
          관측일수=("평균기온", "count")
      )
      .reset_index()
)

annual["연도"] = annual["연도"].astype(int)

# 관측일수가 300일 이상인 연도만 사용
annual = annual[annual["관측일수"] >= 300].copy()

annual = annual.sort_values("연도").reset_index(drop=True)

# =========================================================
# 전체 데이터 확인
# =========================================================

st.subheader("📊 연도별 연평균 기온")

st.dataframe(
    annual,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 학습 / 테스트 데이터
# =========================================================

# 2005년 이전 → 훈련
train = annual[annual["연도"] < 2005].copy()

# 2005년부터 → 테스트
test = annual[annual["연도"] >= 2005].copy()

st.subheader("🧪 훈련용 데이터와 테스트용 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )

with col3:
    st.metric(
        "전체 연도 수",
        f"{len(annual)}개"
    )

st.info(
    "훈련용 데이터는 2005년 이전, 테스트용 데이터는 2005년부터 사용합니다. "
    "테스트용 데이터는 모델 학습에 사용하지 않습니다."
)

# =========================================================
# 연도를 작은 숫자로 변환
# =========================================================

# 고차 다항식을 계산할 때 큰 연도 숫자로 인한
# 수치적 불안정성을 줄이기 위해 연도를 변환합니다.
BASE_YEAR = 1950


def transform_year(year):
    return (np.asarray(year) - BASE_YEAR) / 100


X_train = transform_year(train["연도"])
y_train = train["평균기온"].values

X_test = transform_year(test["연도"])
y_test = test["평균기온"].values

# =========================================================
# 다항회귀 함수
# =========================================================

def polynomial_prediction(train_x, train_y, predict_x, degree):
    """
    np.polyfit을 이용해 다항회귀를 학습하고 예측합니다.
    """

    coefficients = np.polyfit(
        train_x,
        train_y,
        degree
    )

    predictions = np.polyval(
        coefficients,
        predict_x
    )

    return coefficients, predictions


# =========================================================
# 1차 / 3차 / 9차 모델 평가
# =========================================================

results = []
models = {}

for degree in [1, 3, 9]:

    coefficients, test_prediction = polynomial_prediction(
        X_train,
        y_train,
        X_test,
        degree
    )

    # 훈련 데이터 예측
    train_prediction = np.polyval(
        coefficients,
        X_train
    )

    # 테스트 MAE
    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    # 훈련 MAE
    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    # 테스트 MSE
    test_mse = mean_squared_error(
        y_test,
        test_prediction
    )

    # 테스트 R²
    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    # 2050년 예측
    prediction_2050 = np.polyval(
        coefficients,
        transform_year(2050)
    )

    models[degree] = {
        "coefficients": coefficients,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "test_mae": test_mae,
        "train_mae": train_mae,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "prediction_2050": prediction_2050
    }

    results.append({
        "곡선": f"{degree}차",
        "훈련 MAE (℃)": round(train_mae, 3),
        "테스트 MAE (℃)": round(test_mae, 3),
        "테스트 MSE": round(test_mse, 3),
        "테스트 R²": round(test_r2, 3),
        "2050년 예측 (℃)": round(prediction_2050, 1)
    })


results_df = pd.DataFrame(results)

# =========================================================
# 결과 표
# =========================================================

st.subheader("📋 1차·3차·9차 모델 비교")

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"훈련용: {train['연도'].min()}~{train['연도'].max()}년 "
    f"({len(train)}개 연도) · "
    f"테스트용: {test['연도'].min()}~{test['연도'].max()}년 "
    f"({len(test)}개 연도)"
)

# =========================================================
# 가장 좋은 테스트 모델
# =========================================================

best_row = results_df.loc[
    results_df["테스트 MAE (℃)"].idxmin()
]

st.success(
    f"테스트 MAE가 가장 작은 모델은 "
    f"**{best_row['곡선']}**입니다. "
    f"테스트 MAE는 **{best_row['테스트 MAE (℃)']:.3f}℃**입니다."
)

# =========================================================
# 훈련 오차 vs 테스트 오차
# =========================================================

st.subheader("📉 훈련 오차와 테스트 오차 비교")

error_fig = go.Figure()

error_fig.add_trace(
    go.Bar(
        x=results_df["곡선"],
        y=results_df["훈련 MAE (℃)"],
        name="훈련 MAE"
    )
)

error_fig.add_trace(
    go.Bar(
        x=results_df["곡선"],
        y=results_df["테스트 MAE (℃)"],
        name="테스트 MAE"
    )
)

error_fig.update_layout(
    barmode="group",
    xaxis_title="다항식 차수",
    yaxis_title="평균절대오차 MAE (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    error_fig,
    use_container_width=True
)

st.caption(
    "훈련 데이터에 잘 맞는 것과 새로운 테스트 데이터를 잘 예측하는 것은 "
    "서로 다를 수 있습니다."
)

# =========================================================
# 실제 데이터 + 1차 / 3차 / 9차 곡선
# =========================================================

st.subheader("📈 1차·3차·9차 곡선 비교")

fig = go.Figure()

# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=6)
    )
)

# 곡선을 그릴 연도 범위
curve_years = np.linspace(
    annual["연도"].min(),
    2050,
    600
)

curve_x = transform_year(curve_years)

for degree in [1, 3, 9]:

    coefficients = models[degree]["coefficients"]

    curve_y = np.polyval(
        coefficients,
        curve_x
    )

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_y,
            mode="lines",
            name=f"{degree}차 모델"
        )
    )

# 2005년 기준선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년"
)

# 2050년 기준선
fig.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050년"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    legend_title="모델"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 2050년 예측 비교
# =========================================================

st.subheader("🔮 2050년 예측 비교")

prediction_cols = st.columns(3)

for col, degree in zip(prediction_cols, [1, 3, 9]):

    prediction = models[degree]["prediction_2050"]

    with col:
        st.metric(
            f"{degree}차 모델",
            f"{prediction:.1f} ℃"
        )

st.warning(
    "2050년은 모든 모델의 학습 범위를 벗어납니다. "
    "따라서 이 값은 실제 기후 예보가 아니라 회귀식을 학습 범위 밖으로 "
    "연장하여 계산한 외삽값입니다."
)

# =========================================================
# 2050년 예측 차이
# =========================================================

st.subheader("🔍 모델별 2050년 예측 차이")

prediction_df = pd.DataFrame({
    "모델": ["1차", "3차", "9차"],
    "2050년 예측(℃)": [
        models[1]["prediction_2050"],
        models[3]["prediction_2050"],
        models[9]["prediction_2050"]
    ]
})

prediction_fig = go.Figure()

prediction_fig.add_trace(
    go.Bar(
        x=prediction_df["모델"],
        y=prediction_df["2050년 예측(℃)"],
        text=[
            f"{value:.1f}℃"
            for value in prediction_df["2050년 예측(℃)"]
        ],
        textposition="auto"
    )
)

prediction_fig.update_layout(
    xaxis_title="모델",
    yaxis_title="2050년 예측 기온 (℃)"
)

st.plotly_chart(
    prediction_fig,
    use_container_width=True
)

# =========================================================
# 과대적합 설명
# =========================================================

st.subheader("🧠 과대적합 생각해 보기")

st.markdown(
    """
### 과대적합이란?

모델이 **훈련용 데이터에 지나치게 맞춰져서**
새로운 데이터에서는 예측 성능이 떨어지는 현상입니다.

차수가 높아지면 더 복잡한 곡선을 만들 수 있기 때문에
훈련 데이터의 세세한 변화까지 따라갈 수 있습니다.

하지만 복잡한 모델이 항상 새로운 데이터를 더 잘 예측하는 것은 아닙니다.

따라서 모델을 평가할 때는 **학습에 사용하지 않은 테스트 데이터**를
사용해야 합니다.
"""
)

# =========================================================
# 학생용 질문
# =========================================================

st.subheader("✏️ 탐구 질문")

st.markdown(
    """
**1. 훈련용 데이터에 가장 잘 맞는 모델은 몇 차인가요?**

→ 결과표의 훈련 MAE를 비교해 보세요.


**2. 테스트용 데이터의 MAE가 가장 작은 모델은 몇 차인가요?**

→ 결과표의 테스트 MAE를 비교해 보세요.


**3. 훈련용 데이터에 가장 잘 맞는 모델과 테스트용 데이터에서
가장 좋은 모델이 서로 다를 수 있는 이유는 무엇일까요?**

→ 과대적합이라는 개념을 이용해 설명해 보세요.


**4. 1차·3차·9차 모델의 2050년 예측값은 얼마나 다른가요?**

→ 세 모델의 예측값을 비교해 보세요.


**5. 2050년 예측값을 실제 기후 예보라고 할 수 있을까요?**

→ 학습 범위와 외삽의 개념을 이용해 설명해 보세요.
"""
)

# =========================================================
# 핵심 정리
# =========================================================

st.subheader("🎯 오늘의 핵심")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### ① 복잡하다고 항상 좋은 것은 아니다

        높은 차수의 모델은 더 복잡한 곡선을 만들 수 있지만,
        새로운 데이터에서 더 좋은 예측을 한다고 보장할 수 없습니다.
        """
    )

with col2:
    st.markdown(
        """
        ### ② 테스트 데이터가 중요하다

        모델을 평가할 때는 학습에 사용하지 않은
        테스트 데이터를 사용해야 합니다.
        """
    )

with col3:
    st.markdown(
        """
        ### ③ 2050년은 외삽이다

        2005년 이전 데이터로 학습한 모델의
        2050년 예측은 학습 범위 밖의 외삽입니다.
        """
    )

st.divider()

st.caption(
    "🎒 송탄고등학교 · 데이터 과학 · 심화 탐구 프로젝트"
)
