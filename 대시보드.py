from pathlib import Path
import unicodedata

import pandas as pd
import streamlit as st


st.set_page_config(page_title="달빛팹 검사 기록", page_icon="🌙", layout="wide")


@st.cache_data
def load_inspection_data() -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent
    expected_name = unicodedata.normalize("NFC", "달빛팹_검사기록.csv")

    csv_path = next(
        (
            path
            for path in data_dir.glob("*.csv")
            if unicodedata.normalize("NFC", path.name) == expected_name
        ),
        None,
    )

    if csv_path is None:
        raise FileNotFoundError("달빛팹_검사기록.csv 파일을 찾을 수 없습니다.")

    return pd.read_csv(csv_path)


st.title("달빛팹 검사 기록")

try:
    inspection_data = load_inspection_data()
except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"검사 기록을 불러오지 못했습니다: {error}")
    st.stop()

st.sidebar.header("조회 조건")

line_options = ["전체", *sorted(inspection_data["라인"].dropna().unique())]
selected_line = st.sidebar.selectbox("라인", line_options)

result_options = {
    "전체": None,
    "합격만": "합격",
    "불합격만": "불합격",
}
selected_result_label = st.sidebar.selectbox("판정", list(result_options))
selected_result = result_options[selected_result_label]

target_pass_rate = st.sidebar.slider(
    "목표 합격률 (%)",
    min_value=80,
    max_value=99,
    value=95,
    step=1,
)

filtered_data = inspection_data.copy()

if selected_line != "전체":
    filtered_data = filtered_data.loc[filtered_data["라인"] == selected_line]

if selected_result is not None:
    filtered_data = filtered_data.loc[filtered_data["판정"] == selected_result]

total_count = len(filtered_data)
pass_count = int(filtered_data["판정"].eq("합격").sum())
pass_rate = (pass_count / total_count * 100) if total_count else 0.0

metric_total, metric_pass, metric_rate = st.columns(3)
metric_total.metric("전체 건수", f"{total_count:,}건")
metric_pass.metric("합격 건수", f"{pass_count:,}건")
metric_rate.metric("합격률", f"{pass_rate:.1f}%")

if pass_rate >= target_pass_rate:
    st.success(
        f"### ✅ 목표 달성 · 실제 {pass_rate:.1f}% / 목표 {target_pass_rate}%"
    )
else:
    st.error(
        f"### ❌ 목표 미달 · 실제 {pass_rate:.1f}% / 목표 {target_pass_rate}%"
    )

st.subheader("검사 현황")

result_counts = (
    filtered_data["판정"]
    .value_counts()
    .reindex(
        ["합격", "불합격"] if selected_result is None else [selected_result],
        fill_value=0,
    )
    .rename_axis("판정")
    .reset_index(name="건수")
)

failed_by_line = (
    filtered_data.loc[filtered_data["판정"] == "불합격"]
    .groupby("라인")
    .size()
    .rename("불합격 건수")
    .reset_index()
    .sort_values("라인")
)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("#### 판정별 검사 건수")
    st.bar_chart(result_counts, x="판정", y="건수")

with chart_right:
    st.markdown("#### 라인별 불합격 건수")
    if failed_by_line.empty:
        st.info("선택한 조건에 해당하는 불합격 기록이 없습니다.")
    else:
        st.bar_chart(failed_by_line, x="라인", y="불합격 건수")

daily_counts = (
    filtered_data.assign(
        검사일=pd.to_datetime(filtered_data["검사일"], errors="coerce")
    )
    .dropna(subset=["검사일"])
    .groupby("검사일")
    .size()
    .rename("검사 건수")
    .reset_index()
    .sort_values("검사일")
)

st.markdown("#### 검사일별 검사 건수")
if daily_counts.empty:
    st.info("선택한 조건에 해당하는 검사 기록이 없습니다.")
else:
    st.line_chart(daily_counts, x="검사일", y="검사 건수")

with st.expander("검사 기록 표 펼쳐보기"):
    st.dataframe(filtered_data, width="stretch", hide_index=True)
