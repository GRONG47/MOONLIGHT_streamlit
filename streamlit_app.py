from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import urlopen

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


DATA_URL = 'https://raw.githubusercontent.com/sharmaroshan/SECOM-Detecting-Defected-Items/master/uci-secom.csv'
EXPECTED_SHA = 'E8B8187CE276DB148707B5103463AA0F35CB8E521DA2BC697C0FD45A6B06AD0F'
TARGET = 'Pass/Fail'
LOCAL = Path(__file__).with_name('uci-secom.csv')


st.set_page_config(page_title='UCI SECOM ML 분석', layout='wide')


@st.cache_data(show_spinner='데이터를 확인하는 중입니다.')
def load_data() -> pd.DataFrame:
    if LOCAL.exists():
        data = LOCAL.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            data = response.read()
        LOCAL.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest().upper()
    if sha != EXPECTED_SHA:
        raise RuntimeError('CSV 해시가 달라서 실행을 멈췄습니다. 다른 데이터가 섞였을 수 있습니다.')
    return pd.read_csv(io.BytesIO(data))


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    fail = x[y == 1].dropna()
    ok = x[y == 0].dropna()
    if len(fail) < 3 or len(ok) < 3:
        return np.nan
    pooled = np.sqrt((fail.std(ddof=1) ** 2 + ok.std(ddof=1) ** 2) / 2)
    if not pooled or np.isnan(pooled):
        return np.nan
    return float(abs((fail.mean() - ok.mean()) / pooled))


def pick_features(x: pd.DataFrame, y: pd.Series, n: int = 5) -> list[str]:
    rows = []
    miss = x.isna().mean()
    nunique = x.nunique(dropna=True)
    for col in x.columns:
        if miss[col] < 0.5 and nunique[col] > 1:
            rows.append((col, cohen_d(x[col], y)))
    return (
        pd.DataFrame(rows, columns=['sensor', 'score'])
        .dropna()
        .sort_values('score', ascending=False)
        .head(n)['sensor']
        .tolist()
    )


def metrics(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'tn': int(tn),
    }


@st.cache_resource(show_spinner='ML1~ML3 모델을 학습하는 중입니다.')
def run_models(feature_n: int, threshold: float) -> dict[str, object]:
    df = load_data()
    cols = [c for c in df.columns if c not in ['Time', TARGET]]
    x_all = df[cols].apply(pd.to_numeric, errors='coerce')
    y_all = (df[TARGET] == 1).astype(int)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )
    selected = pick_features(x_train_raw, y_train, feature_n)
    med = x_train_raw[selected].median(numeric_only=True)
    x_train = x_train_raw[selected].fillna(med)
    x_test = x_test_raw[selected].fillna(med)

    tree2 = DecisionTreeClassifier(max_depth=2, random_state=42, class_weight='balanced').fit(x_train, y_train)
    depth_rows = []
    for depth in [1, 3, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight='balanced').fit(x_train, y_train)
        depth_rows.append({
            '깊이': '무제한' if depth is None else str(depth),
            '훈련 점수': tree.score(x_train, y_train),
            '시험 점수': tree.score(x_test, y_test),
        })

    one_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced').fit(x_train, y_train)
    forest = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced').fit(x_train, y_train)
    rf_prob = forest.predict_proba(x_test)[:, 1]
    threshold_table = pd.DataFrame([
        {'임계값': t, **metrics(y_test, rf_prob, t)} for t in [0.20, 0.35, 0.50, 0.70, threshold]
    ]).drop_duplicates('임계값').sort_values('임계값')

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {'n_estimators': [50, 200], 'max_depth': [3, 6, None]},
        cv=3,
        scoring='recall',
    ).fit(x_train, y_train)
    logreg = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)),
    ]).fit(x_train, y_train)
    gboost = GradientBoostingClassifier(random_state=42).fit(x_train, y_train)
    model_rows = []
    for name, model in {
        'RandomForest(GridSearch)': grid.best_estimator_,
        'LogisticRegression': logreg,
        'GradientBoosting': gboost,
    }.items():
        prob = model.predict_proba(x_test)[:, 1]
        model_rows.append({'모델': name, **metrics(y_test, prob, threshold)})
    model_table = pd.DataFrame(model_rows).sort_values(['recall', 'f1'], ascending=False)

    perm = permutation_importance(forest, x_test, y_test, n_repeats=10, random_state=42, scoring='recall')
    importance = pd.DataFrame({
        'sensor': selected,
        'forest_importance': forest.feature_importances_,
        'permutation_importance': perm.importances_mean,
    }).sort_values('permutation_importance', ascending=False)

    return {
        'selected': selected,
        'tree_rules': export_text(tree2, feature_names=selected),
        'depth': pd.DataFrame(depth_rows),
        'one_tree_score': one_tree.score(x_test, y_test),
        'forest_score': forest.score(x_test, y_test),
        'threshold_table': threshold_table,
        'grid_params': grid.best_params_,
        'grid_score': grid.best_score_,
        'model_table': model_table,
        'importance': importance,
    }


df = load_data()
feature_n = st.sidebar.slider('모델에 사용할 센서 개수', 3, 12, 5)
threshold = st.sidebar.slider('불량 판정 임계값', 0.05, 0.95, 0.20, 0.05)
result = run_models(feature_n, threshold)

fail = int((df[TARGET] == 1).sum())
ok = int((df[TARGET] == -1).sum())
feature_cols = [c for c in df.columns if c not in ['Time', TARGET]]

st.title('UCI SECOM 머신러닝 분석')
st.caption('ML1, ML2, ML3 모두 같은 CSV 해시를 확인한 뒤 실행합니다.')
st.sidebar.caption(f'SHA256: `{EXPECTED_SHA}`')

tabs = st.tabs(['요약', 'ML1 모델', 'ML2 성적표', 'ML3 해석'])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('전체 행', f'{len(df):,}')
    c2.metric('센서 변수', f'{len(feature_cols):,}')
    c3.metric('Pass / Fail', f'{ok:,} / {fail:,}')
    c4.metric('불량 비율', f'{fail / len(df) * 100:.2f}%')
    st.info('불량이 104개로 적기 때문에 정확도만 보면 속기 쉽습니다. 재현율, 정밀도, 놓친 불량을 같이 봅니다.')
    chart_df = pd.DataFrame({'판정': ['Pass', 'Fail'], '개수': [ok, fail]})
    st.altair_chart(alt.Chart(chart_df).mark_bar().encode(x='판정', y='개수', color='판정'), width='stretch')
    st.write('모델이 선택한 센서:', ', '.join(result['selected']))

with tabs[1]:
    st.subheader('결정트리: if 질문 나무')
    st.code(result['tree_rules'], language='text')
    st.dataframe(result['depth'], width='stretch', hide_index=True)
    c1, c2 = st.columns(2)
    c1.metric('나무 1그루 시험 점수', f"{result['one_tree_score']:.3f}")
    c2.metric('숲 200그루 시험 점수', f"{result['forest_score']:.3f}")

with tabs[2]:
    st.subheader('임계값에 따라 모델 성격이 바뀝니다')
    st.dataframe(result['threshold_table'], width='stretch', hide_index=True)
    row = result['threshold_table'].iloc[(result['threshold_table']['임계값'] - threshold).abs().argmin()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('재현율', f"{row['recall']:.3f}")
    c2.metric('정밀도', f"{row['precision']:.3f}")
    c3.metric('가짜 경보', f"{int(row['fp'])}개")
    c4.metric('놓친 불량', f"{int(row['fn'])}개")

with tabs[3]:
    st.subheader('중요 센서와 최종 모델 비교')
    st.write('GridSearchCV 우승 설정:', result['grid_params'])
    st.caption(f"GridSearchCV 평균 재현율: {result['grid_score']:.3f}")
    st.dataframe(result['model_table'], width='stretch', hide_index=True)
    st.altair_chart(
        alt.Chart(result['importance'])
        .mark_bar()
        .encode(x='permutation_importance', y=alt.Y('sensor', sort='-x')),
        width='stretch',
    )
    st.success(f"현재 임계값 {threshold:.2f} 기준 추천 모델: {result['model_table'].iloc[0]['모델']}")
