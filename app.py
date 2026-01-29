import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="Forward EPS Algo-Trader", layout="wide")
plt.style.use('fivethirtyeight')

# -----------------------------------------------------------
# 2. 데이터 엔진
# -----------------------------------------------------------
@st.cache
def get_forward_eps_data(ticker_name):
    annual_eps_db = {}
    code = ""
    
    if ticker_name == "삼성전자 (반도체)":
        code = "005930.KS"
        annual_eps_db = {2018: 6024, 2019: 3166, 2020: 3841, 2021: 5777, 2022: 8057, 2023: 1680, 2024: 5120, 2025: 4200, 2026: 5500}
    elif ticker_name == "롯데케미칼 (화학)":
        code = "011170.KS"
        annual_eps_db = {2018: 58398, 2019: 21843, 2020: 4083, 2021: 40907, 2022: 7091, 2023: -1874, 2024: -16296, 2025: -43389, 2026: -7256}
    elif ticker_name == "현대차 (자동차)":
        code = "005380.KS"
        annual_eps_db = {2018: 6567, 2019: 13717, 2020: 6153, 2021: 21056, 2022: 29027, 2023: 47360, 2024: 50150, 2025: 43829, 2026: 45000}
    else: # 삼성중공업
        code = "010140.KS"
        annual_eps_db = {2018: -1670, 2019: -2022, 2020: -1931, 2021: -2046, 2022: -680, 2023: 100, 2024: 257, 2025: 640, 2026: 850}

    dates = pd.date_range(start='2018-01-01', end='2026-02-01', freq='MS')
    fwd_eps_list = []
    for d in dates:
        year, month = d.year, d.month
        this_y = annual_eps_db.get(year, 0)
        next_y = annual_eps_db.get(year + 1, this_y)
        fwd_eps = (this_y * (12 - month) / 12.0) + (next_y * month / 12.0)
        fwd_eps_list.append(fwd_eps)

    df_eps = pd.DataFrame({'Fwd_EPS': fwd_eps_list}, index=dates)
    df_eps['Signal'] = np.where((df_eps['Fwd_EPS'] > 0) & (df_eps['Fwd_EPS'] > df_eps['Fwd_EPS'].shift(1)), 1, 0)
    return code, df_eps

# -----------------------------------------------------------
# 3. UI 및 메인 로직
# -----------------------------------------------------------
st.title("📈 12M Forward EPS Algo-Trader")
st.sidebar.header("⚙️ 전략 설정")
ticker = st.sidebar.radio("1. 분석 종목", ["삼성전자 (반도체)", "롯데케미칼 (화학)", "현대차 (자동차)", "삼성중공업 (조선)"])
hedge_choice = st.sidebar.selectbox("2. 헷지(Hedge) 자산 선택", ["미국채 7-10년 (IEF)", "금 (GLD)", "현금 (CMA 3.5%)"])
run_btn = st.sidebar.button("🚀 시뮬레이션 시작")

if run_btn:
    with st.spinner('분석 중...'):
        code, df_eps = get_forward_eps_data(ticker)
        stock = yf.Ticker(code).history(period="10y")['Close']
        stock.index = stock.index.tz_localize(None)
        
        if "미국채 7-10년 (IEF)" in hedge_choice: hedge_data = yf.Ticker("IEF").history(period="10y")['Close']
        elif "금 (GLD)" in hedge_choice: hedge_data = yf.Ticker("GLD").history(period="10y")['Close']
        else: hedge_data = pd.Series(1, index=stock.index)
        
        hedge_data.index = hedge_data.index.tz_localize(None)
        df = pd.DataFrame({'Target': stock}).join(hedge_data.rename('Hedge'), how='outer').join(df_eps, how='outer')
        df = df.fillna(method='ffill').dropna()
        df = df[df.index >= df_eps.index[0]]
        
        df['Ret_T'] = df['Target'].pct_change()
        df['Ret_H'] = (0.035 / 252) if "현금" in hedge_choice else df['Hedge'].pct_change()
        df['Ret_S'] = np.where(df['Signal'].shift(1)==1, df['Ret_T'], df['Ret_H'])
        
        df['Cum_T'], df['Cum_S'] = (1+df['Ret_T']).cumprod(), (1+df['Ret_S']).cumprod()
        
        # --- 지표 계산 ---
        ret_s, ret_bh = (df['Cum_S'].iloc[-1]-1)*100, (df['Cum_T'].iloc[-1]-1)*100
        mdd_s = ((df['Cum_S'] - df['Cum_S'].cummax()) / df['Cum_S'].cummax()).min() * 100
        mdd_bh = ((df['Cum_T'] - df['Cum_T'].cummax()) / df['Cum_T'].cummax()).min() * 100
        
        def calc_sharpe(ret_series):
            if ret_series.std() == 0: return 0
            return (ret_series.mean() * 252) / (ret_series.std() * np.sqrt(252))
        
        sharpe_s, sharpe_bh = calc_sharpe(df['Ret_S']), calc_sharpe(df['Ret_T'])

        # --- 리포트 출력 ---
        st.markdown(f"### 🏆 {ticker} 전략 성과")
        c1, c2, c3 = st.columns(3)
        c1.metric("전략 수익률", f"{ret_s:.2f}%", delta=f"B&H: {ret_bh:.2f}%")
        c2.metric("MDD (안정성)", f"{mdd_s:.2f}%", delta=f"B&H: {mdd_bh:.2f}%")
        c3.metric("샤프 지수 (효율)", f"{sharpe_s:.2f}", delta=f"B&H: {sharpe_bh:.2f}")

        # 1. 자산 성장 그래프
        st.subheader("1. 자산 성장 그래프")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df['Cum_T'], label='Buy & Hold', color='gray', alpha=0.3)
        ax.plot(df.index, df['Cum_S'], label='Strategy', color='red', lw=2)
        ax.fill_between(df.index, df['Cum_S'].min(), df['Cum_S'].max(), where=df['Signal']==1, color='green', alpha=0.05, label='Stock Hold')
        ax.legend(); st.pyplot(fig)

        # 2. 선행 EPS 추이 그래프 (추가됨)
        st.subheader("2. 12M 선행 EPS 추이")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df.index, df['Fwd_EPS'], label='12M Forward EPS', color='blue', lw=2)
        ax2.axhline(0, color='black', lw=1, ls='--') # 0선
        
        # 매수 신호 발생 구간 배경 표시
        y_min, y_max = ax2.get_ylim()
        ax2.fill_between(df.index, y_min, y_max, where=df['Signal']==1, color='green', alpha=0.1, label='Buy Signal')
        
        ax2.set_title(f"EPS Trend & Buy Signals")
        ax2.legend(); st.pyplot(fig2)
