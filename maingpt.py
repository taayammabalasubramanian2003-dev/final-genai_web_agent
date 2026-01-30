import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import os

# =========================
# OPENAI SETUP
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ai_explain(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return "⚠️ AI explanation temporarily unavailable."

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Investment Analyst", layout="wide")

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "👤 Profile", "📊 Analyze Stock", "💼 Portfolio", "📈 Financial Planning"]
)

# =========================
# HOME
# =========================
if page == "🏠 Home":
    st.title("🤖 AI Investment Analyst Agent")
    st.write("""
    This AI agent helps you:
    • Analyze stocks technically & fundamentally  
    • Understand RSI, MACD, trends visually  
    • Get AI-backed BUY / HOLD / WAIT guidance  
    • Build a personalized portfolio  
    • Learn investing step-by-step  
    """)
    st.info("Designed for beginners & long-term investors")

# =========================
# PROFILE
# =========================
if page == "👤 Profile":
    st.header("👤 Investor Profile")

    with st.form("profile"):
        st.session_state.name = st.text_input("Name")
        st.session_state.age = st.number_input("Age", 18, 100)
        st.session_state.income = st.number_input("Monthly Income (₹)")
        st.session_state.savings = st.number_input("Monthly Savings (₹)")
        st.session_state.risk = st.slider("Risk Appetite (%)", 1, 20)
        st.form_submit_button("Save Profile")

# =========================
# ANALYZE STOCK (PHASE 2 + 4)
# =========================
if page == "📊 Analyze Stock":
    st.header("📊 Stock Analysis")

    symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS)")
    mode = st.selectbox("Mode", ["INVESTOR", "TRADER"])

    if st.button("Analyze") and symbol:
        period = "5y" if mode == "INVESTOR" else "6mo"
        interval = "1mo" if mode == "INVESTOR" else "1d"

        df = yf.Ticker(symbol).history(period=period, interval=interval)
        df.reset_index(inplace=True)

        # Moving Averages
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        # Candlestick
        fig = go.Figure()
        fig.add_candlestick(
            x=df["Date"], open=df["Open"],
            high=df["High"], low=df["Low"], close=df["Close"]
        )
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA50"))
        st.plotly_chart(fig, use_container_width=True)

        trend = "BULLISH" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "BEARISH"
        st.success(f"📈 Trend: {trend}")

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))
        rsi_value = round(rsi.iloc[-1], 2)
        st.line_chart(rsi)

        # MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_signal = "BULLISH" if macd.iloc[-1] > signal.iloc[-1] else "BEARISH"
        st.line_chart(pd.DataFrame({"MACD": macd, "Signal": signal}))

        # Fundamental
        info = yf.Ticker(symbol).info
        st.subheader("🏦 Fundamental Analysis")
        st.write("Sector:", info.get("sector"))
        st.write("P/E:", info.get("trailingPE"))
        st.write("EPS:", info.get("trailingEps"))
        st.write("Market Cap:", info.get("marketCap"))

        # AI DECISION
        decision = ai_explain(f"""
        Stock: {symbol}
        Trend: {trend}
        RSI: {rsi_value}
        MACD: {macd_signal}

        Give BUY / HOLD / WAIT with reasoning.
        """)
        st.subheader("🧠 AI Recommendation")
        st.write(decision)

# =========================
# PORTFOLIO (PHASE 3)
# =========================
if page == "💼 Portfolio":
    st.header("💼 Portfolio Allocation")

    capital = st.number_input("Total Investment (₹)", 1000)
    risk = st.session_state.get("risk", 10)

    equity = 40 + risk
    debt = 40 - risk
    gold = 20

    st.write(f"📈 Equity: {equity}%")
    st.write(f"🏦 Debt: {debt}%")
    st.write(f"🪙 Gold: {gold}%")

# =========================
# FINANCIAL PLANNING (PHASE 5)
# =========================
if page == "📈 Financial Planning":
    st.header("📈 Financial Growth")

    monthly = st.number_input("Monthly SIP (₹)", 1000)
    years = st.slider("Years", 1, 30)

    rate = 12 / 100 / 12
    months = years * 12
    fv = monthly * ((1 + rate) ** months - 1) / rate

    st.success(f"Expected Value: ₹{round(fv,2)}")

    st.write(ai_explain("""
    Explain SIP, compounding and long-term investing in simple words.
    """))
