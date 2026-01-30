import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI

# =========================
# OPENAI SETUP
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ai_explain(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly financial advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Investment Analyst", layout="wide")
st.title("🤖 AI Investment Analyst")
st.caption("End-to-End Intelligent Investment Assistant")

# =========================
# SESSION MEMORY (PHASE 7)
# =========================
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR NAVIGATION (WEBSITE STYLE)
# =========================
page = st.sidebar.radio(
    "📍 Navigation",
    ["Home", "Profile", "Analyze Stock", "Portfolio", "AI Decision", "Education", "Financial Planning"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":
    st.header("🏠 Welcome")
    st.write("""
    This AI Agent helps you:
    • Analyze stocks  
    • Build portfolios  
    • Get AI investment advice  
    • Learn investing step-by-step  
    """)
    st.write(ai_explain("Explain what an AI investment advisor does in simple terms."))

# =====================================================
# PROFILE (ONCE)
# =====================================================
elif page == "Profile":
    st.header("👤 Investor Profile")

    with st.form("profile_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", 18, 100)
        income = st.number_input("Monthly Income (₹)")
        savings = st.number_input("Monthly Savings (₹)")
        risk = st.slider("Risk Appetite (1–10)", 1, 10)

        submitted = st.form_submit_button("Save Profile")

    if submitted:
        st.session_state.profile = {
            "name": name,
            "age": age,
            "income": income,
            "savings": savings,
            "risk": risk
        }
        st.success("Profile saved!")

# =====================================================
# PHASE 2 – STOCK ANALYSIS
# =====================================================
elif page == "Analyze Stock":
    st.header("📊 Stock Analysis")

    symbol = st.text_input("Stock Symbol (INFY.NS, TCS.NS)")
    if st.button("Analyze") and symbol:
        df = yf.download(symbol, period="5y")

        if df.empty:
            st.error("Invalid symbol")
            st.stop()

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        fig = go.Figure()
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
        st.plotly_chart(fig, use_container_width=True)

        trend = "Bullish" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Bearish"

        st.session_state.history.append({"stock": symbol, "trend": trend})
        st.success(f"Trend: {trend}")

# =====================================================
# PHASE 3 – PORTFOLIO
# =====================================================
elif page == "Portfolio":
    st.header("💼 Portfolio Allocation")

    amount = st.number_input("Investment Amount (₹)")
    risk = st.session_state.profile.get("risk", 5)

    equity = min(80, risk * 10)
    debt = 100 - equity - 10
    gold = 10

    st.write(f"Equity: {equity}%")
    st.write(f"Debt: {debt}%")
    st.write(f"Gold: {gold}%")

    st.write(ai_explain(
        f"Explain why this allocation suits a risk level of {risk}."
    ))

# =====================================================
# PHASE 4 – AI DECISION
# =====================================================
elif page == "AI Decision":
    st.header("🧠 AI Recommendation")

    if not st.session_state.history:
        st.warning("Analyze a stock first")
    else:
        last = st.session_state.history[-1]
        st.write(ai_explain(
            f"Stock: {last['stock']}, Trend: {last['trend']}. Should user buy, hold or wait?"
        ))

# =====================================================
# PHASE 6 – EDUCATION
# =====================================================
elif page == "Education":
    st.header("🎓 Learn Investing")
    st.write(ai_explain(
        "Explain RSI, MACD, SIP, Compounding, Diversification simply."
    ))

# =====================================================
# PHASE 5 – FINANCIAL PLANNING
# =====================================================
elif page == "Financial Planning":
    st.header("🔮 Financial Growth")

    monthly = st.number_input("Monthly SIP (₹)", value=10000)
    years = st.slider("Years", 1, 30, 10)

    rate = 0.12 / 12
    months = years * 12
    fv = monthly * ((1 + rate) ** months - 1) / rate

    st.success(f"Expected Value: ₹{round(fv, 2)}")
    st.write(ai_explain(
        f"Explain how SIP of {monthly} for {years} years grows to {fv}."
    ))

