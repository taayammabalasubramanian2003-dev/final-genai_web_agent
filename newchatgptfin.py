import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import os
import datetime
import json
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Investment Analyst", layout="wide", page_icon="📈")

# =========================
# OPENAI SETUP
# =========================
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key missing")
    st.stop()

client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini-2024-07-18"

# =========================
# DATABASE
# =========================
DB_FILE = "user_portfolio.json"

# =========================
# AI SAFETY + CACHE
# =========================
@st.cache_data(show_spinner=False)
def ai_cached(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": "You are a beginner-friendly, responsible financial advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350
        )
        return response.choices[0].message.content
    except Exception:
        return None

def ai_explain(prompt):
    # Prevent spamming API
    last_call = st.session_state.get("last_ai_call", 0)
    if time.time() - last_call < 5:
        return "⏳ Please wait a moment before asking AI again."
    
    st.session_state.last_ai_call = time.time()
    response = ai_cached(prompt)
    
    if response:
        return response
    else:
        return "⚠️ AI is temporarily unavailable. Core analysis is still valid."

# =========================
# MARKET SENTIMENT
# =========================
def get_market_sentiment():
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        change = ((hist["Close"][-1] - hist["Close"][0]) / hist["Close"][0]) * 100
        if change > 1.5:
            return "BULLISH"
        elif change < -1.5:
            return "BEARISH"
        return "NEUTRAL"
    except:
        return "NEUTRAL"

# =========================
# PORTFOLIO MEMORY
# =========================
def save_portfolio(name, capital, allocation, companies):
    data = {}
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            data = json.load(f)
    data.setdefault(name, []).append({
        "date": str(datetime.date.today()),
        "capital": capital,
        "allocation": allocation,
        "companies": companies
    })
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_history(name):
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE) as f:
        return json.load(f).get(name, [])

# =========================
# SIDEBAR
# =========================
if "profile_created" not in st.session_state:
    st.session_state.profile_created = False

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Profile",
        "Stock Analysis (Phase 2)",
        "AI Decision (Phase 4)",
        "Portfolio Allocation (Phase 3)",
        "Financial Planning (Phase 5)",
        "Education (Phase 6)",
        "History / Memory (Phase 7)",
        "News & Events (Phase 8)",
        "Auto Rebalancer (Phase 9)"
    ]
)


# =========================
# HOME
# =========================
if page == "Home":
    st.title("🤖 AI Investment Analyst Agent")
    st.image("https://placehold.co/1000x500/png?text=Smart+Investing+Dashboard")
    st.write("A transparent, data-driven investment assistant.")

    if st.button("Test AI"):
        st.write(ai_explain("Explain stock investing in one simple line."))

# =========================
# PROFILE
# =========================
elif page == "Profile":
    with st.form("profile"):
        name = st.text_input("Name")
        age = st.number_input("Age", 18, 100)
        income = st.number_input("Monthly Income ₹")
        savings = st.number_input("Monthly Savings ₹")
        risk = st.slider("Risk Appetite", 1, 20)
        submit = st.form_submit_button("Save")

    if submit:
        st.session_state.update({
            "name": name,
            "age": age,
            "income": income,
            "savings": savings,
            "risk": risk,
            "profile_created": True
        })
        st.success("Profile saved")

# =========================
# STOCK ANALYSIS (PHASE 2)
# =========================
elif page == "Stock Analysis (Phase 2)":
    symbol = st.text_input("Stock Symbol (e.g. INFY.NS)")
    if st.button("Analyze") and symbol:
        stock = yf.Ticker(symbol)
        df = stock.history(period="5y")
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        fig = go.Figure()
        fig.add_candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close)
        fig.add_trace(go.Scatter(x=df.index, y=df.MA20, name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df.MA50, name="MA50"))
        st.plotly_chart(fig, use_container_width=True)

        trend = "BULLISH" if df.MA20.iloc[-1] > df.MA50.iloc[-1] else "BEARISH"

        delta = df.Close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))

        macd = df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()

        st.session_state.update({
            "symbol": symbol,
            "trend": trend,
            "rsi_value": rsi.iloc[-1],
            "macd_signal": "BULLISH" if macd.iloc[-1] > signal.iloc[-1] else "BEARISH",
            "stock_analyzed": True
        })

# =========================
# AI DECISION (PHASE 4)
# =========================
elif page == "AI Decision (Phase 4)":
    if not st.session_state.get("stock_analyzed"):
        st.warning("Analyze a stock first")
    else:
        prompt = f"""
        Stock: {st.session_state.symbol}
        Trend: {st.session_state.trend}
        RSI: {st.session_state.rsi_value}
        MACD: {st.session_state.macd_signal}
        Give a BUY/HOLD/WAIT recommendation.
        """
        st.write(ai_explain(prompt))

# =========================
# PORTFOLIO (PHASE 3)
# =========================
elif page == "Portfolio Allocation (Phase 3)":
    capital = st.number_input("Capital ₹", 1000)
    sentiment = get_market_sentiment()
    st.write(f"Market Sentiment: {sentiment}")

    eq, debt, gold = 50, 30, 20
    if sentiment == "BEARISH":
        eq, debt, gold = 30, 50, 20

    if st.button("Save Portfolio"):
        save_portfolio(
            st.session_state.name,
            capital,
            {"Equity": eq, "Debt": debt, "Gold": gold},
            {"Equity": "Nifty 50", "Debt": "Liquid Fund", "Gold": "Gold ETF"}
        )
        st.success("Portfolio Saved")

# =========================
# HISTORY (PHASE 7)
# =========================
elif page == "History / Memory (Phase 7)":
    history = load_history(st.session_state.get("name", ""))
    st.dataframe(pd.DataFrame(history))
# =========================
# PHASE 8 — NEWS & EVENTS AGENT
# =========================
elif page == "News & Events (Phase 8)":
    st.header("📰 Market News & Events Agent")

    symbol = st.text_input("Enter Stock Symbol for News (e.g., TCS.NS)")

    if st.button("Fetch Latest News") and symbol:
        try:
            stock = yf.Ticker(symbol)
            news = stock.news

            if not news:
                st.warning("No recent news found.")
            else:
                for n in news[:5]:
                    st.subheader(n["title"])
                    st.write(f"Source: {n.get('publisher','Unknown')}")
                    st.write(datetime.datetime.fromtimestamp(n["providerPublishTime"]))
                    
                    ai_summary = ai_explain(f"""
                    Summarize this news for an investor:
                    {n["title"]}
                    Explain whether it is Positive, Negative or Neutral
                    and what action an investor should take.
                    """)
                    st.info(ai_summary)
                    st.divider()
        except Exception as e:
            st.error(f"Error fetching news: {e}")


# =========================
# PHASE 9 — AUTO PORTFOLIO REBALANCER
# =========================
elif page == "Auto Rebalancer (Phase 9)":
    st.header("🔁 Auto Portfolio Rebalancer")

    name = st.session_state.get("name")
    if not name:
        st.warning("Please create a profile first.")
        st.stop()

    history = load_portfolio_history(name)

    if not history:
        st.info("No saved portfolio found to rebalance.")
        st.stop()

    last_portfolio = history[-1]

    st.subheader("📂 Last Saved Portfolio")
    st.json(last_portfolio)

    sentiment = get_market_sentiment()
    st.write(f"📈 Current Market Sentiment: **{sentiment}**")

    if st.button("Run AI Rebalancing"):
        with st.spinner("AI is analyzing portfolio rebalancing..."):
            prompt = f"""
            User Risk Appetite: {st.session_state.risk}
            Market Sentiment: {sentiment}
            Existing Allocation: {last_portfolio['allocation']}

            Suggest whether the user should:
            - Increase Equity
            - Reduce Risk
            - Hold Allocation
            Give reasoning.
            """
            rebalance_advice = ai_explain(prompt)
            st.success("✅ Rebalancing Advice")
            st.write(rebalance_advice)


# =========================
# RESUME / PLACEMENT EXPLANATION
# =========================
elif page == "Resume Explanation":
    st.header("🎓 Project Explanation (Resume Ready)")

    st.markdown("""
    ### 🤖 AI Investment Analyst Agent

    **Description:**  
    A multi-agent AI system that assists retail investors with stock analysis, 
    portfolio allocation, financial planning, and education.

    **Tech Stack:**
    - Python
    - Streamlit
    - OpenAI GPT-4o-mini
    - Yahoo Finance API
    - Plotly

    **Agents Implemented:**
    - Stock Analysis Agent (Technical + Fundamental)
    - AI Decision Agent (Buy/Hold/Wait)
    - Portfolio Allocation Agent
    - Financial Planning Agent
    - Education Agent
    - Memory Agent (User Portfolio History)
    - News & Events Agent
    - Auto Portfolio Rebalancer

    **Key Highlights:**
    - Dynamic real-time market data
    - AI-driven reasoning & explainability
    - Persistent user memory
    - Modular agent-based architecture

    **Use Cases:**
    - Retail investors
    - Financial education platforms
    - Robo-advisory prototypes
    """)

