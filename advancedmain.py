# ==========================================================
# AI INVESTMENT INTELLIGENCE PLATFORM v6.0
# TRUE GENAI MULTI-AGENT SYSTEM
# ==========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import datetime
import json
import os

# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(page_title="AI Multi-Agent Investment Platform", layout="wide")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

MODEL = "gpt-4o-mini"
DB_FILE = "investment_memory.json"

# ==========================================================
# MEMORY AGENT
# ==========================================================

def save_memory(user, record):
    data = {}
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            data = json.load(f)

    if user not in data:
        data[user] = []

    data[user].append(record)

    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_memory(user):
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        data = json.load(f)
        return data.get(user, [])

# ==========================================================
# 1️⃣ MARKET DATA AGENT
# ==========================================================

def market_data_agent(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y")

    if df.empty:
        return None

    return df

# ==========================================================
# 2️⃣ TECHNICAL ANALYSIS AGENT
# ==========================================================

def technical_agent(df):
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    trend = df["MA50"].iloc[-1] > df["MA200"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1] > df["Signal"].iloc[-1]

    return trend, rsi, macd, df

# ==========================================================
# 3️⃣ FUNDAMENTAL AGENT
# ==========================================================

def fundamental_agent(symbol):
    stock = yf.Ticker(symbol)
    info = stock.get_info()
    return {
        "sector": info.get("sector"),
        "pe": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "marketCap": info.get("marketCap")
    }

# ==========================================================
# 4️⃣ SENTIMENT AGENT (GENAI)
# ==========================================================

def sentiment_agent(symbol):
    prompt = f"Provide current market sentiment (Bullish, Bearish, Neutral) for stock {symbol} in one word."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()

# ==========================================================
# 5️⃣ MONTE CARLO RISK SIMULATION AGENT
# ==========================================================

def monte_carlo_agent(df, simulations=200):
    returns = df["Close"].pct_change().dropna()
    last_price = df["Close"].iloc[-1]

    results = []
    for _ in range(simulations):
        price = last_price
        for _ in range(252):
            price *= (1 + np.random.choice(returns))
        results.append(price)

    return np.percentile(results, [5, 50, 95])

# ==========================================================
# 6️⃣ CONFIDENCE AGENT
# ==========================================================

def confidence_agent(trend, rsi, macd, sentiment):
    score = 0

    if trend: score += 1
    if rsi > 50: score += 1
    if macd: score += 1
    if sentiment.lower() == "bullish": score += 1

    confidence = round((score/4)*100, 2)

    return score, confidence

# ==========================================================
# 7️⃣ PORTFOLIO ALLOCATION AGENT
# ==========================================================

def portfolio_agent(risk_level):
    if risk_level >= 15:
        return {"Equity":70, "Debt":20, "Gold":10}
    elif risk_level >= 8:
        return {"Equity":50, "Debt":30, "Gold":20}
    else:
        return {"Equity":30, "Debt":50, "Gold":20}

# ==========================================================
# 8️⃣ REBALANCING AGENT
# ==========================================================

def rebalancing_agent(current_alloc, target_alloc):
    adjustments = {}
    for asset in target_alloc:
        adjustments[asset] = target_alloc[asset] - current_alloc.get(asset, 0)
    return adjustments

# ==========================================================
# 9️⃣ AI ADVISORY AGENT
# ==========================================================

def advisory_agent(symbol, decision, confidence):
    prompt = f"""
    Stock: {symbol}
    Decision: {decision}
    Confidence: {confidence}%
    Explain this investment recommendation briefly and professionally.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"You are a professional financial advisor."},
            {"role":"user","content":prompt}
        ]
    )
    return response.choices[0].message.content

# ==========================================================
# UI START
# ==========================================================

st.title("🤖 AI Multi-Agent Investment Intelligence")

user = st.text_input("Investor Name")
symbol = st.text_input("Stock Symbol", "RELIANCE.NS")
risk_level = st.slider("Risk Appetite (1-20)", 1, 20, 10)

if st.button("Run Full Multi-Agent Analysis"):

    df = market_data_agent(symbol)
    if df is None:
        st.error("Invalid stock symbol.")
        st.stop()

    trend, rsi, macd, df = technical_agent(df)
    fundamentals = fundamental_agent(symbol)
    sentiment = sentiment_agent(symbol)
    mc5, mc50, mc95 = monte_carlo_agent(df)

    score, confidence = confidence_agent(trend, rsi, macd, sentiment)

    if score >= 3:
        decision = "BUY"
    elif score == 2:
        decision = "HOLD"
    else:
        decision = "SELL"

    allocation = portfolio_agent(risk_level)

    explanation = advisory_agent(symbol, decision, confidence)

    # =========================
    # DASHBOARD OUTPUT
    # =========================

    st.subheader("📊 Decision Dashboard")
    st.metric("Final Decision", decision)
    st.metric("Confidence Score", f"{confidence}%")
    st.metric("RSI", round(rsi,2))
    st.metric("Market Sentiment", sentiment)

    st.subheader("🎲 Monte Carlo Projection")
    st.write(f"5% Worst Case: ₹{round(mc5,2)}")
    st.write(f"Median Case: ₹{round(mc50,2)}")
    st.write(f"95% Best Case: ₹{round(mc95,2)}")

    st.subheader("💼 Suggested Allocation")
    fig = go.Figure(data=[go.Pie(labels=list(allocation.keys()), values=list(allocation.values()))])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 AI Explanation")
    st.write(explanation)

    # Save memory
    if user:
        save_memory(user,{
            "date":str(datetime.date.today()),
            "symbol":symbol,
            "decision":decision,
            "confidence":confidence,
            "allocation":allocation
        })

# ==========================================================
# MEMORY VIEW
# ==========================================================

st.divider()
st.subheader("📜 Investment Memory")

if user:
    history = load_memory(user)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("No history found.")
