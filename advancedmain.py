# =====================================================
# AI INVESTMENT INTELLIGENCE PLATFORM v5.0
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import datetime
import json
import os

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="AI Investment Intelligence", layout="wide")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

DB_FILE = "portfolio_memory.json"
MODEL = "gpt-4o-mini"

# =====================================================
# MEMORY AGENT (Persistent Portfolio)
# =====================================================

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

# =====================================================
# MARKET DATA AGENT
# =====================================================

def market_data_agent(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y")

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

    return df

# =====================================================
# MONTE CARLO AGENT
# =====================================================

def monte_carlo_agent(df, simulations=100):
    returns = df["Close"].pct_change().dropna()
    last_price = df["Close"].iloc[-1]

    sim_results = []

    for _ in range(simulations):
        price = last_price
        for _ in range(252):
            price *= (1 + np.random.choice(returns))
        sim_results.append(price)

    return np.percentile(sim_results, [5, 50, 95])

# =====================================================
# CONFIDENCE SCORING AGENT
# =====================================================

def confidence_agent(trend, rsi, macd):
    score = 0
    if trend: score += 1
    if rsi > 50: score += 1
    if macd: score += 1
    confidence = round((score/3)*100,2)
    return score, confidence

# =====================================================
# PORTFOLIO REBALANCING AGENT
# =====================================================

def rebalance_agent(current_alloc, target_alloc):
    adjustments = {}
    for asset in target_alloc:
        adjustments[asset] = target_alloc[asset] - current_alloc.get(asset, 0)
    return adjustments

# =====================================================
# AI ADVISORY AGENT
# =====================================================

def ai_advisor(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"You are a professional financial advisor. Be concise and realistic."},
            {"role":"user","content":prompt}
        ]
    )
    return response.choices[0].message.content

# =====================================================
# UI START
# =====================================================

st.title("🤖 AI Investment Intelligence Platform")

user = st.text_input("Investor Name")
symbol = st.text_input("Stock Symbol", "RELIANCE.NS")

if st.button("Run Full AI Analysis"):

    df = market_data_agent(symbol)

    trend = df["MA50"].iloc[-1] > df["MA200"].iloc[-1]
    rsi_val = df["RSI"].iloc[-1]
    macd_signal = df["MACD"].iloc[-1] > df["Signal"].iloc[-1]

    score, confidence = confidence_agent(trend, rsi_val, macd_signal)

    if score >= 2:
        decision = "BUY"
    elif score == 1:
        decision = "HOLD"
    else:
        decision = "SELL"

    mc_5, mc_50, mc_95 = monte_carlo_agent(df)

    st.subheader("📊 Decision Summary")
    st.metric("Decision", decision)
    st.metric("Confidence %", f"{confidence}%")

    st.subheader("🎲 Monte Carlo Simulation")
    st.write(f"5% Worst Case: ₹{round(mc_5,2)}")
    st.write(f"Median Case: ₹{round(mc_50,2)}")
    st.write(f"95% Best Case: ₹{round(mc_95,2)}")

    explanation = ai_advisor(
        f"Stock: {symbol}, RSI: {rsi_val}, Confidence: {confidence}%, Decision: {decision}. Explain briefly."
    )
    st.subheader("🧠 AI Explanation")
    st.write(explanation)

    # Save memory
    if user:
        save_memory(user,{
            "date":str(datetime.date.today()),
            "symbol":symbol,
            "decision":decision,
            "confidence":confidence
        })

# =====================================================
# MEMORY DASHBOARD
# =====================================================

st.divider()
st.subheader("📜 Portfolio Memory")

if user:
    history = load_memory(user)
    if history:
        st.dataframe(pd.DataFrame(history))
