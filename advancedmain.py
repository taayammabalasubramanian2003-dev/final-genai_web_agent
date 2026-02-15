# =========================
# AI INVESTMENT HIVE - TRUE MULTI AGENT PLATFORM
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import os
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Investment Hive", layout="wide")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("financial-memory")

MODEL = "gpt-4o-mini"

# =========================
# SHARED STATE MODEL
# =========================

class InvestmentState(BaseModel):
    user: str = ""
    symbol: str = ""
    price: float = 0
    trend: str = ""
    rsi: float = 0
    sentiment: str = ""
    memory_context: str = ""
    risk_score: int = 0
    decision: str = ""
    explanation: str = ""
    timestamp: str = ""

# =========================
# AGENTS
# =========================

# -------- DATA AGENT --------
def data_agent(state: InvestmentState):
    stock = yf.Ticker(state.symbol)
    df = stock.history(period="6mo")

    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    state.price = round(df["Close"].iloc[-1], 2)
    state.trend = "BULLISH" if state.price > df["MA50"].iloc[-1] else "BEARISH"
    state.rsi = round(rsi.iloc[-1], 2)
    state.timestamp = str(datetime.datetime.now())

    return state

# -------- SENTIMENT AGENT --------
def sentiment_agent(state: InvestmentState):
    prompt = f"Give overall market sentiment for {state.symbol}. Only BULLISH, BEARISH, or NEUTRAL."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    state.sentiment = response.choices[0].message.content.strip()
    return state

# -------- MEMORY AGENT --------
def memory_agent(state: InvestmentState):
    query = f"{state.symbol} {state.trend} RSI {state.rsi}"

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=2, include_metadata=True)

    if results.matches:
        state.memory_context = results.matches[0].metadata.get("text", "")
    else:
        state.memory_context = "No prior similar case found."

    return state

# -------- RISK AGENT --------
def risk_agent(state: InvestmentState):
    score = 0
    if state.trend == "BULLISH": score += 1
    if state.rsi < 60: score += 1
    if state.sentiment == "BULLISH": score += 1
    state.risk_score = score
    return state

# -------- DECISION AGENT --------
def decision_agent(state: InvestmentState):
    if state.risk_score >= 2:
        state.decision = "BUY"
    elif state.risk_score == 1:
        state.decision = "HOLD"
    else:
        state.decision = "SELL"
    return state

# -------- EXPLANATION AGENT --------
def explanation_agent(state: InvestmentState):
    prompt = f"""
    User: {state.user}
    Stock: {state.symbol}
    Price: {state.price}
    Trend: {state.trend}
    RSI: {state.rsi}
    Sentiment: {state.sentiment}
    Historical Context: {state.memory_context}
    Final Decision: {state.decision}

    Provide professional explanation.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    state.explanation = response.choices[0].message.content
    return state

# -------- CONSOLIDATION AGENT --------
def consolidation_agent(state: InvestmentState):
    return state

# =========================
# MULTI AGENT WORKFLOW
# =========================

workflow = StateGraph(InvestmentState)

workflow.add_node("data", data_agent)
workflow.add_node("sentiment", sentiment_agent)
workflow.add_node("memory", memory_agent)
workflow.add_node("risk", risk_agent)
workflow.add_node("decision", decision_agent)
workflow.add_node("explanation", explanation_agent)
workflow.add_node("consolidate", consolidation_agent)

workflow.set_entry_point("data")
workflow.add_edge("data", "sentiment")
workflow.add_edge("sentiment", "memory")
workflow.add_edge("memory", "risk")
workflow.add_edge("risk", "decision")
workflow.add_edge("decision", "explanation")
workflow.add_edge("explanation", "consolidate")
workflow.add_edge("consolidate", END)

app_graph = workflow.compile()

# =========================
# STREAMLIT UI
# =========================

st.title("🤖 AI Investment Hive - True Multi Agent")

user = st.text_input("Enter Your Name")
symbol = st.text_input("Enter Stock Symbol (NSE/BSE/US)", "RELIANCE.NS")

if st.button("Run Full Multi-Agent Analysis"):

    initial_state = InvestmentState(user=user, symbol=symbol)
    result = app_graph.invoke(initial_state)

    st.success("Multi-Agent Analysis Complete")

    st.subheader("📊 Analysis Result")
    st.metric("Stock Price", f"₹{result.price}")
    st.write("Trend:", result.trend)
    st.write("RSI:", result.rsi)
    st.write("Sentiment:", result.sentiment)
    st.write("Risk Score:", result.risk_score)
    st.write("Decision:", result.decision)

    st.subheader("🧠 Explanation")
    st.write(result.explanation)

    st.subheader("📋 Consolidated Dashboard Summary")
    st.info(f"""
    User: {result.user}
    Stock: {result.symbol}
    Decision: {result.decision}
    Time: {result.timestamp}
    """)
