# =========================================================
# AI INVESTMENT HIVE - TRUE MULTI AGENT (1024 DIM SAFE)
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import datetime
import time

# =============================
# CONFIG
# =============================

st.set_page_config(page_title="AI Investment Hive", layout="wide")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "financial-memory"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1024   # 🔥 MUST MATCH PINECONE
LLM_MODEL = "gpt-4o-mini"

# =============================
# ENSURE INDEX EXISTS
# =============================

existing_indexes = [i.name for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(3)

index = pc.Index(INDEX_NAME)

# =============================
# SHARED STATE
# =============================

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

# =============================
# AGENTS
# =============================

# ---- DATA AGENT ----
def data_agent(state: InvestmentState):
    try:
        stock = yf.Ticker(state.symbol)
        df = stock.history(period="6mo")

        if df.empty:
            return state

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

    except:
        pass

    return state

# ---- SENTIMENT AGENT ----
def sentiment_agent(state: InvestmentState):
    try:
        prompt = f"Give overall market sentiment for {state.symbol}. Only BULLISH, BEARISH, or NEUTRAL."

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        state.sentiment = response.choices[0].message.content.strip()

    except:
        state.sentiment = "NEUTRAL"

    return state

# ---- MEMORY AGENT ----
def memory_agent(state: InvestmentState):
    try:
        query = f"{state.symbol} {state.trend} RSI {state.rsi}"

        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
            dimensions=EMBEDDING_DIM  # 🔥 FIXED
        ).data[0].embedding

        results = index.query(
            vector=embedding,
            top_k=2,
            include_metadata=True
        )

        if results.matches:
            state.memory_context = results.matches[0].metadata.get("text", "")
        else:
            state.memory_context = "No prior similar case found."

    except Exception as e:
        state.memory_context = "Memory system unavailable."

    return state

# ---- RISK AGENT ----
def risk_agent(state: InvestmentState):
    score = 0
    if state.trend == "BULLISH": score += 1
    if state.rsi < 60: score += 1
    if state.sentiment == "BULLISH": score += 1
    state.risk_score = score
    return state

# ---- DECISION AGENT ----
def decision_agent(state: InvestmentState):
    if state.risk_score >= 2:
        state.decision = "BUY"
    elif state.risk_score == 1:
        state.decision = "HOLD"
    else:
        state.decision = "SELL"
    return state

# ---- EXPLANATION AGENT ----
def explanation_agent(state: InvestmentState):
    try:
        prompt = f"""
        User: {state.user}
        Stock: {state.symbol}
        Price: {state.price}
        Trend: {state.trend}
        RSI: {state.rsi}
        Sentiment: {state.sentiment}
        Memory Context: {state.memory_context}
        Final Decision: {state.decision}

        Provide a professional investment explanation.
        """

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        state.explanation = response.choices[0].message.content

        # Save memory safely
        try:
            save_embed = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=prompt,
                dimensions=EMBEDDING_DIM  # 🔥 FIXED
            ).data[0].embedding

            index.upsert([
                (f"{state.symbol}_{time.time()}", save_embed, {"text": prompt})
            ])
        except:
            pass

    except:
        state.explanation = "Explanation service unavailable."

    return state

# ---- CONSOLIDATION AGENT ----
def consolidation_agent(state: InvestmentState):
    return state

# =============================
# LANGGRAPH WORKFLOW
# =============================

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

# =============================
# STREAMLIT UI
# =============================

st.title("🤖 AI Investment Hive - Multi Agent System")

user = st.text_input("Enter Your Name")
symbol = st.text_input("Enter Stock Symbol (NSE/BSE/US)", "RELIANCE.NS")

if st.button("Run Multi-Agent Analysis"):

    if not user:
        st.warning("Please enter your name.")
    else:
        initial_state = InvestmentState(user=user, symbol=symbol)

        with st.spinner("Agents collaborating..."):
            result = app_graph.invoke(initial_state)

        st.success("Multi-Agent Analysis Complete")

        st.subheader("📊 Market Analysis")
        st.metric("Current Price", f"₹{result.price}")
        st.write("Trend:", result.trend)
        st.write("RSI:", result.rsi)
        st.write("Sentiment:", result.sentiment)
        st.write("Risk Score:", result.risk_score)
        st.write("Final Decision:", result.decision)

        st.subheader("🧠 Explanation")
        st.write(result.explanation)

        st.subheader("📋 Consolidated Dashboard Summary")
        st.info(f"""
        👤 User: {result.user}
        📈 Stock: {result.symbol}
        💰 Price: {result.price}
        🎯 Decision: {result.decision}
        🕒 Timestamp: {result.timestamp}
        """)
