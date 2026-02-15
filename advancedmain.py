import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
import datetime
import json

# =========================
# 1. ADVANCED CONFIGURATION
# =========================
st.set_page_config(page_title="AI Financial Hive (Multi-Agent)", layout="wide", page_icon="🧠")

# API Keys Check
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"] # NEW
except:
    st.error("❌ Missing Keys! Add OPENAI_API_KEY and PINECONE_API_KEY to secrets.toml")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# 2. PINECONE VECTOR MEMORY (RAG)
# =========================
# This replaces the simple JSON file with a Semantic Brain
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "financial-memory"

# Ensure Index Exists (Auto-Creation for Demo)
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536, # Standard for OpenAI Embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except:
        pass # Index likely creating

index = pc.Index(INDEX_NAME)

def get_embedding(text):
    """Converts text to Vector using OpenAI"""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def save_memory(user_id, text, metadata):
    """Saves an event/portfolio to Vector DB"""
    vector = get_embedding(text)
    # ID format: user_id_timestamp
    id = f"{user_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    index.upsert(vectors=[(id, vector, metadata)])

def query_memory(user_id, query_text):
    """RAG: Finds similar past events"""
    vector = get_embedding(query_text)
    results = index.query(
        vector=vector,
        top_k=3,
        include_metadata=True,
        filter={"user_id": user_id}
    )
    return [match['metadata']['text'] for match in results['matches']]

# =========================
# 3. AGENT DEFINITIONS (OOP Structure)
# =========================

class MarketAnalystAgent:
    """Agent 1: Specialized in Mathematics & Live Data"""
    def analyze_stock(self, symbol):
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty: return None
        
        # Technical Logic
        df["MA50"] = df["Close"].rolling(50).mean()
        trend = "BULLISH" if df["Close"].iloc[-1] > df["MA50"].iloc[-1] else "BEARISH"
        
        # RSI Logic
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            "price": round(df["Close"].iloc[-1], 2),
            "trend": trend,
            "rsi": round(rsi.iloc[-1], 2),
            "data": df
        }

class PlannerAgent:
    """Agent 2: Specialized in Asset Allocation & Risk"""
    def create_portfolio(self, risk_level, sentiment):
        # Rule-based logic augmented by AI
        if sentiment == "BEARISH":
            return {"Equity": 30, "Debt": 50, "Gold": 20}
        elif risk_level > 15: # Aggressive
            return {"Equity": 70, "Debt": 20, "Gold": 10}
        else: # Balanced
            return {"Equity": 50, "Debt": 30, "Gold": 20}

class AdvisorAgent:
    """Agent 3: The LLM that synthesizes everything"""
    def explain(self, context, prompt):
        full_prompt = f"""
        Context from other Agents: {context}
        User Question: {prompt}
        Answer as a Senior Portfolio Manager. Concise.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.choices[0].message.content

# Instantiate Agents
analyst = MarketAnalystAgent()
planner = PlannerAgent()
advisor = AdvisorAgent()

# =========================
# 4. STREAMLIT FRONTEND (The Interface)
# =========================

st.sidebar.image("https://placehold.co/100x100/png?text=Agent+Hive", caption="Multi-Agent System")
mode = st.sidebar.radio("Active Agent Module", ["Dashboard", "Stock Analyst", "Portfolio Planner", "Memory Search (RAG)"])

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_01" # Simulating a logged-in user

# --- DASHBOARD ---
if mode == "Dashboard":
    st.title("🧠 AI Financial Hive")
    st.markdown("""
    This system uses **advanced vector memory (Pinecone)** and a **Multi-Agent Architecture**.
    
    **Active Agents:**
    * 📊 **MarketAnalyst:** Mathematical computation engine.
    * ⚖️ **Planner:** Asset allocation logic engine.
    * 💬 **Advisor:** Generative reasoning engine.
    * 🧠 **Memory:** RAG system for retrieving past financial context.
    """)
    
    # Live Market Check
    sentiment = "NEUTRAL"
    try:
        nifty = yf.Ticker("^NSEI")
        change = nifty.history(period="2d")["Close"].pct_change().iloc[-1]
        sentiment = "BULLISH 🟢" if change > 0 else "BEARISH 🔴"
    except: pass
    
    st.metric("Global Market Agent Signal", sentiment)

# --- AGENT 1: ANALYST ---
elif mode == "Stock Analyst":
    st.header("📊 Analyst Agent")
    symbol = st.text_input("Ticker Symbol (e.g., AAPL, RELIANCE.NS)")
    
    if st.button("Trigger Analysis"):
        with st.spinner("Analyst Agent is calculating..."):
            data = analyst.analyze_stock(symbol)
            if data:
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", data['price'])
                c2.metric("Trend", data['trend'])
                c3.metric("RSI", data['rsi'])
                
                st.line_chart(data['data']['Close'])
                
                # Handover to Advisor Agent
                explanation = advisor.explain(str(data), f"Is {symbol} a good buy right now?")
                st.info(f"**Advisor Agent:** {explanation}")
                
                # Save to Memory
                save_text = f"Analyzed {symbol}. Trend: {data['trend']}, Price: {data['price']}"
                save_memory(st.session_state.user_id, save_text, {"type": "analysis", "symbol": symbol, "text": save_text})
                st.success("✅ Analysis saved to Long-Term Vector Memory")

# --- AGENT 2: PLANNER ---
elif mode == "Portfolio Planner":
    st.header("⚖️ Planner Agent")
    risk = st.slider("Risk Level (1-20)", 1, 20, 10)
    
    if st.button("Generate Plan"):
        # 1. Check Memory for past plans (RAG)
        past_plans = query_memory(st.session_state.user_id, "portfolio allocation plan")
        if past_plans:
            st.warning("🧠 **Memory Recall:** I found similar plans you created before:")
            for p in past_plans:
                st.caption(f"- {p}")
        
        # 2. Create New Plan
        sentiment = "NEUTRAL" # Simplified
        plan = planner.create_portfolio(risk, sentiment)
        
        # 3. Visualize
        df = pd.DataFrame(list(plan.items()), columns=["Asset", "Pct"])
        fig = go.Figure(data=[go.Pie(labels=df['Asset'], values=df['Pct'])])
        st.plotly_chart(fig)
        
        # 4. Save to Memory
        save_text = f"Portfolio Plan created. Risk: {risk}. Allocation: {plan}"
        save_memory(st.session_state.user_id, save_text, {"type": "portfolio", "risk": risk, "text": save_text})

# --- AGENT 3: MEMORY SEARCH (RAG) ---
elif mode == "Memory Search (RAG)":
    st.header("🧠 Knowledge Retrieval Agent")
    query = st.text_input("Ask your financial history (e.g., 'What stocks did I analyze last week?')")
    
    if st.button("Search Vector DB"):
        results = query_memory(st.session_state.user_id, query)
        if results:
            for r in results:
                st.success(r)
        else:
            st.info("No relevant memories found in Pinecone.")
