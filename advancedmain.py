import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
import datetime
import time
import base64

# =========================
# 1. CONFIGURATION
# =========================
st.set_page_config(page_title="AI Investment Journey", layout="wide", page_icon="🚀")

# --- Load API Keys ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    pc_key = st.secrets["PINECONE_API_KEY"]
except:
    api_key = os.getenv("OPENAI_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    st.error("❌ Critical Error: API Keys missing.")
    st.stop()

client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini"

# =========================
# 2. AGENT CLASSES (The Logic Layer)
# =========================

class MemoryAgent:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            try:
                self.pc.create_index(
                    name=self.index_name, 
                    dimension=1024, 
                    metric='cosine', 
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(2)
            except: pass
        self.index = self.pc.Index(self.index_name)

    def memorize(self, text, metadata):
        try:
            response = client.embeddings.create(input=text, model="text-embedding-3-small", dimensions=1024)
            vector = response.data[0].embedding
            id = f"mem_{int(time.time())}"
            clean_meta = {k: str(v) for k, v in metadata.items() if v is not None}
            clean_meta['text'] = text
            self.index.upsert(vectors=[(id, vector, clean_meta)])
        except: pass

class AnalystAgent:
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # Technicals
            df["MA50"] = df["Close"].rolling(50).mean()
            curr = df["Close"].iloc[-1]
            trend = "BULLISH" if curr > df["MA50"].iloc[-1] else "BEARISH"
            
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                "symbol": symbol, "price": round(curr, 2),
                "trend": trend, "rsi": round(rsi.iloc[-1], 2),
                "history_df": df
            }
        except: return None

    def get_sentiment(self):
        try:
            hist = yf.Ticker("^NSEI").history(period="5d")
            return "BULLISH" if hist["Close"].iloc[-1] > hist["Close"].iloc[0] else "BEARISH"
        except: return "NEUTRAL"

class PlannerAgent:
    def create_allocation(self, risk, sentiment):
        if sentiment == "BEARISH": return {"Equity": 30, "Debt": 50, "Gold": 20}
        if risk >= 15: return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk >= 8: return {"Equity": 50, "Debt": 30, "Gold": 20}
        return {"Equity": 30, "Debt": 50, "Gold": 20}

class ConversationalAgent:
    def chat(self, user_input, history, system_context=""):
        messages = [{"role": "system", "content": f"You are FinBot. Context: {system_context}"}]
        for msg in history:
            if isinstance(msg["content"], str): messages.append(msg)
        messages.append({"role": "user", "content": user_input})
        
        try:
            res = client.chat.completions.create(model=MODEL_VERSION, messages=messages)
            return res.choices[0].message.content
        except: return "I'm having trouble thinking right now."

# --- Initialize Agents ---
memory = MemoryAgent(pc_key)
analyst = AnalystAgent()
planner = PlannerAgent()
tutor = ConversationalAgent()

# =========================
# 3. UI ORCHESTRATOR (Linear Flow)
# =========================

# Session State Init
if "step" not in st.session_state: st.session_state.step = 1
if "profile" not in st.session_state: st.session_state.profile = None
if "analysis" not in st.session_state: st.session_state.analysis = None
if "portfolio" not in st.session_state: st.session_state.portfolio = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🚀 AI Investment Journey")
st.markdown("Follow the steps below to build your personalized financial plan.")

# --- STEP 1: PROFILE ---
st.header("Step 1: Your Investor Profile")
with st.container(border=True):
    if st.session_state.step >= 1:
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", value="Investor")
            income = st.number_input("Monthly Income", value=50000)
        with c2:
            savings = st.number_input("Monthly Savings", value=10000)
            risk = st.slider("Risk Appetite (1=Safe, 20=Risky)", 1, 20, 10)
        
        if st.button("Save Profile & Continue"):
            st.session_state.profile = {"name": name, "risk": risk, "income": income, "savings": savings}
            st.session_state.step = 2
            st.rerun()

    if st.session_state.step > 1:
        st.success(f"✅ Profile Active: {st.session_state.profile['name']} (Risk: {st.session_state.profile['risk']}/20)")

# --- STEP 2: STOCK ANALYSIS & DECISION ---
if st.session_state.step >= 2:
    st.divider()
    st.header("Step 2: Market Analysis & Decision")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            symbol = st.text_input("Enter Stock Symbol", "RELIANCE.NS")
            mode = st.radio("Mode", ["Investor (Long Term)", "Trader (Short Term)"])
            
            if st.button("Analyze Stock"):
                with st.spinner("Agents Analyzing..."):
                    data = analyst.analyze(symbol)
                    if data:
                        st.session_state.analysis = data
                        memory.memorize(f"Analyzed {symbol} at {data['price']}", {"type": "analysis"})
                        st.session_state.step = 3 # Unlock next step
                    else:
                        st.error("Symbol not found.")

        with col2:
            if st.session_state.analysis:
                d = st.session_state.analysis
                # Display Analysis
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"₹{d['price']}")
                c2.metric("Trend", d['trend'])
                c3.metric("RSI", d['rsi'])
                st.line_chart(d['history_df']['Close'])
                
                # AI DECISION LOGIC
                rec = "BUY" if d['trend'] == "BULLISH" and d['rsi'] < 70 else "HOLD"
                color = "green" if rec == "BUY" else "orange"
                st.markdown(f"### 🧠 AI Verdict: :{color}[{rec}]")
                st.caption(f"Reasoning: The trend is {d['trend']} and RSI indicates {d['rsi']} (Momentum).")

# --- STEP 3: PORTFOLIO ALLOCATION ---
if st.session_state.step >= 3:
    st.divider()
    st.header("Step 3: Portfolio Strategy")
    
    with st.container(border=True):
        capital = st.number_input("Investment Capital (₹)", 10000, 10000000, 100000)
        
        if st.button("Generate Allocation Plan"):
            sent = analyst.get_sentiment()
            risk = st.session_state.profile['risk']
            alloc = planner.create_allocation(risk, sent)
            st.session_state.portfolio = {"alloc": alloc, "sentiment": sent, "capital": capital}
            st.session_state.step = 4 # Unlock next step
            
        if st.session_state.portfolio:
            p = st.session_state.portfolio
            st.info(f"Market Sentiment: **{p['sentiment']}** | Strategy: **Dynamic Rebalancing**")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write("#### Recommended Split")
                st.json(p['alloc'])
            with c2:
                fig = go.Figure(data=[go.Pie(labels=list(p['alloc'].keys()), values=list(p['alloc'].values()))])
                st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: CHATBOT TUTOR ---
if st.session_state.step >= 4:
    st.divider()
    st.header("Step 4: Ask the Expert (Chat)")
    
    with st.container(border=True):
        # Context building
        ctx = f"User: {st.session_state.profile['name']}. "
        if st.session_state.analysis:
            ctx += f"Looking at {st.session_state.analysis['symbol']}. "
        if st.session_state.portfolio:
            ctx += f"Portfolio Plan: {st.session_state.portfolio['alloc']}. "
            
        # Chat Interface
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        prompt = st.chat_input("Ask a question about your plan or stocks...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = tutor.chat(prompt, st.session_state.chat_history, system_context=ctx)
                    st.write(resp)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})

# --- STEP 5: FINAL DASHBOARD ---
if st.session_state.step >= 4:
    st.divider()
    st.header("🏁 Session Dashboard")
    st.markdown("Summary of your investment journey today.")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("👤 You")
        st.write(f"**Name:** {st.session_state.profile['name']}")
        st.write(f"**Risk Profile:** {st.session_state.profile['risk']}/20")
    
    with c2:
        st.subheader("📉 Market")
        if st.session_state.analysis:
            st.write(f"**Focus Stock:** {st.session_state.analysis['symbol']}")
            st.write(f"**Trend:** {st.session_state.analysis['trend']}")
        else:
            st.write("No stock analyzed.")
            
    with c3:
        st.subheader("💰 Strategy")
        if st.session_state.portfolio:
            st.write(f"**Capital:** ₹{st.session_state.portfolio['capital']}")
            st.write(f"**Equity Allocation:** {st.session_state.portfolio['alloc']['Equity']}%")
        else:
            st.write("No plan generated.")
            
    if st.button("💾 Save Session to Memory"):
        memory.memorize(f"Session Summary for {st.session_state.profile['name']}", {"type": "session_summary"})
        st.success("Session saved to Pinecone Vector Database!")
