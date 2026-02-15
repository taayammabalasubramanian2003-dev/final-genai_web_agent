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
# 2. AGENT CLASSES
# =========================

class MemoryAgent:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        # Auto-create Index logic
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
        messages = [{"role": "system", "content": f"You are FinBot. Context: {system_context}. Be concise."}]
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
# 3. UI ORCHESTRATOR (Horizontal Layout)
# =========================

# Session State Init
if "profile_created" not in st.session_state: st.session_state.profile_created = False
if "profile" not in st.session_state: st.session_state.profile = None
if "analysis" not in st.session_state: st.session_state.analysis = None
if "portfolio" not in st.session_state: st.session_state.portfolio = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🚀 AI Investment Journey")

# --- HORIZONTAL NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Profile", 
    "📈 Stock Analysis", 
    "💼 Portfolio Plan", 
    "💬 AI Chatbot", 
    "📊 Dashboard"
])

# =========================
# TAB 1: PROFILE
# =========================
with tab1:
    st.subheader("Step 1: Who are you?")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", value="Investor")
            income = st.number_input("Monthly Income", value=50000)
        with c2:
            savings = st.number_input("Monthly Savings", value=10000)
            risk = st.slider("Risk Appetite (1=Safe, 20=Risky)", 1, 20, 10)
        
        if st.button("Save Profile"):
            st.session_state.profile = {"name": name, "risk": risk, "income": income, "savings": savings}
            st.session_state.profile_created = True
            st.success("✅ Profile Saved! You can now move to the 'Stock Analysis' tab.")

# =========================
# TAB 2: STOCK ANALYSIS
# =========================
with tab2:
    if not st.session_state.profile_created:
        st.warning("🔒 Locked. Please save your Profile in the first tab.")
    else:
        st.subheader("Step 2: Market Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            symbol = st.text_input("Enter Stock Symbol", "RELIANCE.NS")
            if st.button("Analyze Stock"):
                with st.spinner("Agents Analyzing..."):
                    data = analyst.analyze(symbol)
                    if data:
                        st.session_state.analysis = data
                        memory.memorize(f"Analyzed {symbol} at {data['price']}", {"type": "analysis"})
                    else:
                        st.error("Symbol not found.")

        with col2:
            if st.session_state.analysis:
                d = st.session_state.analysis
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"₹{d['price']}")
                m2.metric("Trend", d['trend'])
                m3.metric("RSI", d['rsi'])
                
                st.line_chart(d['history_df']['Close'])
                
                # Verdict
                rec = "BUY" if d['trend'] == "BULLISH" and d['rsi'] < 70 else "HOLD"
                color = "green" if rec == "BUY" else "orange"
                st.markdown(f"### 🧠 AI Verdict: :{color}[{rec}]")
                st.success("Analysis Complete! You can now check the 'Portfolio Plan' tab.")

# =========================
# TAB 3: PORTFOLIO
# =========================
with tab3:
    if not st.session_state.analysis:
        st.warning("🔒 Locked. Please Analyze a stock first.")
    else:
        st.subheader("Step 3: Strategic Allocation")
        capital = st.number_input("Investment Capital (₹)", 10000, 10000000, 100000)
        
        if st.button("Generate Strategy"):
            sent = analyst.get_sentiment()
            risk = st.session_state.profile['risk']
            alloc = planner.create_allocation(risk, sent)
            st.session_state.portfolio = {"alloc": alloc, "sentiment": sent, "capital": capital}
            
        if st.session_state.portfolio:
            p = st.session_state.portfolio
            st.info(f"Based on **{p['sentiment']}** Market & **Risk Level {st.session_state.profile['risk']}**")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write("#### Asset Split")
                st.json(p['alloc'])
            with c2:
                fig = go.Figure(data=[go.Pie(labels=list(p['alloc'].keys()), values=list(p['alloc'].values()))])
                st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 4: CHATBOT (Enhanced)
# =========================
with tab4:
    if not st.session_state.portfolio:
        st.warning("🔒 Recommend generating a portfolio plan first for better context.")
    
    st.subheader("💬 AI Financial Tutor")
    
    # Context Construction
    ctx = ""
    if st.session_state.profile: ctx += f"User: {st.session_state.profile['name']}. "
    if st.session_state.analysis: ctx += f"Stock: {st.session_state.analysis['symbol']} ({st.session_state.analysis['trend']}). "
    if st.session_state.portfolio: ctx += f"Plan: {st.session_state.portfolio['alloc']}. "

    # Chat UI
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Quick Suggestion Chips
    st.write("pop-up questions:")
    b1, b2, b3 = st.columns(3)
    if b1.button("Why is the trend Bullish?"):
        prompt = "Why is the trend Bullish?"
    elif b2.button("Explain RSI simply"):
        prompt = "Explain RSI simply"
    elif b3.button("Is my portfolio safe?"):
        prompt = "Is my portfolio safe?"
    else:
        prompt = st.chat_input("Type your question here...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = tutor.chat(prompt, st.session_state.chat_history, system_context=ctx)
                st.write(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()

# =========================
# TAB 5: DASHBOARD
# =========================
with tab5:
    st.subheader("🏁 Session Summary")
    
    if st.session_state.profile:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("👤 Investor", st.session_state.profile['name'])
        with c2:
            val = st.session_state.analysis['symbol'] if st.session_state.analysis else "N/A"
            st.metric("📉 Last Stock", val)
        with c3:
            val = f"₹{st.session_state.portfolio['capital']}" if st.session_state.portfolio else "N/A"
            st.metric("💰 Planned Capital", val)
            
        if st.button("💾 Save Full Session to Memory"):
            memory.memorize(f"Session for {st.session_state.profile['name']}", {"type": "session"})
            st.success("Session Saved!")
    else:
        st.info("Start by creating a profile in Tab 1.")
