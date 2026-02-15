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
from pypdf import PdfReader

# =========================
# 1. SYSTEM CONFIGURATION
# =========================
st.set_page_config(page_title="AI Financial Super App", layout="wide", page_icon="🚀")

# --- Load Secrets ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    pc_key = st.secrets["PINECONE_API_KEY"]
except:
    api_key = os.getenv("OPENAI_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    st.error("❌ Critical: API Keys missing.")
    st.stop()

client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini"

# =========================
# 2. HELPER FUNCTIONS
# =========================
def encode_image(uploaded_file):
    if uploaded_file is not None:
        return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    return None

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages[:5]:
                text += page.extract_text()
            return text
        except: return None
    return None

# =========================
# 3. AGENT CLASSES
# =========================

class MemoryAgent:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        # Auto-heal index
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
            return True
        except: return False

    def recall(self, query, top_k=3):
        try:
            response = client.embeddings.create(input=query, model="text-embedding-3-small", dimensions=1024)
            vector = response.data[0].embedding
            results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
            return [match['metadata']['text'] for match in results['matches']]
        except: return []

class AnalystAgent:
    """📊 THE QUANT: Advanced Technicals with Charts & Reasoning"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # 1. Calc Indicators
            df["MA50"] = df["Close"].rolling(50).mean()
            df["MA200"] = df["Close"].rolling(200).mean()
            
            curr = df["Close"].iloc[-1]
            ma50 = df["MA50"].iloc[-1]
            ma200 = df["MA200"].iloc[-1]
            
            # RSI Calc
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rs = gain.rolling(14).mean() / loss.rolling(14).mean()
            df["RSI"] = 100 - (100 / (1 + rs))
            rsi = df["RSI"].iloc[-1]
            
            # 2. Logic & Reasoning Engine
            signals = []
            
            # Trend Logic
            if curr > ma50:
                trend = "BULLISH"
                signals.append("✅ Price is above 50-Day MA (Short-term Bullish).")
            else:
                trend = "BEARISH"
                signals.append("❌ Price is below 50-Day MA (Short-term Bearish).")
            
            # Golden Cross Logic (Safe Check)
            if pd.notna(ma200):
                if ma50 > ma200:
                    signals.append("✅ Golden Cross active (50 MA > 200 MA).")
            
            # Momentum Logic
            if rsi < 30:
                signals.append("🟢 RSI is Oversold (<30). Potential reversal buy.")
                verdict = "STRONG BUY"
            elif rsi > 70:
                signals.append("🔴 RSI is Overbought (>70). Potential pullback.")
                verdict = "SELL/WAIT"
            else:
                signals.append("⚪ RSI is Neutral.")
                verdict = "BUY" if trend == "BULLISH" else "HOLD"

            reasoning = " ".join(signals)
            
            return {
                "symbol": symbol, 
                "price": round(curr, 2),
                "trend": trend, 
                "rsi": round(rsi, 2),
                "verdict": verdict,
                "reasoning": reasoning,
                "history_df": df
            }
        except Exception as e: 
            print(e)
            return None

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
    def chat(self, user_input, history, image_base64=None, pdf_text=None, system_context=""):
        sys_msg = f"""
        You are FinBot. CONTEXT: {system_context}
        Analyze charts/PDFs if provided. Be concise.
        """
        messages = [{"role": "system", "content": sys_msg}]
        for msg in history:
            if isinstance(msg["content"], str): messages.append(msg)
        
        content_payload = [{"type": "text", "text": user_input}]
        if image_base64:
            content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
        if pdf_text:
            content_payload.append({"type": "text", "text": f"\n\n[DOC]: {pdf_text[:4000]}..."})

        messages.append({"role": "user", "content": content_payload})
        
        try:
            res = client.chat.completions.create(model=MODEL_VERSION, messages=messages, max_tokens=700)
            return res.choices[0].message.content
        except Exception as e: return f"AI Error: {e}"

# --- Initialize ---
memory = MemoryAgent(pc_key)
analyst = AnalystAgent()
planner = PlannerAgent()
tutor = ConversationalAgent()

# =========================
# 4. UI ORCHESTRATOR
# =========================

if "profile_created" not in st.session_state: st.session_state.profile_created = False
if "profile" not in st.session_state: st.session_state.profile = None
if "analysis" not in st.session_state: st.session_state.analysis = None
if "portfolio" not in st.session_state: st.session_state.portfolio = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🚀 AI Financial Super App")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Profile", "📈 Market Analyst", "💼 Strategy", "💬 AI Lab", "📊 Dashboard"
])

# === TAB 1: PROFILE ===
with tab1:
    st.subheader("Identity Layer")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Investor Name", "User")
            income = st.number_input("Monthly Income", value=50000)
        with c2:
            savings = st.number_input("Monthly Savings", value=10000)
            risk = st.slider("Risk Tolerance", 1, 20, 10)
        
        if st.button("Initialize Profile"):
            st.session_state.profile = {"name": name, "risk": risk, "income": income}
            st.session_state.profile_created = True
            st.success("Identity Verified.")

# === TAB 2: MARKET ANALYST ===
with tab2:
    if not st.session_state.profile_created:
        st.warning("Please complete Profile first.")
    else:
        st.subheader("Market Intelligence Layer")
        
        # Search Bar
        c1, c2 = st.columns([1, 3])
        with c1:
            sym = st.text_input("Stock Symbol", "RELIANCE.NS")
            if st.button("Run Deep Analysis"):
                with st.spinner("Analyst Agent working..."):
                    # CLEAR OLD DATA TO PREVENT KEYERROR
                    st.session_state.analysis = None 
                    data = analyst.analyze(sym)
                    if data:
                        st.session_state.analysis = data
                        memory.memorize(f"Analyzed {sym}. Verdict: {data.get('verdict','N/A')}", {"type": "analysis"})
                    else: st.error("Symbol not found")
        
        # Results Display
        with c2:
            if st.session_state.analysis:
                d = st.session_state.analysis
                
                # 1. High Level Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"₹{d['price']}")
                m2.metric("RSI (Momentum)", d['rsi'])
                m3.metric("Trend", d['trend'])
                
                # Safe Verdict
                verdict = d.get('verdict', 'HOLD')
                v_color = "green" if "BUY" in verdict else "red"
                m4.markdown(f"**Verdict:** :{v_color}[{verdict}]")
                
                # 2. Reasoning Box
                reasoning = d.get('reasoning', 'Analysis complete.')
                st.info(f"**💡 Analyst Reasoning:** {reasoning}")

                # 3. Advanced Charts (Plotly)
                tab_chart1, tab_chart2 = st.tabs(["🕯️ Technical Chart", "📉 Momentum (RSI)"])
                
                with tab_chart1:
                    # Candlestick + MA (Safe Check)
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=d['history_df'].index,
                                    open=d['history_df']['Open'], high=d['history_df']['High'],
                                    low=d['history_df']['Low'], close=d['history_df']['Close'], name='Price'))
                    
                    # SAFEGUARD: Only plot MA200 if it exists in columns
                    if 'MA50' in d['history_df'].columns:
                        fig.add_trace(go.Scatter(x=d['history_df'].index, y=d['history_df']['MA50'], 
                                                 line=dict(color='blue', width=1), name='50 Day MA'))
                    
                    if 'MA200' in d['history_df'].columns:
                        fig.add_trace(go.Scatter(x=d['history_df'].index, y=d['history_df']['MA200'], 
                                                 line=dict(color='orange', width=1), name='200 Day MA'))
                    
                    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab_chart2:
                    # RSI Chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=d['history_df'].index, y=d['history_df']['RSI'], 
                                                 line=dict(color='purple', width=2), name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_rsi, use_container_width=True)

# === TAB 3: STRATEGY ===
with tab3:
    if not st.session_state.analysis:
        st.warning("Analyze a stock first.")
    else:
        st.subheader("Strategic Planning")
        cap = st.number_input("Deployment Capital", 100000)
        if st.button("Generate Allocation"):
            sent = analyst.get_sentiment()
            risk = st.session_state.profile['risk']
            alloc =
