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
# 1. ORCHESTRATOR CONFIGURATION
# =========================
st.set_page_config(page_title="AI Investment Hive", layout="wide", page_icon="🧠")

# --- Load API Keys ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    pc_key = st.secrets["PINECONE_API_KEY"]
except:
    api_key = os.getenv("OPENAI_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    st.error("❌ Critical Error: API Keys missing. Please set OPENAI_API_KEY and PINECONE_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini" 

# =========================
# 2. HELPER FUNCTIONS
# =========================
def encode_image(uploaded_file):
    """Converts uploaded file to Base64 for OpenAI Vision"""
    if uploaded_file is not None:
        return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    return None

# =========================
# 3. AGENT DEFINITIONS
# =========================

class MemoryAgent:
    """🧠 THE BRAIN: Vector Database Manager"""
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        # Auto-create Index (1024 dimensions)
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024, 
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(5)
            except: pass
        self.index = self.pc.Index(self.index_name)

    def memorize(self, text, metadata):
        try:
            # Explicitly request 1024 dimensions
            response = client.embeddings.create(input=text, model="text-embedding-3-small", dimensions=1024)
            vector = response.data[0].embedding
            unique_id = f"mem_{int(time.time())}"
            clean_meta = {k: str(v) for k, v in metadata.items() if v is not None}
            clean_meta['text'] = text
            self.index.upsert(vectors=[(unique_id, vector, clean_meta)])
            return True
        except Exception as e:
            st.error(f"Memory Save Error: {e}")
            return False

    def recall(self, query, top_k=3):
        try:
            response = client.embeddings.create(input=query, model="text-embedding-3-small", dimensions=1024)
            query_vector = response.data[0].embedding
            results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            return [match['metadata'] for match in results['matches']]
        except:
            return []

class AnalystAgent:
    """📊 THE QUANT: Technical Analysis"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # Indicators
            df["MA50"] = df["Close"].rolling(50).mean()
            current_price = df["Close"].iloc[-1]
            trend = "BULLISH" if current_price > df["MA50"].iloc[-1] else "BEARISH"
            
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            info = stock.get_info()
            return {
                "symbol": symbol, "price": round(current_price, 2),
                "trend": trend, "rsi": round(rsi.iloc[-1], 2),
                "history_df": df
            }
        except: return None

    def check_market_sentiment(self):
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            change = (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
            return "BULLISH" if change > 0 else "BEARISH"
        except: return "NEUTRAL"

class PlannerAgent:
    """⚖️ THE STRATEGIST: Asset Allocation"""
    def create_allocation(self, risk_level, market_sentiment):
        if market_sentiment == "BEARISH": return {"Equity": 30, "Debt": 50, "Gold": 20}
        if risk_level >= 15: return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk_level >= 8: return {"Equity": 50, "Debt": 30, "Gold": 20}
        else: return {"Equity": 30, "Debt": 50, "Gold": 20}

class ConversationalAgent:
    """🎓 THE TUTOR: Multi-Modal Chat"""
    def chat(self, user_input, history, image_base64=None):
        messages = [{"role": "system", "content": "You are FinBot, an expert financial analyst."}]
        for msg in history:
            if isinstance(msg["content"], str): messages.append(msg)
        
        if image_base64:
            user_message = {"role": "user", "content": [{"type": "text", "text": user_input}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
        else:
            user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        
        try:
            response = client.chat.completions.create(model=MODEL_VERSION, messages=messages, max_tokens=500)
            return response.choices[0].message.content
        except Exception as e: return f"⚠️ Error: {e}"

# --- Initialize ---
memory_node = MemoryAgent(pc_key)
analyst_node = AnalystAgent()
planner_node = PlannerAgent()
chat_node = ConversationalAgent()

# =========================
# 4. FRONTEND ORCHESTRATOR
# =========================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "profile" not in st.session_state: st.session_state.profile = {"name": "Investor", "risk": 10}
if "last_analysis" not in st.session_state: st.session_state.last_analysis = None

st.sidebar.title("🤖 Agent Navigation")
page = st.sidebar.radio("Module", ["Home", "Profile Setup", "Stock Analysis", "AI Decision", "Portfolio Allocation", "Financial Planning", "Education (Chatbot)", "Memory Logs"])

if page == "Home":
    st.title("🤖 AI Investment Hive")
    st.image("https://placehold.co/1000x400/png?text=Agent+Orchestrator+Active", caption="System Online")
    st.success("System Status: All Agents Operational")

elif page == "Profile Setup":
    st.header("👤 Profile Agent")
    # FIXED: Renamed form to 'profile_form' to avoid key collision
    with st.form("profile_form"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", "Investor"))
        risk = st.slider("Risk Appetite", 1, 20, st.session_state.profile.get("risk", 10))
        income = st.number_input("Monthly Income", value=st.session_state.profile.get("income", 50000))
        savings = st.number_input("Monthly Savings", value=st.session_state.profile.get("savings", 10000))
        if st.form_submit_button("Save Profile"):
            st.session_state.profile = {"name": name, "risk": risk, "income": income, "savings": savings}
            st.success("Profile Updated!")

elif page == "Stock Analysis":
    st.header("📊 Analyst Agent")
    symbol = st.text_input("Symbol", "RELIANCE.NS")
    if st.button("Run Analysis"):
        with st.status("Analyst working..."):
            data = analyst_node.analyze(symbol)
            if data:
                st.session_state.last_analysis = data
                st.metric("Price", f"₹{data['price']}")
                st.line_chart(data['history_df']['Close'])
                memory_node.memorize(f"Analyzed {symbol} at {data['price']}", {"type": "analysis"})
            else: st.error("Symbol not found.")

elif page == "AI Decision":
    st.header("🧠 Decision Agent")
    if st.session_state.last_analysis:
        data = st.session_state.last_analysis
        rec = "BUY" if data['trend'] == "BULLISH" and data['rsi'] < 70 else "HOLD"
        st.subheader(f"Recommendation: {rec}")
        st.write(f"Trend: {data['trend']}, RSI: {data['rsi']}")
        if st.button("Ask Advisor Why?"):
            st.write(chat_node.chat(f"Why {rec} for {data['symbol']}?", []))
    else: st.warning("Run Analysis first.")

elif page == "Portfolio Allocation":
    st.header("💼 Planner Agent")
    cap = st.number_input("Capital", 10000)
    if st.button("Generate Plan"):
        sent = analyst_node.check_market_sentiment()
        risk = st.session_state.profile.get("risk", 10)
        alloc = planner_node.create_allocation(risk, sent)
        st.write(f"Market: {sent}")
        st.json(alloc)
        fig = go.Figure(data=[go.Pie(labels=list(alloc.keys()), values=list(alloc.values()))])
        st.plotly_chart(fig)
        memory_node.memorize(f"Plan: {alloc}", {"type": "plan"})

elif page == "Financial Planning":
    st.header("🔮 Wealth Agent")
    sip = st.number_input("SIP", 5000)
    yrs = st.slider("Years", 1, 30, 10)
    future = sip * (((1+0.01)**(yrs*12)-1)/0.01)
    st.metric("Future Value", f"₹{future:,.0f}")
    st.area_chart([sip * (((1+0.01)**m-1)/0.01) for m in range(1, yrs*12+1)])

elif page == "Education (Chatbot)":
    st.header("🎓 Tutor Agent")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str): st.write(msg["content"])
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Chart", type=["jpg", "png"])
        img_b64 = encode_image(uploaded_file)
    
    prompt = st.chat_input("Ask FinBot...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            resp = chat_node.chat(prompt, st.session_state.chat_history[:-1], img_b64)
            st.write(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})

elif page == "Memory Logs":
    st.header("📜 Agent Memory")
    if st.button("Refresh"):
        logs = memory_node.recall("investment", 5)
        for l in logs: st.info(l.get('text'))
