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
import json

# =========================
# 1. ORCHESTRATOR CONFIGURATION
# =========================
st.set_page_config(page_title="AI Investment Intelligence (v5.0)", layout="wide", page_icon="🧠")

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
# 3. AGENT DEFINITIONS (The "Workers")
# =========================

class MemoryAgent:
    """🧠 THE BRAIN: Manages Long-Term Vector Memory (Pinecone)"""
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        # Check/Create Index
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
    """📊 THE QUANT: Technicals + Monte Carlo + Confidence"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None

            # 1. Technical Indicators
            df["MA50"] = df["Close"].rolling(50).mean()
            df["MA200"] = df["Close"].rolling(200).mean()
            current_price = df["Close"].iloc[-1]
            
            # Trend Logic
            trend = "BULLISH" if current_price > df["MA50"].iloc[-1] else "BEARISH"
            golden_cross = df["MA50"].iloc[-1] > df["MA200"].iloc[-1]
            
            # RSI Logic
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]

            # MACD Logic
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_bullish = macd.iloc[-1] > signal.iloc[-1]

            # 2. Confidence Scoring Agent Logic
            score = 0
            if trend == "BULLISH": score += 1
            if golden_cross: score += 1
            if rsi_val < 70 and rsi_val > 30: score += 1 # Healthy range
            if macd_bullish: score += 1
            
            confidence = round((score / 4) * 100, 2)
            
            # 3. Monte Carlo Simulation (Predictive)
            returns = df["Close"].pct_change().dropna()
            simulations = 100
            sim_results = []
            for _ in range(simulations):
                price = current_price
                for _ in range(252): # 1 trading year
                    price *= (1 + np.random.choice(returns))
                sim_results.append(price)
            
            mc_low, mc_med, mc_high = np.percentile(sim_results, [5, 50, 95])

            return {
                "symbol": symbol, 
                "price": round(current_price, 2),
                "trend": trend, 
                "rsi": round(rsi_val, 2),
                "confidence": confidence,
                "mc_forecast": {"low": round(mc_low, 2), "med": round(mc_med, 2), "high": round(mc_high, 2)},
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
    """⚖️ THE STRATEGIST: Asset Allocation & Rebalancing"""
    def create_allocation(self, risk_level, market_sentiment):
        if market_sentiment == "BEARISH": return {"Equity": 30, "Debt": 50, "Gold": 20}
        
        if risk_level >= 15: return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk_level >= 8: return {"Equity": 50, "Debt": 30, "Gold": 20}
        else: return {"Equity": 30, "Debt": 50, "Gold": 20}

    def rebalance(self, current_holdings, target_allocation, total_value):
        """Calculates buy/sell amounts to match target"""
        adjustments = {}
        for asset, target_pct in target_allocation.items():
            target_amt = total_value * (target_pct / 100)
            current_amt = current_holdings.get(asset, 0)
            diff = target_amt - current_amt
            action = "BUY" if diff > 0 else "SELL"
            adjustments[asset] = {"Action": action, "Amount": abs(round(diff, 2))}
        return adjustments

class ConversationalAgent:
    """🎓 THE TUTOR: Multi-Modal Chat"""
    def chat(self, user_input, history, image_base64=None, system_context=""):
        messages = [{"role": "system", "content": f"You are FinBot (Investment Intelligence). Context: {system_context}"}]
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
if "user_profile_data" not in st.session_state: st.session_state.user_profile_data = {"name": "Investor", "risk": 10}
if "stock_analysis_data" not in st.session_state: st.session_state.stock_analysis_data = None

st.sidebar.title("🤖 Investment Intelligence")
page = st.sidebar.radio("Modules", ["Home", "Profile Setup", "Stock Analysis", "AI Decision", "Portfolio & Rebalance", "Financial Planning", "Education (Vision)", "Memory Logs"])

# --- HOME ---
if page == "Home":
    st.title("🤖 AI Investment Intelligence Platform v5.0")
    st.image("https://placehold.co/1000x400/png?text=Advanced+Multi-Agent+System", caption="System Online")
    st.info("System Capabilities: Monte Carlo Simulation • Confidence Scoring • Auto-Rebalancing • Vision Analysis • Vector Memory")

# --- PROFILE ---
elif page == "Profile Setup":
    st.header("👤 Profile Agent")
    with st.form("profile_form_ui"):
        name = st.text_input("Name", value=st.session_state.user_profile_data.get("name", "Investor"))
        risk = st.slider("Risk Appetite", 1, 20, st.session_state.user_profile_data.get("risk", 10))
        income = st.number_input("Monthly Income", value=st.session_state.user_profile_data.get("income", 50000))
        savings = st.number_input("Monthly Savings", value=st.session_state.user_profile_data.get("savings", 10000))
        if st.form_submit_button("Save Profile"):
            st.session_state.user_profile_data = {"name": name, "risk": risk, "income": income, "savings": savings}
            st.success("Profile Updated!")

# --- STOCK ANALYSIS ---
elif page == "Stock Analysis":
    st.header("📊 Analyst Agent")
    symbol = st.text_input("Symbol", "RELIANCE.NS")
    if st.button("Run Advanced Analysis"):
        with st.status("Analyst Agent is running protocols..."):
            data = analyst_node.analyze(symbol)
            if data:
                st.session_state.stock_analysis_data = data
                st.success("Analysis Complete")
                
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"₹{data['price']}")
                c2.metric("Trend", data['trend'])
                c3.metric("RSI", data['rsi'])
                c4.metric("Confidence", f"{data['confidence']}%")
                
                st.line_chart(data['history_df']['Close'])
                
                # Monte Carlo Display
                st.subheader("🎲 Monte Carlo Forecast (1 Year)")
                mc = data['mc_forecast']
                col1, col2, col3 = st.columns(3)
                col1.error(f"Worst Case (5%): ₹{mc['low']}")
                col2.info(f"Median Case: ₹{mc['med']}")
                col3.success(f"Best Case (95%): ₹{mc['high']}")
                
                memory_node.memorize(f"Analyzed {symbol}. Price: {data['price']}. Confidence: {data['confidence']}%.", {"type": "analysis"})
            else: st.error("Symbol not found.")

# --- AI DECISION ---
elif page == "AI Decision":
    st.header("🧠 Decision Agent")
    if st.session_state.stock_analysis_data:
        data = st.session_state.stock_analysis_data
        rec = "BUY" if data['confidence'] > 70 else "HOLD" if data['confidence'] > 40 else "SELL"
        color = "green" if rec == "BUY" else "orange" if rec == "HOLD" else "red"
        
        st.markdown(f"<h1 style='color:{color};'>Recommendation: {rec}</h1>", unsafe_allow_html=True)
        st.write(f"**Confidence Score:** {data['confidence']}% based on Multi-Factor Analysis (Trend, RSI, Golden Cross, MACD).")
        
        if st.button("Get Detailed Report"):
            ctx = f"Stock: {data['symbol']}, Price: {data['price']}, Trend: {data['trend']}, RSI: {data['rsi']}, Monte Carlo High: {data['mc_forecast']['high']}"
            st.write(chat_node.chat(f"Write a investment report for {rec}.", [], system_context=ctx))
    else: st.warning("Run Analysis first.")

# --- PORTFOLIO & REBALANCE ---
elif page == "Portfolio & Rebalance":
    st.header("⚖️ Planner Agent")
    mode = st.radio("Mode", ["New Portfolio", "Auto-Rebalance"])
    
    if mode == "New Portfolio":
        cap = st.number_input("Capital", 10000)
        if st.button("Generate Strategy"):
            sent = analyst_node.check_market_sentiment()
            risk = st.session_state.user_profile_data.get("risk", 10)
            alloc = planner_node.create_allocation(risk, sent)
            st.write(f"Market Sentiment: **{sent}**")
            fig = go.Figure(data=[go.Pie(labels=list(alloc.keys()), values=list(alloc.values()))])
            st.plotly_chart(fig)
            memory_node.memorize(f"Plan: {alloc}", {"type": "plan"})
            
    elif mode == "Auto-Rebalance":
        st.subheader("🔄 Portfolio Rebalancing")
        cap = st.number_input("Total Portfolio Value", 100000)
        c1, c2, c3 = st.columns(3)
        curr_eq = c1.number_input("Current Equity ₹", value=40000)
        curr_db = c2.number_input("Current Debt ₹", value=40000)
        curr_gd = c3.number_input("Current Gold ₹", value=20000)
        
        if st.button("Calculate Rebalancing"):
            sent = analyst_node.check_market_sentiment()
            risk = st.session_state.user_profile_data.get("risk", 10)
            target = planner_node.create_allocation(risk, sent)
            
            st.write("### Target Allocation")
            st.json(target)
            
            current_holdings = {"Equity": curr_eq, "Debt": curr_db, "Gold": curr_gd}
            adjustments = planner_node.rebalance(current_holdings, target, cap)
            
            st.subheader("🛠️ Required Actions")
            for asset, action in adjustments.items():
                color = "green" if action['Action'] == "BUY" else "red"
                st.markdown(f"**{asset}**: :{color}[{action['Action']}] ₹{action['Amount']}")

# --- FINANCIAL PLANNING ---
elif page == "Financial Planning":
    st.header("🔮 Wealth Agent")
    sip = st.number_input("SIP", 5000)
    yrs = st.slider("Years", 1, 30, 10)
    future = sip * (((1+0.12/12)**(yrs*12)-1)/(0.12/12)) # Assumed 12%
    st.metric("Proj. Wealth (12%)", f"₹{future:,.0f}")
    st.area_chart([sip * (((1+0.12/12)**m-1)/(0.12/12)) for m in range(1, yrs*12+1)])

# --- EDUCATION (VISION) ---
elif page == "Education (Vision)":
    st.header("🎓 Tutor Agent (Multi-Modal)")
    
    # Context Injection
    ctx_note = ""
    if st.session_state.stock_analysis_data:
        d = st.session_state.stock_analysis_data
        ctx_note = f"User is looking at {d['symbol']} (Confidence: {d['confidence']}%)."
        st.info(f"💡 Context: Analyzing **{d['symbol']}**")

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
            resp = chat_node.chat(prompt, st.session_state.chat_history[:-1], img_b64, system_context=ctx_note)
            st.write(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})

# --- MEMORY LOGS ---
elif page == "Memory Logs":
    st.header("📜 Agent Memory")
    if st.button("Refresh"):
        logs = memory_node.recall("investment", 5)
        for l in logs: st.info(l.get('text'))

st.sidebar.divider()
st.sidebar.caption("System: Multi-Agent RAG + Vision v5.0")
