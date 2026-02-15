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
    """📊 THE QUANT: Returns MACD, RSI, and Trend"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # Indicators
            df["MA50"] = df["Close"].rolling(50).mean()
            curr = df["Close"].iloc[-1]
            trend = "BULLISH" if curr > df["MA50"].iloc[-1] else "BEARISH"
            
            # RSI
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rs = gain.rolling(14).mean() / loss.rolling(14).mean()
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]

            # MACD
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_val = macd.iloc[-1]
            signal_val = signal.iloc[-1]
            
            # Verdict Logic
            verdict = "HOLD"
            if trend == "BULLISH" and rsi_val < 70 and macd_val > signal_val:
                verdict = "BUY"
            elif trend == "BEARISH" or rsi_val > 70:
                verdict = "SELL"

            return {
                "symbol": symbol, 
                "price": round(curr, 2),
                "trend": trend, 
                "rsi": round(rsi_val, 2),
                "macd": round(macd_val, 2),
                "verdict": verdict,
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
        # FIX: Syntax error resolved here
        if sentiment == "BEARISH": 
            alloc = {"Equity": 30, "Debt": 50, "Gold": 20}
        elif risk >= 15: 
            alloc = {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk >= 8: 
            alloc = {"Equity": 50, "Debt": 30, "Gold": 20}
        else: 
            alloc = {"Equity": 30, "Debt": 50, "Gold": 20}
        return alloc

class ConversationalAgent:
    def chat(self, user_input, history, image_base64=None, pdf_text=None, system_context=""):
        sys_msg = f"""
        You are FinBot. CONTEXT: {system_context}
        
        INSTRUCTIONS:
        1. Answer the user's question clearly.
        2. At the very end of your response, ALWAYS suggest 3 short follow-up questions the user might want to ask next. Format them like this:
           
           *Follow-up Suggestions:*
           1. [Question 1]
           2. [Question 2]
           3. [Question 3]
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

# State Management
if "profile_created" not in st.session_state: st.session_state.profile_created = False
if "profile" not in st.session_state: st.session_state.profile = {}
if "analysis" not in st.session_state: st.session_state.analysis = None
if "portfolio" not in st.session_state: st.session_state.portfolio = None
if "sip_plan" not in st.session_state: st.session_state.sip_plan = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🚀 AI Financial Super App")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Profile", "📈 Market", "💼 Strategy", "🔮 SIP Planner", "💬 AI Lab", "📊 Dashboard"
])

# === TAB 1: PROFILE ===
with tab1:
    st.subheader("Identity Layer")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", "Investor")
            age = st.number_input("Age", 18, 100, 25)
            mode = st.selectbox("Trading Mode", ["Investor (Long Term)", "Trader (Short Term)"])
        with c2:
            income = st.number_input("Monthly Income", value=50000)
            risk = st.slider("Risk Tolerance (1-20)", 1, 20, 10)
        
        if st.button("Save Profile"):
            st.session_state.profile = {"name": name, "age": age, "mode": mode, "risk": risk}
            st.session_state.profile_created = True
            st.success("Profile Secured.")

# === TAB 2: MARKET ANALYST ===
with tab2:
    if not st.session_state.profile_created:
        st.warning("Complete Profile first.")
    else:
        st.subheader("Market Intelligence")
        col1, col2 = st.columns([1, 3])
        with col1:
            sym = st.text_input("Stock Symbol", "RELIANCE.NS")
            if st.button("Run Analysis"):
                with st.spinner("Analyzing..."):
                    st.session_state.analysis = None # Reset
                    data = analyst.analyze(sym)
                    if data:
                        st.session_state.analysis = data
                        memory.memorize(f"Analyzed {sym}. Verdict: {data['verdict']}", {"type": "analysis"})
                    else: st.error("Symbol not found")
        
        with col2:
            if st.session_state.analysis:
                d = st.session_state.analysis
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"₹{d['price']}")
                m2.metric("RSI", d['rsi'])
                m3.metric("MACD", d['macd'])
                
                v_color = "green" if d['verdict'] == "BUY" else "red"
                m4.markdown(f"### :{v_color}[{d['verdict']}]")
                
                # Candlestick Chart
                fig = go.Figure(data=[go.Candlestick(x=d['history_df'].index,
                                open=d['history_df']['Open'], high=d['history_df']['High'],
                                low=d['history_df']['Low'], close=d['history_df']['Close'])])
                st.plotly_chart(fig, use_container_width=True)

# === TAB 3: STRATEGY ===
with tab3:
    if not st.session_state.analysis:
        st.warning("Analyze a stock first.")
    else:
        st.subheader("Portfolio Allocation")
        cap = st.number_input("Capital (₹)", 100000)
        if st.button("Generate Plan"):
            sent = analyst.get_sentiment()
            risk = st.session_state.profile['risk']
            alloc = planner.create_allocation(risk, sent)
            st.session_state.portfolio = {"alloc": alloc, "cap": cap, "sent": sent}
        
        if st.session_state.portfolio:
            p = st.session_state.portfolio
            c1, c2 = st.columns(2)
            with c1: st.json(p['alloc'])
            with c2: 
                fig = go.Figure(data=[go.Pie(labels=list(p['alloc'].keys()), values=list(p['alloc'].values()))])
                st.plotly_chart(fig, use_container_width=True)

# === TAB 4: SIP PLANNER (New) ===
with tab4:
    st.subheader("Wealth Builder (SIP)")
    s1, s2 = st.columns(2)
    with s1:
        sip_amt = st.number_input("Monthly SIP", 5000)
        sip_yrs = st.slider("Duration (Years)", 1, 30, 10)
    
    # Calculate
    future = sip_amt * (((1+0.12/12)**(sip_yrs*12)-1)/(0.12/12))
    
    if st.button("Save SIP Plan"):
        st.session_state.sip_plan = {"amount": sip_amt, "years": sip_yrs, "future_val": round(future, 2)}
        st.success("SIP Plan Saved to Dashboard!")
    
    st.metric("Projected Wealth (12%)", f"₹{future:,.0f}")
    st.area_chart([sip_amt * (((1+0.12/12)**m-1)/(0.12/12)) for m in range(1, sip_yrs*12+1)])

# === TAB 5: AI LAB ===
with tab5:
    st.subheader("💬 AI Lab")
    ctx = ""
    if st.session_state.profile: ctx += f"User: {st.session_state.profile['name']}. "
    if st.session_state.analysis: ctx += f"Stock: {st.session_state.analysis['symbol']} ({st.session_state.analysis['verdict']}). "
    
    with st.expander("Uploads"):
        img_file = st.file_uploader("Chart Image", type=['png', 'jpg'])
        img_b64 = encode_image(img_file) if img_file else None
        
        pdf_file = st.file_uploader("Report PDF", type=['pdf'])
        pdf_txt = extract_text_from_pdf(pdf_file) if pdf_file else None

    chat_box = st.container(height=400)
    for msg in st.session_state.chat_history:
        with chat_box.chat_message(msg["role"]):
            st.write(msg["content"])

    q = st.chat_input("Ask FinBot...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with chat_box.chat_message("user"): st.write(q)
        
        with chat_box.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = tutor.chat(q, st.session_state.chat_history, img_b64, pdf_txt, ctx)
                st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# === TAB 6: DASHBOARD (Requested Update) ===
with tab6:
    st.subheader("🏁 Master Dashboard")
    
    if st.session_state.profile:
        # ROW 1: PROFILE
        with st.container(border=True):
            st.markdown("### 👤 User Profile")
            p = st.session_state.profile
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Name", p['name'])
            c2.metric("Age", p['age'])
            c3.metric("Mode", p['mode'])
            c4.metric("Risk Score", f"{p['risk']}/20")

        # ROW 2: MARKET DATA
        if st.session_state.analysis:
            with st.container(border=True):
                st.markdown("### 📉 Market Analysis")
                d = st.session_state.analysis
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Stock", d['symbol'])
                k2.metric("Price", f"₹{d['price']}")
                k3.metric("RSI", d['rsi'])
                k4.metric("MACD", d['macd'])
                
                color = "green" if d['verdict'] == "BUY" else "red"
                k5.markdown(f"**Rec:** :{color}[{d['verdict']}]")
        
        # ROW 3: STRATEGY & WEALTH
        c_left, c_right = st.columns(2)
        
        with c_left:
            with st.container(border=True):
                st.markdown("### 💼 Portfolio Strategy")
                if st.session_state.portfolio:
                    plan = st.session_state.portfolio
                    st.write(f"**Capital:** ₹{plan['cap']}")
                    st.json(plan['alloc'])
                else:
                    st.caption("No portfolio generated.")

        with c_right:
            with st.container(border=True):
                st.markdown("### 🔮 SIP Wealth Plan")
                if st.session_state.sip_plan:
                    sip = st.session_state.sip_plan
                    st.metric("Monthly Investment", f"₹{sip['amount']}")
                    st.metric("Duration", f"{sip['years']} Years")
                    st.metric("Projected Value", f"₹{sip['future_val']:,.0f}")
                else:
                    st.caption("No SIP plan saved.")

        st.divider()
        if st.button("Sync Dashboard to Memory"):
            memory.memorize(f"Dashboard Snapshot for {st.session_state.profile['name']}", {"type": "dashboard"})
            st.success("Snapshot saved to Pinecone.")
            
    else:
        st.info("Start at Tab 1 to populate the Dashboard.")
