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
    """📊 THE QUANT: Deep Technical & Fundamental Analysis"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # --- Technicals ---
            df["MA50"] = df["Close"].rolling(50).mean()
            df["MA200"] = df["Close"].rolling(200).mean()
            curr = df["Close"].iloc[-1]
            trend = "BULLISH" if curr > df["MA50"].iloc[-1] else "BEARISH"
            
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rs = gain.rolling(14).mean() / loss.rolling(14).mean()
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            # --- Fundamentals ---
            info = stock.info
            fundamentals = {
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "MarketCap": info.get("marketCap", 0),
                "PE_Ratio": info.get("trailingPE", 0),
                "EPS": info.get("trailingEps", 0),
                "52WeekHigh": info.get("fiftyTwoWeekHigh", 0),
                "52WeekLow": info.get("fiftyTwoWeekLow", 0)
            }

            # Verdict Logic
            verdict = "HOLD"
            if trend == "BULLISH" and rsi.iloc[-1] < 70:
                verdict = "BUY"
            elif trend == "BEARISH" or rsi.iloc[-1] > 70:
                verdict = "SELL"

            return {
                "symbol": symbol, 
                "price": round(curr, 2),
                "trend": trend, 
                "rsi": round(rsi.iloc[-1], 2),
                "macd": round(macd.iloc[-1], 2),
                "verdict": verdict,
                "fundamentals": fundamentals,
                "history_df": df
            }
        except Exception as e:
            return None

    def get_sentiment(self):
        try:
            hist = yf.Ticker("^NSEI").history(period="5d")
            return "BULLISH" if hist["Close"].iloc[-1] > hist["Close"].iloc[0] else "BEARISH"
        except: return "NEUTRAL"

class PlannerAgent:
    """⚖️ THE STRATEGIST: Risk-Adjusted Recommendations"""
    
    def recommend_assets(self, risk_score):
        """Returns specific companies based on Risk Profile (1-20)"""
        if risk_score <= 7: # Conservative
            equity = ["HDFC Bank", "ITC", "HUL"]
            debt = ["HDFC Liquid Fund", "SBI Gilt Fund", "ICICI Pru Savings"]
            gold = ["Nippon Gold BeES", "SBI Gold ETF", "HDFC Gold ETF"]
        elif risk_score <= 14: # Moderate
            equity = ["Reliance Industries", "Infosys", "L&T"]
            debt = ["Aditya Birla Corporate Bond", "Kotak Bond", "Axis Strategic Bond"]
            gold = ["Sovereign Gold Bond", "Kotak Gold ETF", "Axis Gold ETF"]
        else: # Aggressive
            equity = ["Adani Enterprises", "Tata Motors", "Zomato"]
            debt = ["Credit Risk Funds", "Dynamic Bond Funds", "Hybrid Funds"]
            gold = ["Digital Gold", "Gold Futures", "Quant Gold Fund"]
            
        return {"Equity": equity, "Debt": debt, "Gold": gold}

    def create_allocation(self, risk, sentiment):
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
        1. Answer clearly.
        2. Suggest 3 short follow-up questions at the end.
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
        except: return "AI currently offline."

# --- Initialize ---
memory = MemoryAgent(pc_key)
analyst = AnalystAgent()
planner = PlannerAgent()
tutor = ConversationalAgent()

# =========================
# 4. UI ORCHESTRATOR
# =========================

# --- SESSION STATE ---
if "profile_created" not in st.session_state: st.session_state.profile_created = False
if "profile" not in st.session_state: st.session_state.profile = {}
if "activity_log" not in st.session_state: st.session_state.activity_log = [] # Stores history of actions
if "current_analysis" not in st.session_state: st.session_state.current_analysis = None
if "current_portfolio" not in st.session_state: st.session_state.current_portfolio = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🚀 AI Financial Super App")

# --- NAVIGATION ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Profile", "📈 Deep Analysis", "💼 Portfolio Plan", "🔮 SIP Planner", "💬 AI Lab", "📊 Dashboard"
])

# === TAB 1: PROFILE ===
with tab1:
    st.subheader("Identity Layer")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", "Investor")
            age = st.number_input("Age", 25)
            mode = st.selectbox("Mode", ["Investor", "Trader"])
        with c2:
            income = st.number_input("Income", 50000)
            risk = st.slider("Risk (1-20)", 1, 20, 10)
        
        if st.button("Save Profile"):
            st.session_state.profile = {"name": name, "age": age, "mode": mode, "risk": risk}
            st.session_state.profile_created = True
            st.success("Profile Secured.")

# === TAB 2: DEEP ANALYSIS ===
with tab2:
    if not st.session_state.profile_created:
        st.warning("Complete Profile first.")
    else:
        st.subheader("Market Intelligence")
        col1, col2 = st.columns([1, 3])
        with col1:
            sym = st.text_input("Symbol", "RELIANCE.NS")
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    data = analyst.analyze(sym)
                    if data:
                        st.session_state.current_analysis = data
                        # Log to History
                        entry = {
                            "time": datetime.datetime.now().strftime("%H:%M:%S"),
                            "type": "Analysis",
                            "item": sym,
                            "detail": f"Verdict: {data['verdict']} @ {data['price']}"
                        }
                        st.session_state.activity_log.append(entry)
                        memory.memorize(f"Analyzed {sym}. Verdict: {data['verdict']}", {"type": "analysis"})
                    else: st.error("Symbol not found")
        
        with col2:
            if st.session_state.current_analysis:
                d = st.session_state.current_analysis
                f = d['fundamentals']
                
                # Metrics Row 1
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"₹{d.get('price',0)}")
                m2.metric("RSI", d.get('rsi',0))
                m3.metric("MACD", d.get('macd','N/A'))
                color = "green" if d['verdict'] == "BUY" else "red"
                m4.markdown(f"### :{color}[{d['verdict']}]")
                
                # Fundamentals Expandable
                with st.expander("📚 Fundamental Data (Deep Dive)", expanded=True):
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Sector", f['Sector'])
                    f2.metric("P/E Ratio", f['PE_Ratio'])
                    f3.metric("Market Cap", f"{f['MarketCap']/1e9:.2f}B")
                    f4.metric("52W High", f['52WeekHigh'])
                
                # Chart
                fig = go.Figure(data=[go.Candlestick(x=d['history_df'].index,
                                open=d['history_df']['Open'], high=d['history_df']['High'],
                                low=d['history_df']['Low'], close=d['history_df']['Close'])])
                st.plotly_chart(fig, use_container_width=True)

# === TAB 3: PORTFOLIO & SUGGESTIONS ===
with tab3:
    if not st.session_state.current_analysis:
        st.warning("Analyze a stock first.")
    else:
        st.subheader("Strategic Planning")
        cap = st.number_input("Capital (₹)", 100000)
        
        if st.button("Generate Strategy"):
            sent = analyst.get_sentiment()
            risk = st.session_state.profile['risk']
            
            # Get Allocation %
            alloc = planner.create_allocation(risk, sent)
            # Get Specific Companies
            recs = planner.recommend_assets(risk)
            
            st.session_state.current_portfolio = {"alloc": alloc, "recs": recs, "cap": cap}
            
            # Log to History
            st.session_state.activity_log.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "type": "Portfolio",
                "item": f"Risk Level {risk}",
                "detail": f"Equity: {alloc['Equity']}%"
            })
        
        if st.session_state.current_portfolio:
            p = st.session_state.current_portfolio
            
            # Allocation Chart
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("#### Asset Split")
                st.json(p['alloc'])
            with c2:
                fig = go.Figure(data=[go.Pie(labels=list(p['alloc'].keys()), values=list(p['alloc'].values()))])
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("🏆 Recommended Assets (Based on Risk)")
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.info(f"**Equity Picks**\n" + "\n".join([f"- {x}" for x in p['recs']['Equity']]))
            with r2:
                st.warning(f"**Debt/Bond Picks**\n" + "\n".join([f"- {x}" for x in p['recs']['Debt']]))
            with r3:
                st.success(f"**Gold Options**\n" + "\n".join([f"- {x}" for x in p['recs']['Gold']]))

# === TAB 4: SIP PLANNER ===
with tab4:
    st.subheader("Wealth Builder")
    s1, s2 = st.columns(2)
    with s1:
        sip_amt = st.number_input("Monthly SIP", 5000)
        sip_yrs = st.slider("Duration (Years)", 1, 30, 10)
    future = sip_amt * (((1+0.12/12)**(sip_yrs*12)-1)/(0.12/12))
    st.metric("Projected Wealth (12%)", f"₹{future:,.0f}")
    st.area_chart([sip_amt * (((1+0.12/12)**m-1)/(0.12/12)) for m in range(1, sip_yrs*12+1)])

# === TAB 5: AI LAB ===
with tab5:
    st.subheader("💬 AI Lab")
    ctx = f"User: {st.session_state.profile.get('name')}."
    
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

# === TAB 6: ACTIVITY DASHBOARD (TIMELINE) ===
with tab6:
    st.subheader("🏁 Session Timeline")
    
    # 1. Profile Summary
    if st.session_state.profile:
        with st.container(border=True):
            st.markdown("### 👤 User Identity")
            p = st.session_state.profile
            c1, c2, c3 = st.columns(3)
            c1.metric("Name", p['name'])
            c2.metric("Risk Score", f"{p['risk']}/20")
            c3.metric("Mode", p['mode'])

    st.divider()
    
    # 2. Activity Log (Timeline)
    st.markdown("### 📜 Activity History (Use 1, Use 2...)")
    
    if len(st.session_state.activity_log) == 0:
        st.info("No activity yet. Analyze a stock to start tracking.")
    else:
        # Loop through log in reverse to show newest first
        for i, log in enumerate(reversed(st.session_state.activity_log)):
            with st.container(border=True):
                c_time, c_type, c_desc = st.columns([1, 1, 4])
                c_time.caption(f"⏰ {log['time']}")
                c_type.markdown(f"**{log['type']}**")
                c_desc.write(f"{log['item']} - {log['detail']}")
    
    st.divider()
    if st.button("Save Full Timeline to Memory"):
        memory.memorize(f"Timeline for {st.session_state.profile.get('name')}", {"type": "timeline"})
        st.success("History Saved!")
