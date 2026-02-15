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
from pypdf import PdfReader  # NEW: For PDF Analysis

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
    """Converts image to Base64 for Vision API"""
    if uploaded_file is not None:
        return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    return None

def extract_text_from_pdf(uploaded_file):
    """Extracts text from uploaded PDF for RAG"""
    if uploaded_file is not None:
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages[:5]: # Limit to first 5 pages to save tokens
                text += page.extract_text()
            return text
        except: return None
    return None

# =========================
# 3. ADVANCED AGENT CLASSES
# =========================

class MemoryAgent:
    """🧠 THE BRAIN: Pinecone Vector Database"""
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        # Auto-heal index
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024, # Optimized for text-embedding-3-small
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(2)
            except: pass
        self.index = self.pc.Index(self.index_name)

    def memorize(self, text, metadata):
        """Saves context to Long-Term Memory"""
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
        """Retrieves relevant past info"""
        try:
            response = client.embeddings.create(input=query, model="text-embedding-3-small", dimensions=1024)
            vector = response.data[0].embedding
            results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
            return [match['metadata']['text'] for match in results['matches']]
        except: return []

class AnalystAgent:
    """📊 THE QUANT: Market Data & Technicals"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # Indicators
            df["MA50"] = df["Close"].rolling(50).mean()
            curr = df["Close"].iloc[-1]
            trend = "BULLISH" if curr > df["MA50"].iloc[-1] else "BEARISH"
            
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rs = gain.rolling(14).mean() / loss.rolling(14).mean()
            rsi = 100 - (100 / (1 + rs))
            
            return {
                "symbol": symbol, "price": round(curr, 2),
                "trend": trend, "rsi": round(rsi.iloc[-1], 2),
                "history_df": df
            }
        except: return None

    def get_sentiment(self):
        try:
            # Simple heuristic using Nifty 50
            hist = yf.Ticker("^NSEI").history(period="5d")
            return "BULLISH" if hist["Close"].iloc[-1] > hist["Close"].iloc[0] else "BEARISH"
        except: return "NEUTRAL"

class PlannerAgent:
    """⚖️ THE STRATEGIST: Risk & Allocation"""
    def create_allocation(self, risk, sentiment):
        if sentiment == "BEARISH": return {"Equity": 30, "Debt": 50, "Gold": 20}
        if risk >= 15: return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk >= 8: return {"Equity": 50, "Debt": 30, "Gold": 20}
        return {"Equity": 30, "Debt": 50, "Gold": 20}

class ConversationalAgent:
    """🎓 THE TUTOR: Multi-Modal (Text + Vision + PDF)"""
    def chat(self, user_input, history, image_base64=None, pdf_text=None, system_context=""):
        
        # Dynamic System Prompt
        sys_msg = f"""
        You are FinBot, an advanced AI Financial Analyst.
        SYSTEM CONTEXT: {system_context}
        
        CAPABILITIES:
        1. Analyze Charts (if image provided).
        2. Analyze Annual Reports/Documents (if PDF text provided).
        3. Answer general financial questions.
        
        INSTRUCTIONS:
        - Be concise and professional.
        - If a PDF is uploaded, answer based on that specific text.
        """
        
        messages = [{"role": "system", "content": sys_msg}]
        
        # Add Conversation History (Text Only)
        for msg in history:
            if isinstance(msg["content"], str): messages.append(msg)
        
        # Build User Message Payload
        content_payload = [{"type": "text", "text": user_input}]
        
        if image_base64:
            content_payload.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        if pdf_text:
            content_payload.append({
                "type": "text", 
                "text": f"\n\n[ATTACHED DOCUMENT CONTENT]: {pdf_text[:4000]}..." # Truncate for limits
            })

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

# State Init
if "profile_created" not in st.session_state: st.session_state.profile_created = False
if "profile" not in st.session_state: st.session_state.profile = None
if "analysis" not in st.session_state: st.session_state.analysis = None
if "portfolio" not in st.session_state: st.session_state.portfolio = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🚀 AI Financial Super App")

# --- HORIZONTAL TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Profile", "📈 Market", "💼 Strategy", "💬 AI Lab", "📊 Dashboard"
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
            st.success("Identity Verified. Access granted to Market Layer.")

# === TAB 2: MARKET ANALYST ===
with tab2:
    if not st.session_state.profile_created:
        st.warning("Please complete Profile first.")
    else:
        st.subheader("Market Intelligence Layer")
        c1, c2 = st.columns([1, 2])
        with c1:
            sym = st.text_input("Stock Symbol", "RELIANCE.NS")
            if st.button("Run Analysis Protocol"):
                with st.spinner("Analyst Agent working..."):
                    data = analyst.analyze(sym)
                    if data:
                        st.session_state.analysis = data
                        memory.memorize(f"Analyzed {sym} at {data['price']}", {"type": "analysis"})
                    else: st.error("Symbol not found")
        
        with c2:
            if st.session_state.analysis:
                d = st.session_state.analysis
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"₹{d['price']}")
                m2.metric("Trend", d['trend'])
                m3.metric("RSI", d['rsi'])
                st.line_chart(d['history_df']['Close'])
                
                rec = "BUY" if d['trend'] == "BULLISH" and d['rsi'] < 70 else "HOLD"
                st.info(f"**AI Verdict:** {rec} (Based on Trend + Momentum)")

# === TAB 3: STRATEGY ===
with tab3:
    if not st.session_state.analysis:
        st.warning("Analyze a stock first.")
    else:
        st.subheader("Strategic Planning Layer")
        cap = st.number_input("Deployment Capital", 100000)
        if st.button("Generate Allocation"):
            sent = analyst.get_sentiment()
            risk = st.session_state.profile['risk']
            alloc = planner.create_allocation(risk, sent)
            st.session_state.portfolio = {"alloc": alloc, "cap": cap, "sent": sent}
        
        if st.session_state.portfolio:
            p = st.session_state.portfolio
            st.write(f"**Market Context:** {p['sent']}")
            c1, c2 = st.columns(2)
            with c1: st.json(p['alloc'])
            with c2: 
                fig = go.Figure(data=[go.Pie(labels=list(p['alloc'].keys()), values=list(p['alloc'].values()))])
                st.plotly_chart(fig, use_container_width=True)

# === TAB 4: AI LAB (CHAT + VISION + PDF) ===
with tab4:
    st.subheader("💬 Conversational Intelligence (Multi-Modal)")
    
    # Context Injection
    ctx = ""
    if st.session_state.profile: ctx += f"User: {st.session_state.profile['name']}. "
    if st.session_state.analysis: ctx += f"Focus Stock: {st.session_state.analysis['symbol']}. "
    
    # 1. File Uploads (Sidebar within Tab)
    with st.expander("📂 Upload Data (Charts or Reports)", expanded=True):
        uc1, uc2 = st.columns(2)
        with uc1:
            img_file = st.file_uploader("Upload Chart (Image)", type=['png', 'jpg'])
            img_b64 = encode_image(img_file) if img_file else None
        with uc2:
            pdf_file = st.file_uploader("Upload Report (PDF)", type=['pdf'])
            pdf_txt = extract_text_from_pdf(pdf_file) if pdf_file else None
            if pdf_txt: st.success("PDF Content Extracted!")

    # 2. Chat Interface
    chat_container = st.container(height=400)
    for msg in st.session_state.chat_history:
        with chat_container.chat_message(msg["role"]):
            st.write(msg["content"])

    # 3. Input Area
    user_q = st.chat_input("Ask about markets, your plan, or uploaded files...")
    
    if user_q:
        # User Msg
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with chat_container.chat_message("user"): st.write(user_q)
        
        # AI Msg
        with chat_container.chat_message("assistant"):
            with st.spinner("Processing (Vision + Text)..."):
                reply = tutor.chat(user_q, st.session_state.chat_history, img_b64, pdf_txt, ctx)
                st.write(reply)
        
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# === TAB 5: DASHBOARD ===
with tab5:
    st.subheader("🏁 Mission Control")
    if st.session_state.profile:
        c1, c2, c3 = st.columns(3)
        c1.metric("User", st.session_state.profile['name'])
        c2.metric("Last Asset", st.session_state.analysis['symbol'] if st.session_state.analysis else "None")
        c3.metric("Plan Value", f"₹{st.session_state.portfolio['cap']}" if st.session_state.portfolio else "0")
        
        st.divider()
        st.write("### 🧠 Long-Term Memory Logs (Pinecone)")
        if st.button("Sync Memory"):
            logs = memory.recall("investment", 5)
            for l in logs: st.info(l)
    else:
        st.info("System Idle. Start at Tab 1.")
