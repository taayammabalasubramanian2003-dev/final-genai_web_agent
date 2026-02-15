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
# Tries to get keys from Streamlit Secrets first, then Environment Variables
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
    """Converts uploaded file bytes to Base64 string for OpenAI Vision"""
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
        
        # Check/Create Index (1024 dims for compatibility with 'text-embedding-3-small')
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
            # Force 1024 dimensions to match your specific index
            response = client.embeddings.create(input=text, model="text-embedding-3-small", dimensions=1024)
            vector = response.data[0].embedding
            unique_id = f"mem_{int(time.time())}"
            # Clean metadata to simple strings to avoid Pinecone errors
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
    """📊 THE QUANT: Handles Technical & Fundamental Data"""
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
                "symbol": symbol,
                "price": round(current_price, 2),
                "trend": trend,
                "rsi": round(rsi.iloc[-1], 2),
                "sector": info.get('sector', 'Unknown'),
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
    """⚖️ THE STRATEGIST: Manages Risk & Allocation"""
    def create_allocation(self, risk_level, market_sentiment):
        if market_sentiment == "BEARISH":
            return {"Equity": 30, "Debt": 50, "Gold": 20}
        
        if risk_level >= 15: return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk_level >= 8: return {"Equity": 50, "Debt": 30, "Gold": 20}
        else: return {"Equity": 30, "Debt": 50, "Gold": 20}

class ConversationalAgent:
    """🎓 THE TUTOR (Multi-Modal): Handles Text, Vision & Context"""
    def chat(self, user_input, history, image_base64=None, system_context=""):
        # 1. System Prompt with Dynamic Context
        # This prompts the AI to act like a real agent (memorable & proactive)
        system_msg = f"""
        You are FinBot, an expert financial analyst agent.
        
        CONTEXT FROM SYSTEM:
        {system_context}
        
        INSTRUCTIONS:
        1. Answer the user's financial questions accurately.
        2. If an image is provided, analyze the chart/table in detail.
        3. ALWAYS ask a follow-up question to keep the conversation engaging.
        4. Be concise and professional.
        """
        
        messages = [{"role": "system", "content": system_msg}]
        
        # 2. Add History (Text only to save tokens)
        for msg in history:
            if isinstance(msg["content"], str): 
                messages.append(msg)

        # 3. Construct Current Message (Text + Optional Image)
        if image_base64:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        else:
            user_message = {"role": "user", "content": user_input}
            
        messages.append(user_message)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Agent Error: {e}"

# --- Initialize Agents ---
memory_node = MemoryAgent(pc_key)
analyst_node = AnalystAgent()
planner_node = PlannerAgent()
chat_node = ConversationalAgent()

# =========================
# 4. FRONTEND ORCHESTRATOR
# =========================

# --- SHARED STATE (The "Bus" connecting all agents) ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "user_profile_data" not in st.session_state: st.session_state.user_profile_data = {"name": "Investor", "risk": 10} 
if "stock_analysis_data" not in st.session_state: st.session_state.stock_analysis_data = None

st.sidebar.title("🤖 Agent Navigation")
page = st.sidebar.radio("Select Module", [
    "Home", "Profile Setup", "Stock Analysis", "AI Decision", 
    "Portfolio Allocation", "Financial Planning", "Education (Chatbot)", "Memory Logs"
])

# --- HOME ---
if page == "Home":
    st.title("🤖 AI Investment Hive")
    st.image("https://placehold.co/1000x400/png?text=Smart+Investing+Dashboard", caption="Orchestrator Online")
    st.markdown("### Status: **ONLINE 🟢**")
    
    # Dashboard Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Profile", st.session_state.user_profile_data['name'])
    
    if st.session_state.stock_analysis_data:
        c2.metric("Last Analyzed", st.session_state.stock_analysis_data['symbol'])
        c3.metric("Last Price", f"₹{st.session_state.stock_analysis_data['price']}")
    else:
        c2.metric("Last Analyzed", "None")
        c3.metric("Status", "Idle")

# --- PROFILE ---
elif page == "Profile Setup":
    st.header("👤 Profile Agent")
    # FIX: Renamed form to 'profile_form_ui' to avoid collision with session_state keys
    with st.form("profile_form_ui"):
        name = st.text_input("Name", value=st.session_state.user_profile_data.get("name", "Investor"))
        risk = st.slider("Risk Appetite", 1, 20, st.session_state.user_profile_data.get("risk", 10))
        income = st.number_input("Monthly Income", value=st.session_state.user_profile_data.get("income", 50000))
        savings = st.number_input("Monthly Savings", value=st.session_state.user_profile_data.get("savings", 10000))
        
        if st.form_submit_button("Save Profile"):
            # Update the specific data dictionary
            st.session_state.user_profile_data = {"name": name, "risk": risk, "income": income, "savings": savings}
            st.success("Profile updated successfully.")

# --- STOCK ANALYSIS ---
elif page == "Stock Analysis":
    st.header("📊 Analyst Agent")
    symbol = st.text_input("Enter Symbol", "RELIANCE.NS")
    if st.button("Run Analysis"):
        with st.status("Analyst working..."):
            data = analyst_node.analyze(symbol)
            if data:
                st.session_state.stock_analysis_data = data # Store in Shared State
                st.success("Analysis Complete")
                st.metric("Price", f"₹{data['price']}")
                st.line_chart(data['history_df']['Close'])
                
                c1, c2 = st.columns(2)
                c1.metric("Trend", data['trend'])
                c2.metric("RSI", data['rsi'])
                
                memory_node.memorize(f"Analyzed {symbol} at {data['price']}", {"type": "analysis"})
            else:
                st.error("Symbol not found.")

# --- AI DECISION ---
elif page == "AI Decision":
    st.header("🧠 Decision Agent")
    if st.session_state.stock_analysis_data:
        data = st.session_state.stock_analysis_data
        st.subheader(f"Verdict for {data['symbol']}")
        
        # Logic
        rec = "BUY" if data['trend'] == "BULLISH" and data['rsi'] < 70 else "HOLD"
        color = "green" if rec == "BUY" else "orange"
        
        st.markdown(f"### Recommendation: :{color}[{rec}]")
        st.write(f"**Why?** Trend is **{data['trend']}** | RSI: **{data['rsi']}**")
        
        if st.button("Ask Advisor for Detail"):
            # Uses Chat Node but with specific context injection
            ctx = f"Stock: {data['symbol']}, Price: {data['price']}, Trend: {data['trend']}, RSI: {data['rsi']}"
            exp = chat_node.chat(f"Explain why {rec} is the recommendation.", [], system_context=ctx)
            st.write(exp)
    else:
        st.warning("⚠️ No data found. Please run 'Stock Analysis' first.")

# --- PORTFOLIO ---
elif page == "Portfolio Allocation":
    st.header("💼 Planner Agent")
    capital = st.number_input("Capital", 10000)
    
    if st.button("Generate Strategy"):
        sent = analyst_node.check_market_sentiment()
        risk = st.session_state.user_profile_data.get("risk", 10)
        alloc = planner_node.create_allocation(risk, sent)
        
        st.write(f"Market Context: **{sent}** | Risk Profile: **{risk}/20**")
        
        fig = go.Figure(data=[go.Pie(labels=list(alloc.keys()), values=list(alloc.values()))])
        st.plotly_chart(fig)
        
        memory_node.memorize(f"Plan: {alloc} for {capital}", {"type": "plan"})
        st.success("Plan generated and saved.")

# --- FINANCIAL PLANNING ---
elif page == "Financial Planning":
    st.header("🔮 Wealth Agent")
    sip = st.number_input("SIP Amount", 5000)
    yrs = st.slider("Years", 1, 30, 10)
    
    future = sip * (((1+0.01)**(yrs*12)-1)/0.01)
    st.metric("Future Value", f"₹{future:,.0f}")
    st.area_chart([sip * (((1+0.01)**m-1)/0.01) for m in range(1, yrs*12+1)])

# --- EDUCATION (CHATBOT) ---
elif page == "Education (Chatbot)":
    st.header("🎓 Conversational Tutor Agent")
    
    # 1. Inject Context (Connectivity Feature)
    context_note = ""
    if st.session_state.stock_analysis_data:
        d = st.session_state.stock_analysis_data
        context_note = f"User is currently analyzing {d['symbol']} which is trading at {d['price']}."
        st.info(f"💡 FinBot is aware you are looking at **{d['symbol']}**.")

    # 2. Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str): st.write(msg["content"])

    # 3. Vision Input
    with st.sidebar:
        st.subheader("📷 Visual Input")
        uploaded_file = st.file_uploader("Upload Chart", type=["jpg", "png", "jpeg"])
        img_b64 = encode_image(uploaded_file)

    # 4. Chat Logic
    prompt = st.chat_input("Ask FinBot...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the 'context_note' so the agent knows about Reliance/etc.
                response = chat_node.chat(prompt, st.session_state.chat_history[:-1], img_b64, system_context=context_note)
                st.write(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- MEMORY LOGS ---
elif page == "Memory Logs":
    st.header("📜 Agent Memory")
    if st.button("Refresh"):
        logs = memory_node.recall("investment", 5)
        for l in logs: st.info(l.get('text'))

st.sidebar.divider()
st.sidebar.caption("System: Multi-Agent RAG + Vision v4.0")
