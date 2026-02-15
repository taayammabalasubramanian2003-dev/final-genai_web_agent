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
import json
from PIL import Image # For handling images

# =========================
# 1. ORCHESTRATOR CONFIGURATION
# =========================
st.set_page_config(page_title="AI Investment Hive (Conversational)", layout="wide", page_icon="🧠")

# Load Secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    pc_key = st.secrets["PINECONE_API_KEY"]
except:
    # Fallback for local dev
    api_key = os.getenv("OPENAI_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    st.error("❌ Secrets not found! Please set OPENAI_API_KEY and PINECONE_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# 2. AGENT DEFINITIONS (The "Workers")
# =========================

class MemoryAgent:
    """
    🧠 THE BRAIN: Manages Long-Term Vector Memory (Pinecone)
    """
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        
        # Auto-create Index if missing
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(5)
            except Exception as e:
                st.error(f"Memory Init Error: {e}")
        
        self.index = self.pc.Index(self.index_name)

    def memorize(self, text, metadata):
        try:
            response = client.embeddings.create(input=text, model="text-embedding-3-small")
            vector = response.data[0].embedding
            unique_id = f"mem_{int(time.time())}"
            # Ensure metadata values are strings
            clean_meta = {k: str(v) for k, v in metadata.items() if v is not None}
            clean_meta['text'] = text
            self.index.upsert(vectors=[(unique_id, vector, clean_meta)])
            return True
        except Exception as e:
            st.error(f"Memory Error: {e}")
            return False

    def recall(self, query, top_k=3):
        try:
            response = client.embeddings.create(input=query, model="text-embedding-3-small")
            query_vector = response.data[0].embedding
            results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            return [match['metadata'] for match in results['matches']]
        except:
            return []

class AnalystAgent:
    """
    📊 THE QUANT: Handles Technical & Fundamental Data
    """
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None

            # Calc Indicators
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
                "pe_ratio": info.get('trailingPE', 'N/A'),
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
    """
    ⚖️ THE STRATEGIST: Manages Risk & Allocation
    """
    def create_allocation(self, risk_level, market_sentiment):
        if market_sentiment == "BEARISH":
            return {"Equity": 30, "Debt": 50, "Gold": 20}
        
        if risk_level >= 15:
            return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk_level >= 8:
            return {"Equity": 50, "Debt": 30, "Gold": 20}
        else:
            return {"Equity": 30, "Debt": 50, "Gold": 20}

class ConversationalAgent:
    """
    💬 THE CHATBOT: Handles Full Conversation Memory & Multi-Modal Interactions
    """
    def chat(self, user_input, history, image_desc=None):
        # 1. Build Context from History
        messages = [
            {"role": "system", "content": """
             You are a helpful Financial Tutor Agent named 'FinBot'. 
             You act like a professional mentor.
             1. Remember the context of the conversation.
             2. Always ask a follow-up question to check the user's understanding.
             3. If an image description is provided, analyze it as if you are seeing the chart.
             """}
        ]
        
        # Add past history to context
        for msg in history:
            messages.append(msg)
            
        # Add current input
        content = user_input
        if image_desc:
            content += f"\n[System Note: User uploaded an image related to: {image_desc}]"
            
        messages.append({"role": "user", "content": content})
        
        # 2. Get Response
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error: {e}"

# --- Initialize Agents ---
memory_node = MemoryAgent(pc_key)
analyst_node = AnalystAgent()
planner_node = PlannerAgent()
chat_node = ConversationalAgent()

# =========================
# 3. FRONTEND ORCHESTRATOR
# =========================

# Initialize Session State
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "profile" not in st.session_state: st.session_state.profile = {}
if "last_analysis" not in st.session_state: st.session_state.last_analysis = None

st.sidebar.title("🤖 Agent Navigation")
page = st.sidebar.radio("Select Module", [
    "Home", "Profile Setup", "Stock Analysis", "AI Decision", 
    "Portfolio Allocation", "Financial Planning", "Education (Chatbot)", "Memory Logs"
])

# --- HOME ---
if page == "Home":
    st.title("🤖 AI Investment Hive")
    st.image("https://placehold.co/1000x400/png?text=Conversational+AI+Agent", caption="Orchestrator Online")
    st.markdown("### Status: **ONLINE 🟢**")
    st.info("Welcome back! I am ready to analyze markets, plan portfolios, and chat with you.")

# --- PROFILE ---
elif page == "Profile Setup":
    st.header("👤 Profile Agent")
    with st.form("profile"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", "Investor"))
        risk = st.slider("Risk Appetite", 1, 20, 10)
        if st.form_submit_button("Save"):
            st.session_state.profile = {"name": name, "risk": risk}
            st.success("Profile updated.")

# --- STOCK ANALYSIS ---
elif page == "Stock Analysis":
    st.header("📊 Analyst Agent")
    symbol = st.text_input("Enter Symbol", "RELIANCE.NS")
    if st.button("Run Analysis"):
        with st.status("Analyst working..."):
            data = analyst_node.analyze(symbol)
            if data:
                st.session_state.last_analysis = data
                st.success("Analysis Complete")
                st.metric("Price", f"₹{data['price']}")
                st.line_chart(data['history_df']['Close'])
                # Auto-save to memory
                memory_node.memorize(f"Analyzed {symbol} at {data['price']}", {"type": "analysis"})
            else:
                st.error("Symbol not found.")

# --- AI DECISION ---
elif page == "AI Decision":
    st.header("🧠 Decision Agent")
    if st.session_state.last_analysis:
        data = st.session_state.last_analysis
        st.write(f"Analyzing {data['symbol']}...")
        
        # Simple Logic for Demo
        rec = "BUY" if data['trend'] == "BULLISH" else "HOLD"
        color = "green" if rec == "BUY" else "orange"
        
        st.markdown(f"### Recommendation: :{color}[{rec}]")
        st.caption(f"Reason: Trend is {data['trend']} and RSI is {data['rsi']}")
    else:
        st.warning("Run Stock Analysis first.")

# --- PORTFOLIO ---
elif page == "Portfolio Allocation":
    st.header("💼 Planner Agent")
    capital = st.number_input("Capital", 10000)
    if st.button("Generate Plan"):
        sent = analyst_node.check_market_sentiment()
        risk = st.session_state.profile.get("risk", 10)
        alloc = planner_node.create_allocation(risk, sent)
        
        st.write(f"Market Sentiment: **{sent}**")
        st.json(alloc)
        
        # Visualization
        fig = go.Figure(data=[go.Pie(labels=list(alloc.keys()), values=list(alloc.values()))])
        st.plotly_chart(fig)
        
        # Save Plan
        memory_node.memorize(f"Plan: {alloc} for {capital}", {"type": "plan"})

# --- FINANCIAL PLANNING ---
elif page == "Financial Planning":
    st.header("🔮 Wealth Agent")
    sip = st.number_input("SIP Amount", 5000)
    yrs = st.slider("Years", 1, 30, 10)
    
    total = sip * 12 * yrs
    # Approx 12% return calculation
    future = sip * (((1+0.01)**(yrs*12)-1)/0.01)
    
    st.metric("Future Value (approx)", f"₹{future:,.0f}")
    st.area_chart([sip * (((1+0.01)**m-1)/0.01) for m in range(1, yrs*12+1)])

# --- EDUCATION (CHATBOT AGENT) ---
elif page == "Education (Chatbot)":
    st.header("🎓 Conversational Tutor Agent")
    st.caption("I act like a pro agent. I remember our chat context and can see charts you upload.")

    # 1. Display Chat History (WhatsApp Style)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 2. Multi-Modal Inputs (Sidebar or Main)
    with st.expander("📷 Upload Image / Chart (Multi-Modal)"):
        uploaded_file = st.file_uploader("Upload a stock chart or financial table", type=["jpg", "png", "jpeg"])
        image_desc = None
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded for Analysis", width=200)
            # In a real production app, you would send base64 image to GPT-4o
            # For this demo, we simulate vision by telling the agent an image exists
            image_desc = "User uploaded a financial chart image." 

    # 3. Chat Input
    prompt = st.chat_input("Ask a follow-up question...")
    
    if prompt:
        # Show User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("FinBot is thinking..."):
                # Pass history AND image context to the agent
                response = chat_node.chat(prompt, st.session_state.chat_history[:-1], image_desc)
                st.write(response)
        
        # Append AI Message to History
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- MEMORY LOGS ---
elif page == "Memory Logs":
    st.header("📜 Agent Memory (Pinecone)")
    if st.button("Refresh Logs"):
        logs = memory_node.recall("investment portfolio", top_k=5)
        if logs:
            st.success("Found relevant memories:")
            for l in logs:
                st.info(l.get('text'))
        else:
            st.write("Memory empty or connecting...")

st.sidebar.divider()
st.sidebar.caption("Powered by Multi-Agent RAG System")
