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
st.set_page_config(page_title="AI Investment Hive (Multi-Modal)", layout="wide", page_icon="🧠")

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
# 2. HELPER FUNCTIONS (Tools)
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
    """
    🧠 THE BRAIN: Manages Long-Term Vector Memory (Pinecone)
    """
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        
        # Auto-create Index logic (Updated to 1024 to match your setup)
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024, # FIXED: Changed to 1024 to match your index
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(5) # Wait for init
            except Exception as e:
                pass 
        
        self.index = self.pc.Index(self.index_name)

    def memorize(self, text, metadata):
        try:
            # FIXED: Request 1024 dimensions explicitly
            response = client.embeddings.create(
                input=text, 
                model="text-embedding-3-small",
                dimensions=1024 
            )
            vector = response.data[0].embedding
            unique_id = f"mem_{int(time.time())}"
            
            # Clean metadata to simple strings
            clean_meta = {k: str(v) for k, v in metadata.items() if v is not None}
            clean_meta['text'] = text
            
            self.index.upsert(vectors=[(unique_id, vector, clean_meta)])
            return True
        except Exception as e:
            st.error(f"Memory Save Error: {e}")
            return False

    def recall(self, query, top_k=3):
        try:
            # FIXED: Request 1024 dimensions explicitly
            response = client.embeddings.create(
                input=query, 
                model="text-embedding-3-small",
                dimensions=1024
            )
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
    🎓 THE TUTOR (Multi-Modal): Handles Text & Vision
    """
    def chat(self, user_input, history, image_base64=None):
        # 1. System Prompt
        messages = [{
            "role": "system", 
            "content": "You are FinBot, an expert financial analyst. Analyze text questions and financial charts. Be helpful, concise, and professional."
        }]
        
        # 2. Add History (Text only context to save tokens/complexity)
        for msg in history:
            if isinstance(msg["content"], str): # Only add text history
                messages.append(msg)

        # 3. Construct Current User Message (Text + Optional Image)
        if image_base64:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        else:
            user_message = {"role": "user", "content": user_input}
            
        messages.append(user_message)
        
        # 4. Call OpenAI
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

# Initialize Session State Variables Safely
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "profile" not in st.session_state: st.session_state.profile = {"name": "Investor", "risk": 10}
if "last_analysis" not in st.session_state: st.session_state.last_analysis = None

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
    st.info("Welcome back! I am ready to analyze markets, plan portfolios, and chat with you.")

# --- PROFILE ---
elif page == "Profile Setup":
    st.header("👤 Profile Agent")
    with st.form("profile"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", "Investor"))
        risk = st.slider("Risk Appetite", 1, 20, 10)
        income = st.number_input("Monthly Income", value=50000)
        savings = st.number_input("Monthly Savings", value=10000)
        if st.form_submit_button("Save Profile"):
            st.session_state.profile = {"name": name, "risk": risk, "income": income, "savings": savings}
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
                
                # Visual Technicals
                c1, c2 = st.columns(2)
                c1.metric("Trend Direction", data['trend'])
                c2.metric("RSI Strength", data['rsi'])
                
                # Auto-save
                memory_node.memorize(f"Analyzed {symbol} at {data['price']}", {"type": "analysis"})
            else:
                st.error("Symbol not found.")

# --- AI DECISION ---
elif page == "AI Decision":
    st.header("🧠 Decision Agent")
    if st.session_state.last_analysis:
        data = st.session_state.last_analysis
        st.subheader(f"Verdict for {data['symbol']}")
        
        # Logic
        rec = "BUY" if data['trend'] == "BULLISH" and data['rsi'] < 70 else "HOLD"
        color = "green" if rec == "BUY" else "orange"
        
        st.markdown(f"### Recommendation: :{color}[{rec}]")
        st.write(f"**Why?** The trend is currently **{data['trend']}** and the RSI is **{data['rsi']}**. The technical indicators suggest this action.")
        
        with st.expander("See Advanced Explanation"):
            # Use the Chat Agent for a one-off explanation
            exp = chat_node.chat(f"Explain why {rec} is good for {data['symbol']} with RSI {data['rsi']}", [])
            st.write(exp)
    else:
        st.warning("Run Stock Analysis first.")

# --- PORTFOLIO ---
elif page == "Portfolio Allocation":
    st.header("💼 Planner Agent")
    capital = st.number_input("Capital", 10000)
    
    # Custom vs AI
    mode = st.radio("Strategy Mode", ["AI Recommendation", "Build My Own"])
    
    if mode == "AI Recommendation":
        if st.button("Generate Strategy"):
            sent = analyst_node.check_market_sentiment()
            risk = st.session_state.profile.get("risk", 10)
            alloc = planner_node.create_allocation(risk, sent)
            
            st.write(f"Market Context: **{sent}**")
            
            # Pie Chart
            fig = go.Figure(data=[go.Pie(labels=list(alloc.keys()), values=list(alloc.values()))])
            st.plotly_chart(fig)
            
            st.success("Strategy Generated based on Sentiment & Risk Profile.")
            memory_node.memorize(f"AI Plan: {alloc} for {capital}", {"type": "plan"})
            
    else: # Build My Own
        c1, c2, c3 = st.columns(3)
        u_eq = c1.number_input("Equity %", 0, 100, 50)
        u_db = c2.number_input("Debt %", 0, 100, 30)
        u_gd = c3.number_input("Gold %", 0, 100, 20)
        
        if u_eq + u_db + u_gd == 100:
            if st.button("Analyze My Plan"):
                msg = f"My plan: Equity {u_eq}%, Debt {u_db}%, Gold {u_gd}%. Risk level {st.session_state.profile.get('risk', 10)}. Is this safe?"
                advice = chat_node.chat(msg, [])
                st.info(advice)
                memory_node.memorize(f"User Plan: Equity {u_eq}, Debt {u_db}", {"type": "plan"})
        else:
            st.error("Total must be 100%")

# --- FINANCIAL PLANNING ---
elif page == "Financial Planning":
    st.header("🔮 Wealth Agent")
    sip = st.number_input("SIP Amount", 5000)
    yrs = st.slider("Years", 1, 30, 10)
    ret = st.slider("Expected Return (%)", 5, 20, 12)
    
    total_invested = sip * 12 * yrs
    future_value = sip * (((1+ret/100/12)**(yrs*12)-1)/(ret/100/12))
    
    c1, c2 = st.columns(2)
    c1.metric("Total Invested", f"₹{total_invested:,.0f}")
    c2.metric("Expected Wealth", f"₹{future_value:,.0f}")
    
    st.area_chart([sip * (((1+ret/100/12)**m-1)/(ret/100/12)) for m in range(1, yrs*12+1)])
    
    if st.button("Get Advice"):
        advice = chat_node.chat(f"Is a SIP of {sip} for {yrs} years good for wealth creation?", [])
        st.write(advice)

# --- EDUCATION (CHATBOT) ---
elif page == "Education (Chatbot)":
    st.header("🎓 Conversational Tutor Agent")
    st.info("I can see! Upload a chart and I will analyze it.")

    # 1. Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            # If msg content is a list (multimodal), just show the text part for history
            if isinstance(msg["content"], list):
                st.write(msg["content"][0]["text"])
            else:
                st.write(msg["content"])

    # 2. Image Input (Sidebar or Main)
    with st.sidebar:
        st.subheader("📷 Visual Input")
        uploaded_file = st.file_uploader("Upload Chart", type=["jpg", "png", "jpeg"])
        img_b64 = None
        if uploaded_file:
            st.image(uploaded_file, caption="Ready to Analyze")
            img_b64 = encode_image(uploaded_file)

    # 3. Chat Input
    prompt = st.chat_input("Ask a question about finance or the image...")
    
    if prompt:
        # Display User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("FinBot is analyzing..."):
                response = chat_node.chat(prompt, st.session_state.chat_history[:-1], img_b64)
                st.write(response)
        
        # Save AI Response
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- MEMORY LOGS ---
elif page == "Memory Logs":
    st.header("📜 Agent Memory")
    if st.button("Refresh from Pinecone"):
        logs = memory_node.recall("investment analysis", top_k=5)
        if logs:
            st.success("Found relevant memories:")
            for l in logs:
                st.info(l.get('text'))
        else:
            st.write("Memory empty.")

st.sidebar.divider()
st.sidebar.caption("System: Multi-Agent RAG + Vision")
