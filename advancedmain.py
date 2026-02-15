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

# =========================
# 1. ORCHESTRATOR CONFIGURATION
# =========================
st.set_page_config(page_title="AI Investment Hive (Multi-Agent)", layout="wide", page_icon="🧠")

# Load Secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    pc_key = st.secrets["PINECONE_API_KEY"]
except:
    st.error("❌ Secrets not found! Please set OPENAI_API_KEY and PINECONE_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# 2. AGENT DEFINITIONS (The "Workers")
# =========================

class MemoryAgent:
    """
    🧠 THE BRAIN: Manages Long-Term Vector Memory (Pinecone)
    Role: Stores experiences and retrieves relevant context for other agents.
    """
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "financial-memory"
        
        # Auto-create Index if it doesn't exist
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536, # Must match OpenAI embedding
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(10) # Wait for initialization
            except Exception as e:
                st.error(f"Memory Agent Initialization Failed: {e}")
        
        self.index = self.pc.Index(self.index_name)

    def memorize(self, text, metadata):
        """Encodes and saves an event"""
        try:
            # 1. Convert text to vector
            response = client.embeddings.create(input=text, model="text-embedding-3-small")
            vector = response.data[0].embedding
            
            # 2. Generate ID
            unique_id = f"mem_{int(time.time())}"
            
            # 3. Clean Metadata
            clean_meta = {k: str(v) for k, v in metadata.items()}
            clean_meta['text'] = text
            
            # 4. Upsert
            self.index.upsert(vectors=[(unique_id, vector, clean_meta)])
            return True
        except Exception as e:
            st.error(f"Memory Error: {e}")
            return False

    def recall(self, query, top_k=3):
        """Retrieves similar past events"""
        try:
            # 1. Convert query to vector
            response = client.embeddings.create(input=query, model="text-embedding-3-small")
            query_vector = response.data[0].embedding
            
            # 2. Search Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            # 3. Format Output
            return [match['metadata'] for match in results['matches']]
        except Exception as e:
            st.error(f"Recall Error: {e}")
            return []

class AnalystAgent:
    """
    📊 THE QUANT: Handles Technical & Fundamental Data
    Role: Fetches live data, calculates indicators, and detects trends.
    """
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            
            if df.empty: return None

            # Technical Indicators
            df["MA50"] = df["Close"].rolling(50).mean()
            current_price = df["Close"].iloc[-1]
            trend = "BULLISH" if current_price > df["MA50"].iloc[-1] else "BEARISH"
            
            # RSI Calculation
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD Calculation
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_sig = "BUY" if macd.iloc[-1] > signal.iloc[-1] else "SELL"

            # Fundamental Data
            info = stock.get_info()
            
            return {
                "symbol": symbol,
                "price": round(current_price, 2),
                "trend": trend,
                "rsi": round(rsi.iloc[-1], 2),
                "macd": macd_sig,
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "sector": info.get('sector', 'Unknown'),
                "history_df": df
            }
        except: return None

    def check_market_sentiment(self):
        """Checks Nifty 50 for global sentiment"""
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            change = (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
            return "BULLISH" if change > 0 else "BEARISH"
        except: return "NEUTRAL"

class PlannerAgent:
    """
    ⚖️ THE STRATEGIST: Manages Risk & Allocation
    Role: Decides how to split money based on rules and sentiment.
    """
    def create_allocation(self, risk_level, market_sentiment):
        # Logic: If market is bad, be defensive (Gold/Debt). If good, be aggressive (Equity).
        if market_sentiment == "BEARISH":
            return {"Equity": 30, "Debt": 50, "Gold": 20}
        
        # Risk-based logic
        if risk_level >= 15: # High Risk
            return {"Equity": 70, "Debt": 20, "Gold": 10}
        elif risk_level >= 8: # Moderate
            return {"Equity": 50, "Debt": 30, "Gold": 20}
        else: # Conservative
            return {"Equity": 30, "Debt": 50, "Gold": 20}

class AdvisorAgent:
    """
    💬 THE COMMUNICATOR: Synthesis & Explanation
    Role: Uses LLM to explain the findings of other agents to the user.
    """
    def explain(self, context, question):
        prompt = f"""
        You are a Senior Financial Advisor Agent. 
        Analyze the data provided by the Analyst and Planner agents.
        
        CONTEXT FROM AGENTS:
        {context}
        
        USER QUESTION:
        {question}
        
        Provide a concise, professional answer (max 3 sentences).
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# --- Initialize The Hive ---
memory_node = MemoryAgent(pc_key)
analyst_node = AnalystAgent()
planner_node = PlannerAgent()
advisor_node = AdvisorAgent()

# =========================
# 3. FRONTEND ORCHESTRATOR (UI)
# =========================

# Session State
if "profile" not in st.session_state: st.session_state.profile = {}
if "last_analysis" not in st.session_state: st.session_state.last_analysis = None

st.sidebar.title("🤖 Agent Navigation")
page = st.sidebar.radio("Select Module", [
    "Home", "Profile Setup", "Stock Analysis", "AI Decision", 
    "Portfolio Allocation", "Financial Planning", "Education", "Memory Logs"
])

# --- HOME ---
if page == "Home":
    st.title("🤖 AI Investment Hive")
    st.image("https://placehold.co/1000x400/png?text=Multi-Agent+System+Active", caption="Orchestrator Online")
    st.markdown("""
    ### System Status: **ONLINE 🟢**
    
    This platform uses a **Multi-Agent Architecture**:
    1.  **Analyst Agent:** Scans markets & calculates indicators.
    2.  **Planner Agent:** Strategizes based on your risk profile.
    3.  **Advisor Agent:** Explains logic in plain English.
    4.  **Memory Agent:** Stores your history in a **Pinecone Vector Database**.
    """)

# --- PROFILE ---
elif page == "Profile Setup":
    st.header("👤 Profile Agent")
    with st.form("profile"):
        name = st.text_input("Name", value=st.session_state.profile.get("name", "Investor"))
        risk = st.slider("Risk Appetite", 1, 20, 10)
        income = st.number_input("Monthly Income", value=50000)
        if st.form_submit_button("Save Profile"):
            st.session_state.profile = {"name": name, "risk": risk, "income": income}
            st.success("Profile updated and broadcast to all agents.")

# --- STOCK ANALYSIS ---
elif page == "Stock Analysis":
    st.header("📊 Analyst Agent Dashboard")
    symbol = st.text_input("Enter Symbol (e.g., RELIANCE.NS)")
    
    if st.button("Run Analysis Protocol"):
        with st.status("🔄 Orchestrator: Coordinating Agents...", expanded=True) as status:
            
            st.write("1️⃣ **Analyst Agent:** Fetching market data...")
            data = analyst_node.analyze(symbol)
            
            if data:
                st.session_state.last_analysis = data # Store for other agents
                st.write("2️⃣ **Analyst Agent:** Calculating RSI, MACD, Trends...")
                st.write("3️⃣ **Advisor Agent:** Synthesizing report...")
                status.update(label="✅ Analysis Complete", state="complete")
                
                # UI Display
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"₹{data['price']}")
                c2.metric("Trend", data['trend'])
                c3.metric("RSI", data['rsi'])
                st.line_chart(data['history_df']['Close'])
                
                # Memory Save
                mem_text = f"Analyzed {symbol}. Price: {data['price']}. Trend: {data['trend']}."
                memory_node.memorize(mem_text, {"type": "analysis", "symbol": symbol})
                st.toast("💾 Saved to Long-Term Memory")
                
            else:
                st.error("Analyst Agent could not find symbol.")
                status.update(label="❌ Failed", state="error")

# --- AI DECISION ---
elif page == "AI Decision":
    st.header("🧠 Decision Agent")
    if not st.session_state.last_analysis:
        st.warning("⚠️ Waiting for Analyst Agent data. Go to 'Stock Analysis' first.")
    else:
        data = st.session_state.last_analysis
        
        # Agent Logic
        score = 0
        if data['trend'] == "BULLISH": score += 1
        if data['rsi'] < 70 and data['rsi'] > 30: score += 1
        if data['macd'] == "BUY": score += 1
        
        recommendation = "BUY" if score >= 2 else "HOLD"
        
        st.info(f"**Agent Consensus:** {recommendation}")
        
        # Explanation from LLM Agent
        exp = advisor_node.explain(str(data), f"Why is the recommendation {recommendation}?")
        st.write(f"**Advisor Insight:** {exp}")

# --- PORTFOLIO ALLOCATION ---
elif page == "Portfolio Allocation":
    st.header("💼 Portfolio Planner Agent")
    
    # 1. RAG MEMORY CHECK
    st.subheader("🧠 Memory Recall")
    past_plans = memory_node.recall("portfolio allocation plan", top_k=1)
    if past_plans:
        st.success(f"**I remember your last plan:** {past_plans[0].get('text', 'No text')}")
    else:
        st.caption("No prior plans found in vector memory.")

    capital = st.number_input("Investment Amount", value=100000)
    
    if st.button("Generate Strategy"):
        # 2. Check Market
        sentiment = analyst_node.check_market_sentiment()
        st.write(f"📉 **Market Context:** {sentiment}")
        
        # 3. Planner Logic
        risk = st.session_state.profile.get("risk", 10)
        allocation = planner_node.create_allocation(risk, sentiment)
        
        # 4. Display
        df = pd.DataFrame(list(allocation.items()), columns=["Asset", "Pct"])
        fig = go.Figure(data=[go.Pie(labels=df['Asset'], values=df['Pct'])])
        st.plotly_chart(fig)
        
        # 5. Save to Memory
        mem_text = f"Portfolio Plan: {allocation} for Capital: {capital}. Market: {sentiment}"
        memory_node.memorize(mem_text, {"type": "portfolio", "risk": risk})
        st.success("✅ Plan executed and saved to neural memory.")

# --- FINANCIAL PLANNING ---
elif page == "Financial Planning":
    st.header("🔮 Wealth Projection Agent")
    sip = st.number_input("Monthly SIP", value=5000)
    yrs = st.slider("Years", 1, 30, 10)
    rate = st.slider("Rate (%)", 5, 20, 12)
    
    future_val = sip * (((1+rate/100/12)**(yrs*12)-1)/(rate/100/12))
    st.metric("Future Value", f"₹{future_val:,.0f}")
    
    chart_data = [sip * (((1+rate/100/12)**m-1)/(rate/100/12)) for m in range(1, yrs*12+1)]
    st.area_chart(chart_data)
    
    if st.button("Get Advisor Opinion"):
        st.write(advisor_node.explain(f"SIP: {sip}, Years: {yrs}", "Is this a good plan for retirement?"))

# --- EDUCATION ---
elif page == "Education":
    st.header("🎓 Education Agent")
    q = st.text_input("Ask about finance (e.g., 'What is RSI?')")
    if st.button("Ask Tutor"):
        ans = advisor_node.explain("You are a helpful tutor.", q)
        st.write(ans)

# --- HISTORY ---
elif page == "History / Memory":
    st.header("📜 Vector Database Logs")
    if st.button("Load Memories from Pinecone"):
        # Fetch generic recent memories
        mems = memory_node.recall("portfolio analysis stock", top_k=10)
        if mems:
            for m in mems:
                st.info(f"📅 **Memory:** {m.get('text')}")
        else:
            st.warning("Pinecone Index appears empty. Save some portfolios first!")

st.sidebar.divider()
st.sidebar.caption("System: Multi-Agent RAG v2.0")
