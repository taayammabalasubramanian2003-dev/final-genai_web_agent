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

# =========================
# 1. SYSTEM CONFIGURATION
# =========================
st.set_page_config(page_title="AI Financial Hive (Advanced)", layout="wide", page_icon="🧠")

# Load Keys
try:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    pc_key = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
except:
    st.error("❌ Missing API Keys! Please add OPENAI_API_KEY and PINECONE_API_KEY to secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini"

# =========================
# 2. MEMORY AGENT (Pinecone / RAG)
# =========================
class MemoryAgent:
    def __init__(self, api_key, index_name="financial-memory"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Check if index exists, if not create it (Serverless)
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            try:
                self.pc.create_index(
                    name=index_name,
                    dimension=1536, # OpenAI embedding size
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(2) # Wait for init
            except Exception as e:
                st.warning(f"Index creation skipped/failed: {e}")

        self.index = self.pc.Index(index_name)

    def get_embedding(self, text):
        """Converts text concept into vector numbers"""
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding

    def store_memory(self, user_name, text, metadata):
        """Saves an experience to the Vector Brain"""
        try:
            vector = self.get_embedding(text)
            # Create unique ID based on time
            id = f"{user_name}_{int(time.time())}"
            
            # Metadata must be simple strings/numbers for Pinecone
            clean_meta = {k: str(v) for k, v in metadata.items()}
            clean_meta["text"] = text
            clean_meta["user"] = user_name
            
            self.index.upsert(vectors=[(id, vector, clean_meta)])
            return True
        except Exception as e:
            return f"Memory Error: {e}"

    def retrieve_memory(self, user_name, query, top_k=2):
        """RAG: Finds relevant past memories"""
        try:
            vector = self.get_embedding(query)
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter={"user": user_name} # Only search this user's data
            )
            return [match['metadata']['text'] for match in results['matches']]
        except:
            return []

# Initialize Memory Agent
memory_agent = MemoryAgent(pc_key)

# =========================
# 3. SPECIALIZED AGENTS (Logic Layer)
# =========================

class MarketAnalystAgent:
    """Agent responsible for Mathematics & Data Fetching"""
    def analyze(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty: return None
            
            # Technical Calcs
            current_price = df["Close"].iloc[-1]
            ma50 = df["Close"].rolling(50).mean().iloc[-1]
            trend = "BULLISH" if current_price > ma50 else "BEARISH"
            
            # RSI Calc
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                "symbol": symbol,
                "price": round(current_price, 2),
                "trend": trend,
                "rsi": round(rsi.iloc[-1], 2),
                "ma50": round(ma50, 2),
                "chart_data": df["Close"]
            }
        except: return None

class PlannerAgent:
    """Agent responsible for Rules & Risk Logic"""
    def allocate(self, risk_level, market_sentiment):
        # Rule Engine
        if market_sentiment == "BEARISH":
            # Defensive mode
            return {"Equity": 30, "Debt": 50, "Gold": 20}
        elif risk_level >= 15:
            # Aggressive mode
            return {"Equity": 70, "Debt": 20, "Gold": 10}
        else:
            # Balanced mode
            return {"Equity": 50, "Debt": 30, "Gold": 20}

class AdvisorAgent:
    """Agent responsible for Synthesis & Communication (LLM)"""
    def synthesize(self, context, user_query):
        system_prompt = """
        You are a Senior Investment Advisor Agent.
        Use the provided Context (Data, Trends, Risks) to answer the User.
        Do not make up numbers. Use the data provided.
        Be professional but concise.
        """
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {user_query}"}
            ]
        )
        return response.choices[0].message.content

# Initialize Agents
analyst = MarketAnalystAgent()
planner = PlannerAgent()
advisor = AdvisorAgent()

# =========================
# 4. ORCHESTRATOR (Streamlit UI)
# =========================

# Session State
if "user_name" not in st.session_state: st.session_state.user_name = "Guest"
if "analysis_cache" not in st.session_state: st.session_state.analysis_cache = {}

st.sidebar.image("https://placehold.co/100x100/png?text=Agent+Hive", caption="Multi-Agent System")
st.sidebar.title("Agent Navigation")
mode = st.sidebar.radio("Active Module", ["Dashboard", "Analyst Agent", "Planner Agent", "Memory Agent"])

# --- DASHBOARD ---
if mode == "Dashboard":
    st.title("🧠 AI Financial Hive (Multi-Agent)")
    st.markdown("""
    This system uses an advanced **Agent-to-Agent (A2A)** workflow with **Vector Memory**.
    
    ### 🔗 Active Protocol:
    1.  **Analyst Agent:** Fetches raw data & computes indicators.
    2.  **Memory Agent (RAG):** Checks Pinecone for your past similar moves.
    3.  **Planner Agent:** optimizing allocation based on rules.
    4.  **Advisor Agent:** Synthesizes all inputs into advice.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📡 **Market Signal:** Live Nifty 50 Trend")
        # Quick Sentiment Check
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            change = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
            sentiment = "BULLISH 🟢" if change > 0 else "BEARISH 🔴"
            st.metric("Nifty 50 Sentiment", sentiment, f"{change*100:.2f}%")
        except:
            st.warning("Market Data Unavailable")

# --- ANALYST AGENT ---
elif mode == "Analyst Agent":
    st.header("📊 Market Analyst Agent")
    st.caption("Capabilities: Technical Analysis, Pattern Recognition")
    
    symbol = st.text_input("Ticker Symbol", "TCS.NS")
    
    if st.button("Run Analysis Protocol"):
        with st.spinner("Analyst is computing..."):
            data = analyst.analyze(symbol)
            
            if data:
                # 1. Display Data
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"₹{data['price']}")
                c2.metric("Trend", data['trend'])
                c3.metric("RSI", data['rsi'])
                st.line_chart(data['chart_data'])
                
                # 2. Advisor Synthesis
                st.subheader("📝 Advisor Insight")
                explanation = advisor.synthesize(
                    context=str(data),
                    user_query=f"Is {symbol} a good buy given this technical data?"
                )
                st.write(explanation)
                
                # 3. Save to Long-Term Memory
                mem_text = f"Analyzed {symbol} at ₹{data['price']}. Trend was {data['trend']}."
                memory_agent.store_memory(st.session_state.user_name, mem_text, {"type": "analysis", "symbol": symbol})
                st.toast("✅ Analysis saved to Vector Memory")
                
            else:
                st.error("Symbol not found.")

# --- PLANNER AGENT ---
elif mode == "Planner Agent":
    st.header("⚖️ Portfolio Planner Agent")
    st.caption("Capabilities: Risk Assessment, Asset Allocation, RAG Memory")
    
    st.session_state.user_name = st.text_input("User Name (for Memory Recall)", st.session_state.user_name)
    risk = st.slider("Risk Tolerance (1-20)", 1, 20, 10)
    
    if st.button("Generate Strategy"):
        # 1. RAG Step: Retrieve Past Context
        st.subheader("🧠 Memory Retrieval")
        past_memories = memory_agent.retrieve_memory(st.session_state.user_name, "portfolio plan risk", top_k=1)
        
        if past_memories:
            st.info(f"**I remember you previously:** {past_memories[0]}")
        else:
            st.caption("No prior relevant history found.")
            
        # 2. Planning Step
        # Get sentiment dynamically
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            change = (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
            sentiment = "BULLISH" if change > 0 else "BEARISH"
        except: sentiment = "NEUTRAL"
        
        allocation = planner.allocate(risk, sentiment)
        
        # 3. Visualization
        st.subheader("📍 Recommended Allocation")
        df = pd.DataFrame(list(allocation.items()), columns=["Asset", "Percentage"])
        fig = go.Figure(data=[go.Pie(labels=df['Asset'], values=df['Percentage'])])
        st.plotly_chart(fig)
        
        # 4. Save to Memory
        mem_text = f"Created Portfolio Plan. Risk: {risk}. Market: {sentiment}. Allocation: {allocation}"
        memory_agent.store_memory(st.session_state.user_name, mem_text, {"type": "plan", "risk": risk})
        st.success("Strategy executed and saved to long-term memory.")

# --- MEMORY AGENT ---
elif mode == "Memory Agent":
    st.header("🧠 Knowledge & Memory Agent")
    st.caption("Interface to the Pinecone Vector Database")
    
    query = st.text_input("Ask your financial history (Semantic Search)")
    if st.button("Search Memory"):
        if query:
            with st.spinner("Searching neural database..."):
                results = memory_agent.retrieve_memory(st.session_state.user_name, query, top_k=3)
                if results:
                    for i, r in enumerate(results):
                        st.success(f"**Result {i+1}:** {r}")
                else:
                    st.warning("No memories found matching that context.")

st.sidebar.divider()
st.sidebar.caption(f"Powered by {MODEL_VERSION} & Pinecone Vector DB")
