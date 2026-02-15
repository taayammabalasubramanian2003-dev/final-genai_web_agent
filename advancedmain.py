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
# 1. SYSTEM CONFIGURATION & SETUP
# =========================
st.set_page_config(page_title="AI Investment Analyst (Advanced)", layout="wide", page_icon="📈")

# --- Load API Keys ---
try:
    # Try getting from Streamlit secrets (Cloud deployment)
    api_key = st.secrets["OPENAI_API_KEY"]
    pc_key = st.secrets["PINECONE_API_KEY"]
except:
    # Fallback to environment variables (Local testing)
    api_key = os.getenv("OPENAI_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")

if not api_key or not pc_key:
    st.error("❌ Missing Keys! Please set OPENAI_API_KEY and PINECONE_API_KEY in Streamlit Secrets.")
    st.stop()

# Initialize Clients
client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini"

# =========================
# 2. THE MULTI-AGENT BACKEND (Classes)
# =========================

class MemoryAgent:
    """
    The Brain: Handles Long-Term Vector Memory (Pinecone) & Short-Term Context
    """
    def __init__(self, api_key, index_name="financial-memory"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Auto-create index if missing
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            try:
                self.pc.create_index(
                    name=index_name,
                    dimension=1536, # OpenAI embedding size
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                time.sleep(2) # Allow initialization
            except Exception as e:
                st.warning(f"Memory initialization note: {e}")

        self.index = self.pc.Index(index_name)

    def get_embedding(self, text):
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding

    def store_memory(self, user_name, text, metadata):
        """Saves portfolio/analysis to Vector DB"""
        try:
            vector = self.get_embedding(text)
            id = f"{user_name}_{int(time.time())}"
            # Ensure metadata is flat and string-based for safety
            clean_meta = {k: str(v) for k, v in metadata.items() if v is not None}
            clean_meta["text"] = text
            clean_meta["user"] = user_name
            self.index.upsert(vectors=[(id, vector, clean_meta)])
            return True
        except Exception as e:
            st.error(f"Memory Error: {e}")
            return False

    def retrieve_memory(self, user_name, query, top_k=5):
        """Retrieves relevant past history"""
        try:
            vector = self.get_embedding(query)
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter={"user": user_name}
            )
            # Return list of dictionaries for tabular display
            return [
                {
                    "Date": datetime.datetime.fromtimestamp(int(match['id'].split('_')[1])).strftime('%Y-%m-%d'),
                    "Summary": match['metadata']['text'],
                    "Type": match['metadata'].get('type', 'General')
                } 
                for match in results['matches']
            ]
        except:
            return []

class AnalystAgent:
    """
    The Quant: Handles Technical Analysis & Data Fetching
    """
    def analyze_stock(self, symbol, period, interval):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
            if df.empty: return None, None

            df.reset_index(inplace=True)
            
            # --- Technical Calculations ---
            # Moving Averages
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()
            
            # Trend
            trend = "BULLISH" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "BEARISH"
            
            # RSI
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rs = gain.rolling(14).mean() / loss.rolling(14).mean()
            df["RSI"] = 100 - (100 / (1 + rs))
            rsi_val = df["RSI"].iloc[-1]
            
            # MACD
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            macd_signal = "BULLISH" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "BEARISH"

            # Fundamental Snapshot
            info = stock.get_info()
            fundamentals = {
                "Sector": info.get("sector", "N/A"),
                "PE": info.get("trailingPE", "N/A"),
                "EPS": info.get("trailingEps", "N/A"),
                "MarketCap": info.get("marketCap", "N/A")
            }

            results = {
                "symbol": symbol,
                "price": df["Close"].iloc[-1],
                "trend": trend,
                "rsi": rsi_val,
                "macd_signal": macd_signal,
                "fundamentals": fundamentals
            }
            return df, results
        except Exception as e:
            return None, str(e)

    def get_market_sentiment(self):
        """Checks Nifty 50 for Global Sentiment"""
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            if hist.empty: return "NEUTRAL"
            change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
            if change > 1.5: return "BULLISH"
            elif change < -1.5: return "BEARISH"
            else: return "NEUTRAL"
        except: return "NEUTRAL"

class AdvisorAgent:
    """
    The Communicator: Generates explanations using LLM
    """
    def generate_response(self, system_role, user_prompt):
        try:
            response = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ AI Error: {e}"

# --- Instantiate Agents ---
memory_agent = MemoryAgent(pc_key)
analyst_agent = AnalystAgent()
advisor_agent = AdvisorAgent()

# =========================
# 3. SESSION STATE MANAGEMENT
# =========================
if "profile_created" not in st.session_state: st.session_state.profile_created = False
if "chat_history" not in st.session_state: st.session_state.chat_history = [] # For Education Chatbot

# =========================
# 4. FRONTEND UI (Streamlit)
# =========================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Profile", 
    "Stock Analysis (Phase 2)", 
    "AI Decision (Phase 4)", 
    "Portfolio Allocation (Phase 3)", 
    "Financial Planning (Phase 5)", 
    "Education (Phase 6)", 
    "History / Memory (Phase 7)"
])

# --- HOME PAGE ---
if page == "Home":
    st.title("🤖 AI Investment Analyst Agent")
    st.image("https://placehold.co/1000x400/png?text=Smart+Investing+Dashboard", caption="Smart Investing for Everyone")
    st.markdown("""
    ### Welcome to your Advanced Personal Investment Assistant
    Powered by **Multi-Agent Architecture** & **Pinecone Vector Memory**.
    
    **Capabilities:**
    * 🧠 **Memory Agent:** Remembers your portfolio history semantically.
    * 📊 **Analyst Agent:** Performs real-time technical & fundamental analysis.
    * 💬 **Advisor Agent:** Explains complex data in simple English.
    """)
    
    if st.button("🧪 Test System Connection"):
        with st.spinner("Pinging Agents..."):
            res = advisor_agent.generate_response("You are a helper.", "Say 'System Online'")
            st.success(f"✅ {res}")

# --- PROFILE PAGE ---
elif page == "Profile":
    st.header("👤 Investor Profile")
    with st.form("profile_form"):
        st.session_state.name = st.text_input("Your Name", value=st.session_state.get("name", ""))
        st.session_state.age = st.number_input("Age", 18, 100, value=st.session_state.get("age", 25))
        st.session_state.income = st.number_input("Monthly Income (₹)", 0, value=st.session_state.get("income", 50000))
        st.session_state.savings = st.number_input("Monthly Savings (₹)", 0, value=st.session_state.get("savings", 10000))
        st.session_state.risk = st.slider("Risk Appetite (1=Safe, 20=Risky)", 1, 20, value=st.session_state.get("risk", 10))
        if st.form_submit_button("Save Profile"):
            st.session_state.profile_created = True
            st.success("✅ Profile Saved! Agents updated.")

# --- STOCK ANALYSIS PAGE ---
elif page == "Stock Analysis (Phase 2)":
    st.header("📊 Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., INFY.NS, RELIANCE.NS, TSLA)")
    mode = st.selectbox("Mode", ["INVESTOR", "TRADER"])
    
    if st.button("Analyze Stock") and symbol:
        st.session_state.symbol = symbol
        period = "5y" if mode == "INVESTOR" else "6mo"
        interval = "1mo" if mode == "INVESTOR" else "1d"
        
        with st.spinner("Analyst Agent is crunching numbers..."):
            df, results = analyst_agent.analyze_stock(symbol, period, interval)
            
            if df is None:
                st.error("❌ Stock not found.")
            else:
                st.session_state.analysis_results = results # Save for Phase 4
                st.session_state.stock_analyzed = True
                
                # Visuals
                st.subheader("🕯️ Price Action & Indicators")
                fig = go.Figure()
                fig.add_candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")
                fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA50", line=dict(color='orange')))
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Trend", results['trend'])
                c2.metric("RSI", f"{results['rsi']:.2f}")
                c3.metric("MACD Signal", results['macd_signal'])
                
                st.subheader("📉 RSI Indicator")
                st.line_chart(df.set_index("Date")["RSI"])
                
                st.divider()
                st.subheader("🏦 Fundamental Snapshot")
                f = results['fundamentals']
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Sector", f['Sector'])
                fc2.metric("P/E Ratio", f['PE'])
                fc3.metric("EPS", f['EPS'])
                fc4.metric("Market Cap", f"₹{f['MarketCap']}")

# --- AI DECISION PAGE ---
elif page == "AI Decision (Phase 4)":
    st.header("🧠 AI Decision Agent")
    if not st.session_state.get("stock_analyzed"):
        st.warning("⚠️ Please run Stock Analysis first.")
    else:
        res = st.session_state.analysis_results
        
        # Score Logic
        score = 0
        reasons = []
        if res['trend'] == "BULLISH": score+=1; reasons.append("✅ Trend is Bullish (MA20 > MA50)")
        else: reasons.append("❌ Trend is Bearish")
        if res['rsi'] > 50: score+=1; reasons.append("✅ RSI indicates momentum (>50)")
        else: reasons.append("❌ RSI indicates weakness")
        if res['macd_signal'] == "BULLISH": score+=1; reasons.append("✅ MACD is positive")
        else: reasons.append("❌ MACD is negative")
        
        decision = "BUY" if score >= 2 else "HOLD" if score == 1 else "WAIT/SELL"
        color = "green" if decision == "BUY" else "orange" if decision == "HOLD" else "red"
        
        st.markdown(f"<h2 style='color:{color};'>Recommendation: {decision}</h2>", unsafe_allow_html=True)
        st.write("### Reasoning:")
        for r in reasons: st.write(r)
        
        st.divider()
        st.subheader("🤖 Agent Explanation")
        with st.spinner("Advisor Agent is writing report..."):
            prompt = f"""
            Analyze {res['symbol']}. Price: {res['price']}. Trend: {res['trend']}. RSI: {res['rsi']}.
            Explain simply why the recommendation is {decision}. Max 3 sentences.
            """
            explanation = advisor_agent.generate_response("You are a financial expert.", prompt)
            st.write(explanation)

# --- PORTFOLIO ALLOCATION PAGE ---
elif page == "Portfolio Allocation (Phase 3)":
    st.header("💼 Intelligent Portfolio Manager")
    
    if not st.session_state.profile_created:
        st.warning("⚠️ Create Profile first.")
        st.stop()
        
    # Memory Check (RAG)
    past_plans = memory_agent.retrieve_memory(st.session_state.name, "portfolio allocation", top_k=1)
    if past_plans:
        st.info(f"📂 **Memory Agent found past plan:** {past_plans[0]['Summary']}")
    
    capital = st.number_input("Investment Amount (₹)", 1000, step=500, value=100000)
    sentiment = analyst_agent.get_market_sentiment()
    st.write(f"**Market Sentiment:** {sentiment}")
    
    # Strategy Logic
    if sentiment == "BEARISH": rec = {"Equity": 30, "Debt": 50, "Gold": 20}
    elif sentiment == "BULLISH": rec = {"Equity": 60, "Debt": 30, "Gold": 10}
    else: rec = {"Equity": 50, "Debt": 30, "Gold": 20}
    
    st.subheader("📊 Asset Allocation Strategy")
    c1, c2, c3 = st.columns(3)
    eq = c1.number_input("Equity %", value=rec['Equity'])
    db = c2.number_input("Debt %", value=rec['Debt'])
    gd = c3.number_input("Gold %", value=rec['Gold'])
    
    if eq+db+gd != 100: st.error("Total must be 100%")
    else:
        # Chart
        fig = go.Figure(data=[go.Pie(labels=["Equity", "Debt", "Gold"], values=[eq, db, gd], hole=.3)])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🏢 Company Selection")
        custom = st.checkbox("Choose custom companies")
        if not custom:
            companies = {"Equity": "Nifty 50 ETF", "Debt": "Liquid Fund", "Gold": "Gold BeES"}
            st.json(companies)
        else:
            col1, col2, col3 = st.columns(3)
            e_c = col1.text_input("Equity", "Reliance")
            d_c = col2.text_input("Debt", "SBI Fund")
            g_c = col3.text_input("Gold", "SGB")
            companies = {"Equity": e_c, "Debt": d_c, "Gold": g_c}
            
            if st.button("Analyze Custom Plan"):
                with st.spinner("Advisor Agent checking risks..."):
                    p = f"User wants to invest in {companies}. Risk level {st.session_state.risk}. Is this safe?"
                    st.write(advisor_agent.generate_response("Financial Risk Analyst", p))

        if st.button("💾 Save Portfolio to Memory"):
            text_record = f"Portfolio: {eq}% Equity ({companies['Equity']}), {db}% Debt, {gd}% Gold. Capital: {capital}. Date: {datetime.date.today()}"
            meta = {"type": "portfolio", "capital": capital, "equity": eq}
            success = memory_agent.store_memory(st.session_state.name, text_record, meta)
            if success: st.success("✅ Saved to Pinecone Vector Database!")

# --- FINANCIAL PLANNING PAGE ---
elif page == "Financial Planning (Phase 5)":
    st.header("🔮 SIP Calculator")
    sip = st.number_input("Monthly SIP (₹)", value=5000)
    yrs = st.slider("Years", 1, 30, 10)
    ret = st.slider("Exp Return %", 5, 20, 12)
    
    total = sip * 12 * yrs
    future = sip * (((1+ret/100/12)**(yrs*12)-1)/(ret/100/12))
    
    c1, c2 = st.columns(2)
    c1.metric("Invested", f"₹{total:,.0f}")
    c2.metric("Future Value", f"₹{future:,.0f}")
    
    vals = [sip * (((1+ret/100/12)**m-1)/(ret/100/12)) for m in range(1, yrs*12+1)]
    st.area_chart(vals)
    
    if st.button("Get AI Advice"):
        st.write(advisor_agent.generate_response("Financial Planner", f"Explain benefit of SIP of {sip} for {yrs} years at {ret}% return."))

# --- EDUCATION PAGE (Chatbot) ---
elif page == "Education (Phase 6)":
    st.header("🎓 Education Agent")
    
    # Predefined
    topic = st.selectbox("Learn Topic:", ["RSI", "MACD", "Candlestick", "SIP"])
    if st.button("Explain"):
        st.write(advisor_agent.generate_response("Teacher", f"Explain {topic} simply."))
        
    st.divider()
    st.subheader("💬 Chat with Finance Tutor")
    
    # Chat Interface
    user_input = st.text_input("Ask a follow-up question:")
    if st.button("Ask"):
        st.session_state.chat_history.append(f"User: {user_input}")
        # Send history context to AI
        context = "\n".join(st.session_state.chat_history[-5:]) # Last 5 messages
        reply = advisor_agent.generate_response("Tutor", f"History: {context}\nUser asks: {user_input}")
        st.session_state.chat_history.append(f"AI: {reply}")
    
    # Display Chat
    for msg in st.session_state.chat_history:
        st.text(msg)

# --- HISTORY PAGE ---
elif page == "History / Memory (Phase 7)":
    st.header("📜 Investment Memory (Vector DB)")
    if not st.session_state.profile_created:
        st.warning("Login (Profile) to see memory.")
        st.stop()
        
    if st.button("🔄 Load Memory from Pinecone"):
        history = memory_agent.retrieve_memory(st.session_state.name, "portfolio analysis investment", top_k=10)
        if history:
            df = pd.DataFrame(history)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No memories found in Pinecone.")

st.sidebar.divider()
st.sidebar.caption(f"System: Multi-Agent RAG | Model: {MODEL_VERSION}")
