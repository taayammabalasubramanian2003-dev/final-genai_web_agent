import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import os
import datetime
import json

# =========================
# 1. SETUP & CONFIG
# =========================
st.set_page_config(page_title="AI Investment Analyst", layout="wide", page_icon="📈")

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("❌ OpenAI API key not found. Please check Streamlit Secrets.")
        st.stop()

client = OpenAI(api_key=api_key)
MODEL_VERSION = "gpt-4o-mini"

# =========================
# 2. HELPER FUNCTIONS
# =========================
def ai_explain(prompt):
    """Get explanation from OpenAI"""
    try:
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": "You are a smart financial advisor. Keep answers concise, helpful, and data-driven."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

def get_market_sentiment():
    """Checks Nifty 50 to see if market is Bullish or Bearish today (Dynamic Factor)"""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        if hist.empty: return "NEUTRAL"
        
        # Compare today's close vs 5 days ago
        change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
        if change > 1.5: return "BULLISH"
        elif change < -1.5: return "BEARISH"
        else: return "NEUTRAL"
    except:
        return "NEUTRAL"

# File handling for "Memory" (Simulating a database with JSON)
DB_FILE = "user_portfolio.json"

def save_portfolio_to_db(user_name, date, allocation, capital):
    """Saves user investment plan to a local file"""
    data = {}
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            data = json.load(f)
    
    data[user_name] = {
        "date": str(date),
        "allocation": allocation,
        "capital": capital,
        "status": "Active"
    }
    
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

def load_portfolio_from_db(user_name):
    """Loads user investment plan"""
    if not os.path.exists(DB_FILE):
        return None
    with open(DB_FILE, "r") as f:
        data = json.load(f)
    return data.get(user_name)

# =========================
# 3. SIDEBAR & SESSION
# =========================
if "profile_created" not in st.session_state:
    st.session_state.profile_created = False
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", "Profile", "Stock Analysis", "AI Decision", 
    "Portfolio Allocation", "Financial Planning", "Education", "History"
])

# =========================
# PAGE: HOME
# =========================
if page == "Home":
    st.title("🤖 AI Investment Analyst Agent")
    
    # ✅ FIXED IMAGE (Using a reliable placeholder service)
    st.image("https://placehold.co/800x400/png?text=Smart+Investing+Dashboard", caption="Smart Investing for Everyone")
    
    st.markdown("""
    ### Welcome to your Personal Investment Assistant
    We use **Real-Time Market Data** + **OpenAI Logic** to guide you.
    
    **New Features:**
    * **🧠 Memory:** We remember your last month's investment.
    * **⚡ Dynamic:** Portfolios adjust based on today's Nifty trend.
    * **🎓 Q&A:** Ask any doubt in the Education section.
    """)

# =========================
# PAGE: PROFILE
# =========================
elif page == "Profile":
    st.header("👤 Investor Profile")
    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.get("name", ""))
        age = st.number_input("Age", 18, 100, value=st.session_state.get("age", 25))
        risk = st.slider("Risk Appetite (1=Low, 20=High)", 1, 20, value=st.session_state.get("risk", 10))
        submitted = st.form_submit_button("Save Profile")

    if submitted:
        st.session_state.name = name
        st.session_state.age = age
        st.session_state.risk = risk
        st.session_state.profile_created = True
        st.success("✅ Profile Saved!")

# =========================
# PAGE: STOCK ANALYSIS
# =========================
elif page == "Stock Analysis":
    st.header("📊 Stock Analysis")
    symbol = st.text_input("Enter Symbol (e.g., TATAMOTORS.NS, INFY.NS)")
    
    if st.button("Analyze") and symbol:
        st.session_state.symbol = symbol
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        
        if df.empty:
            st.error("No data found.")
        else:
            # Simple MA Analysis
            df["MA50"] = df["Close"].rolling(50).mean()
            current_price = df["Close"].iloc[-1]
            ma50 = df["MA50"].iloc[-1]
            
            trend = "BULLISH" if current_price > ma50 else "BEARISH"
            
            st.metric("Current Price", f"₹{round(current_price, 2)}")
            st.metric("Trend (vs 50 DMA)", trend)
            
            st.line_chart(df["Close"])
            
            # Save for AI Decision
            st.session_state.trend = trend
            st.session_state.stock_analyzed = True
            st.session_state.current_price = current_price

# =========================
# PAGE: AI DECISION
# =========================
elif page == "AI Decision":
    st.header("🧠 AI Buy/Sell Decision")
    if not st.session_state.get("stock_analyzed"):
        st.warning("Please analyze a stock first.")
    else:
        symbol = st.session_state.symbol
        trend = st.session_state.trend
        
        prompt = f"The stock {symbol} is currently {trend}. The price is {st.session_state.current_price}. Should I Buy, Hold, or Sell for a 6-month horizon? Give 1 reason."
        
        st.subheader(f"Advice for {symbol}")
        with st.spinner("Analyzing..."):
            advice = ai_explain(prompt)
            st.info(advice)

# =========================
# PAGE: PORTFOLIO (DYNAMIC & MEMORY)
# =========================
elif page == "Portfolio Allocation":
    st.header("💼 Dynamic Portfolio Manager")
    
    if not st.session_state.profile_created:
        st.warning("Please create a profile first.")
        st.stop()

    name = st.session_state.name
    
    # 1. CHECK MEMORY (Did user invest last month?)
    saved_portfolio = load_portfolio_from_db(name)
    
    if saved_portfolio:
        st.info(f"📂 **Welcome back, {name}!** We found your investment from **{saved_portfolio['date']}**.")
        st.write("### Your Last Allocation:")
        st.json(saved_portfolio['allocation'])
        
        action = st.radio("What would you like to do?", ["Review Current Plan", "Create New Plan for This Month"])
        
        if action == "Review Current Plan":
            st.subheader("🕵️ Performance Review")
            st.write("Comparing your entry date with today's market...")
            
            # Simulated check (In real app, we would fetch historical prices of that date)
            prompt = f"User invested in {saved_portfolio['allocation']} on {saved_portfolio['date']}. Today is {datetime.date.today()}. Market is volatile. Suggest if they should HOLD this portfolio or REBALANCE (change it). Keep it short."
            review = ai_explain(prompt)
            st.write(review)
            st.stop() # Stop here if reviewing
            
    # 2. CREATE NEW / DYNAMIC ALLOCATION
    st.divider()
    st.subheader("🆕 Create New Allocation")
    capital = st.number_input("Investment Amount (₹)", value=50000)
    
    # --- DYNAMIC FACTOR ---
    sentiment = get_market_sentiment()
    st.write(f"**Market Sentiment Today:** {sentiment}")
    
    if sentiment == "BEARISH":
        st.caption("⚠️ Market is down. AI suggests increasing SAFE assets (Gold/Debt).")
        rec_equity = 30
        rec_debt = 50
        rec_gold = 20
    elif sentiment == "BULLISH":
        st.caption("🚀 Market is up. AI suggests increasing EQUITY for growth.")
        rec_equity = 60
        rec_debt = 30
        rec_gold = 10
    else:
        st.caption("⚖️ Market is neutral. Standard balanced approach.")
        rec_equity = 50
        rec_debt = 30
        rec_gold = 20

    # Allow user to customize
    c1, c2, c3 = st.columns(3)
    user_equity = c1.number_input("Equity %", value=rec_equity)
    user_debt = c2.number_input("Debt %", value=rec_debt)
    user_gold = c3.number_input("Gold %", value=rec_gold)
    
    total = user_equity + user_debt + user_gold
    if total != 100:
        st.error(f"Total must be 100%. Current: {total}%")
    else:
        st.success("Allocation is Valid ✅")
        
        # Display Specific Recommendations
        st.subheader("📌 Recommended Instruments")
        st.write(f"**Equity (₹{capital * user_equity/100:.0f}):** Nifty 50 ETF, ICICI Bank")
        st.write(f"**Debt (₹{capital * user_debt/100:.0f}):** Liquid Funds")
        st.write(f"**Gold (₹{capital * user_gold/100:.0f}):** Gold BeES")
        
        if st.button("✅ Confirm & Save Investment"):
            allocation = {"Equity": user_equity, "Debt": user_debt, "Gold": user_gold}
            save_portfolio_to_db(name, datetime.date.today(), allocation, capital)
            st.balloons()
            st.success("Investment Plan Saved! We will remember this for next month.")

# =========================
# PAGE: EDUCATION (WITH Q&A)
# =========================
elif page == "Education":
    st.header("🎓 Investor Education & Q&A")
    
    # Standard Topics
    topic = st.selectbox("Quick Topics:", ["What is SIP?", "How does Inflation hurt?", "What is a PE Ratio?"])
    if st.button("Explain Topic"):
        st.write(ai_explain(f"Explain {topic} simply."))

    st.divider()
    
    # ✅ NEW: ASK ANY QUESTION
    st.subheader("🙋‍♂️ Ask a specific question")
    user_q = st.text_input("Type your doubt here (e.g., 'Is it safe to invest in Adani now?')")
    
    if st.button("Get AI Answer"):
        if user_q:
            with st.spinner("Thinking..."):
                answer = ai_explain(f"User Question: {user_q}. Answer as a responsible financial guide.")
                st.write(answer)
        else:
            st.warning("Please type a question.")

# =========================
# OTHER PAGES
# =========================
elif page == "Financial Planning":
    st.header("🔮 Simple SIP Calculator")
    amt = st.number_input("Monthly Amount", value=5000)
    yrs = st.slider("Years", 1, 30, 10)
    ret = 0.12 # 12% avg
    val = amt * (((1+ret/12)**(yrs*12)-1)/(ret/12))
    st.metric(f"Value after {yrs} Years", f"₹{val:,.0f}")

elif page == "History":
    st.header("📜 Investment History")
    name = st.session_state.get("name")
    if name:
        data = load_portfolio_from_db(name)
        if data:
            st.write(data)
        else:
            st.info("No saved history found.")
    else:
        st.warning("Please create a profile to see history.")
