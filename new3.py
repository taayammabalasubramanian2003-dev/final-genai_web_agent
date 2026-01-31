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
    """Checks Nifty 50 to see if market is Bullish or Bearish today"""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        if hist.empty: return "NEUTRAL"
        
        change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
        if change > 1.5: return "BULLISH"
        elif change < -1.5: return "BEARISH"
        else: return "NEUTRAL"
    except:
        return "NEUTRAL"

# --- DATABASE FUNCTIONS (UPDATED FOR HISTORY LIST) ---
DB_FILE = "user_portfolio.json"

def save_portfolio_to_db(user_name, date, allocation, capital, companies):
    """Saves user investment plan to a local file (Appends to list)"""
    data = {}
    
    # Load existing DB
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = {} # Handle empty/corrupt file

    # Initialize user list if not exists
    if user_name not in data:
        data[user_name] = []
    
    # Create new entry
    new_entry = {
        "date": str(date),
        "capital": capital,
        "allocation": allocation,
        "companies": companies, # User selected companies
        "status": "Active"
    }
    
    # Append and Save
    data[user_name].append(new_entry)
    
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_portfolio_history(user_name):
    """Loads all past investments for a user"""
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        try:
            data = json.load(f)
            return data.get(user_name, [])
        except:
            return []

# =========================
# 3. SIDEBAR & SESSION
# =========================
if "profile_created" not in st.session_state:
    st.session_state.profile_created = False

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
    st.image("https://placehold.co/800x400/png?text=Smart+Investing+Dashboard", caption="Smart Investing for Everyone")
    st.markdown("""
    ### Welcome to your Personal Investment Assistant
    **New Updates:**
    * **📝 Custom Choices:** Choose your own companies for the portfolio.
    * **📜 Tabular History:** View all your past investments in a clean table.
    * **🧠 AI Review:** Click a button to review any past decision.
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
    symbol = st.text_input("Enter Symbol (e.g., TATAMOTORS.NS)")
    if st.button("Analyze") and symbol:
        st.session_state.symbol = symbol
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        if not df.empty:
            st.session_state.trend = "BULLISH" if df["Close"].iloc[-1] > df["Close"].rolling(50).mean().iloc[-1] else "BEARISH"
            st.session_state.current_price = df["Close"].iloc[-1]
            st.session_state.stock_analyzed = True
            st.line_chart(df["Close"])
            st.success(f"Analyzed {symbol}. Trend: {st.session_state.trend}")
        else:
            st.error("Invalid Symbol")

# =========================
# PAGE: AI DECISION
# =========================
elif page == "AI Decision":
    st.header("🧠 AI Buy/Sell Decision")
    if st.session_state.get("stock_analyzed"):
        if st.button("Get AI Advice"):
            prompt = f"Stock: {st.session_state.symbol}, Price: {st.session_state.current_price}, Trend: {st.session_state.trend}. Buy, Hold or Sell?"
            st.write(ai_explain(prompt))
    else:
        st.warning("Analyze a stock first.")

# =========================
# PAGE: PORTFOLIO (UPDATED WITH COMPANY SELECTION)
# =========================
elif page == "Portfolio Allocation":
    st.header("💼 Create Investment Portfolio")
    
    if not st.session_state.profile_created:
        st.warning("Please create a profile first.")
        st.stop()

    name = st.session_state.name
    capital = st.number_input("Investment Amount (₹)", value=50000)
    
    # Dynamic Sentiment Logic
    sentiment = get_market_sentiment()
    st.info(f"Market Sentiment: **{sentiment}**")
    
    # Suggested Split
    if sentiment == "BEARISH":
        rec = {"Equity": 30, "Debt": 50, "Gold": 20}
    elif sentiment == "BULLISH":
        rec = {"Equity": 60, "Debt": 30, "Gold": 10}
    else:
        rec = {"Equity": 50, "Debt": 30, "Gold": 20}

    st.subheader("1️⃣ Asset Allocation (%)")
    c1, c2, c3 = st.columns(3)
    eq_pct = c1.number_input("Equity %", value=rec["Equity"])
    db_pct = c2.number_input("Debt %", value=rec["Debt"])
    gd_pct = c3.number_input("Gold %", value=rec["Gold"])

    if eq_pct + db_pct + gd_pct != 100:
        st.error("Total must be 100%")
        st.stop()

    st.subheader("2️⃣ Choose Companies / Funds")
    st.caption("We suggest some leaders, but you can type your own choice.")
    
    # User Input for Specific Companies
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Equity Choice**")
        eq_choice = st.text_input("Stock Name", value="Nifty 50 ETF" if eq_pct > 0 else "None")
    
    with col2:
        st.markdown("**Debt Choice**")
        db_choice = st.text_input("Fund Name", value="Liquid Fund" if db_pct > 0 else "None")
        
    with col3:
        st.markdown("**Gold Choice**")
        gd_choice = st.text_input("ETF Name", value="Gold BeES" if gd_pct > 0 else "None")

    # Review Button
    if st.button("💾 Save & Review Portfolio"):
        companies = {
            "Equity": eq_choice,
            "Debt": db_choice,
            "Gold": gd_choice
        }
        allocation = {"Equity": eq_pct, "Debt": db_pct, "Gold": gd_pct}
        
        # Save to DB
        save_portfolio_to_db(name, datetime.date.today(), allocation, capital, companies)
        
        st.success("✅ Portfolio Saved to History!")
        
        # AI Immediate Review
        st.subheader("🤖 AI Instant Review")
        with st.spinner("Analyzing your choices..."):
            prompt = f"""
            User Portfolio Review:
            Capital: {capital}
            Allocation: {allocation}
            Selected Companies: {companies}
            Market Sentiment: {sentiment}
            
            Rate this portfolio (1-10) and give 1 suggestion to improve.
            """
            st.write(ai_explain(prompt))

# =========================
# PAGE: HISTORY (TABULAR + REVIEW)
# =========================
elif page == "History":
    st.header("📜 Investment History")
    
    name = st.session_state.get("name")
    if not name:
        st.warning("Please create a profile to see history.")
        st.stop()
        
    history = load_portfolio_history(name)
    
    if not history:
        st.info("No records found. Go to 'Portfolio Allocation' to add one.")
    else:
        # Convert JSON List to DataFrame for Table
        table_data = []
        for i, entry in enumerate(history):
            row = {
                "ID": i, # Hidden ID for selection
                "Date": entry['date'],
                "Capital (₹)": entry['capital'],
                "Equity %": entry['allocation']['Equity'],
                "Debt %": entry['allocation']['Debt'],
                "Gold %": entry['allocation']['Gold'],
                "Equity Co.": entry['companies'].get('Equity', '-'),
                "Debt Co.": entry['companies'].get('Debt', '-'),
                "Gold Co.": entry['companies'].get('Gold', '-')
            }
            table_data.append(row)
            
        df = pd.DataFrame(table_data)
        
        # Display Table (Exclude ID from view if possible, but keep for logic)
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True)
        
        st.divider()
        st.subheader("🧐 Review Past Portfolio")
        
        # Selectbox to choose which entry to review
        options = [f"{row['Date']} - ₹{row['Capital (₹)']}" for row in table_data]
        selected_option = st.selectbox("Select a Record to Review with AI:", options)
        
        if st.button("Analyze Selected Record"):
            # Find the selected data
            index = options.index(selected_option)
            selected_record = history[index]
            
            with st.spinner("AI is analyzing past performance potential..."):
                prompt = f"""
                Review this past portfolio created on {selected_record['date']}:
                Capital: {selected_record['capital']}
                Allocation: {selected_record['allocation']}
                Companies Chosen: {selected_record['companies']}
                
                1. Was this a balanced choice?
                2. What are the risks of these specific companies?
                """
                st.write(ai_explain(prompt))

# =========================
# PAGE: EDUCATION
# =========================
elif page == "Education":
    st.header("🎓 Education & Q&A")
    q = st.text_input("Ask a question:")
    if st.button("Ask"):
        st.write(ai_explain(q))

# =========================
# PAGE: FINANCIAL PLANNING
# =========================
elif page == "Financial Planning":
    st.header("🔮 SIP Calculator")
    amt = st.number_input("Monthly Amount", value=5000)
    yrs = st.slider("Years", 1, 30, 10)
    val = amt * (((1+0.12/12)**(yrs*12)-1)/(0.12/12))
    st.metric(f"Value after {yrs} Years", f"₹{val:,.0f}")
