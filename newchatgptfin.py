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
MODEL_VERSION = "gpt-4o-mini-2024-07-18"

# Database File
DB_FILE = "user_portfolio.json"

# =========================
# 2. HELPER FUNCTIONS
# =========================
def ai_explain(prompt):
    """Get explanation from OpenAI"""
    try:
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": "You are a smart, responsible financial advisor. Keep answers concise, helpful, and data-driven."},
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

def save_portfolio_to_db(user_name, date, allocation, capital, companies):
    """Saves user investment plan to a local file (Appends to list)"""
    data = {}
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = {} 

    if user_name not in data:
        data[user_name] = []
    
    new_entry = {
        "date": str(date),
        "capital": capital,
        "allocation": allocation,
        "companies": companies,
        "status": "Active"
    }
    
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
# 3. SIDEBAR NAVIGATION
# =========================
if "profile_created" not in st.session_state:
    st.session_state.profile_created = False

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Profile",
        "Stock Analysis (Phase 2)",
        "AI Decision (Phase 4)",
        "Portfolio Allocation (Phase 3)",
        "Financial Planning (Phase 5)",
        "Education (Phase 6)",
        "History / Memory (Phase 7)",
        "News & Events (Phase 8)",
        "Auto Rebalancer (Phase 9)"
    ]
)


# =========================
# PAGE: HOME
# =========================
if page == "Home":
    st.title("🤖 AI Investment Analyst Agent")
    # ✅ Fixed Image
    st.image("https://placehold.co/1000x500/png?text=Smart+Investing+Dashboard", caption="Smart Investing for Everyone")
    
    st.markdown("""
    ### Welcome to your Personal Investment Assistant
    This tool combines real-time market data with OpenAI's intelligence to help you make better financial decisions.
    
    **New Features:**
    * **🧠 Memory:** We remember your portfolios month-to-month.
    * **⚡ Dynamic:** Allocation adjusts based on today's Market Sentiment.
    * **📝 Custom Choices:** Replace AI suggestions with your own companies and get them analyzed.
    * **📜 Tabular History:** Review past decisions with one click.
    """)
    
    st.subheader("🧪 OpenAI Connection Test")
    if st.button("Test AI Connection"):
        with st.spinner("Connecting to OpenAI..."):
            test_response = ai_explain("Explain stock investing to a beginner in one simple line.")
            st.success("✅ OpenAI Connected Successfully!")
            st.write(f"**AI says:** {test_response}")

# =========================
# PAGE: PROFILE
# =========================
elif page == "Profile":
    st.header("👤 Investor Profile")
    with st.form("profile_form"):
        name = st.text_input("Your Name", value=st.session_state.get("name", ""))
        age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.get("age", 25))
        income = st.number_input("Monthly Income (₹)", min_value=0, value=st.session_state.get("income", 50000))
        savings = st.number_input("Monthly Savings (₹)", min_value=0, value=st.session_state.get("savings", 10000))
        risk = st.slider("Risk Appetite (1=Low, 20=High)", 1, 20, value=st.session_state.get("risk", 10))
        
        submitted = st.form_submit_button("Save Profile")

    if submitted:
        st.session_state.name = name
        st.session_state.age = age
        st.session_state.income = income
        st.session_state.savings = savings
        st.session_state.risk = risk
        st.session_state.profile_created = True
        st.success("✅ Profile saved successfully! You can now access other features.")

# =========================
# PAGE: STOCK ANALYSIS (PHASE 2)
# =========================
elif page == "Stock Analysis (Phase 2)":
    st.header("📊 Stock Analysis")
    
    symbol = st.text_input("Enter Stock Symbol (e.g., INFY.NS, RELIANCE.NS, TSLA)")
    mode = st.selectbox("Mode", ["INVESTOR", "TRADER"])

    if st.button("Analyze Stock") and symbol:
        st.session_state.symbol = symbol  # Save for other phases
        
        period = "5y" if mode == "INVESTOR" else "6mo"
        interval = "1mo" if mode == "INVESTOR" else "1d"

        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                st.error("❌ No data found. Please check the symbol.")
            else:
                df.reset_index(inplace=True)

                # --- 1. MOVING AVERAGES ---
                df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()

                # --- 2. CANDLESTICK CHART ---
                st.subheader("🕯️ Price Action & Moving Averages")
                fig = go.Figure()
                fig.add_candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")
                fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20 (Blue)", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA50 (Orange)", line=dict(color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                trend = "BULLISH" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "BEARISH"
                st.info(f"📈 **Trend Direction:** {trend}")

                # --- 3. RSI ---
                delta = df["Close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                rs = gain.rolling(14).mean() / loss.rolling(14).mean()
                df["RSI"] = 100 - (100 / (1 + rs))
                rsi_val = df["RSI"].iloc[-1]
                
                st.subheader("📉 RSI Indicator")
                st.line_chart(df.set_index("Date")["RSI"])
                st.write(f"**Current RSI:** {round(rsi_val, 2)}")

                # --- 4. MACD ---
                ema12 = df["Close"].ewm(span=12, adjust=False).mean()
                ema26 = df["Close"].ewm(span=26, adjust=False).mean()
                df["MACD"] = ema12 - ema26
                df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
                macd_val = df["MACD"].iloc[-1]
                signal_val = df["Signal"].iloc[-1]
                macd_signal = "BULLISH" if macd_val > signal_val else "BEARISH"

                st.subheader("📊 MACD Indicator")
                st.line_chart(df.set_index("Date")[["MACD", "Signal"]])
                st.write(f"**MACD Signal:** {macd_signal}")

                # Save Analysis to Session State for Phase 4
                st.session_state.trend = trend
                st.session_state.rsi_value = rsi_val
                st.session_state.macd_signal = macd_signal
                st.session_state.stock_analyzed = True
                
                # --- 5. FUNDAMENTAL ANALYSIS ---
                st.divider()
                st.header("🏦 Fundamental Snapshot")
                info = stock.get_info()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sector", info.get("sector", "N/A"))
                c2.metric("P/E Ratio", round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A")
                c3.metric("EPS", info.get("trailingEps", "N/A"))
                c4.metric("Market Cap", f"{info.get('marketCap', 0) / 10**9:.2f}B")

        except Exception as e:
            st.error(f"Error analyzing stock: {e}")

# =========================
# PAGE: AI DECISION (PHASE 4)
# =========================
elif page == "AI Decision (Phase 4)":
    st.header("🧠 AI Decision Agent")

    if not st.session_state.get("stock_analyzed"):
        st.warning("⚠️ Please analyze a stock in the 'Stock Analysis' page first.")
    else:
        symbol = st.session_state.symbol
        trend = st.session_state.trend
        rsi = st.session_state.rsi_value
        macd = st.session_state.macd_signal
        
        st.write(f"### Analyzing: **{symbol}**")
        
        # Scoring Logic
        score = 0
        reasons = []

        if trend == "BULLISH":
            score += 1
            reasons.append("✅ Trend is Bullish (MA20 > MA50)")
        else:
            reasons.append("❌ Trend is Bearish")

        if rsi > 50:
            score += 1
            reasons.append("✅ RSI indicates buying momentum (>50)")
        else:
            reasons.append("❌ RSI indicates weakness (<50)")

        if macd == "BULLISH":
            score += 1
            reasons.append("✅ MACD is above Signal Line")
        else:
            reasons.append("❌ MACD is below Signal Line")

        # Decision
        if score >= 2:
            decision = "BUY"
            color = "green"
        elif score == 1:
            decision = "HOLD"
            color = "orange"
        else:
            decision = "WAIT / SELL"
            color = "red"

        st.markdown(f"<h2 style='color:{color};'>AI Recommendation: {decision}</h2>", unsafe_allow_html=True)
        
        st.subheader("📝 Reasoning")
        for r in reasons:
            st.write(r)

        st.divider()
        st.subheader("🤖 OpenAI Detailed Explanation")
        with st.spinner("Asking OpenAI..."):
            prompt = f"""
            I am analyzing stock {symbol}.
            Technical Indicators:
            - Trend: {trend}
            - RSI: {rsi}
            - MACD: {macd}
            
            Based on these, explain in simple words why a user should {decision}. 
            Keep it under 3 sentences.
            """
            explanation = ai_explain(prompt)
            st.write(explanation)

# =========================
# PAGE: PORTFOLIO (PHASE 3) - ENHANCED
# =========================
elif page == "Portfolio Allocation (Phase 3)":
    st.header("💼 Intelligent Portfolio Manager")
    
    if not st.session_state.profile_created:
        st.warning("⚠️ Please complete your Profile first.")
        st.stop()
        
    name = st.session_state.name
    
    # 1. CHECK MONTH-TO-MONTH TRACKING
    history = load_portfolio_history(name)
    if history:
        last_entry = history[-1]
        st.info(f"📂 **Welcome back, {name}!** We found your last investment plan from **{last_entry['date']}**.")
        st.caption("You can either continue with that plan or create a new one based on today's market.")
        st.divider()

    capital = st.number_input("Total Investment Amount (₹)", min_value=1000, step=500, value=100000)
    
    # 2. DYNAMIC MARKET SENTIMENT
    sentiment = get_market_sentiment()
    st.write(f"**Current Market Sentiment:** {sentiment}")
    
    # Logic for Recommendation
    if sentiment == "BEARISH":
        st.caption("⚠️ Market is volatile. AI suggests increasing SAFE assets (Gold/Debt).")
        rec_alloc = {"Equity": 30, "Debt": 50, "Gold ETF": 20}
    elif sentiment == "BULLISH":
        st.caption("🚀 Market is positive. AI suggests increasing EQUITY for growth.")
        rec_alloc = {"Equity": 60, "Debt": 30, "Gold ETF": 10}
    else:
        rec_alloc = {"Equity": 50, "Debt": 30, "Gold ETF": 20}
    
    # 3. ASSET ALLOCATION
    st.subheader("📊 Asset Allocation Strategy")
    c1, c2, c3 = st.columns(3)
    eq_pct = c1.number_input("Equity %", value=rec_alloc["Equity"])
    db_pct = c2.number_input("Debt %", value=rec_alloc["Debt"])
    gd_pct = c3.number_input("Gold %", value=rec_alloc["Gold ETF"])
    
    if eq_pct + db_pct + gd_pct != 100:
        st.error("Total allocation must equal 100%.")
        st.stop()
        
    # Chart
    alloc_df = pd.DataFrame([
        {"Asset": "Equity", "Amount": capital * eq_pct/100},
        {"Asset": "Debt", "Amount": capital * db_pct/100},
        {"Asset": "Gold ETF", "Amount": capital * gd_pct/100}
    ])
    fig = go.Figure(data=[go.Pie(labels=alloc_df['Asset'], values=alloc_df['Amount'], hole=.3)])
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. COMPANY SELECTION (AI vs CUSTOM)
    st.subheader("🏢 Company & Fund Selection")
    
    custom_choice = st.checkbox("📝 I want to choose my own companies/funds")
    
    if not custom_choice:
        # AI Suggestions
        st.info("🤖 **AI Recommended Leaders**")
        st.markdown(f"**Equity:** Nifty 50 ETF (Top 50 Companies)")
        st.markdown(f"**Debt:** HDFC Liquid Fund (Stability)")
        st.markdown(f"**Gold:** Nippon Gold BeES (Safety)")
        
        final_companies = {
            "Equity": "Nifty 50 ETF", 
            "Debt": "HDFC Liquid Fund", 
            "Gold ETF": "Nippon Gold BeES"
        }
        
    else:
        # Custom Inputs
        st.warning("⚠️ You are choosing your own stocks. AI will review them before saving.")
        col1, col2, col3 = st.columns(3)
        eq_choice = col1.text_input("Your Equity Choice", value="Reliance")
        db_choice = col2.text_input("Your Debt Choice", value="SBI Liquid Fund")
        gd_choice = col3.text_input("Your Gold Choice", value="Sovereign Gold Bond")
        
        final_companies = {"Equity": eq_choice, "Debt": db_choice, "Gold ETF": gd_choice}
        
        if st.button("🔍 Analyze My Custom Plan"):
            with st.spinner("AI is analyzing your specific companies..."):
                prompt = f"""
                User wants to invest ₹{capital} with this split:
                Equity: {eq_pct}% in {eq_choice}
                Debt: {db_pct}% in {db_choice}
                Gold: {gd_pct}% in {gd_choice}
                
                Market Sentiment is {sentiment}.
                
                Briefly analyze the risk of {eq_choice} and if this portfolio is balanced.
                """
                analysis = ai_explain(prompt)
                st.write(analysis)

    # 5. SAVE BUTTON
    st.divider()
    if st.button("💾 Confirm & Save Portfolio"):
        allocation = {"Equity": eq_pct, "Debt": db_pct, "Gold ETF": gd_pct}
        save_portfolio_to_db(name, datetime.date.today(), allocation, capital, final_companies)
        st.success("✅ Portfolio Saved to History! We will remember this for your next month's review.")
        st.balloons()

# =========================
# PAGE: FINANCIAL PLANNING (PHASE 5)
# =========================
elif page == "Financial Planning (Phase 5)":
    st.header("🔮 SIP Calculator & Planner")
    
    monthly_sip = st.number_input("Monthly SIP Amount (₹)", value=5000)
    years = st.slider("Investment Duration (Years)", 1, 30, 10)
    expected_return = st.slider("Expected Annual Return (%)", 5, 20, 12)
    
    months = years * 12
    rate = expected_return / 100 / 12
    
    future_value = monthly_sip * ((1 + rate) ** months - 1) / rate
    invested_amount = monthly_sip * months
    wealth_gained = future_value - invested_amount
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Invested Amount", f"₹{invested_amount:,.0f}")
    c2.metric("Wealth Gained", f"₹{wealth_gained:,.0f}", delta=f"{expected_return}% Return")
    c3.metric("Total Value", f"₹{future_value:,.0f}")
    
    # Growth Chart
    st.subheader("📈 Wealth Growth Chart")
    values = [monthly_sip * ((1 + rate) ** m - 1) / rate for m in range(1, months + 1)]
    chart_df = pd.DataFrame({"Month": range(1, months + 1), "Portfolio Value": values})
    st.area_chart(chart_df.set_index("Month"))
    
    st.subheader("🤖 AI Advice on Compounding")
    if st.button("Ask AI about this Plan"):
        explanation = ai_explain(f"Explain how investing ₹{monthly_sip} monthly for {years} years creates wealth using the power of compounding.")
        st.write(explanation)

# =========================
# PAGE: EDUCATION (PHASE 6) - ENHANCED
# =========================
elif page == "Education (Phase 6)":
    st.header("🎓 Investor Education")
    
    # Standard Topics
    topic = st.selectbox("Select a topic to learn:", [
        "What is RSI?",
        "What is MACD?",
        "Importance of Moving Averages",
        "What is a Candlestick Pattern?",
        "Risk Management basics"
    ])
    
    if st.button("Explain Topic"):
        with st.spinner("AI Teacher is writing..."):
            lesson = ai_explain(f"Explain '{topic}' to a beginner investor in simple English. Use examples.")
            st.markdown(f"### {topic}")
            st.write(lesson)

    st.divider()
    # ✅ Q&A FEATURE
    st.subheader("🙋‍♂️ Ask a Question")
    user_q = st.text_input("Type your specific doubt here (e.g., 'Is Adani Power a good buy right now?')")
    if st.button("Get Answer"):
        if user_q:
            with st.spinner("Thinking..."):
                ans = ai_explain(user_q)
                st.write(ans)

# =========================
# PAGE: HISTORY (PHASE 7) - ENHANCED
# =========================
elif page == "History / Memory (Phase 7)":
    st.header("📜 Investment History")
    
    name = st.session_state.get("name")
    if not name:
        st.warning("Please create a profile to see history.")
        st.stop()
        
    history = load_portfolio_history(name)
    
    if not history:
        st.info("No records found. Go to 'Portfolio Allocation' to save your first plan.")
    else:
        # Convert to clean table
        table_data = []
        for i, entry in enumerate(history):
            row = {
                "ID": i,
                "Date": entry['date'],
                "Capital": f"₹{entry['capital']}",
                "Equity %": entry['allocation']['Equity'],
                "Companies": str(entry['companies'])
            }
            table_data.append(row)
            
        df = pd.DataFrame(table_data)
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True)
        
        st.divider()
        st.subheader("🧐 Review Past Decisions")
        
        options = [f"{row['Date']} - {row['Capital']}" for row in table_data]
        selected_option = st.selectbox("Select a record to review:", options)
        
        if st.button("Analyze Selected Record"):
            idx = options.index(selected_option)
            record = history[idx]
            
            with st.spinner("AI is analyzing your past performance potential..."):
                prompt = f"""
                Review this past portfolio created on {record['date']}:
                Allocation: {record['allocation']}
                Companies: {record['companies']}
                
                Was this a balanced choice? Risk level?
                """
                st.write(ai_explain(prompt))

# =========================
# FOOTER
# =========================
st.sidebar.divider()
st.sidebar.caption("Powered by OpenAI GPT-4o-mini & Yahoo Finance")



# =========================
# PHASE 8 — NEWS & EVENTS AGENT
# =========================
elif page == "News & Events (Phase 8)":
    st.header("📰 Market News & Events Agent")

    symbol = st.text_input("Enter Stock Symbol for News (e.g., TCS.NS)")

    if st.button("Fetch Latest News") and symbol:
        try:
            stock = yf.Ticker(symbol)
            news = stock.news

            if not news:
                st.warning("No recent news found.")
            else:
                for n in news[:5]:
                    st.subheader(n["title"])
                    st.write(f"Source: {n.get('publisher','Unknown')}")
                    st.write(datetime.datetime.fromtimestamp(n["providerPublishTime"]))
                    
                    ai_summary = ai_explain(f"""
                    Summarize this news for an investor:
                    {n["title"]}
                    Explain whether it is Positive, Negative or Neutral
                    and what action an investor should take.
                    """)
                    st.info(ai_summary)
                    st.divider()
        except Exception as e:
            st.error(f"Error fetching news: {e}")


# =========================
# PHASE 9 — AUTO PORTFOLIO REBALANCER
# =========================
elif page == "Auto Rebalancer (Phase 9)":
    st.header("🔁 Auto Portfolio Rebalancer")

    name = st.session_state.get("name")
    if not name:
        st.warning("Please create a profile first.")
        st.stop()

    history = load_portfolio_history(name)

    if not history:
        st.info("No saved portfolio found to rebalance.")
        st.stop()

    last_portfolio = history[-1]

    st.subheader("📂 Last Saved Portfolio")
    st.json(last_portfolio)

    sentiment = get_market_sentiment()
    st.write(f"📈 Current Market Sentiment: **{sentiment}**")

    if st.button("Run AI Rebalancing"):
        with st.spinner("AI is analyzing portfolio rebalancing..."):
            prompt = f"""
            User Risk Appetite: {st.session_state.risk}
            Market Sentiment: {sentiment}
            Existing Allocation: {last_portfolio['allocation']}

            Suggest whether the user should:
            - Increase Equity
            - Reduce Risk
            - Hold Allocation
            Give reasoning.
            """
            rebalance_advice = ai_explain(prompt)
            st.success("✅ Rebalancing Advice")
            st.write(rebalance_advice)


# =========================
# RESUME / PLACEMENT EXPLANATION
# =========================
elif page == "Resume Explanation":
    st.header("🎓 Project Explanation (Resume Ready)")

    st.markdown("""
    ### 🤖 AI Investment Analyst Agent

    **Description:**  
    A multi-agent AI system that assists retail investors with stock analysis, 
    portfolio allocation, financial planning, and education.

    **Tech Stack:**
    - Python
    - Streamlit
    - OpenAI GPT-4o-mini
    - Yahoo Finance API
    - Plotly

    **Agents Implemented:**
    - Stock Analysis Agent (Technical + Fundamental)
    - AI Decision Agent (Buy/Hold/Wait)
    - Portfolio Allocation Agent
    - Financial Planning Agent
    - Education Agent
    - Memory Agent (User Portfolio History)
    - News & Events Agent
    - Auto Portfolio Rebalancer

    **Key Highlights:**
    - Dynamic real-time market data
    - AI-driven reasoning & explainability
    - Persistent user memory
    - Modular agent-based architecture

    **Use Cases:**
    - Retail investors
    - Financial education platforms
    - Robo-advisory prototypes
    """)

