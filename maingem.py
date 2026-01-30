import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import os
import datetime

# =========================
# 1. OPENAI SETUP
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Try getting from streamlit secrets if env var not set
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("❌ OpenAI API key not found. Please check Streamlit Secrets.")
        st.stop()

# Configure OpenAI Client
client = OpenAI(api_key=api_key)

# ✅ MODEL VERSION AS REQUESTED
MODEL_VERSION = "gpt-4o-mini-2024-07-18"

def ai_explain(prompt):
    """
    Function to get explanation from OpenAI GPT-4o-mini
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": "You are a helpful financial investment assistant. Keep answers concise and beginner-friendly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# =========================
# 2. PAGE CONFIG & SESSION STATE
# =========================
st.set_page_config(page_title="AI Investment Analyst", layout="wide", page_icon="📈")

# Initialize Session State Variables
if "profile_created" not in st.session_state:
    st.session_state.profile_created = False
if "history" not in st.session_state:
    st.session_state.history = []  # Phase 7 Memory

# =========================
# 3. SIDEBAR NAVIGATION
# =========================
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
        "History / Memory (Phase 7)"
    ]
)

# =========================
# PAGE: HOME
# =========================
if page == "Home":
    st.title("🤖 AI Investment Analyst Agent")
    st.image("https://images.unsplash.com/photo-1611974765270-ca1258634369?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", caption="Smart Investing for Everyone")
    st.markdown("""
    ### Welcome to your Personal Investment Assistant
    This tool combines real-time market data with OpenAI's intelligence to help you make better financial decisions.
    
    **Get Started:**
    1. Go to **Profile** to set your details.
    2. Use **Stock Analysis** to check market trends.
    3. Check **AI Decision** for buy/sell ratings.
    4. Explore **Portfolio Allocation** for asset distribution.
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
                
                # Add to History (Phase 7)
                log_entry = {
                    "Date": str(datetime.date.today()),
                    "Stock": symbol, 
                    "Trend": trend, 
                    "RSI": round(rsi_val, 2)
                }
                st.session_state.history.append(log_entry)

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
# PAGE: PORTFOLIO (PHASE 3)
# =========================
elif page == "Portfolio Allocation (Phase 3)":
    st.header("💼 Portfolio Allocation")
    
    if not st.session_state.profile_created:
        st.warning("⚠️ Please complete your Profile first.")
    else:
        capital = st.number_input("Total Investment Amount (₹)", min_value=1000, step=500, value=100000)
        risk = st.session_state.risk
        
        assets = st.multiselect(
            "Choose Asset Classes",
            ["Equity", "Debt", "Gold ETF"],
            default=["Equity", "Debt", "Gold ETF"]
        )
        
        if st.button("Generate Allocation"):
            # Allocation Logic based on Risk
            if risk <= 5: # Conservative
                alloc = {"Equity": 30, "Debt": 50, "Gold ETF": 20}
            elif risk <= 12: # Moderate
                alloc = {"Equity": 50, "Debt": 30, "Gold ETF": 20}
            else: # Aggressive
                alloc = {"Equity": 70, "Debt": 20, "Gold ETF": 10}
                
            st.subheader("📊 Recommended Split")
            
            # Display Allocation
            data = []
            for asset in assets:
                pct = alloc.get(asset, 0)
                amt = capital * (pct / 100)
                st.success(f"**{asset}**: {pct}% → ₹{amt:,.0f}")
                data.append({"Asset": asset, "Amount": amt})
            
            # Pie Chart
            df_chart = pd.DataFrame(data)
            fig = go.Figure(data=[go.Pie(labels=df_chart['Asset'], values=df_chart['Amount'], hole=.3)])
            st.plotly_chart(fig)
            
            st.subheader("💡 Investment Suggestions")
            if "Equity" in assets:
                st.markdown("**Equity:** Nifty 50 ETF, HDFC Bank, TCS")
            if "Debt" in assets:
                st.markdown("**Debt:** Liquid Funds, Corporate Bond Funds")
            if "Gold ETF" in assets:
                st.markdown("**Gold:** Gold BeES, Sovereign Gold Bond")

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
# PAGE: EDUCATION (PHASE 6)
# =========================
elif page == "Education (Phase 6)":
    st.header("🎓 Investor Education")
    
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

# =========================
# PAGE: HISTORY (PHASE 7)
# =========================
elif page == "History / Memory (Phase 7)":
    st.header("📜 Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history found. Go to 'Stock Analysis' to start.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History Cleared!")
            st.rerun()

# =========================
# FOOTER
# =========================
st.sidebar.divider()
st.sidebar.caption("Powered by OpenAI GPT-4o-mini & Yahoo Finance")
