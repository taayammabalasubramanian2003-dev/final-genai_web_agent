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

