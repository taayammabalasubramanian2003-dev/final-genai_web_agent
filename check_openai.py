import streamlit as st
import os
from openai import OpenAI, AuthenticationError, RateLimitError

st.title("🕵️ OpenAI Key & Model Checker")

# 1. Try to get API Key from Streamlit Secrets or Input
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ OpenAI API Key found in Secrets")
except:
    api_key = st.text_input("Enter OpenAI API Key (sk-...)", type="password")

if st.button("Check OpenAI Key"):
    if not api_key:
        st.error("❌ Please enter a key or set it in secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)
    
    # Common models to test
    models_to_test = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]

    st.write("### 🔍 Testing Models...")
    
    for model in models_to_test:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            st.success(f"✅ **{model}** is WORKING!")
        except AuthenticationError:
            st.error("❌ Invalid API Key")
            break
        except RateLimitError:
            st.warning(f"⚠️ {model}: Quota Exceeded (Check Billing)")
        except Exception as e:
            st.warning(f"🚫 {model}: Access Denied or Error ({e})")
