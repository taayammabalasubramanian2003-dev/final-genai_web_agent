import streamlit as st
from openai import OpenAI, AuthenticationError, RateLimitError
import os

st.title("🕵️ OpenAI Model Checker")

# 1. Get API Key
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.text_input("Enter OpenAI API Key (sk-...)", type="password")

if st.button("Check My Key"):
    if not api_key:
        st.error("❌ No API Key found.")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    
    # List of models to test
    candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    st.write("### 🔍 Testing your key against models...")
    
    for model in candidates:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            st.success(f"✅ **{model}** is WORKING")
        except AuthenticationError:
            st.error("❌ Invalid API Key")
            break
        except RateLimitError:
            st.warning(f"⚠️ {model}: Quota Exceeded (Billing Issue)")
        except Exception as e:
            st.warning(f"🚫 {model}: Access Denied")
