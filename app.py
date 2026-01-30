import streamlit as st
import google.generativeai as genai
import os

st.title("🧪 Gemini API App")

# 1. Load Gemini Key securely
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local testing if secrets.toml is missing
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found. Please add it to Streamlit Secrets.")
    st.stop()

# 2. Configure Gemini
genai.configure(api_key=api_key)

# 3. USE THE WORKING MODEL
model_name = "models/gemini-2.5-flash"
model = genai.GenerativeModel(model_name)

st.success(f"✅ Connected to {model_name}")

# 4. Input and Response
user_input = st.text_input("Ask something:")
if st.button("Generate"):
    try:
        with st.spinner("Thinking..."):
            response = model.generate_content(user_input)
            st.write(response.text)
    except Exception as e:
        st.error(f"Error: {e}")
