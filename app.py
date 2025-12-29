# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# FINAL STABLE VERSION
# ==========================================

import streamlit as st
import pandas as pd
import os

# ---------- OPTIONAL LLM (Groq) ----------
USE_LLM = "GROQ_API_KEY" in st.secrets or os.getenv("GROQ_API_KEY")

if USE_LLM:
    from groq import Groq
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")))

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Recommendation System",
    layout="wide"
)

st.title("🤖 Agentic AI – Mutual Fund Recommendation System")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Mutual Funds Data.csv")
    df.columns = df.columns.str.strip()  # safety
    return df

df = load_data()

# ---------- AUTO COLUMN DETECTION (NO KEYERROR EVER) ----------
def find_col(keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

COL = {
    "fund": find_col("fund"),
    "category": find_col("category"),
    "risk": find_col("risk"),
    "expense": find_col("expense"),
    "ret1y": find_col("1"),
    "ret3y": find_col("3"),
}

missing = [k for k, v in COL.items() if v is None]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Investor Preferences")

risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
preference = st.sidebar.selectbox("Primary Preference", ["Stability", "Growth", "Tax Saving"])
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

get_reco = st.sidebar.button("Get Recommendations")

# ---------- AGENT: RECOMMENDATION ----------
def recommendation_agent(data):
    filtered = data.copy()

    risk_map = {"Low": 1, "Medium": 2, "High": 3}
    filtered = filtered[filtered[COL["risk"]] <= risk_map[risk_profile]]

    if preference == "Stability":
        filtered = filtered.sort_values(by=COL["expense"])
    elif preference == "Growth":
        filtered = filtered.sort_values(by=[COL["ret3y"], COL["ret1y"]], ascending=False)
    elif preference == "Tax Saving":
        filtered = filtered[filtered[COL["category"]].str.contains("ELSS", case=False, na=False)]

    return filtered.head(top_k)

# ---------- SESSION STATE ----------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ---------- RUN ONLY ON BUTTON ----------
if get_reco:
    st.session_state.recommendations = recommendation_agent(df)

# ---------- DISPLAY ----------
if st.session_state.recommendations is not None:
    st.subheader(f"📌 Top {len(st.session_state.recommendations)} Recommended Mutual Funds")

    for _, row in st.session_state.recommendations.iterrows():
        st.markdown(f"""
**{row[COL['fund']]}**
- Category: {row[COL['category']]}
- Risk Level: {row[COL['risk']]}
- 1Y Return: {row[COL['ret1y']]}%
- 3Y Return: {row[COL['ret3y']]}%
- Expense Ratio: {row[COL['expense']]}%
---
""")

# ---------- CHATBOT ----------
st.markdown("### 💬 Chat with the Agent")

user_q = st.text_input(
    "Ask follow-ups (e.g. 'why stability funds?', 'compare first two funds')"
)

if user_q and st.session_state.recommendations is not None:
    if USE_LLM:
        context = st.session_state.recommendations.to_string(index=False)
        prompt = f"""
You are a financial assistant.
Answer ONLY using the data below.

DATA:
{context}

QUESTION:
{user_q}
"""
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        st.success(response.choices[0].message.content)
    else:
        st.info(
            "LLM not configured. This system uses dataset-based recommendations. "
            "Add GROQ_API_KEY to enable explanations."
        )
