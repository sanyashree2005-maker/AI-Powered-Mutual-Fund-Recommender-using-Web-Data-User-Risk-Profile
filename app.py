# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# FINAL CRASH-PROOF VERSION
# ==========================================

import streamlit as st
import pandas as pd
import os

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Recommendation System",
    layout="wide"
)

st.title("🤖 Agentic AI – Mutual Fund Recommendation System")

# ------------------------------------------
# Load Dataset
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mutual Funds Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ------------------------------------------
# Safe Column Resolver (NO KeyError)
# ------------------------------------------
def col(name):
    for c in df.columns:
        if name.lower() in c.lower():
            return c
    return None

COL_FUND = col("fund")
COL_CAT = col("category")
COL_RISK = col("risk")
COL_EXP = col("expense")
COL_1Y = col("1")
COL_3Y = col("3")

required = [COL_FUND, COL_CAT, COL_RISK, COL_EXP, COL_1Y, COL_3Y]
if any(c is None for c in required):
    st.error("Dataset columns not compatible. Please check CSV.")
    st.stop()

# ------------------------------------------
# Sidebar Controls
# ------------------------------------------
st.sidebar.header("Investor Preferences")

risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
preference = st.sidebar.selectbox("Primary Preference", ["Stability", "Growth", "Tax Saving"])
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

run_btn = st.sidebar.button("Get Recommendations")

# ------------------------------------------
# Recommendation Agent (Deterministic)
# ------------------------------------------
def recommendation_agent(data):
    df = data.copy()

    risk_map = {"Low": 1, "Medium": 2, "High": 3}
    df = df[df[COL_RISK] <= risk_map[risk_profile]]

    if preference == "Stability":
        df = df.sort_values(by=COL_EXP)
    elif preference == "Growth":
        df = df.sort_values(by=[COL_3Y, COL_1Y], ascending=False)
    elif preference == "Tax Saving":
        df = df[df[COL_CAT].str.contains("ELSS", case=False, na=False)]

    return df.head(top_k)

# ------------------------------------------
# Session State
# ------------------------------------------
if "reco" not in st.session_state:
    st.session_state.reco = None

# ------------------------------------------
# Run Recommendation ONLY on Button
# ------------------------------------------
if run_btn:
    st.session_state.reco = recommendation_agent(df)

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.reco is not None:
    st.subheader(f"📌 Top {len(st.session_state.reco)} Recommended Mutual Funds")

    for _, r in st.session_state.reco.iterrows():
        st.markdown(f"""
**{r[COL_FUND]}**
- Category: {r[COL_CAT]}
- Risk Level: {r[COL_RISK]}
- 1Y Return: {r[COL_1Y]}%
- 3Y Return: {r[COL_3Y]}%
- Expense Ratio: {r[COL_EXP]}%
---
""")

# ------------------------------------------
# Chat Agent (LLM OPTIONAL, SAFE)
# ------------------------------------------
st.markdown("### 💬 Chat with the Agent")

query = st.text_input(
    "Ask follow-ups (e.g. 'why stability funds?', 'compare first two funds')"
)

if query and st.session_state.reco is not None:

    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        st.info(
            "LLM not configured. "
            "Answering using dataset logic:\n\n"
            "These funds match your selected risk profile and preference "
            "based on expense ratio, returns, and category filters."
        )
    else:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            context = st.session_state.reco[[COL_FUND, COL_CAT, COL_RISK, COL_1Y, COL_3Y, COL_EXP]].to_string(index=False)

            prompt = f"""
You are a financial assistant.
Answer strictly using the data below.

DATA:
{context}

QUESTION:
{query}
"""

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            st.success(response.choices[0].message.content)

        except Exception as e:
            st.warning(
                "LLM temporarily unavailable.\n\n"
                "This does NOT affect recommendations.\n"
                "Please try again later."
            )
