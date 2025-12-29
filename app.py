# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# FINAL SAFE VERSION (CSV-Agnostic)
# ==========================================

import streamlit as st
import pandas as pd
from groq import Groq

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Recommendation System",
    layout="wide"
)

st.title("🤖 Agentic AI – Mutual Fund Recommendation System")
st.caption(
    "Recommendations are deterministic. The agent only explains and answers follow-ups."
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mutual Funds Data.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

df = load_data()

# ------------------------------------------
# 🔍 Dynamic Column Detection (KEY FIX)
# ------------------------------------------
def find_col(keyword_list):
    for col in df.columns:
        for k in keyword_list:
            if k in col:
                return col
    return None

COL_FUND = find_col(["scheme", "fund"])
COL_CAT = find_col(["category"])
COL_RISK = find_col(["risk"])
COL_1Y = find_col(["1", "one"])
COL_3Y = find_col(["3", "three"])
COL_EXP = find_col(["expense"])

required = {
    "Fund Name": COL_FUND,
    "Category": COL_CAT,
    "Risk": COL_RISK,
    "1Y Return": COL_1Y,
    "3Y Return": COL_3Y,
    "Expense": COL_EXP,
}

missing = [k for k, v in required.items() if v is None]
if missing:
    st.error(f"Dataset missing required information: {missing}")
    st.stop()

# ------------------------------------------
# Sidebar – Preferences
# ------------------------------------------
st.sidebar.header("Investor Preferences")

risk_profile = st.sidebar.selectbox(
    "Risk Profile", ["Low", "Medium", "High"]
)

preference = st.sidebar.selectbox(
    "Primary Preference", ["Stability", "Growth", "Tax Saving"]
)

top_k = st.sidebar.slider(
    "Number of Recommendations", 1, 10, 5
)

get_reco = st.sidebar.button("Get Recommendations")

RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}

# ------------------------------------------
# Recommendation Agent (Deterministic)
# ------------------------------------------
def recommendation_agent(data: pd.DataFrame):
    data = data.copy()

    data = data[data[COL_RISK] <= RISK_MAP[risk_profile]]

    if preference == "Stability":
        data = data.sort_values(
            by=[COL_RISK, COL_EXP],
            ascending=[True, True]
        )

    elif preference == "Growth":
        data = data.sort_values(
            by=[COL_3Y, COL_1Y],
            ascending=False
        )

    elif preference == "Tax Saving":
        data = data[
            data[COL_CAT].astype(str).str.contains("elss", case=False, na=False)
        ]

    return data.head(top_k)

# ------------------------------------------
# Run Recommendations
# ------------------------------------------
if get_reco:
    recs = recommendation_agent(df)

    if recs.empty:
        st.warning("No funds matched your criteria.")
        st.stop()

    st.session_state["recs"] = recs

    st.subheader(f"📌 Top {len(recs)} Recommended Mutual Funds")

    for _, r in recs.iterrows():
        st.markdown(
            f"""
**{r[COL_FUND]}**
- Category: {r[COL_CAT]}
- Risk Level: {r[COL_RISK]}
- 1Y Return: {r[COL_1Y]}%
- 3Y Return: {r[COL_3Y]}%
- Expense Ratio: {r[COL_EXP]}%
---
"""
        )

# ------------------------------------------
# Chatbot – Explanation Agent (LLM ONLY)
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Chat with the Agent")

question = st.text_input(
    "Ask follow-ups (e.g. why stability?, compare first two funds)"
)

client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def explanation_agent(q, df_rec):
    snapshot = df.to_string(index=False)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain recommendations strictly using given data. "
                    "No new recommendations. No hallucinations."
                )
            },
            {
                "role": "user",
                "content": f"Question: {q}\n\nRecommended Funds:\n{snapshot}"
            }
        ],
        temperature=0.3
    )
    return response.
