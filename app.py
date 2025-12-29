# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset + Context-Aware Chatbot
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

# ------------------------------------------
# Load Dataset
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mutual Funds Data.csv")

df = load_data()

# ------------------------------------------
# Sidebar – Investor Profile
# ------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox(
    "Risk Profile",
    ["Low", "Medium", "High"]
)

preference = st.sidebar.selectbox(
    "Primary Preference",
    ["Stability", "Growth", "Tax Saving"]
)

top_k = st.sidebar.slider(
    "Number of Recommendations",
    1, 10, 5
)

run_reco = st.sidebar.button("Get Recommendations")

# ------------------------------------------
# Risk Mapping
# ------------------------------------------
RISK_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

# ------------------------------------------
# Recommendation Engine
# ------------------------------------------
def recommend_funds(df, risk_profile, preference, top_k):
    filtered = df.copy()

    filtered = filtered[filtered["Risk Level"] <= RISK_MAP[risk_profile]]

    if preference == "Growth":
        filtered = filtered.sort_values(
            by=["3Y Return (%)", "1Y Return (%)"],
            ascending=False
        )

    elif preference == "Stability":
        filtered = filtered.sort_values(
            by=["Risk Level", "Expense Ratio (%)"],
            ascending=[True, True]
        )

    elif preference == "Tax Saving":
        filtered = filtered[
            filtered["Category"].str.contains("ELSS", case=False, na=False)
        ]

    return filtered.head(top_k)

# ------------------------------------------
# Session State
# ------------------------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ------------------------------------------
# Generate Recommendations
# ------------------------------------------
if run_reco:
    st.session_state.recommendations = recommend_funds(
        df, risk_profile, preference, top_k
    )

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.recommendations is not None:

    recos = st.session_state.recommendations

    st.subheader(f"📌 Top {len(recos)} Recommended Mutual Funds")

    for _, row in recos.iterrows():
        st.markdown(
            f"""
**{row['Fund Name']}**
- Category: {row['Category']}
- Risk Level: {row['Risk Level']}
- 1Y Return: {row['1Y Return (%)']}%
- 3Y Return: {row['3Y Return (%)']}%
- Expense Ratio: {row['Expense Ratio (%)']}%
---
"""
        )

# ------------------------------------------
# Chatbot Section (Context-Aware)
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Ask Follow-up Questions")

user_query = st.text_input(
    "Ask about recommendations (e.g. 'why is this fund recommended?')"
)

if user_query and st.session_state.recommendations is not None:

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    context = st.session_state.recommendations.to_string(index=False)

    prompt = f"""
You are a mutual fund assistant.

ONLY answer using the following recommendation data.
Do NOT invent funds.
Do NOT give financial advice.

Recommendation Data:
{context}

User Question:
{user_query}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    st.write(response.choices[0].message.content)

elif user_query:
    st.info("Please generate recommendations first.")
