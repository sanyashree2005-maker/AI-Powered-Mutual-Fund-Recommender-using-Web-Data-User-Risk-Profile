# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset + Safe LLM Chat Agent
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

# Normalize column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.lower()

# Column mapping (safe)
COL = {
    "name": "scheme name",
    "category": "category",
    "risk": "risk",
    "one_year": "1y return",
    "three_year": "3y return",
    "expense": "expense ratio"
}

# ------------------------------------------
# Sidebar – Investor Preferences
# ------------------------------------------
st.sidebar.header("Investor Preferences")

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

get_reco = st.sidebar.button("Get Recommendations")

# ------------------------------------------
# Recommendation Agent (Deterministic)
# ------------------------------------------
def recommendation_agent(data):
    filtered = data.copy()

    # Risk mapping (safe)
    RISK_MAP = {
        "Low": ["Low"],
        "Medium": ["Low", "Moderate"],
        "High": ["Low", "Moderate", "High"]
    }

    if COL["risk"] in filtered.columns:
        filtered = filtered[filtered[COL["risk"]].isin(RISK_MAP[risk_profile])]

    # Preference logic
    if preference == "Stability":
        filtered = filtered.sort_values(by=COL["expense"])
    elif preference == "Growth":
        filtered = filtered.sort_values(by=COL["three_year"], ascending=False)
    elif preference == "Tax Saving":
        filtered = filtered[
            filtered[COL["category"]].str.contains("ELSS", case=False, na=False)
        ]

    return filtered.head(top_k)

# ------------------------------------------
# Store Recommendations
# ------------------------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if get_reco:
    st.session_state.recommendations = recommendation_agent(df)

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.recommendations is not None:
    st.subheader(f"📌 Top {len(st.session_state.recommendations)} Mutual Funds")

    for _, row in st.session_state.recommendations.iterrows():
        st.markdown(
            f"""
**{row.get(COL['name'], 'N/A')}**
- Category: {row.get(COL['category'], 'N/A')}
- Risk: {row.get(COL['risk'], 'N/A')}
- 1Y Return: {row.get(COL['one_year'], 'N/A')}
- 3Y Return: {row.get(COL['three_year'], 'N/A')}
- Expense Ratio: {row.get(COL['expense'], 'N/A')}
---
"""
        )

# ------------------------------------------
# Chat with Agent (Groq – SAFE MODE)
# ------------------------------------------
st.markdown("### 💬 Chat with the Agent")

user_query = st.text_input(
    "Ask follow-ups (e.g. 'Why these funds?' or 'Compare top 2')"
)

if user_query and st.session_state.recommendations is not None:
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        context = st.session_state.recommendations.to_string(index=False)

        prompt = f"""
You are a financial assistant.
Answer ONLY using the data below.
Do NOT invent fund names.

DATA:
{context}

QUESTION:
{user_query}
"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        st.success(response.choices[0].message.content)

    except Exception:
        st.warning(
            "LLM temporarily unavailable. Recommendations are still valid."
        )

elif user_query:
    st.info("Please generate recommendations first.")
