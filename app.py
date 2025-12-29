# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Deterministic Reco + LLM Explanation Agent
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
# Load CSV (FIXED FILE)
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mutual Funds Data.csv")

df = load_data()

# ------------------------------------------
# Column Normalization & Mapping (CRITICAL)
# ------------------------------------------
df.columns = [c.strip().lower() for c in df.columns]

COLUMN_MAP = {
    "scheme name": "fund_name",
    "fund name": "fund_name",
    "1 yr return (%)": "1y_return",
    "3 yr return (%)": "3y_return",
    "expense ratio (%)": "expense_ratio",
    "risk level": "risk_level",
    "category": "category",
}

for original, standardized in COLUMN_MAP.items():
    if original in df.columns:
        df.rename(columns={original: standardized}, inplace=True)

REQUIRED_COLS = {
    "fund_name",
    "category",
    "risk_level",
    "1y_return",
    "3y_return",
    "expense_ratio",
}

missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

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
# Deterministic Recommendation Agent
# ------------------------------------------
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}

def recommendation_agent(data: pd.DataFrame):
    filtered = data.copy()

    # Risk filtering
    filtered = filtered[
        filtered["risk_level"] <= RISK_MAP[risk_profile]
    ]

    # Preference logic
    if preference == "Stability":
        filtered = filtered.sort_values(
            by=["risk_level", "expense_ratio"],
            ascending=[True, True]
        )

    elif preference == "Growth":
        filtered = filtered.sort_values(
            by=["3y_return", "1y_return"],
            ascending=False
        )

    elif preference == "Tax Saving":
        filtered = filtered[
            filtered["category"].str.contains("elss", case=False, na=False)
        ]

    return filtered.head(top_k)

# ------------------------------------------
# Run Recommendations ONLY on Click
# ------------------------------------------
if get_reco:
    recs = recommendation_agent(df)

    if recs.empty:
        st.warning("No funds matched your criteria.")
        st.stop()

    st.session_state["recommendations"] = recs

    st.subheader(f"📌 Top {len(recs)} Recommended Mutual Funds")

    for _, row in recs.iterrows():
        st.markdown(
            f"""
**{row['fund_name']}**
- Category: {row['category']}
- Risk Level: {row['risk_level']}
- 1Y Return: {row['1y_return']}%
- 3Y Return: {row['3y_return']}%
- Expense Ratio: {row['expense_ratio']}%
---
"""
        )

# ------------------------------------------
# Chatbot (LLM Explanation Agent ONLY)
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Chat with the Agent")

user_question = st.text_input(
    "Ask follow-ups (e.g. why stability?, compare first two funds)"
)

client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def explanation_agent(question: str, rec_df: pd.DataFrame):
    snapshot = rec_df.to_string(index=False)

    system_prompt = """
You are a mutual fund explanation agent.
Rules:
- Use ONLY the provided data
- Do NOT recommend new funds
- Do NOT hallucinate
- Be clear and factual
"""

    user_prompt = f"""
User question:
{question}

Recommended funds:
{snapshot}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

if user_question:
    if "recommendations" not in st.session_state:
        st.warning("Please generate recommendations first.")
    elif client is None:
        st.info(
            "LLM not configured. "
            "Recommendations are based on Risk Level, Returns, and Expense Ratio."
        )
    else:
        with st.spinner("Agent answering..."):
            answer = explanation_agent(
                user_question,
                st.session_state["recommendations"]
            )
        st.markdown(answer)
