# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset + Tool-Orchestrated LLM Chat Agent
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
    "Deterministic recommendations from CSV + LLM-powered follow-up chatbot"
)

# ------------------------------------------
# Load Groq Client (SAFE)
# ------------------------------------------
client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
    min_value=1,
    max_value=10,
    value=5
)

run_button = st.sidebar.button("Get Recommendations")

# ------------------------------------------
# Upload Dataset
# ------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Mutual Fund CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------------------------------
# Column Normalization (VERY IMPORTANT)
# ------------------------------------------
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

REQUIRED_COLS = {
    "fund_name",
    "category",
    "risk_level",
    "1y_return",
    "3y_return",
    "expense_ratio"
}

missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ------------------------------------------
# Deterministic Recommendation Agent
# ------------------------------------------
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}

def recommendation_agent(data):
    filtered = data.copy()

    # Risk filter
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

    if filtered.empty:
        return pd.DataFrame()

    return filtered.head(top_k)

# ------------------------------------------
# Run Recommendation ONLY on Button Click
# ------------------------------------------
if run_button:
    recommendations = recommendation_agent(df)

    if recommendations.empty:
        st.warning("No funds matched your criteria.")
        st.stop()

    st.session_state["recommendations"] = recommendations

    st.subheader(f"📌 Top {len(recommendations)} Recommended Mutual Funds")

    for _, row in recommendations.iterrows():
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
# Chatbot Agent (LLM – SAFE, OPTIONAL)
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Chat with the Agent")

user_question = st.text_input(
    "Ask follow-ups (e.g. why stability?, compare first two funds)"
)

def chat_agent(question, rec_df):
    if client is None:
        return (
            "LLM is currently unavailable.\n\n"
            "Recommendations are generated deterministically "
            "from risk level, returns, and expense ratio."
        )

    snapshot = rec_df.to_string(index=False)

    system_prompt = """
You are a mutual fund explanation agent.
Answer ONLY from the provided data.
Do NOT hallucinate.
Be concise and factual.
"""

    user_prompt = f"""
User question:
{question}

Current recommended funds:
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
    else:
        with st.spinner("Agent thinking..."):
            answer = chat_agent(
                user_question,
                st.session_state["recommendations"]
            )
        st.markdown(answer)
