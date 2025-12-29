# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# CSV-Driven + Safe Chatbot
# ==========================================

import streamlit as st
import pandas as pd

# Optional LLM
try:
    from groq import Groq
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

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
    df.columns = df.columns.str.strip()  # SAFETY
    return df

df = load_data()

# ------------------------------------------
# Column Mapping (CRITICAL FIX)
# ------------------------------------------
COLUMN_MAP = {
    "fund": "Fund Name",
    "category": "Category",
    "risk": "Risk",
    "ret1y": "1Y Return",
    "ret3y": "3Y Return",
    "expense": "Expense Ratio"
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

generate = st.sidebar.button("Get Recommendations")

# ------------------------------------------
# Risk Mapping
# ------------------------------------------
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}

# ------------------------------------------
# Session State
# ------------------------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ------------------------------------------
# Recommendation Agent (NO LLM)
# ------------------------------------------
def recommendation_agent(df, risk, preference, k):
    data = df.copy()

    # Risk filter
    data = data[data[COLUMN_MAP["risk"]] <= RISK_MAP[risk]]

    # Preference logic
    if preference == "Growth":
        data = data.sort_values(
            by=[COLUMN_MAP["ret3y"], COLUMN_MAP["ret1y"]],
            ascending=False
        )

    elif preference == "Stability":
        data = data.sort_values(
            by=[COLUMN_MAP["risk"], COLUMN_MAP["expense"]],
            ascending=[True, True]
        )

    elif preference == "Tax Saving":
        data = data[
            data[COLUMN_MAP["category"]]
            .str.contains("ELSS", case=False, na=False)
        ]

    return data.head(k)

# ------------------------------------------
# Generate Recommendations ONLY ON USER ACTION
# ------------------------------------------
if generate:
    st.session_state.recommendations = recommendation_agent(
        df, risk_profile, preference, top_k
    )

# ------------------------------------------
# Show Decision Variables
# ------------------------------------------
if st.session_state.recommendations is not None:
    st.markdown("### 📊 Decision Variables Used")
    st.write([
        COLUMN_MAP["risk"],
        COLUMN_MAP["category"],
        COLUMN_MAP["ret1y"],
        COLUMN_MAP["ret3y"],
        COLUMN_MAP["expense"]
    ])

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.recommendations is not None:

    recos = st.session_state.recommendations
    st.markdown(f"### 📌 Top {len(recos)} Recommended Mutual Funds")

    for _, row in recos.iterrows():
        st.markdown(
            f"""
**{row[COLUMN_MAP["fund"]]}**
- Category: {row[COLUMN_MAP["category"]]}
- Risk: {row[COLUMN_MAP["risk"]]}
- 1Y Return: {row[COLUMN_MAP["ret1y"]]}%
- 3Y Return: {row[COLUMN_MAP["ret3y"]]}%
- Expense Ratio: {row[COLUMN_MAP["expense"]]}%
---
"""
        )

# ------------------------------------------
# Chatbot – Follow-ups & Intent
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Chat with the Agent")

user_query = st.text_input(
    "Ask follow-ups or type 'recommend based on stability'"
)

if user_query:

    # Intent-based recommendation
    if "recommend" in user_query.lower():
        st.session_state.recommendations = recommendation_agent(
            df, risk_profile, preference, top_k
        )
        st.success("Recommendations updated based on your query.")

    elif st.session_state.recommendations is None:
        st.info("Please generate recommendations first.")

    else:
        if LLM_AVAILABLE:
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                context = st.session_state.recommendations.to_string(index=False)

                prompt = f"""
You are an explanation agent.
Use ONLY the data below. Do not recommend new funds.

DATA:
{context}

Question:
{user_query}
"""

                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )

                st.write(response.choices[0].message.content)

            except:
                st.warning("LLM unavailable. Showing recommendations only.")
        else:
            st.warning("LLM not configured.")
