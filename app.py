# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Robust CSV + Chatbot + Safe LLM
# ==========================================

import streamlit as st
import pandas as pd

# Optional LLM (safe)
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
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df = load_data()

# ------------------------------------------
# Auto Column Detection (CRITICAL FIX)
# ------------------------------------------
def find_column(possible_names):
    for col in df.columns:
        for name in possible_names:
            if name in col:
                return col
    return None

COL_FUND = find_column(["fund"])
COL_CATEGORY = find_column(["category"])
COL_RISK = find_column(["risk"])
COL_RET1Y = find_column(["1y"])
COL_RET3Y = find_column(["3y"])
COL_EXPENSE = find_column(["expense"])

required = [COL_FUND, COL_CATEGORY, COL_RISK, COL_RET1Y, COL_RET3Y, COL_EXPENSE]
if any(c is None for c in required):
    st.error("Dataset columns not recognized. Please check CSV structure.")
    st.stop()

# ------------------------------------------
# Sidebar – Investor Preferences
# ------------------------------------------
st.sidebar.header("Investor Preferences")

risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
preference = st.sidebar.selectbox(
    "Primary Preference", ["Stability", "Growth", "Tax Saving"]
)
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

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
def recommendation_agent(data, risk, pref, k):
    df = data.copy()

    df = df[df[COL_RISK] <= RISK_MAP[risk]]

    if pref == "Growth":
        df = df.sort_values(by=[COL_RET3Y, COL_RET1Y], ascending=False)

    elif pref == "Stability":
        df = df.sort_values(by=[COL_RISK, COL_EXPENSE])

    elif pref == "Tax Saving":
        df = df[df[COL_CATEGORY].str.contains("elss", case=False, na=False)]

    return df.head(k)

# ------------------------------------------
# Generate Recommendations (ONLY ON CLICK)
# ------------------------------------------
if generate:
    st.session_state.recommendations = recommendation_agent(
        df, risk_profile, preference, top_k
    )

# ------------------------------------------
# Decision Variables
# ------------------------------------------
st.markdown("### 📊 Decision Variables Used")
st.write([COL_RISK, COL_CATEGORY, COL_RET1Y, COL_RET3Y, COL_EXPENSE])

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.recommendations is not None:
    recos = st.session_state.recommendations

    st.markdown(f"### 📌 Top {len(recos)} Recommended Mutual Funds")

    for _, row in recos.iterrows():
        st.markdown(
            f"""
**{row[COL_FUND]}**
- Category: {row[COL_CATEGORY]}
- Risk Level: {row[COL_RISK]}
- 1Y Return: {row[COL_RET1Y]}%
- 3Y Return: {row[COL_RET3Y]}%
- Expense Ratio: {row[COL_EXPENSE]}%
---
"""
        )

# ------------------------------------------
# Chatbot – Follow-ups & Intent
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Chat with the Agent")

user_query = st.text_input(
    "Ask follow-ups or type: 'recommend based on stability'"
)

if user_query:

    if "recommend" in user_query.lower():
        st.session_state.recommendations = recommendation_agent(
            df, risk_profile, preference, top_k
        )
        st.success("Recommendations updated based on your request.")

    elif st.session_state.recommendations is None:
        st.info("Please generate recommendations first.")

    else:
        if LLM_AVAILABLE:
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                context = st.session_state.recommendations.to_string(index=False)

                prompt = f"""
You are an explanation agent.
Use ONLY the data below.

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
                st.warning("LLM unavailable. Showing dataset-based results only.")
        else:
            st.warning("LLM not configured.")
