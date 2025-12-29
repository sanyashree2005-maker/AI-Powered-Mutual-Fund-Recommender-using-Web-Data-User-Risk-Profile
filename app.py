# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset + Query Agent (NO LLM)
# ==========================================

import streamlit as st
import pandas as pd

# ------------------------------------------
# Page Configuration
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
    df.columns = [c.strip().lower() for c in df.columns]
    return df

df = load_data()

# ------------------------------------------
# Auto Column Detection
# ------------------------------------------
def find_col(keywords):
    for col in df.columns:
        for kw in keywords:
            if kw in col:
                return col
    return None

COL_FUND = find_col(["fund", "scheme", "name"])
COL_CATEGORY = find_col(["category"])
COL_RISK = find_col(["risk"])
COL_1Y = find_col(["1y"])
COL_3Y = find_col(["3y"])
COL_EXPENSE = find_col(["expense"])

# ------------------------------------------
# Sidebar – Investor Profile
# ------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
investment_horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
preference = st.sidebar.multiselect(
    "Preferences", ["Stability", "Growth", "Tax Saving"], default=["Stability"]
)
top_k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# ------------------------------------------
# Recommendation Agent
# ------------------------------------------
filtered = df.copy()

if "Stability" in preference:
    filtered = filtered[filtered[COL_RISK] <= 2]
    filtered["score"] = (5 - filtered[COL_RISK]) + (1 / filtered[COL_EXPENSE])

elif "Growth" in preference:
    filtered["score"] = filtered[COL_3Y] * 1.5 + filtered[COL_1Y]

elif "Tax Saving" in preference:
    filtered = filtered[
        filtered[COL_CATEGORY].str.contains("elss", case=False, na=False)
    ]
    filtered["score"] = filtered[COL_3Y]

top_funds = filtered.sort_values("score", ascending=False).head(top_k)

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
st.subheader(f"📌 Top {len(top_funds)} Recommended Mutual Funds")

for _, row in top_funds.iterrows():
    st.markdown(
        f"""
**{row[COL_FUND]}**
- Category: {row[COL_CATEGORY]}
- Risk Level: {row[COL_RISK]}
- 1Y Return: {row[COL_1Y]}%
- 3Y Return: {row[COL_3Y]}%
- Expense Ratio: {row[COL_EXPENSE]}%
---
"""
    )

# ==========================================
# 🔎 QUERY AGENT (THIS IS WHAT YOU WERE MISSING)
# ==========================================
st.markdown("## 🔎 Ask Questions About the Dataset")

query = st.text_input(
    "Try: 'low risk funds', 'highest return fund', 'elss funds', 'lowest expense fund'"
)

if query:
    q = query.lower()
    result = df.copy()

    if "low risk" in q:
        result = result[result[COL_RISK] <= 2]

    if "elss" in q or "tax" in q:
        result = result[result[COL_CATEGORY].str.contains("elss", case=False, na=False)]

    if "highest return" in q:
        result = result.sort_values(COL_3Y, ascending=False).head(1)

    if "lowest expense" in q:
        result = result.sort_values(COL_EXPENSE).head(1)

    st.subheader("📊 Query Result")

    if result.empty:
        st.warning("No matching results found.")
    else:
        for _, row in result.head(5).iterrows():
            st.markdown(
                f"""
**{row[COL_FUND]}**
- Category: {row[COL_CATEGORY]}
- Risk Level: {row[COL_RISK]}
- 3Y Return: {row[COL_3Y]}%
- Expense Ratio: {row[COL_EXPENSE]}%
---
"""
            )

# ------------------------------------------
# Footer
# ------------------------------------------
st.caption(
    "This system uses agent-based deterministic decision logic on structured financial data."
)
