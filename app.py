# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset-Driven (Robust Column Detection)
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
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

df = load_data()

# ------------------------------------------
# Auto-detect Columns (CRITICAL FIX)
# ------------------------------------------
def find_column(keywords):
    for col in df.columns:
        for kw in keywords:
            if kw in col:
                return col
    return None

COL_FUND = find_column(["fund", "scheme", "name"])
COL_CATEGORY = find_column(["category"])
COL_RISK = find_column(["risk"])
COL_1Y = find_column(["1y", "1 yr", "one year"])
COL_3Y = find_column(["3y", "3 yr", "three year"])
COL_EXPENSE = find_column(["expense"])

required = {
    "Fund Name": COL_FUND,
    "Category": COL_CATEGORY,
    "Risk": COL_RISK,
    "1Y Return": COL_1Y,
    "3Y Return": COL_3Y,
    "Expense Ratio": COL_EXPENSE,
}

missing = [k for k, v in required.items() if v is None]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ------------------------------------------
# Sidebar – Investor Profile
# ------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox(
    "Risk Profile", ["Low", "Medium", "High"]
)

investment_horizon = st.sidebar.selectbox(
    "Investment Horizon", ["Short", "Medium", "Long"]
)

preference = st.sidebar.multiselect(
    "Preferences",
    ["Stability", "Growth", "Tax Saving"],
    default=["Stability"]
)

top_k = st.sidebar.slider(
    "Number of recommendations", 1, 10, 5
)

# ------------------------------------------
# Recommendation Logic (NO LLM)
# ------------------------------------------
filtered_df = df.copy()

# Stability → low risk, low expense
if "Stability" in preference:
    filtered_df = filtered_df[filtered_df[COL_RISK] <= 2]
    filtered_df["score"] = (5 - filtered_df[COL_RISK]) + (1 / filtered_df[COL_EXPENSE])

# Growth → higher returns
elif "Growth" in preference:
    filtered_df["score"] = filtered_df[COL_3Y] * 1.5 + filtered_df[COL_1Y]

# Tax Saving → ELSS funds
elif "Tax Saving" in preference:
    filtered_df = filtered_df[
        filtered_df[COL_CATEGORY].str.contains("elss", case=False, na=False)
    ]
    filtered_df["score"] = filtered_df[COL_3Y]

# Fallback
if filtered_df.empty:
    st.warning("No funds matched the selected criteria.")
    st.stop()

# Rank & select
top_funds = filtered_df.sort_values("score", ascending=False).head(top_k)

# ------------------------------------------
# Display Results
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

st.caption(
    "Recommendations are generated directly from structured mutual fund data "
    "using deterministic agent-style decision logic."
)
