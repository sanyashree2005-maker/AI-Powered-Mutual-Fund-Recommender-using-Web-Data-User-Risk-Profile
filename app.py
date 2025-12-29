# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset-Driven | Button-Triggered | Stable
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
    return pd.read_csv("Mutual Funds Data.csv")

df = load_data()

# ------------------------------------------
# Column Mapping (VERY IMPORTANT)
# ------------------------------------------
COL_FUND = "Fund Name"
COL_CATEGORY = "Category"
COL_RISK = "Risk Level"
COL_1Y = "1Y Return (%)"
COL_3Y = "3Y Return (%)"
COL_EXPENSE = "Expense Ratio (%)"

# ------------------------------------------
# Sidebar – Investor Profile
# ------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox(
    "Risk Profile",
    ["Low", "Medium", "High"]
)

investment_horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short", "Medium", "Long"]
)

preference = st.sidebar.selectbox(
    "Primary Preference",
    ["Stability", "Growth", "Tax Saving"]
)

top_k = st.sidebar.slider(
    "Number of Recommendations",
    1, 10, 5
)

generate = st.sidebar.button("🔍 Get Recommendations")

# ------------------------------------------
# Risk Mapping
# ------------------------------------------
RISK_MAP = {
    "Low": 2,
    "Medium": 3,
    "High": 5
}

# ------------------------------------------
# Show Decision Variables
# ------------------------------------------
with st.expander("📊 Decision Variables Used"):
    st.write("""
- Risk Level  
- 1-Year Return  
- 3-Year Return  
- Expense Ratio  
- Fund Category  
- User Risk Profile  
- User Preference  
""")

# ------------------------------------------
# Recommendation Logic (ONLY ON CLICK)
# ------------------------------------------
if generate:

    filtered = df.copy()

    # Risk-based filtering
    filtered = filtered[filtered[COL_RISK] <= RISK_MAP[risk_profile]]

    # Preference-based logic
    if preference == "Stability":
        filtered = filtered.sort_values(
            by=[COL_RISK, COL_EXPENSE],
            ascending=[True, True]
        )

    elif preference == "Growth":
        filtered = filtered.sort_values(
            by=[COL_3Y, COL_1Y],
            ascending=False
        )

    elif preference == "Tax Saving":
        filtered = filtered[
            filtered[COL_CATEGORY].str.contains("ELSS", case=False, na=False)
        ]

    # Fallback
    if filtered.empty:
        st.warning("No mutual funds match the selected criteria.")
        st.stop()

    top_funds = filtered.head(top_k)

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

# ------------------------------------------
# Footer
# ------------------------------------------
st.caption(
    "This system generates recommendations using structured mutual fund data "
    "and deterministic decision logic triggered by user intent."
)
