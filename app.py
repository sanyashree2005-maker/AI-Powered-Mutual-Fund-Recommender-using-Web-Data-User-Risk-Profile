# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset-Driven with Defined Variables
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
# Decision Variables (Explicit Definition)
# ------------------------------------------
with st.expander("📊 Decision Variables Used for Recommendation"):
    st.markdown("""
The recommendation system uses the following **dataset variables**:

- **Fund Name** → Identification of the mutual fund  
- **Category** → Used to identify Tax Saving (ELSS) funds  
- **Risk Level** → Used to filter funds for Stability  
- **1Y Return (%)** → Used for short-term performance evaluation  
- **3Y Return (%)** → Used for growth-based ranking  
- **Expense Ratio (%)** → Used to assess cost efficiency  

The system **does not use any external knowledge or LLM reasoning**.
All recommendations are **data-driven and deterministic**.
""")

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

investment_horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short", "Medium", "Long"]
)

preference = st.sidebar.multiselect(
    "Preferences",
    ["Stability", "Growth", "Tax Saving"],
    default=["Stability"]
)

top_k = st.sidebar.slider(
    "Number of recommendations",
    1, 10, 5
)

# ------------------------------------------
# Recommendation Logic
# ------------------------------------------
filtered_df = df.copy()

if "Stability" in preference:
    filtered_df = filtered_df[filtered_df["Risk Level"] <= 2]

if "Growth" in preference:
    filtered_df = filtered_df.sort_values(
        by=["3Y Return (%)", "1Y Return (%)"],
        ascending=False
    )

if "Tax Saving" in preference:
    filtered_df = filtered_df[
        filtered_df["Category"].str.contains("ELSS", case=False, na=False)
    ]

if filtered_df.empty:
    st.warning("No funds matched the selected criteria.")
    st.stop()

top_funds = filtered_df.head(top_k)

# ------------------------------------------
# Display Results
# ------------------------------------------
st.subheader(f"📌 Top {len(top_funds)} Recommended Mutual Funds")

for _, row in top_funds.iterrows():
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
# Footer
# ------------------------------------------
st.caption(
    "Recommendations are generated using explicitly defined dataset variables."
)
