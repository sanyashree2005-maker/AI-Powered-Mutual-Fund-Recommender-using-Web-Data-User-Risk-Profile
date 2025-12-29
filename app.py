import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Recommendation System",
    layout="wide"
)

st.title("🤖 Agentic AI – Mutual Fund Recommendation System")

# -------------------------------------------------
# DATASET (GITHUB RAW URL)
# -------------------------------------------------
DATA_URL = "Mutual Funds Data.csv"
# 🔴 Replace with your actual GitHub RAW CSV URL

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# -------------------------------------------------
# SIDEBAR – INVESTOR PROFILE (HARD CONSTRAINTS)
# -------------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox(
    "Risk Profile", ["Low", "Moderate", "High"]
)

investment_horizon = st.sidebar.selectbox(
    "Investment Horizon", ["Short", "Medium", "Long"]
)

preference = st.sidebar.multiselect(
    "Preferences",
    ["Stability", "Growth", "Tax Saving"],
    max_selections=1
)
preference = preference[0] if preference else None

# 🔢 TOP-K CONTROL (FIX FOR “TOP 2” ISSUE)
top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# 🧠 OPTIONAL LLM EXPLANATION
use_llm_explanation = st.sidebar.checkbox(
    "Enable AI Explanation",
    value=False
)

# -------------------------------------------------
# CORE RECOMMENDATION LOGIC
# -------------------------------------------------
def apply_preference_filter(df, preference):
    if preference == "Stability":
        return df.sort_values(
            by=["risk_level", "fund_size_cr", "returns_3yr"],
            ascending=[True, False, False]
        )

    if preference == "Growth":
        return df.sort_values(
            by=["returns_3yr", "returns_5yr", "sharpe"],
            ascending=[False, False, False]
        )

    if preference == "Tax Saving":
        return df[df["category"].str.contains("ELSS", case=False, na=False)]

    return df

# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------
user_query = st.chat_input("Ask anything about mutual funds...")

# -------------------------------------------------
# LLM (EXPLANATION ONLY – OPTIONAL)
# -------------------------------------------------
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0.3
)

# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if user_query and preference:

    filtered_df = apply_preference_filter(df, preference)

    if filtered_df.empty:
        st.error("No mutual funds match the selected preference.")
        st.stop()

    top_funds = filtered_df.head(top_k)[[
        "scheme_name",
        "category",
        "risk_level",
        "returns_1yr",
        "returns_3yr",
        "returns_5yr",
        "expense_ratio"
    ]]

    # -------------------------
    # DISPLAY RECOMMENDATIONS
    # -------------------------
    st.subheader(f"📌 Top {top_k} Recommended Mutual Funds")

    for _, row in top_funds.iterrows():
        st.markdown(
            f"""
**{row['scheme_name']}**  
• Category: {row['category']}  
• Risk Level: {row['risk_level']}  
• 1Y Return: {row['returns_1yr']}%  
• 3Y Return: {row['returns_3yr']}%  
• Expense Ratio: {row['expense_ratio']}%
"""
        )

    # -------------------------
    # OPTIONAL LLM EXPLANATION
    # -------------------------
    if use_llm_explanation:
        context = top_funds[[
            "scheme_name",
            "category",
            "risk_level",
            "returns_3yr",
            "expense_ratio"
        ]].to_string(index=False)

        explanation_prompt = f"""
Using ONLY the data below, explain briefly why these funds match the user's profile.

User Profile:
- Risk: {risk_profile}
- Horizon: {investment_horizon}
- Preference: {preference}

Rules:
- No new fund names
- No predictions
- No investment advice

Data:
{context}
"""

        try:
            explanation = llm.invoke(explanation_prompt)
            st.subheader("🧠 Explanation")
            st.write(explanation.content)
        except Exception:
            st.subheader("🧠 Explanation")
            st.info(
                "Explanation is temporarily unavailable due to LLM limits. "
                "The recommendations above are generated directly from the dataset."
            )

elif user_query and not preference:
    st.warning("Please select one preference: Stability, Growth, or Tax Saving.")
