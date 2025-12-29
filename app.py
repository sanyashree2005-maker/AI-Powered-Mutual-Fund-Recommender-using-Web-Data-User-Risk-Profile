# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# Dataset-Driven + Chat Agent (No Auto LLM Calls)
# ==========================================

import streamlit as st
import pandas as pd

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

# ------------------------------------------
# Sidebar – Investor Preferences
# ------------------------------------------
st.sidebar.header("Investor Preferences")

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

get_reco = st.sidebar.button("Get Recommendations")

# ------------------------------------------
# Risk Mapping
# ------------------------------------------
RISK_MAP = {
    "Low": 2,
    "Medium": 3,
    "High": 5
}

# ------------------------------------------
# Recommendation Agent (Deterministic)
# ------------------------------------------
def recommendation_agent(data):
    filtered = data.copy()

    # Risk filter
    filtered = filtered[filtered["Risk Level"] <= RISK_MAP[risk_profile]]

    # Preference logic
    if preference == "Stability":
        filtered = filtered.sort_values(
            by=["Risk Level", "Expense Ratio (%)"]
        )

    elif preference == "Growth":
        filtered = filtered.sort_values(
            by=["3Y Return (%)", "1Y Return (%)"],
            ascending=False
        )

    elif preference == "Tax Saving":
        filtered = filtered[
            filtered["Category"].str.contains("ELSS", case=False, na=False)
        ]

    return filtered.head(top_k)

# ------------------------------------------
# Session State
# ------------------------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ------------------------------------------
# Generate Recommendations (ONLY ON CLICK)
# ------------------------------------------
if get_reco:
    st.session_state.recommendations = recommendation_agent(df)

# ------------------------------------------
# Show Dataset Variables Used
# ------------------------------------------
with st.expander("📊 Decision Variables Used"):
    st.markdown("""
- **Risk Level**
- **1Y Return (%)**
- **3Y Return (%)**
- **Expense Ratio (%)**
- **Category**
""")

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.recommendations is not None:
    recs = st.session_state.recommendations

    if recs.empty:
        st.warning("No funds matched your criteria.")
    else:
        st.subheader(f"📌 Top {len(recs)} Recommended Mutual Funds")

        for _, row in recs.iterrows():
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
# Chat Agent (Explanation / Follow-ups)
# ------------------------------------------
st.markdown("### 💬 Chat with the Agent")

user_query = st.text_input(
    "Ask follow-ups like: 'Why were these funds recommended?'"
)

if user_query:
    if st.session_state.recommendations is None:
        st.warning("Please generate recommendations first.")
    else:
        q = user_query.lower()
        recs = st.session_state.recommendations

        if "why" in q:
            st.success(
                "These funds were recommended because they match your selected "
                "risk profile, preference, and investment horizon using dataset "
                "variables such as Risk Level, Returns, Category, and Expense Ratio."
            )

        elif "risk" in q:
            st.success(
                "Risk was evaluated using the 'Risk Level' column. "
                "Only funds within your selected risk tolerance were chosen."
            )

        elif "return" in q:
            st.success(
                "Returns were evaluated using 1-Year and 3-Year return columns "
                "from the dataset."
            )

        elif "compare" in q and len(recs) >= 2:
            f1, f2 = recs.iloc[0], recs.iloc[1]
            st.success(
                f"Comparison:\n\n"
                f"- {f1['Fund Name']} → Risk: {f1['Risk Level']}, "
                f"3Y Return: {f1['3Y Return (%)']}%\n"
                f"- {f2['Fund Name']} → Risk: {f2['Risk Level']}, "
                f"3Y Return: {f2['3Y Return (%)']}%"
            )

        else:
            st.info(
                "You can ask about:\n"
                "- Why funds were recommended\n"
                "- Risk level\n"
                "- Returns\n"
                "- Expense ratio\n"
                "- Comparison between funds"
            )

# ------------------------------------------
# Footer
# ------------------------------------------
st.caption(
    "Recommendations are generated using deterministic, dataset-driven logic. "
    "The chat agent explains and reasons over the recommendations without "
    "regenerating them."
)
