# ==========================================
# Agentic AI – Mutual Fund Recommendation System
# CSV-Driven + Safe Explanation Agent
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
    return pd.read_csv("Mutual Funds Data.csv")

df = load_data()

# ------------------------------------------
# Sidebar – Controls
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

    data = data[data["Risk Level"] <= RISK_MAP[risk]]

    if preference == "Growth":
        data = data.sort_values(
            by=["3Y Return (%)", "1Y Return (%)"],
            ascending=False
        )

    elif preference == "Stability":
        data = data.sort_values(
            by=["Risk Level", "Expense Ratio (%)"],
            ascending=[True, True]
        )

    elif preference == "Tax Saving":
        data = data[
            data["Category"].str.contains("ELSS", case=False, na=False)
        ]

    return data.head(k)

# ------------------------------------------
# Generate Recommendations ONLY ON CLICK
# ------------------------------------------
if generate:
    st.session_state.recommendations = recommendation_agent(
        df, risk_profile, preference, top_k
    )

# ------------------------------------------
# Display Recommendation Variables
# ------------------------------------------
if st.session_state.recommendations is not None:
    st.markdown("### 📊 Decision Variables Used")
    st.write(
        [
            "Risk Level",
            "Category",
            "1Y Return (%)",
            "3Y Return (%)",
            "Expense Ratio (%)"
        ]
    )

# ------------------------------------------
# Display Recommendations
# ------------------------------------------
if st.session_state.recommendations is not None:

    recos = st.session_state.recommendations

    st.markdown(f"### 📌 Top {len(recos)} Recommended Mutual Funds")

    for _, row in recos.iterrows():
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
# Chatbot (Explanation Agent)
# ------------------------------------------
st.markdown("---")
st.subheader("💬 Ask Follow-up Questions")

user_query = st.text_input(
    "Ask about recommendations or type 'recommend based on stability'"
)

if user_query:

    # Intent trigger
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

Use ONLY the data below.
Do not recommend new funds.

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
                st.warning(
                    "LLM is temporarily unavailable. "
                    "You can still view recommendations."
                )
        else:
            st.warning(
                "LLM not configured. Chatbot explanation is disabled."
            )
