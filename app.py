# =========================================
# Agentic AI – Mutual Fund Market Intelligence
# =========================================

import streamlit as st
from langchain_groq import ChatGroq

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Market Intelligence",
    layout="wide"
)

st.title("📈 Agentic AI – Mutual Fund Market Intelligence")
st.caption("Ask anything about mutual funds, market trends, and investment strategies")

# -------------------------------
# LOAD GROQ LLM
# -------------------------------
llm = ChatGroq(
    model="llama3-70b-8192",   # most stable Groq model
    temperature=0
)

# -------------------------------
# SAFE LLM CALL (NO CACHE)
# -------------------------------
def safe_llm_call(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception:
        return "⚠️ The system is temporarily unavailable. Please try again."

# -------------------------------
# SIDEBAR – INVESTOR PROFILE
# -------------------------------
st.sidebar.header("Investor Profile")

risk = st.sidebar.selectbox(
    "Risk Profile",
    ["Low", "Medium", "High"]
)

horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short", "Medium", "Long"]
)

amount = st.sidebar.number_input(
    "Investment Amount",
    min_value=500,
    step=500
)

# -------------------------------
# SESSION MEMORY (FOLLOW-UPS)
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# USER INPUT
# -------------------------------
query = st.text_input("Ask anything about mutual funds")

submit = st.button("Submit")

# -------------------------------
# AGENT LOGIC
# -------------------------------
if submit and query:
    context = f"""
You are a financial market intelligence assistant.

Investor profile:
- Risk: {risk}
- Horizon: {horizon}
- Amount: {amount}

Previous conversation:
{st.session_state.chat_history}

User question:
{query}

Rules:
- Explain clearly
- No guaranteed returns
- Educational purpose only
"""

    answer = safe_llm_call(context)

    st.session_state.chat_history.append(
        f"User: {query}\nAgent: {answer}"
    )

# -------------------------------
# DISPLAY RESPONSE
# -------------------------------
if st.session_state.chat_history:
    st.subheader("Agent Response")
    st.write(st.session_state.chat_history[-1].split("Agent:")[1])

# -------------------------------
# DISCLAIMER
# -------------------------------
st.markdown("---")
st.caption("⚠️ Educational use only. Not financial advice.")
