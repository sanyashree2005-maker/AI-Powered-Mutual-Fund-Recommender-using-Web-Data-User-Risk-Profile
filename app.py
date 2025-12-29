import streamlit as st
from langchain_groq import ChatGroq
from datetime import datetime
from functools import lru_cache

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Market Intelligence",
    layout="wide"
)

# =============================
# LOAD GROQ LLM (STABLE MODEL)
# =============================
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0
)

# =============================
# SESSION MEMORY (FOR FOLLOW-UPS)
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# SAFE LLM CALL (NO CRASH)
# =============================
@lru_cache(maxsize=100)
def safe_llm_call(prompt: str) -> str:
    try:
        return llm.invoke(prompt).content
    except Exception:
        return "⚠️ The system is temporarily unavailable. Please try again."

# =============================
# AGENTS
# =============================
def intent_agent(query: str) -> str:
    prompt = f"""
    Classify the user's intent into ONE word:
    market | recommendation | explanation | comparison | exit

    Query: {query}
    """
    return safe_llm_call(prompt).lower()

def market_agent(query: str) -> str:
    prompt = f"""
    You are a mutual fund market intelligence assistant.
    Answer like a finance website.
    Avoid exact numbers unless certain.

    Question:
    {query}
    """
    return safe_llm_call(prompt)

def recommendation_agent(query: str, profile: dict) -> str:
    prompt = f"""
    Investor Profile:
    Risk Level: {profile['risk']}
    Investment Horizon: {profile['horizon']}
    Investment Amount: {profile['amount']}

    Task:
    Recommend suitable mutual fund categories and examples.
    Explain briefly.

    User Question:
    {query}
    """
    return safe_llm_call(prompt)

def explanation_agent(query: str, last_response: str) -> str:
    prompt = f"""
    Previous response:
    {last_response}

    Follow-up question:
    {query}

    Explain clearly in simple terms.
    """
    return safe_llm_call(prompt)

# =============================
# UI
# =============================
st.title("📈 Agentic AI – Mutual Fund Market Intelligence")
st.caption(
    f"Market insights refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
)

# Sidebar – Investor Profile
st.sidebar.header("Investor Profile")
risk = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
amount = st.sidebar.number_input(
    "Investment Amount",
    min_value=1000,
    step=500
)

query = st.text_input("Ask anything about mutual funds")

if st.button("Submit") and query:

    profile = {
        "risk": risk,
        "horizon": horizon,
        "amount": amount
    }

    intent = intent_agent(query)

    if intent == "exit":
        response = "Thank you! Feel free to ask anytime."

    elif intent == "recommendation":
        response = recommendation_agent(query, profile)

    elif intent in ["explanation", "comparison"] and st.session_state.chat_history:
        response = explanation_agent(
            query,
            st.session_state.chat_history[-1]
        )

    else:
        response = market_agent(query)

    st.session_state.chat_history.append(response)

    st.subheader("Agent Response")
    st.write(response)
