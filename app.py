import streamlit as st
from langchain_openai import ChatOpenAI
from datetime import datetime

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Market Intelligence",
    layout="wide"
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# =============================
# SESSION MEMORY (FOR FOLLOW UPS)
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# AGENTS
# =============================
def intent_agent(query):
    prompt = f"""
    Classify the user's intent into one of:
    - market
    - recommendation
    - explanation
    - comparison
    - exit

    Query: {query}
    """
    return llm.invoke(prompt).content.strip().lower()

def market_agent(query):
    prompt = f"""
    You are a financial market intelligence assistant.
    Answer using general market knowledge.
    Avoid exact numbers unless confident.

    Question: {query}
    """
    return llm.invoke(prompt).content

def recommendation_agent(query, profile):
    prompt = f"""
    User Profile:
    Risk: {profile['risk']}
    Horizon: {profile['horizon']}
    Amount: {profile['amount']}

    Task:
    Recommend suitable mutual fund categories and examples.
    No guarantees, no financial advice disclaimer needed.

    Question: {query}
    """
    return llm.invoke(prompt).content

def explanation_agent(query, last_response):
    prompt = f"""
    The user is asking a follow-up question.

    Previous Answer:
    {last_response}

    Follow-up Question:
    {query}

    Explain clearly and simply.
    """
    return llm.invoke(prompt).content

# =============================
# UI
# =============================
st.title("📈 Agentic AI – Mutual Fund Market Intelligence")
st.caption(f"Market data refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

st.sidebar.header("Investor Profile")
risk = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
amount = st.sidebar.number_input("Investment Amount", min_value=1000, step=500)

query = st.text_input("Ask anything about mutual funds")

if st.button("Submit") and query:

    profile = {
        "risk": risk,
        "horizon": horizon,
        "amount": amount
    }

    intent = intent_agent(query)

    if intent == "exit":
        response = "Thank you! Let me know if you need anything else."

    elif intent == "market":
        response = market_agent(query)

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
