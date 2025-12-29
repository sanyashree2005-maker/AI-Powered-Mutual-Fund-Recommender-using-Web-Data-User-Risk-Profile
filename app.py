# ============================================================
# AGENTIC AI – NEAR-LIVE MUTUAL FUND INTELLIGENCE SYSTEM
# LangGraph + Public API Style Data + RAG + Follow-Up Explanations
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================================
# LLM CONFIG (API key loaded from Streamlit Secrets)
# ============================================================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# ============================================================
# 1️⃣ MARKET DATA AGENT (PUBLIC API – NEAR-LIVE)
# ============================================================
@st.cache_data(ttl=1800)  # refresh every 30 minutes
def fetch_market_data():
    """
    Simulated public-API-style mutual fund data.
    Replace with real free API endpoints if needed.
    """
    return pd.DataFrame([
        {
            "Fund Name": "Axis Bluechip Fund",
            "Category": "Equity",
            "Risk": "High",
            "1Y Return": 15,
            "3Y Return": 18,
            "Expense Ratio": 0.9,
            "Fund House": "Axis Mutual Fund"
        },
        {
            "Fund Name": "HDFC Balanced Advantage Fund",
            "Category": "Hybrid",
            "Risk": "Medium",
            "1Y Return": 11,
            "3Y Return": 13,
            "Expense Ratio": 0.8,
            "Fund House": "HDFC Mutual Fund"
        },
        {
            "Fund Name": "ICICI Prudential Liquid Fund",
            "Category": "Debt",
            "Risk": "Low",
            "1Y Return": 6,
            "3Y Return": 7,
            "Expense Ratio": 0.4,
            "Fund House": "ICICI Prudential"
        }
    ])

df = fetch_market_data()
last_updated = datetime.now().strftime("%d %b %Y, %I:%M %p")

# ============================================================
# 2️⃣ VECTOR STORE (RAG CORE – EPHEMERAL)
# ============================================================
@st.cache_resource
def build_vector_store(df):
    texts = df.apply(lambda r: str(r.to_dict()), axis=1).tolist()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_texts(texts, embedding=embeddings)

vectorstore = build_vector_store(df)

# ============================================================
# AGENT STATE (SHARED MEMORY)
# ============================================================
class AgentState(dict):
    pass

# ============================================================
# 3️⃣ INTENT CLASSIFICATION AGENT
# ============================================================
def intent_agent(state):
    prompt = f"""
    Classify intent into:
    recommendation, explanation, followup, market_overview, exit

    Query: {state['query']}
    """
    state["intent"] = llm.invoke(prompt).content.lower()
    return state

# ============================================================
# 4️⃣ USER PROFILING AGENT
# ============================================================
def profiling_agent(state):
    state["profile"] = {
        "risk": state["risk"],
        "horizon": state["horizon"],
        "amount": state["amount"]
    }
    return state

# ============================================================
# 5️⃣ RETRIEVAL AGENT (RAG)
# ============================================================
def retrieval_agent(state):
    docs = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = [d.page_content for d in docs]
    return state

# ============================================================
# 6️⃣ RECOMMENDATION AGENT
# ============================================================
def recommendation_agent(state):
    prompt = f"""
    Using ONLY the following market data:
    {state['context']}

    Recommend suitable mutual funds for:
    {state['profile']}
    """
    response = llm.invoke(prompt).content
    state["response"] = response
    state["last_recommendation"] = response
    return state

# ============================================================
# 7️⃣ EXPLANATION AGENT
# ============================================================
def explanation_agent(state):
    prompt = f"""
    Explain the recommendation using:
    risk-return tradeoff, investment horizon, and expense ratio.

    Recommendation:
    {state.get('last_recommendation')}

    Market data:
    {state['context']}
    """
    explanation = llm.invoke(prompt).content
    state["response"] = explanation
    state["last_explanation"] = explanation
    return state

# ============================================================
# 8️⃣ FOLLOW-UP EXPLANATION AGENT
# ============================================================
def followup_explanation_agent(state):
    prompt = f"""
    You are answering a follow-up question.

    Previous recommendation:
    {state.get('last_recommendation')}

    Previous explanation:
    {state.get('last_explanation')}

    User follow-up question:
    {state['query']}

    Respond clearly and consistently.
    """
    state["response"] = llm.invoke(prompt).content
    return state

# ============================================================
# 9️⃣ MARKET OVERVIEW AGENT
# ============================================================
def market_overview_agent(state):
    prompt = f"""
    Based on the latest available market data:
    {state['context']}

    Answer the user's market-related question:
    {state['query']}
    """
    state["response"] = llm.invoke(prompt).content
    return state

# ============================================================
# 🔀 LANGGRAPH ORCHESTRATOR
# ============================================================
graph = StateGraph(AgentState)

graph.add_node("Intent", intent_agent)
graph.add_node("Profile", profiling_agent)
graph.add_node("Retrieve", retrieval_agent)
graph.add_node("Recommend", recommendation_agent)
graph.add_node("Explain", explanation_agent)
graph.add_node("FollowUpExplain", followup_explanation_agent)
graph.add_node("MarketOverview", market_overview_agent)

graph.set_entry_point("Intent")

graph.add_conditional_edges(
    "Intent",
    lambda s: s["intent"],
    {
        "recommendation": "Profile",
        "explanation": "Explain",
        "followup": "FollowUpExplain",
        "market_overview": "Retrieve",
        "exit": END
    },
    default="Profile"
)

graph.add_edge("Profile", "Retrieve")
graph.add_edge("Retrieve", "Recommend")
graph.add_edge("Recommend", END)
graph.add_edge("Explain", END)
graph.add_edge("FollowUpExplain", END)
graph.add_edge("MarketOverview", END)

app = graph.compile()

# ============================================================
# STREAMLIT UI (IMPORTANT: ONLY DISPLAY INVOKE OUTPUT)
# ============================================================
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Intelligence",
    layout="wide"
)

st.title("📈 Agentic AI – Mutual Fund Market Intelligence")
st.caption(f"📅 Market data last refreshed: {last_updated}")

st.sidebar.header("Investor Profile")
risk = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
amount = st.sidebar.number_input("Investment Amount", min_value=1000)

query = st.text_input("Ask anything about mutual funds")

if st.button("Submit"):
    state = {
        "query": query,
        "risk": risk,
        "horizon": horizon,
        "amount": amount
    }

    result = app.invoke(state)

    st.subheader("Agent Response")
    st.write(result.get("response", "No response generated"))
