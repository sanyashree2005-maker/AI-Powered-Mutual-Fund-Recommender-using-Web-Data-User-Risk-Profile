# ============================================================
# AGENTIC AI – NEAR-LIVE MUTUAL FUND INTELLIGENCE SYSTEM
# LangGraph + RAG + Follow-Up Explanations
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================================
# LLM CONFIG (API KEY FROM STREAMLIT SECRETS)
# ============================================================
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ============================================================
# MARKET DATA AGENT (NEAR-LIVE – API STYLE)
# ============================================================
@st.cache_data(ttl=1800)
def fetch_market_data():
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
# VECTOR STORE (RAG CORE)
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
# AGENT STATE
# ============================================================
class AgentState(dict):
    pass

# ============================================================
# INTENT CLASSIFICATION AGENT (SAFE)
# ============================================================
def intent_agent(state):
    prompt = f"""
    Classify intent into EXACTLY one of:
    recommendation, explanation, followup, market_overview, exit

    If unsure, choose recommendation.

    Query: {state['query']}
    """
    intent = llm.invoke(prompt).content.lower().strip()

    if intent not in [
        "recommendation",
        "explanation",
        "followup",
        "market_overview",
        "exit"
    ]:
        intent = "recommendation"

    state["intent"] = intent
    return state

# ============================================================
# USER PROFILING AGENT
# ============================================================
def profiling_agent(state):
    state["profile"] = {
        "risk": state["risk"],
        "horizon": state["horizon"],
        "amount": state["amount"]
    }
    return state

# ============================================================
# RETRIEVAL AGENT (RAG)
# ============================================================
def retrieval_agent(state):
    docs = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = [d.page_content for d in docs]
    return state

# ============================================================
# RECOMMENDATION AGENT
# ============================================================
def recommendation_agent(state):
    prompt = f"""
    Using ONLY the following market data:
    {state['context']}

    Recommend suitable mutual funds for:
    {state['profile']}
    """
    state["response"] = llm.invoke(prompt).content
    state["last_recommendation"] = state["response"]
    return state

# ============================================================
# EXPLANATION AGENT
# ============================================================
def explanation_agent(state):
    prompt = f"""
    Explain the recommendation using:
    risk-return tradeoff, horizon, and expense ratio.

    Recommendation:
    {state.get('last_recommendation')}

    Market data:
    {state['context']}
    """
    state["response"] = llm.invoke(prompt).content
    state["last_explanation"] = state["response"]
    return state

# ============================================================
# FOLLOW-UP EXPLANATION AGENT
# ============================================================
def followup_explanation_agent(state):
    prompt = f"""
    Previous recommendation:
    {state.get('last_recommendation')}

    Previous explanation:
    {state.get('last_explanation')}

    Follow-up question:
    {state['query']}
    """
    state["response"] = llm.invoke(prompt).content
    return state

# ============================================================
# MARKET OVERVIEW AGENT
# ============================================================
def market_overview_agent(state):
    prompt = f"""
    Based on the latest publicly available market data:
    {state['context']}

    Answer:
    {state['query']}
    """
    state["response"] = llm.invoke(prompt).content
    return state

# ============================================================
# LANGGRAPH ORCHESTRATOR (FINAL FIXED)
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
    }
)

graph.add_edge("Profile", "Retrieve")
graph.add_edge("Retrieve", "Recommend")
graph.add_edge("Retrieve", "MarketOverview")
graph.add_edge("Recommend", END)
graph.add_edge("Explain", END)
graph.add_edge("FollowUpExplain", END)
graph.add_edge("MarketOverview", END)

app = graph.compile()

# ============================================================
# STREAMLIT UI
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

if st.button("Submit") and query:
    state = {
        "query": query,
        "risk": risk,
        "horizon": horizon,
        "amount": amount
    }

    result = app.invoke(state)

    st.subheader("Agent Response")
    st.write(result.get("response", "No response generated"))
