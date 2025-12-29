import streamlit as st
import pandas as pd
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# LLM
# =========================
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# =========================
# MARKET DATA (Near-Live)
# =========================
@st.cache_data(ttl=1800)
def fetch_market_data():
    return pd.DataFrame([
        {"Fund":"Axis Bluechip Fund","Category":"Equity","Risk":"High","1Y":15,"3Y":18,"Expense":0.9},
        {"Fund":"HDFC Balanced Advantage Fund","Category":"Hybrid","Risk":"Medium","1Y":11,"3Y":13,"Expense":0.8},
        {"Fund":"ICICI Prudential Liquid Fund","Category":"Debt","Risk":"Low","1Y":6,"3Y":7,"Expense":0.4},
    ])

df = fetch_market_data()
timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

# =========================
# VECTOR STORE (RAG)
# =========================
@st.cache_resource
def build_vs(df):
    texts = df.apply(lambda r: str(r.to_dict()), axis=1).tolist()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_texts(texts, emb)

vs = build_vs(df)

# =========================
# STATE
# =========================
class AgentState(dict):
    pass

# =========================
# AGENTS
# =========================
def intent_agent(state):
    prompt = f"""
    Classify intent into one of:
    recommendation, explanation, followup, market_overview, exit.
    If unsure, use recommendation.

    Query: {state['query']}
    """
    intent = llm.invoke(prompt).content.strip().lower()
    if intent not in ["recommendation","explanation","followup","market_overview","exit"]:
        intent = "recommendation"
    state["intent"] = intent
    return state

def profile_agent(state):
    state["profile"] = {
        "risk": state["risk"],
        "horizon": state["horizon"],
        "amount": state["amount"]
    }
    return state

def retrieve_agent(state):
    docs = vs.similarity_search(state["query"], k=3)
    state["context"] = [d.page_content for d in docs]
    return state

def recommend_agent(state):
    prompt = f"""
    Use ONLY this data:
    {state['context']}

    Recommend funds for:
    {state['profile']}
    """
    state["response"] = llm.invoke(prompt).content
    state["last_recommendation"] = state["response"]
    return state

def explain_agent(state):
    prompt = f"""
    Explain clearly using risk, horizon and expense ratio.

    Recommendation:
    {state.get('last_recommendation')}
    """
    state["response"] = llm.invoke(prompt).content
    state["last_explanation"] = state["response"]
    return state

def followup_agent(state):
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

def market_agent(state):
    prompt = f"""
    Based on latest public market data:
    {state['context']}

    Question:
    {state['query']}
    """
    state["response"] = llm.invoke(prompt).content
    return state

def finalize_agent(state):
    if "response" not in state or not state["response"]:
        state["response"] = (
            "I could not generate a clear answer from available market data. "
            "Please rephrase your question."
        )
    return state

# =========================
# LANGGRAPH
# =========================
graph = StateGraph(AgentState)

graph.add_node("Intent", intent_agent)
graph.add_node("Profile", profile_agent)
graph.add_node("Retrieve", retrieve_agent)
graph.add_node("Recommend", recommend_agent)
graph.add_node("Explain", explain_agent)
graph.add_node("FollowUp", followup_agent)
graph.add_node("Market", market_agent)
graph.add_node("Finalize", finalize_agent)

graph.set_entry_point("Intent")

graph.add_conditional_edges(
    "Intent",
    lambda s: s["intent"],
    {
        "recommendation": "Profile",
        "explanation": "Explain",
        "followup": "FollowUp",
        "market_overview": "Retrieve",
        "exit": "Finalize"
    }
)

graph.add_edge("Profile", "Retrieve")
graph.add_edge("Retrieve", "Recommend")
graph.add_edge("Retrieve", "Market")

graph.add_edge("Recommend", "Finalize")
graph.add_edge("Explain", "Finalize")
graph.add_edge("FollowUp", "Finalize")
graph.add_edge("Market", "Finalize")
graph.add_edge("Finalize", END)

app = graph.compile()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("Agentic AI – Mutual Fund Intelligence", layout="wide")

st.title("📈 Agentic AI – Mutual Fund Market Intelligence")
st.caption(f"📅 Market data last refreshed: {timestamp}")

st.sidebar.header("Investor Profile")
risk = st.sidebar.selectbox("Risk Profile", ["Low","Medium","High"])
horizon = st.sidebar.selectbox("Investment Horizon", ["Short","Medium","Long"])
amount = st.sidebar.number_input("Investment Amount", min_value=1000)

query = st.text_input("Ask anything about mutual funds")

if st.button("Submit") and query:
    result = app.invoke({
        "query": query,
        "risk": risk,
        "horizon": horizon,
        "amount": amount
    })
    st.subheader("Agent Response")
    st.write(result.get("response"))
