# ============================================================
# AGENTIC AI – MUTUAL FUND RECOMMENDER
# LangGraph + RAG + Web-Scraped Data
# Streamlit Deployment Ready
# ============================================================

import streamlit as st
import pandas as pd

from langgraph.graph import StateGraph, END
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ============================================================
# LLM CONFIG (Loaded from Streamlit Secrets)
# ============================================================
llm = OpenAI(temperature=0)

# ============================================================
# WEB-SCRAPED / SIMULATED DATA (ACADEMIC SAFE)
# ============================================================
@st.cache_data
def load_fund_data():
    data = [
        {
            "Fund Name": "Axis Bluechip Fund",
            "Category": "Equity",
            "Risk": "High",
            "1Y Return": "15%",
            "3Y Return": "18%",
            "5Y Return": "20%",
            "Expense Ratio": "0.9%",
            "Fund House": "Axis Mutual Fund"
        },
        {
            "Fund Name": "HDFC Balanced Advantage",
            "Category": "Hybrid",
            "Risk": "Medium",
            "1Y Return": "11%",
            "3Y Return": "13%",
            "5Y Return": "14%",
            "Expense Ratio": "0.8%",
            "Fund House": "HDFC Mutual Fund"
        },
        {
            "Fund Name": "ICICI Prudential Liquid Fund",
            "Category": "Debt",
            "Risk": "Low",
            "1Y Return": "6%",
            "3Y Return": "7%",
            "5Y Return": "7.5%",
            "Expense Ratio": "0.4%",
            "Fund House": "ICICI Prudential"
        }
    ]
    return pd.DataFrame(data)

df = load_fund_data()

# ============================================================
# VECTOR STORE (RAG CORE)
# ============================================================
@st.cache_resource
def build_vector_store(df):
    texts = df.apply(lambda r: str(r.to_dict()), axis=1).tolist()
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_texts(texts, embeddings)

vectorstore = build_vector_store(df)

# ============================================================
# AGENT STATE
# ============================================================
class AgentState(dict):
    pass

# ============================================================
# 1️⃣ INTENT CLASSIFICATION AGENT
# ============================================================
def intent_agent(state):
    prompt = f"""
    Classify the user intent into:
    recommendation, comparison, explanation, exit

    Query: {state['query']}
    """
    state["intent"] = llm(prompt).strip().lower()
    return state

# ============================================================
# 2️⃣ USER PROFILING AGENT
# ============================================================
def profiling_agent(state):
    state["profile"] = {
        "risk": state["risk"],
        "horizon": state["horizon"],
        "amount": state["amount"]
    }
    return state

# ============================================================
# 3️⃣ RETRIEVAL AGENT (RAG)
# ============================================================
def retrieval_agent(state):
    docs = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = [d.page_content for d in docs]
    return state

# ============================================================
# 4️⃣ RECOMMENDATION AGENT
# ============================================================
def recommendation_agent(state):
    if not state["context"]:
        state["response"] = "This information is not available in the dataset."
        return state

    prompt = f"""
    Using ONLY the following retrieved data:
    {state['context']}

    Recommend suitable mutual funds for this investor profile:
    {state['profile']}
    """
    state["response"] = llm(prompt)
    return state

# ============================================================
# 5️⃣ EXPLANATION AGENT
# ============================================================
def explanation_agent(state):
    prompt = f"""
    Explain why the recommended funds are suitable.
    Base your reasoning only on:
    risk, returns, expense ratio, and investment horizon.

    Data:
    {state['context']}
    """
    state["response"] = llm(prompt)
    return state

# ============================================================
# 6️⃣ CONTINUATION AGENT
# ============================================================
def continuation_agent(state):
    state["continue"] = True
    return state

# ============================================================
# LANGGRAPH ORCHESTRATOR
# ============================================================
graph = StateGraph(AgentState)

graph.add_node("Intent", intent_agent)
graph.add_node("Profile", profiling_agent)
graph.add_node("Retrieve", retrieval_agent)
graph.add_node("Recommend", recommendation_agent)
graph.add_node("Explain", explanation_agent)
graph.add_node("Continue", continuation_agent)

graph.set_entry_point("Intent")

graph.add_conditional_edges(
    "Intent",
    lambda s: s["intent"],
    {
        "recommendation": "Profile",
        "explanation": "Retrieve",
        "exit": END
    }
)

graph.add_edge("Profile", "Retrieve")
graph.add_edge("Retrieve", "Recommend")
graph.add_edge("Recommend", "Continue")
graph.add_edge("Explain", "Continue")
graph.add_edge("Continue", END)

app = graph.compile()

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config("Agentic Mutual Fund AI", layout="wide")
st.title("🤖 Agentic AI – Mutual Fund Recommender")

st.sidebar.header("Investor Profile")
risk = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
amount = st.sidebar.number_input("Investment Amount", min_value=1000)

query = st.text_input("Ask your question")

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
