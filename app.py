# ============================================================
# AGENTIC AI – MUTUAL FUND RECOMMENDER
# Web Scraping + LangGraph + RAG
# Colab Compatible
# ============================================================

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, END
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ============================================================
# LLM CONFIG
# ============================================================
llm = OpenAI(temperature=0)

# ============================================================
# 1️⃣ WEB SCRAPING AGENT (DATA SOURCE)
# ============================================================
def scrape_mutual_funds():
    """
    Scrapes sample mutual fund data from a public webpage.
    (For academic use only)
    """
    url = "https://example.com/mutual-funds"  # placeholder
    funds = []

    # ---- DEMO DATA (replace with real scraping target) ----
    funds.append({
        "Fund Name": "Axis Bluechip Fund",
        "Category": "Equity",
        "Risk": "High",
        "1Y Return": "15%",
        "3Y Return": "18%",
        "Expense Ratio": "0.9%",
        "Fund House": "Axis MF"
    })

    funds.append({
        "Fund Name": "HDFC Balanced Advantage",
        "Category": "Hybrid",
        "Risk": "Medium",
        "1Y Return": "11%",
        "3Y Return": "13%",
        "Expense Ratio": "0.8%",
        "Fund House": "HDFC MF"
    })

    return pd.DataFrame(funds)

# ============================================================
# LOAD + CACHE DATA
# ============================================================
@st.cache_data
def load_data():
    return scrape_mutual_funds()

data = load_data()

# ============================================================
# VECTOR STORE (RAG CORE)
# ============================================================
@st.cache_resource
def create_vector_store(df):
    texts = df.apply(lambda r: str(r.to_dict()), axis=1).tolist()
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_texts(texts, embeddings)

vectorstore = create_vector_store(data)

# ============================================================
# AGENT STATE
# ============================================================
class AgentState(dict):
    pass

# ============================================================
# 2️⃣ INTENT CLASSIFICATION AGENT
# ============================================================
def intent_agent(state):
    prompt = f"""
    Classify intent into:
    recommendation, comparison, explanation, exit

    Query: {state['query']}
    """
    state["intent"] = llm(prompt).strip().lower()
    return state

# ============================================================
# 3️⃣ USER PROFILING AGENT
# ============================================================
def profiling_agent(state):
    state["profile"] = {
        "risk": state["risk"],
        "horizon": state["horizon"],
        "amount": state["amount"]
    }
    return state

# ============================================================
# 4️⃣ RETRIEVAL AGENT (RAG)
# ============================================================
def retrieval_agent(state):
    docs = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = [d.page_content for d in docs]
    return state

# ============================================================
# 5️⃣ RECOMMENDATION AGENT
# ============================================================
def recommendation_agent(state):
    if not state["context"]:
        state["response"] = "This information is not available in the dataset."
        return state

    prompt = f"""
    Using ONLY the following data:
    {state['context']}

    Recommend mutual funds suitable for this profile:
    {state['profile']}
    """
    state["response"] = llm(prompt)
    return state

# ============================================================
# 6️⃣ EXPLANATION AGENT
# ============================================================
def explanation_agent(state):
    prompt = f"""
    Explain why these funds were recommended.
    Use risk-return tradeoff and expense ratio.

    Data:
    {state['context']}
    """
    state["response"] = llm(prompt)
    return state

# ============================================================
# 7️⃣ CONTINUATION AGENT
# ============================================================
def continuation_agent(state):
    state["continue"] = True
    return state

# ============================================================
# ORCHESTRATOR (LANGGRAPH)
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
st.title("🤖 Agentic AI – Mutual Fund Recommender (Web Data)")

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
