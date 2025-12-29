# =========================================================
# AGENTIC AI – MUTUAL FUND RECOMMENDER (LANGGRAPH)
# Web-Scraped | RAG | Groq | Streamlit
# =========================================================

import os
import requests
import bs4
import streamlit as st
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =========================================================
# ENV + LLM
# =========================================================

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-8b-8192",
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# STATE DEFINITION
# =========================================================

class AgentState(TypedDict):
    messages: List[Any]
    intent: str
    user_profile: Dict[str, Any]
    documents: List[Document]
    response: str
    continue_chat: bool

# =========================================================
# AGENT 1 — INTENT CLASSIFIER
# =========================================================

def intent_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify user intent: recommendation, comparison, explanation, market, exit"),
        ("human", "{input}")
    ])

    result = llm.invoke(
        prompt.format(input=state["messages"][-1].content)
    )

    state["intent"] = result.content.lower()
    return state

# =========================================================
# AGENT 2 — USER PROFILING
# =========================================================

def user_profile_agent(state: AgentState):
    state["user_profile"] = {
        "risk": st.session_state.get("risk"),
        "horizon": st.session_state.get("horizon"),
        "preferences": st.session_state.get("preferences")
    }
    return state

# =========================================================
# AGENT 3 — WEB SCRAPER + RAG RETRIEVER
# =========================================================

def scrape_funds():
    url = "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/equity.html"
    html = requests.get(url, timeout=10).text
    soup = bs4.BeautifulSoup(html, "html.parser")

    docs = []
    rows = soup.select("table tbody tr")[:15]

    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) >= 5:
            docs.append(
                Document(
                    page_content=f"Fund {cols[0]}, Category {cols[1]}, 1Y Return {cols[2]}, 3Y Return {cols[3]}, Risk {cols[4]}"
                )
            )
    return docs

def retrieval_agent(state: AgentState):
    try:
        docs = scrape_funds()
        vectordb = Chroma.from_documents(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        query = state["messages"][-1].content
        state["documents"] = retriever.invoke(query)
    except Exception:
        state["documents"] = []
    return state

# =========================================================
# AGENT 4 — RECOMMENDATION
# =========================================================

def recommendation_agent(state: AgentState):
    if not state["documents"]:
        state["response"] = "Relevant mutual fund data is not available from current sources."
        return state

    context = "\n".join(d.page_content for d in state["documents"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Recommend mutual funds strictly using provided data."),
        ("human", "User Profile: {profile}\nData:\n{context}")
    ])

    result = llm.invoke(
        prompt.format(profile=state["user_profile"], context=context)
    )

    state["response"] = result.content
    return state

# =========================================================
# AGENT 5 — EXPLANATION
# =========================================================

def explanation_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain recommendations using risk, horizon, and expense logic."),
        ("human", "{text}")
    ])
    result = llm.invoke(prompt.format(text=state["response"]))
    state["response"] = result.content
    return state

# =========================================================
# AGENT 6 — COMPARISON
# =========================================================

def comparison_agent(state: AgentState):
    context = "\n".join(d.page_content for d in state["documents"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare mutual funds strictly using given data."),
        ("human", context)
    ])
    state["response"] = llm.invoke(prompt).content
    return state

# =========================================================
# AGENT 7 — FOLLOW-UP CONTROLLER
# =========================================================

def continuation_agent(state: AgentState):
    state["continue_chat"] = True
    state["messages"].append(
        AIMessage(content="Would you like another recommendation or market analysis?")
    )
    return state

# =========================================================
# ORCHESTRATOR (LANGGRAPH)
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("intent", intent_agent)
graph.add_node("profile", user_profile_agent)
graph.add_node("retrieve", retrieval_agent)
graph.add_node("recommend", recommendation_agent)
graph.add_node("explain", explanation_agent)
graph.add_node("compare", comparison_agent)
graph.add_node("continue", continuation_agent)

graph.set_entry_point("intent")

graph.add_edge("intent", "profile")
graph.add_edge("profile", "retrieve")

graph.add_conditional_edges(
    "retrieve",
    lambda s: "compare" if "compare" in s["intent"]
    else "recommend"
)

graph.add_edge("recommend", "explain")
graph.add_edge("explain", "continue")
graph.add_edge("compare", "continue")
graph.add_edge("continue", END)

app_graph = graph.compile()

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Agentic Mutual Fund AI", layout="wide")
st.title("🤖 Agentic AI – Mutual Fund Recommender")

with st.sidebar:
    st.session_state["risk"] = st.selectbox("Risk Profile", ["Low", "Medium", "High"])
    st.session_state["horizon"] = st.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
    st.session_state["preferences"] = st.multiselect("Preferences", ["Tax Saving", "Growth", "Stability"])

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask anything about mutual funds...")

if user_input:
    state = {
        "messages": st.session_state.chat + [HumanMessage(content=user_input)],
        "intent": "",
        "user_profile": {},
        "documents": [],
        "response": "",
        "continue_chat": False
    }

    result = app_graph.invoke(state)

    st.session_state.chat.extend([
        HumanMessage(content=user_input),
        AIMessage(content=result["response"])
    ])

for msg in st.session_state.chat:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
