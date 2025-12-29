# =========================================================
# Agentic AI – Mutual Fund Recommender (FINAL FIXED VERSION)
# =========================================================

import streamlit as st
import requests
import bs4
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Agentic AI – Mutual Fund Recommender", layout="wide")
st.title("🤖 Agentic AI – Mutual Fund Recommender")

# ------------------ LLM (NO STREAMING) ------------------
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-8b-8192",
    temperature=0.2,
    streaming=False
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ STATE ------------------
class AgentState(TypedDict):
    messages: List[Any]
    intent: str
    user_profile: Dict[str, Any]
    documents: List[Document]
    response: str

# ------------------ AGENTS ------------------

def intent_agent(state: AgentState):
    if not state.get("messages"):
        state["intent"] = "recommendation"
        return state

    last_msg = state["messages"][-1].content.strip()
    if not last_msg:
        state["intent"] = "recommendation"
        return state

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return ONLY one word: recommendation, comparison, explanation, market, exit"),
        ("human", last_msg)
    ])

    result = llm.invoke(prompt.format_messages())
    state["intent"] = result.content.strip().lower()
    return state


def user_profile_agent(state: AgentState):
    state["user_profile"] = {
        "risk": st.session_state.get("risk"),
        "horizon": st.session_state.get("horizon"),
        "preferences": st.session_state.get("preferences"),
    }
    return state


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
                    page_content=f"Fund {cols[0]}, Category {cols[1]}, "
                                 f"1Y {cols[2]}, 3Y {cols[3]}, Risk {cols[4]}"
                )
            )
    return docs


def retrieval_agent(state: AgentState):
    try:
        docs = scrape_funds()
        if not docs:
            state["documents"] = []
            return state

        vectordb = Chroma.from_documents(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        state["documents"] = retriever.invoke(state["messages"][-1].content)
    except Exception:
        state["documents"] = []
    return state


def recommendation_agent(state: AgentState):
    if not state.get("documents"):
        state["response"] = "No sufficient mutual fund data available from public sources."
        return state

    context = "\n".join(d.page_content for d in state["documents"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Recommend mutual funds strictly using the provided data."),
        ("human", f"User Profile: {state['user_profile']}\n\nData:\n{context}")
    ])

    result = llm.invoke(prompt.format_messages())
    state["response"] = result.content
    return state


def explanation_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain the recommendation clearly and safely."),
        ("human", state["response"])
    ])

    result = llm.invoke(prompt.format_messages())
    state["response"] = result.content
    return state


def comparison_agent(state: AgentState):
    if not state.get("documents"):
        state["response"] = "No sufficient data available for comparison."
        return state

    context = "\n".join(d.page_content for d in state["documents"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare mutual funds using only the given data."),
        ("human", context)
    ])

    result = llm.invoke(prompt.format_messages())
    state["response"] = result.content
    return state

# ------------------ LANGGRAPH ------------------
graph = StateGraph(AgentState)

graph.add_node("intent", intent_agent)
graph.add_node("profile", user_profile_agent)
graph.add_node("retrieve", retrieval_agent)
graph.add_node("recommend", recommendation_agent)
graph.add_node("explain", explanation_agent)
graph.add_node("compare", comparison_agent)

graph.set_entry_point("intent")
graph.add_edge("intent", "profile")
graph.add_edge("profile", "retrieve")

graph.add_conditional_edges(
    "retrieve",
    lambda s: "compare" if "compare" in s["intent"] else "recommend"
)

graph.add_edge("recommend", "explain")
graph.add_edge("explain", END)
graph.add_edge("compare", END)

app_graph = graph.compile()

# ------------------ STREAMLIT UI ------------------
with st.sidebar:
    st.session_state["risk"] = st.selectbox("Risk Profile", ["Low", "Medium", "High"])
    st.session_state["horizon"] = st.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
    st.session_state["preferences"] = st.multiselect(
        "Preferences", ["Growth", "Stability", "Tax Saving"]
    )

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask anything about mutual funds...")

if user_input and user_input.strip():
    state = {
        "messages": st.session_state.chat + [HumanMessage(content=user_input)],
        "intent": "",
        "user_profile": {},
        "documents": [],
        "response": ""
    }

    result = app_graph.invoke(state)

    st.session_state.chat.extend([
        HumanMessage(content=user_input),
        AIMessage(content=result["response"])
    ])

for msg in st.session_state.chat:
    st.chat_message("user").write(msg.content)
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
