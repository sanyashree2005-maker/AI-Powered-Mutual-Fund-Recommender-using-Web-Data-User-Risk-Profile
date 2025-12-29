# =====================================================
# Agentic AI – Mutual Fund Recommendation System
# (Streamlit-safe, Recommender-grade)
# =====================================================

import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Recommender",
    layout="wide"
)
st.title("🤖 Agentic AI – Mutual Fund Recommendation System")

# -----------------------------------------------------
# LLM (GROQ – SUPPORTED MODEL)
# -----------------------------------------------------
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=0.1,   # 🔑 low temperature = less generic output
    streaming=False
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------------------------
# AGENT 1: INTENT CLASSIFICATION
# -----------------------------------------------------
def intent_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Classify the intent. Return ONLY one word:\n"
            "recommendation, explanation, comparison, market, exit"
        ),
        ("human", query)
    ])
    return llm.invoke(prompt.format_messages()).content.strip().lower()

# -----------------------------------------------------
# AGENT 2: USER PROFILING
# -----------------------------------------------------
def user_profile_agent() -> Dict:
    return {
        "risk": st.session_state["risk"],
        "horizon": st.session_state["horizon"],
        "preferences": st.session_state["preferences"]
    }

# -----------------------------------------------------
# AGENT 3: WEB DATA SCRAPING
# -----------------------------------------------------
def scrape_mutual_funds() -> List[Document]:
    url = "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/equity.html"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    docs = []
    rows = soup.select("table tbody tr")[:15]

    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) >= 5:
            docs.append(
                Document(
                    page_content=(
                        f"Fund Name: {cols[0]}, "
                        f"Category: {cols[1]}, "
                        f"1Y Return: {cols[2]}, "
                        f"3Y Return: {cols[3]}, "
                        f"Risk Level: {cols[4]}"
                    )
                )
            )
    return docs

# -----------------------------------------------------
# AGENT 4: RETRIEVAL (RAG)
# -----------------------------------------------------
def retrieval_agent(query: str) -> List[Document]:
    docs = scrape_mutual_funds()
    if not docs:
        return []

    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return retriever.invoke(query)

# -----------------------------------------------------
# AGENT 5: RECOMMENDATION (STRICT & RANKED)
# -----------------------------------------------------
def recommendation_agent(profile: Dict, docs: List[Document]) -> str:
    if not docs:
        return "Mutual fund data is not available from current public sources."

    context = "\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a mutual fund recommendation engine.\n"
            "Rules:\n"
            "1. Select ONLY fund names present in the data\n"
            "2. Rank the funds from best to worst\n"
            "3. Give 1–2 concrete reasons per fund\n"
            "4. NO generic financial advice\n"
            "5. If data is insufficient, clearly say so"
        ),
        (
            "human",
            f"""
User Profile:
Risk: {profile['risk']}
Investment Horizon: {profile['horizon']}
Preferences: {profile['preferences']}

Available Mutual Fund Data:
{context}

Return the answer strictly in this format:
1. Fund Name – Reason
2. Fund Name – Reason
"""
        )
    ])

    return llm.invoke(prompt.format_messages()).content

# -----------------------------------------------------
# AGENT 6: EXPLANATION
# -----------------------------------------------------
def explanation_agent(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain the recommendation clearly using the given context."),
        ("human", text)
    ])
    return llm.invoke(prompt.format_messages()).content

# -----------------------------------------------------
# AGENT 7: COMPARISON
# -----------------------------------------------------
def comparison_agent(docs: List[Document]) -> str:
    if not docs:
        return "Not enough mutual fund data to perform comparison."

    context = "\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the mutual funds strictly using the given data."),
        ("human", context)
    ])
    return llm.invoke(prompt.format_messages()).content

# -----------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------
def orchestrator(query: str) -> str:
    if not query or not query.strip():
        return "Please enter a valid mutual fund related question."

    intent = intent_agent(query)
    profile = user_profile_agent()
    docs = retrieval_agent(query)

    if intent == "comparison":
        return comparison_agent(docs)

    if intent == "explanation":
        return explanation_agent(query)

    if intent == "exit":
        return "Thank you for using the Mutual Fund Recommendation System."

    recommendation = recommendation_agent(profile, docs)
    return recommendation

# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------
with st.sidebar:
    st.session_state["risk"] = st.selectbox("Risk Profile", ["Low", "Medium", "High"])
    st.session_state["horizon"] = st.selectbox(
        "Investment Horizon", ["Short", "Medium", "Long"]
    )
    st.session_state["preferences"] = st.multiselect(
        "Preferences", ["Growth", "Stability", "Tax Saving"]
    )

if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

user_input = st.chat_input("Ask anything about mutual funds...")

if user_input and user_input.strip():
    if st.session_state.last_query != user_input:
        st.session_state.last_query = user_input
        response = orchestrator(user_input)
        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("assistant", response))

for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)
