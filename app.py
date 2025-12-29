# =====================================================
# Agentic AI – Mutual Fund Recommendation System
# (Streamlit-safe, Recommendation-grade)
# =====================================================

import streamlit as st
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
    page_title="Agentic AI – Mutual Fund Recommendation System",
    layout="wide"
)
st.title("🤖 Agentic AI – Mutual Fund Recommendation System")

# -----------------------------------------------------
# LLM (GROQ – SUPPORTED MODEL)
# -----------------------------------------------------
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=0.1,
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
# AGENT 3: STATIC MUTUAL FUND DATA (OPTION 1 FIX)
# -----------------------------------------------------
def load_mutual_fund_data() -> List[Document]:
    funds = [
        "Fund Name: SBI Magnum Low Duration Fund, Category: Debt, 1Y Return: 7%, 3Y Return: 6.8%, Risk Level: Low",
        "Fund Name: HDFC Balanced Advantage Fund, Category: Hybrid, 1Y Return: 12%, 3Y Return: 14%, Risk Level: Moderate",
        "Fund Name: ICICI Prudential Bluechip Fund, Category: Equity, 1Y Return: 15%, 3Y Return: 16%, Risk Level: Moderate",
        "Fund Name: Axis Long Term Equity Fund, Category: ELSS, 1Y Return: 18%, 3Y Return: 20%, Risk Level: High",
        "Fund Name: Kotak Equity Opportunities Fund, Category: Equity, 1Y Return: 16%, 3Y Return: 17%, Risk Level: Moderate"
    ]

    return [Document(page_content=fund) for fund in funds]

# -----------------------------------------------------
# AGENT 4: RETRIEVAL (RAG)
# -----------------------------------------------------
def retrieval_agent(query: str) -> List[Document]:
    docs = load_mutual_fund_data()
    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return retriever.invoke(query)

# -----------------------------------------------------
# AGENT 5: RECOMMENDATION (STRICT & RANKED)
# -----------------------------------------------------
def recommendation_agent(profile: Dict, docs: List[Document]) -> str:
    context = "\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a mutual fund recommendation engine.\n"
            "Rules:\n"
            "1. Select ONLY fund names present in the data\n"
            "2. Rank funds from best to worst\n"
            "3. Match the user's risk profile and preferences\n"
            "4. Give 1 short reason per fund\n"
            "5. NO generic advice"
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

Return output strictly in this format:
1. Fund Name – Reason
2. Fund Name – Reason
"""
        )
    ])

    return llm.invoke(prompt.format_messages()).content

# -----------------------------------------------------
# AGENT 6: COMPARISON
# -----------------------------------------------------
def comparison_agent(docs: List[Document]) -> str:
    context = "\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the mutual funds using only the given data."),
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

    if intent == "exit":
        return "Thank you for using the Mutual Fund Recommendation System."

    return recommendation_agent(profile, docs)

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
