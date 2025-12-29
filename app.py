# =====================================================
# Agentic AI – Mutual Fund Recommendation System
# LangGraph Orchestrated | CSV Driven | Streamlit Safe
# =====================================================

import streamlit as st
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Recommendation System",
    layout="wide"
)

st.title("🤖 Agentic AI – Mutual Fund Recommendation System")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mutual Funds Data.csv")

df = load_data()

# -----------------------------------------------------
# 🔴 COLUMN MAPPING (CRITICAL FIX)
# -----------------------------------------------------
COLUMN_MAP = {
    "fund": "Scheme Name",
    "category": "Category",
    "risk": "Risk",
    "ret_1y": "1Y Return (%)",
    "ret_3y": "3Y Return (%)",
    "expense": "Expense Ratio (%)",
}

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])

preference = st.sidebar.selectbox(
    "Preference", ["Stability", "Growth", "Tax Saving"]
)

top_k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# -----------------------------------------------------
# LANGGRAPH STATE
# -----------------------------------------------------
class MFState(TypedDict):
    df: pd.DataFrame
    risk: str
    preference: str
    top_k: int

# -----------------------------------------------------
# AGENT 1: RISK FILTER
# -----------------------------------------------------
def risk_agent(state: MFState):
    df = state["df"]

    if state["risk"] == "Low":
        df = df[df[COLUMN_MAP["risk"]] <= 2]
    elif state["risk"] == "Medium":
        df = df[df[COLUMN_MAP["risk"]] <= 3]

    return {**state, "df": df}

# -----------------------------------------------------
# AGENT 2: SCORING
# -----------------------------------------------------
def scoring_agent(state: MFState):
    df = state["df"].copy()
    df["Score"] = 0.0

    if state["preference"] == "Stability":
        df["Score"] = (
            (5 - df[COLUMN_MAP["risk"]]) * 2
            + (1 / df[COLUMN_MAP["expense"]])
        )

    elif state["preference"] == "Growth":
        df["Score"] = (
            df[COLUMN_MAP["ret_3y"]] * 1.5
            + df[COLUMN_MAP["ret_1y"]]
        )

    elif state["preference"] == "Tax Saving":
        df["Score"] = df[COLUMN_MAP["category"]].str.contains(
            "ELSS", case=False, na=False
        ).astype(int) * 10

    return {**state, "df": df}

# -----------------------------------------------------
# AGENT 3: RANKING
# -----------------------------------------------------
def ranking_agent(state: MFState):
    df = state["df"].sort_values("Score", ascending=False)
    return {**state, "df": df.head(state["top_k"])}

# -----------------------------------------------------
# BUILD LANGGRAPH (NO PREGEL, NO STREAM)
# -----------------------------------------------------
graph = StateGraph(MFState)
graph.add_node("risk", risk_agent)
graph.add_node("score", scoring_agent)
graph.add_node("rank", ranking_agent)

graph.set_entry_point("risk")
graph.add_edge("risk", "score")
graph.add_edge("score", "rank")

app_graph = graph.compile()

# -----------------------------------------------------
# RUN GRAPH
# -----------------------------------------------------
result = app_graph.invoke({
    "df": df,
    "risk": risk_profile,
    "preference": preference,
    "top_k": top_k
})

final_df = result["df"]

# -----------------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------------
st.subheader(f"📌 Top {len(final_df)} Recommended Mutual Funds")

for _, row in final_df.iterrows():
    st.markdown(
        f"""
**{row[COLUMN_MAP["fund"]]}**
- Category: {row[COLUMN_MAP["category"]]}
- Risk: {row[COLUMN_MAP["risk"]]}
- 1Y Return: {row[COLUMN_MAP["ret_1y"]]}%
- 3Y Return: {row[COLUMN_MAP["ret_3y"]]}%
- Expense Ratio: {row[COLUMN_MAP["expense"]]}%
---
"""
    )

st.caption("Deterministic agent-based recommendation using LangGraph orchestration.")
