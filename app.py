# =====================================================
# Agentic AI – Mutual Fund Recommendation System
# Robust CSV + LangGraph Orchestration
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
    df = pd.read_csv("Mutual Funds Data.csv")
    df.columns = [c.strip().lower() for c in df.columns]  # normalize
    return df

df = load_data()

# -----------------------------------------------------
# 🔍 AUTO COLUMN DETECTION (CRITICAL)
# -----------------------------------------------------
def find_column(keywords):
    for col in df.columns:
        for kw in keywords:
            if kw in col:
                return col
    return None

COL_FUND = find_column(["scheme", "fund", "name"])
COL_CATEGORY = find_column(["category"])
COL_RISK = find_column(["risk"])
COL_1Y = find_column(["1y", "1 yr", "one year"])
COL_3Y = find_column(["3y", "3 yr", "three year"])
COL_EXPENSE = find_column(["expense"])

missing = [k for k, v in {
    "Fund Name": COL_FUND,
    "Category": COL_CATEGORY,
    "Risk": COL_RISK,
    "1Y Return": COL_1Y,
    "3Y Return": COL_3Y,
    "Expense Ratio": COL_EXPENSE
}.items() if v is None]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
preference = st.sidebar.selectbox("Preference", ["Stability", "Growth", "Tax Saving"])
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
        df = df[df[COL_RISK] <= 2]
    elif state["risk"] == "Medium":
        df = df[df[COL_RISK] <= 3]

    return {**state, "df": df}

# -----------------------------------------------------
# AGENT 2: SCORING
# -----------------------------------------------------
def scoring_agent(state: MFState):
    df = state["df"].copy()

    if state["preference"] == "Stability":
        df["score"] = (5 - df[COL_RISK]) + (1 / df[COL_EXPENSE])

    elif state["preference"] == "Growth":
        df["score"] = df[COL_3Y] * 1.5 + df[COL_1Y]

    elif state["preference"] == "Tax Saving":
        df["score"] = df[COL_CATEGORY].str.contains(
            "elss", case=False, na=False
        ).astype(int) * 10

    return {**state, "df": df}

# -----------------------------------------------------
# AGENT 3: RANKING
# -----------------------------------------------------
def ranking_agent(state: MFState):
    df = state["df"].sort_values("score", ascending=False)
    return {**state, "df": df.head(state["top_k"])}

# -----------------------------------------------------
# BUILD LANGGRAPH (SAFE MODE)
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
**{row[COL_FUND]}**
- Category: {row[COL_CATEGORY]}
- Risk: {row[COL_RISK]}
- 1Y Return: {row[COL_1Y]}%
- 3Y Return: {row[COL_3Y]}%
- Expense Ratio: {row[COL_EXPENSE]}%
---
"""
    )

st.caption("Deterministic agent-based recommendation using LangGraph orchestration.")
