# =====================================================
# Agentic AI – Mutual Fund Recommendation System
# LangGraph Orchestrated | Dataset-Driven | Streamlit-Safe
# =====================================================

import streamlit as st
import pandas as pd
from typing import TypedDict, List
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
# DECISION VARIABLES (EXPLICIT FOR VIVA)
# -----------------------------------------------------
with st.expander("📊 Decision Variables Used"):
    st.markdown("""
The recommendation system uses the following **dataset variables**:

- **Fund Name**
- **Category**
- **Risk Level**
- **1Y Return (%)**
- **3Y Return (%)**
- **Expense Ratio (%)**

Recommendations are generated using **deterministic scoring**, not LLM reasoning.
""")

# -----------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mutual Funds Data.csv")

df = load_data()

# -----------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------
st.sidebar.header("Investor Profile")

risk_profile = st.sidebar.selectbox(
    "Risk Profile", ["Low", "Medium", "High"]
)

preference = st.sidebar.multiselect(
    "Preferences",
    ["Stability", "Growth", "Tax Saving"],
    default=["Stability"],
    max_selections=1
)
preference = preference[0]

top_k = st.sidebar.slider(
    "Number of recommendations",
    1, 10, 5
)

# -----------------------------------------------------
# LANGGRAPH STATE
# -----------------------------------------------------
class MFState(TypedDict):
    df: pd.DataFrame
    risk: str
    preference: str
    top_k: int

# -----------------------------------------------------
# AGENT 1: RISK FILTER AGENT
# -----------------------------------------------------
def risk_agent(state: MFState):
    df = state["df"]

    if state["risk"] == "Low":
        df = df[df["Risk Level"] <= 2]
    elif state["risk"] == "Medium":
        df = df[df["Risk Level"] <= 3]

    return {**state, "df": df}

# -----------------------------------------------------
# AGENT 2: SCORING AGENT
# -----------------------------------------------------
def scoring_agent(state: MFState):
    df = state["df"].copy()
    df["Score"] = 0.0

    if state["preference"] == "Stability":
        df["Score"] += (5 - df["Risk Level"]) * 2
        df["Score"] += (1 / df["Expense Ratio (%)"]) * 0.5

    elif state["preference"] == "Growth":
        df["Score"] += df["3Y Return (%)"] * 1.5
        df["Score"] += df["1Y Return (%)"]

    elif state["preference"] == "Tax Saving":
        df["Score"] += df["Category"].str.contains(
            "ELSS", case=False, na=False
        ).astype(int) * 10

    return {**state, "df": df}

# -----------------------------------------------------
# AGENT 3: RANKING AGENT
# -----------------------------------------------------
def ranking_agent(state: MFState):
    df = state["df"].sort_values("Score", ascending=False)
    df = df.head(state["top_k"])
    return {**state, "df": df}

# -----------------------------------------------------
# BUILD LANGGRAPH (SAFE MODE)
# -----------------------------------------------------
graph = StateGraph(MFState)

graph.add_node("risk_agent", risk_agent)
graph.add_node("scoring_agent", scoring_agent)
graph.add_node("ranking_agent", ranking_agent)

graph.set_entry_point("risk_agent")
graph.add_edge("risk_agent", "scoring_agent")
graph.add_edge("scoring_agent", "ranking_agent")

app_graph = graph.compile()

# -----------------------------------------------------
# RUN ORCHESTRATOR
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
if final_df.empty:
    st.warning("No funds matched the selected criteria.")
else:
    st.subheader(f"📌 Top {len(final_df)} Recommended Mutual Funds")

    for _, row in final_df.iterrows():
        st.markdown(
            f"""
**{row['Fund Name']}**
- Category: {row['Category']}
- Risk Level: {row['Risk Level']}
- 1Y Return: {row['1Y Return (%)']}%
- 3Y Return: {row['3Y Return (%)']}%
- Expense Ratio: {row['Expense Ratio (%)']}%
---
"""
        )

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.caption(
    "LangGraph is used as an orchestration framework to coordinate autonomous, "
    "deterministic agents operating on structured data."
)
