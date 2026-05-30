# pages/1_Government.py
# ---------------------------------------------
# GOVERNMENT PORTAL
# ---------------------------------------------

import streamlit as st
import json
import os
import sys
import pandas as pd
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.agents.mobility_agent   import run_mobility_agent, load_all_neighbourhood_scores
from src.services.openai_service import ask_ai, ask_ai_with_history

# -- Page config --------------------------------------------------------------

st.set_page_config(page_title="Government Portal", layout="wide")

# -- Helper -------------------------------------------------------------------

def clean_ai_response(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    skip_phrases = [
        "let me analyze", "looking at the data", "i need to find",
        "the data provided", "i don't see", "the user is asking",
        "the neighborhood data", "i should", "let me check",
        "based on the data", "looking at", "the question",
        "so the answer", "to answer this", "first i",
        "the context", "the user", "i need to",
    ]
    lines  = text.strip().split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(phrase in line.lower() for phrase in skip_phrases):
            continue
        result.append(line)
    return " ".join(result).strip()

# -- Header -------------------------------------------------------------------

st.title("Government Portal")
st.caption("City planner view -- Toronto transit analysis")
st.divider()

# -- Load data ----------------------------------------------------------------

if "mobility" not in st.session_state:
    if st.button("Run Analysis", use_container_width=False):
        with st.spinner("Loading transit data..."):
            try:
                mobility = run_mobility_agent()
                scores   = load_all_neighbourhood_scores()
                st.session_state["mobility"]  = mobility
                st.session_state["scores"]    = scores
                st.session_state["chat"]      = []
                st.session_state["gov_error"] = None

                prompt = f"""
Toronto transit data:
Zone breakdown: {json.dumps(mobility.get('zone_breakdown', {}))}
Transit deserts: {json.dumps(mobility.get('transit_deserts', []))}
Worst connected: {json.dumps(mobility.get('neighbourhood_rankings', {}).get('worst_connected', []))}

In 3-4 sentences only, what should the city prioritize for transit investment?
Answer directly. No reasoning.
"""
                raw = ask_ai(
                    "You are a Toronto city planning advisor. "
                    "Answer in 3-4 sentences only. No reasoning. No bullet points.",
                    prompt,
                    max_tokens=300,
                )
                st.session_state["gov_summary"] = clean_ai_response(raw)

            except FileNotFoundError as e:
                st.session_state["gov_error"] = str(e)
        st.rerun()
    else:
        st.info("Click Run Analysis to start.")
        st.stop()

# -- Error --------------------------------------------------------------------

if st.session_state.get("gov_error"):
    st.error(st.session_state["gov_error"])
    st.info("Run first:\n\n```\npython src/agents/mobility_agent.py --store\n```")
    st.stop()

# -- Data ---------------------------------------------------------------------

mobility = st.session_state["mobility"]
scores   = st.session_state["scores"]
rankings = mobility.get("neighbourhood_rankings", {})

# AI summary
if "gov_summary" in st.session_state:
    st.subheader("AI Recommendation")
    st.info(st.session_state["gov_summary"])
    st.divider()

# 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("TTC Network")
    rs = mobility.get("route_summary", {})
    st.metric("Total Routes", rs.get("total_routes", 0))
    st.metric("Total Stops",  rs.get("total_stops",  0))
    st.write(f"Bus:       {rs.get('bus_routes', 0)} routes")
    st.write(f"Subway:    {rs.get('subway_routes', 0)} routes")
    st.write(f"Streetcar: {rs.get('streetcar_routes', 0)} routes")

with col2:
    st.subheader("Transit Deserts")
    deserts = mobility.get("transit_deserts", [])
    if deserts:
        for d in deserts:
            severity = d.get("severity", "Unknown")
            area     = d.get("area",     "Unknown")
            stops    = d.get("stop_count", 0)
            color    = (
                "red"    if severity == "Critical" else
                "orange" if severity == "High"     else
                "blue"
            )
            st.markdown(f":{color}[{severity}] **{area}** -- {stops} stops")
    else:
        st.success("No critical transit deserts found")

with col3:
    st.subheader("Recommendations")
    for rec in mobility.get("recommendations", []):
        st.write(f"- {rec}")

st.divider()

# Best and worst
col4, col5 = st.columns(2)

with col4:
    st.markdown("**Best Connected**")
    best = rankings.get("best_connected", [])
    if best:
        st.dataframe(
            pd.DataFrame(best)[["neighbourhood", "score", "rating", "stops"]],
            use_container_width=True,
            hide_index=True,
        )

with col5:
    st.markdown("**Worst Connected**")
    worst = rankings.get("worst_connected", [])
    if worst:
        st.dataframe(
            pd.DataFrame(worst)[["neighbourhood", "score", "rating", "stops"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# Zone chart
st.subheader("TTC Stops by Zone")
zone_data = mobility.get("zone_breakdown", {})
if zone_data:
    zone_df = pd.DataFrame(
        list(zone_data.items()),
        columns=["Zone", "Stops"]
    ).sort_values("Stops", ascending=False)
    st.bar_chart(zone_df.set_index("Zone"))

st.divider()

# All neighbourhoods
st.subheader("All 158 Neighbourhoods")
valid_scores  = {k: v for k, v in scores.items() if "error" not in v}
sorted_scores = sorted(
    valid_scores.items(),
    key=lambda x: x[1].get("connectivity_score", 0),
    reverse=True
)
all_df = pd.DataFrame([
    {
        "Neighbourhood": k,
        "Score":         v["connectivity_score"],
        "Rating":        v["rating"],
        "Stops":         v["stop_count"],
        "Subway":        "Yes" if v["has_subway"]    else "No",
        "Streetcar":     "Yes" if v["has_streetcar"] else "No",
        "Bus":           "Yes" if v["has_bus"]       else "No",
    }
    for k, v in sorted_scores
])
st.dataframe(all_df, use_container_width=True, hide_index=True)

st.divider()

# -- Chat ---------------------------------------------------------------------

st.subheader("Ask a Question")
st.caption(
    "Ask anything about Toronto transit. "
    "Example: How is transit in Mimico? "
    "Which area needs the most investment?"
)

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about Toronto transit...")

if user_input:
    st.session_state["chat"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Find matching neighbourhoods from the question
    question_lower = user_input.lower()
    mentioned_nb   = {}
    for name, data in scores.items():
        if any(word in name.lower() for word in question_lower.split()
               if len(word) > 3):
            mentioned_nb[name] = {
                "stop_count":         data.get("stop_count"),
                "connectivity_score": data.get("connectivity_score"),
                "rating":             data.get("rating"),
                "has_subway":         data.get("has_subway"),
                "has_streetcar":      data.get("has_streetcar"),
                "has_bus":            data.get("has_bus"),
                "sample_stops":       data.get("sample_stops", [])[:5],
            }

    # Build context
    context = f"""
Zone breakdown: {json.dumps(mobility.get('zone_breakdown', {}))}
Transit deserts: {json.dumps(mobility.get('transit_deserts', []))}
Route summary: {json.dumps(mobility.get('route_summary', {}))}
Best connected: {json.dumps(rankings.get('best_connected', []))}
Worst connected: {json.dumps(rankings.get('worst_connected', []))}

Neighbourhoods matching the question:
{json.dumps(mentioned_nb, indent=2) if mentioned_nb else "No exact match -- use all scores below"}

All 158 neighbourhood scores:
{json.dumps({k: {"score": v.get("connectivity_score"), "stops": v.get("stop_count"), "rating": v.get("rating"), "sample_stops": v.get("sample_stops", [])[:3]} for k, v in scores.items()})}
"""

    messages = [
        {
            "role":    "system",
            "content": (
                "You are a Toronto city planning advisor. "
                "You have TTC data for all 158 Toronto neighbourhoods. "
                "When asked about an area search for partial name matches. "
                "Lakeshore matches Humber Bay Shores or New Toronto. "
                "Mimico matches Mimico-Queensway. "
                "Always give specific numbers. "
                "Answer directly. No reasoning out loud."
            ),
        },
        {
            "role":    "user",
            "content": f"Data:\n{context}\n\nQuestion: {user_input}",
        },
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            raw    = ask_ai_with_history(messages, max_tokens=300)
            answer = clean_ai_response(raw)
            st.write(answer)

    st.session_state["chat"].append({"role": "assistant", "content": answer})
