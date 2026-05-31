# pages/1_Government.py
# ---------------------------------------------
# GOVERNMENT PORTAL
# ---------------------------------------------
import sys
print(f"[GOV PAGE LOADED] Python path: {sys.argv}", flush=True)
import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import importlib
import src.agents.government_agent as _gov_module
importlib.reload(_gov_module)

run_government_agent    = _gov_module.run_government_agent
ask_government_question = _gov_module.ask_government_question
load_urban_profiles     = _gov_module.load_urban_profiles
get_investment_priorities = _gov_module.get_investment_priorities
get_chart_data          = _gov_module.get_chart_data
from src.agents.mobility_agent import load_all_neighbourhood_scores

# -- Page config --------------------------------------------------------------

st.set_page_config(page_title="Government Portal", layout="wide")

# -- Chart renderer -----------------------------------------------------------

def render_chart(chart_info: dict):
    if not chart_info or not chart_info.get("data"):
        return
    st.markdown(f"**{chart_info['title']}**")
    if chart_info["chart_type"] == "bar":
        chart_df = pd.DataFrame(
            list(chart_info["data"].items()),
            columns=[chart_info["x_label"], chart_info["y_label"]]
        ).set_index(chart_info["x_label"])
        st.bar_chart(chart_df)
    elif chart_info["chart_type"] == "multi_bar":
        rows = []
        for ward, metrics in chart_info["data"].items():
            for metric, value in metrics.items():
                rows.append({"Ward": ward, "Metric": metric, "Value": value})
        if rows:
            pivot_df = pd.DataFrame(rows).pivot(
                index="Metric", columns="Ward", values="Value"
            )
            st.bar_chart(pivot_df)

# -- Header -------------------------------------------------------------------

st.title("Government Portal")
st.caption("City planner view -- Toronto transit and urban analysis")
st.divider()

# -- Load data ----------------------------------------------------------------

if "gov_data" not in st.session_state:
    if st.button("Run Analysis", use_container_width=False):
        with st.spinner("Running analysis..."):
            try:
                gov_data       = run_government_agent()
                scores         = load_all_neighbourhood_scores()
                urban_profiles = load_urban_profiles()

                st.session_state["gov_data"]       = gov_data
                st.session_state["scores"]         = scores
                st.session_state["urban_profiles"] = urban_profiles
                st.session_state["chat"]           = []
                st.session_state["gov_error"]      = None

                # Generate summary from pure data -- no AI call
                _priorities = gov_data.get("investment_priorities", [])
                _avgs       = gov_data.get("city_averages", {})

                if _priorities:
                    top  = _priorities[0]
                    top2 = _priorities[1] if len(_priorities) > 1 else None

                    summary = (
                        f"Toronto's top investment priority is {top['ward']}, "
                        f"which has {top['parks']} parks "
                        f"(city average: {_avgs.get('avg_parks', 0):.0f}) "
                        f"and {top['bike_lanes']} bike lane segments "
                        f"(city average: {_avgs.get('avg_bike_lanes', 0):.0f}). "
                    )
                    if top2:
                        summary += (
                            f"{top2['ward']} follows with "
                            f"{top2['transit_stops']} transit stops "
                            f"(city average: {_avgs.get('avg_transit_stops', 0):.0f}) "
                            f"and {top2['businesses']} active businesses "
                            f"(city average: {_avgs.get('avg_businesses', 0):.0f}). "
                        )
                    summary += (
                        f"Across {gov_data.get('total_wards', 0)} wards analyzed, "
                        f"bike lane infrastructure and park coverage "
                        f"are the most widespread gaps."
                    )
                    st.session_state["gov_summary"] = summary

            except FileNotFoundError as e:
                st.session_state["gov_error"] = str(e)
        st.rerun()
    else:
        st.info("Click Run Analysis to start.")
        st.stop()

# -- Error --------------------------------------------------------------------

if st.session_state.get("gov_error"):
    st.error(st.session_state["gov_error"])
    st.info(
        "Run this first:\n\n"
        "```\npython src/agents/mobility_agent.py --store\n```"
    )
    st.stop()

# -- Data ---------------------------------------------------------------------

gov_data       = st.session_state["gov_data"]
scores         = st.session_state["scores"]
urban_profiles = st.session_state.get("urban_profiles", {})
priorities     = gov_data.get("investment_priorities", [])
city_avgs      = gov_data.get("city_averages", {})

valid_scores  = {k: v for k, v in scores.items() if "error" not in v}
sorted_scores = sorted(
    valid_scores.items(),
    key=lambda x: x[1].get("connectivity_score", 0),
    reverse=True
)

# -- AI Summary ---------------------------------------------------------------

if "gov_summary" in st.session_state:
    st.subheader("AI Recommendation")
    st.info(st.session_state["gov_summary"])
    st.divider()

# -- Top metrics row ----------------------------------------------------------

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Total Wards",          gov_data.get("total_wards", 0))
with col_b:
    st.metric("Total Neighbourhoods", gov_data.get("total_neighbourhoods", 0))
with col_c:
    st.metric("Avg Transit Stops",    city_avgs.get("avg_transit_stops", 0))
with col_d:
    st.metric("Avg Bike Lanes",       city_avgs.get("avg_bike_lanes", 0))

st.divider()

# -- Tabs ---------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Transit", "Urban Profiles", "All Neighbourhoods"])

# =============================================================================
# TAB 1 -- TRANSIT
# =============================================================================

with tab1:

    st.subheader("Investment Priorities")
    st.caption(
        "Wards ranked by need across transit, parks, "
        "bike lanes and business activity"
    )

    if priorities:
        priority_df = pd.DataFrame([
            {
                "Ward":             p["ward"],
                "Priority Score":   p["investment_score"],
                "Transit Stops":    p["transit_stops"],
                "Parks":            p["parks"],
                "Bike Lanes":       p["bike_lanes"],
                "Businesses":       p["businesses"],
                "Dev Applications": p["dev_applications"],
                "Vibrancy":         p["vibrancy"],
                "Economic":         p["economic"],
                "Reasons":          ", ".join(p["reasons"]),
            }
            for p in priorities
        ])
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Best Connected Neighbourhoods**")
        best_5 = sorted_scores[:5]
        if best_5:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Neighbourhood": k,
                        "Score":         v["connectivity_score"],
                        "Stops":         v["stop_count"],
                        "Subway":        "Yes" if v["has_subway"]    else "No",
                        "Streetcar":     "Yes" if v["has_streetcar"] else "No",
                    }
                    for k, v in best_5
                ]),
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.markdown("**Worst Connected Neighbourhoods**")
        worst_5 = sorted_scores[-5:][::-1]
        if worst_5:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Neighbourhood": k,
                        "Score":         v["connectivity_score"],
                        "Stops":         v["stop_count"],
                        "Subway":        "Yes" if v["has_subway"]    else "No",
                        "Streetcar":     "Yes" if v["has_streetcar"] else "No",
                    }
                    for k, v in worst_5
                ]),
                use_container_width=True,
                hide_index=True,
            )

# =============================================================================
# TAB 2 -- URBAN PROFILES
# =============================================================================

with tab2:

    if not urban_profiles:
        st.warning(
            "No urban profile data found. "
            "Add data/urban_profiles_payload.json to your project."
        )
    else:
        ward_rows = []
        for ward, data in urban_profiles.items():
            ward_rows.append({
                "Ward":             ward,
                "Parks":            data.get("total_parks", 0),
                "Transit Stops":    data.get("total_active_transit_stops", 0),
                "Bike Lanes":       data.get("total_bike_lane_segments", 0),
                "Road Segments":    data.get("total_road_segments", 0),
                "Businesses":       data.get("total_active_businesses", 0),
                "Cultural Spots":   data.get("total_cultural_hotspots", 0),
                "Ice Rinks":        data.get("total_outdoor_ice_rinks", 0),
                "Dev Applications": data.get("total_development_applications", 0),
                "Recreation":       data.get("recreation_deficit_score", ""),
                "Vibrancy":         data.get("community_vibrancy_score", ""),
                "Transit Label":    data.get("transit_connectivity", ""),
                "Economic":         data.get("economic_vitality_score", ""),
            })

        ward_df = pd.DataFrame(ward_rows)

        st.subheader("Ward Comparison")
        st.dataframe(ward_df, use_container_width=True, hide_index=True)

        st.divider()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Parks per Ward**")
            st.bar_chart(
                ward_df[["Ward", "Parks"]]
                .sort_values("Parks", ascending=False)
                .set_index("Ward")
            )
        with col4:
            st.markdown("**Bike Lanes per Ward**")
            st.bar_chart(
                ward_df[["Ward", "Bike Lanes"]]
                .sort_values("Bike Lanes", ascending=False)
                .set_index("Ward")
            )

        st.divider()

        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Transit Stops per Ward**")
            st.bar_chart(
                ward_df[["Ward", "Transit Stops"]]
                .sort_values("Transit Stops", ascending=False)
                .set_index("Ward")
            )
        with col6:
            st.markdown("**Active Businesses per Ward**")
            st.bar_chart(
                ward_df[["Ward", "Businesses"]]
                .sort_values("Businesses", ascending=False)
                .set_index("Ward")
            )

        st.divider()

        st.subheader("Wards Needing Attention")
        col7, col8, col9 = st.columns(3)
        with col7:
            st.markdown("**Lowest Transit Stops**")
            st.dataframe(
                ward_df[["Ward", "Transit Stops"]].nsmallest(5, "Transit Stops"),
                use_container_width=True, hide_index=True,
            )
        with col8:
            st.markdown("**Fewest Parks**")
            st.dataframe(
                ward_df[["Ward", "Parks"]].nsmallest(5, "Parks"),
                use_container_width=True, hide_index=True,
            )
        with col9:
            st.markdown("**Fewest Bike Lanes**")
            st.dataframe(
                ward_df[["Ward", "Bike Lanes"]].nsmallest(5, "Bike Lanes"),
                use_container_width=True, hide_index=True,
            )

# =============================================================================
# TAB 3 -- ALL NEIGHBOURHOODS
# =============================================================================

with tab3:

    st.subheader("All 158 Neighbourhoods -- TTC Scores")
    all_df = pd.DataFrame([
        {
            "Neighbourhood": k,
            "Score":         v["connectivity_score"],
            "Stops":         v["stop_count"],
            "Subway":        "Yes" if v["has_subway"]    else "No",
            "Streetcar":     "Yes" if v["has_streetcar"] else "No",
            "Bus":           "Yes" if v["has_bus"]       else "No",
        }
        for k, v in sorted_scores
    ])
    st.dataframe(all_df, use_container_width=True, hide_index=True)

# =============================================================================
# CHAT
# =============================================================================

st.divider()

# Reset button -- wipes all cached chat history
col_reset, _ = st.columns([1, 8])
with col_reset:
    if st.button("Clear Chat"):
        st.session_state["chat"] = []
        st.rerun()

st.subheader("Ask a Question")
st.caption(
    "Examples: Which wards need the most investment? "
    "How is transit around Etobicoke? "
    "I am planning to build in Willowdale, what should I know?"
)

if "chat" not in st.session_state:
    st.session_state["chat"] = []

# Render chat history
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask about Toronto city planning...")

if user_input:
    st.session_state["chat"].append({
        "role":    "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            answer = ask_government_question(
                user_input,
                chat_history=[
                    m for m in st.session_state["chat"][:-1]
                    if m["role"] == "user"
                ],
            )

            # Debug -- prints to terminal to verify answer source
            print(f"\n[Streamlit Chat] Answer:\n{answer}\n")

            

            st.write(answer)
            

    st.session_state["chat"].append({
        "role":    "assistant",
        "content": answer
    })
