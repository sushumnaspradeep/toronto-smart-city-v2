# pages/2_Constructor.py
# ---------------------------------------------
# CONSTRUCTOR PORTAL
# ---------------------------------------------

import streamlit as st
import os
import sys
import pandas as pd
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import importlib
import src.agents.constructor_agent as _con_module
importlib.reload(_con_module)

get_construction_profile  = _con_module.get_construction_profile
ask_constructor_question  = _con_module.ask_constructor_question
get_area_permit_stats     = _con_module.get_area_permit_stats
load_permits_summary      = _con_module.load_permits_summary

# -- Page config --------------------------------------------------------------

st.set_page_config(page_title="Constructor Portal", layout="wide")

# -- Header -------------------------------------------------------------------

st.title("Constructor Portal")
st.caption("Site assessment tool for developers and constructors in Toronto")
st.divider()

# -- Search -------------------------------------------------------------------

col_search, col_btn = st.columns([4, 1])
with col_search:
    area_input = st.text_input(
        "Enter a neighbourhood or area",
        placeholder="e.g. Mimico, Willowdale, Scarborough, Etobicoke",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("Assess Site", use_container_width=True)

# -- Run assessment -----------------------------------------------------------

if search_btn and area_input:
    with st.spinner(f"Assessing {area_input}..."):
        profile = get_construction_profile(area_input)
        st.session_state["con_profile"]  = profile
        st.session_state["con_area"]     = area_input
        st.session_state["con_chat"]     = []

elif not search_btn and "con_profile" not in st.session_state:
    st.info("Enter a neighbourhood name above to get a site assessment.")
    st.stop()

if "con_profile" not in st.session_state:
    st.stop()

# -- Load profile -------------------------------------------------------------

profile  = st.session_state["con_profile"]
area     = st.session_state["con_area"]
avgs     = profile.get("city_averages", {
    "transit": 360.7, "parks": 71.6,
    "bike_lanes": 61.4, "businesses": 5465.4,
})
ward     = profile.get("ward")
ttc      = profile.get("ttc")
permits  = profile.get("permits", {})
risk     = profile.get("investment_risk", "Unknown")

# -- Risk badge ---------------------------------------------------------------

st.subheader(f"Site Assessment: {area}")

risk_color = {
    "Low":     "green",
    "Medium":  "orange",
    "High":    "red",
}.get(risk, "blue")

st.markdown(
    f"**Investment Risk:** :{risk_color}[{risk}]  "
    f"{'✅' if risk == 'Low' else '⚠️' if risk == 'Medium' else '🔴'}"
)
if profile.get("investment_note"):
    st.caption(profile["investment_note"])

st.divider()

# -- Metrics row --------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    stops = ttc["stops"] if ttc else 0
    delta = stops - avgs["transit"]
    st.metric(
        "TTC Stops",
        stops,
        delta=f"{delta:+.0f} vs avg",
        delta_color="normal",
    )

with col2:
    parks = ward["parks"] if ward else 0
    delta = parks - avgs["parks"]
    st.metric(
        "Parks",
        parks,
        delta=f"{delta:+.0f} vs avg",
        delta_color="normal",
    )

with col3:
    bikes = ward["bike_lanes"] if ward else 0
    delta = bikes - avgs["bike_lanes"]
    st.metric(
        "Bike Lanes",
        bikes,
        delta=f"{delta:+.0f} vs avg",
        delta_color="normal",
    )

with col4:
    biz   = ward["businesses"] if ward else 0
    delta = biz - avgs["businesses"]
    st.metric(
        "Businesses",
        f"{biz:,}",
        delta=f"{delta:+.0f} vs avg",
        delta_color="normal",
    )

with col5:
    st.metric(
        "Nearby Permits",
        permits.get("total_nearby", 0),
        delta="within 2km",
        delta_color="off",
    )

st.divider()

# -- Tabs ---------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Transit & Roads", "Ward Profile", "Permits"])

# =============================================================================
# TAB 1 -- TRANSIT & ROADS
# =============================================================================

with tab1:

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("TTC Access")
        if ttc:
            st.write(f"**Neighbourhood:** {ttc['neighbourhood']}")
            st.write(f"**Total Stops:** {ttc['stops']}")
            st.write(f"**Subway:** {'✅ Yes' if ttc['has_subway'] else '❌ No'}")
            st.write(f"**Streetcar:** {'✅ Yes' if ttc['has_streetcar'] else '❌ No'}")
            st.write(f"**Bus:** {'✅ Yes' if ttc['has_bus'] else '❌ No'}")

            if ttc.get("sample_stops"):
                st.markdown("**Key Stops:**")
                for stop in ttc["sample_stops"]:
                    st.write(f"- {stop}")

            all_matches = profile.get("all_ttc_matches", [])
            if len(all_matches) > 1:
                st.divider()
                st.markdown("**Other Nearby Neighbourhoods:**")
                for m in all_matches:
                    if m["neighbourhood"] != ttc["neighbourhood"]:
                        st.write(
                            f"- {m['neighbourhood']}: "
                            f"{m['stops']} stops"
                        )
        else:
            st.warning("No TTC data found for this area.")

    with col_b:
        st.subheader("Road Access & Permit Times")
        if ward:
            st.write(f"**Ward:** {ward['name']}")
            st.write(f"**Road Segments:** {ward['road_segments']:,}")
            city_roads = 2565
            road_delta = ward["road_segments"] - city_roads
            st.write(
                f"**vs City Average:** {road_delta:+,} "
                f"({'above' if road_delta >= 0 else 'below'} avg {city_roads:,})"
            )
            st.write(f"**Ice Rinks:** {ward.get('ice_rinks', 0)}")
        else:
            st.warning("No ward data found for this area.")

        # Approval times
        approval_times = profile.get("approval_times", [])
        if approval_times:
            st.divider()
            st.markdown("**Permit Approval Times**")
            approval_df = pd.DataFrame(approval_times)[
                ["fsa", "median_days", "mean_days", "p90_days", "permit_count"]
            ]
            approval_df.columns = [
                "FSA", "Median Days", "Mean Days", "90th Pct Days", "Permits"
            ]
            st.dataframe(approval_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                f"No approval time data found for this area. "
                f"FSA codes: {', '.join(profile.get('area_fsas', []))}"
            )

# =============================================================================
# TAB 2 -- WARD PROFILE
# =============================================================================

with tab2:

    if ward:
        st.subheader(f"{ward['name']} Ward Profile")

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("**Infrastructure**")
            data_rows = [
                {"Metric": "Transit Stops",   "Value": ward["transit_stops"],  "City Avg": avgs["transit"],    "Status": "✅" if ward["transit_stops"]  >= avgs["transit"]    else "⚠️"},
                {"Metric": "Parks",            "Value": ward["parks"],          "City Avg": avgs["parks"],      "Status": "✅" if ward["parks"]          >= avgs["parks"]      else "⚠️"},
                {"Metric": "Bike Lanes",       "Value": ward["bike_lanes"],     "City Avg": avgs["bike_lanes"], "Status": "✅" if ward["bike_lanes"]     >= avgs["bike_lanes"] else "⚠️"},
                {"Metric": "Road Segments",    "Value": ward["road_segments"],  "City Avg": 2565,               "Status": "✅" if ward["road_segments"]  >= 2565               else "⚠️"},
            ]
            st.dataframe(
                pd.DataFrame(data_rows),
                use_container_width=True,
                hide_index=True,
            )

        with col_d:
            st.markdown("**Community**")
            community_rows = [
                {"Metric": "Active Businesses", "Value": f"{ward['businesses']:,}", "City Avg": f"{avgs['businesses']:,.0f}", "Status": "✅" if ward["businesses"] >= avgs["businesses"] else "⚠️"},
                {"Metric": "Cultural Hotspots", "Value": ward["cultural_spots"],    "City Avg": "N/A",                         "Status": "ℹ️"},
                {"Metric": "Ice Rinks",         "Value": ward.get("ice_rinks", 0), "City Avg": "N/A",                         "Status": "ℹ️"},
                {"Metric": "Dev Applications",  "Value": ward["dev_apps"],          "City Avg": "N/A",                         "Status": "✅" if ward["dev_apps"] == 0 else "ℹ️"},
            ]
            st.dataframe(
                pd.DataFrame(community_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # Bar chart comparing ward to city averages
        st.markdown("**Ward vs City Averages**")
        chart_data = pd.DataFrame({
            "Metric":   ["Transit Stops", "Parks", "Bike Lanes"],
            ward["name"]: [
                ward["transit_stops"],
                ward["parks"],
                ward["bike_lanes"],
            ],
            "City Average": [
                avgs["transit"],
                avgs["parks"],
                avgs["bike_lanes"],
            ],
        }).set_index("Metric")
        st.bar_chart(chart_data)

    else:
        st.warning("No ward profile data found for this area.")

# =============================================================================
# TAB 3 -- PERMITS
# =============================================================================

with tab3:

    st.subheader("Nearby Permit Activity")
    st.caption(f"Active permits within 2km of {area}")

    nearby_list = permits.get("nearby_list", [])

    if nearby_list:
        permits_df = pd.DataFrame(nearby_list)[
            ["permit_id", "address", "fsa", "distance_km"]
        ]
        permits_df.columns = ["Permit ID", "Address", "FSA", "Distance (km)"]
        st.dataframe(permits_df, use_container_width=True, hide_index=True)
    else:
        st.info("No permits found within 2km of this area.")

    st.divider()

    # City-wide permit summary
    st.subheader("City-Wide Permit Summary")
    summary = load_permits_summary()

    if summary:
        col_e, col_f = st.columns(2)

        with col_e:
            st.markdown("**Permit Types**")
            p_types = summary.get("permit_type_counts", {})
            if p_types:
                types_df = pd.DataFrame(
                    sorted(p_types.items(), key=lambda x: x[1], reverse=True)[:10],
                    columns=["Type", "Count"]
                )
                st.dataframe(types_df, use_container_width=True, hide_index=True)

        with col_f:
            st.markdown("**Status Breakdown**")
            status = summary.get("status_breakdown", {})
            if status:
                status_df = pd.DataFrame(
                    sorted(status.items(), key=lambda x: x[1], reverse=True)[:10],
                    columns=["Status", "Count"]
                )
                st.dataframe(status_df, use_container_width=True, hide_index=True)

# =============================================================================
# AI SUMMARY
# =============================================================================

st.divider()
st.subheader("AI Site Assessment")

if f"ai_summary_{area}" not in st.session_state:
    with st.spinner("Generating assessment..."):
        summary_q = (
            f"I am planning to build in {area}. "
            f"Give me a complete site assessment covering transit, "
            f"parks, bike infrastructure, businesses, road access, "
            f"permit activity and investment risk."
        )
        summary_answer = ask_constructor_question(summary_q, area_name=area)
        st.session_state[f"ai_summary_{area}"] = summary_answer

import re
summary_text = st.session_state[f"ai_summary_{area}"]
# Italicize all (Source: ...) citations
summary_text = re.sub(
    r'\(Source: ([^)]+)\)',
    r'*(Source: \1)*',
    summary_text
)
st.info(summary_text)
# =============================================================================
# CHAT
# =============================================================================

st.divider()

col_reset, _ = st.columns([1, 8])
with col_reset:
    if st.button("Clear Chat"):
        st.session_state["con_chat"] = []
        st.rerun()

st.subheader("Ask a Question")
st.caption(
    f"Ask anything about building in {area}. "
    "Examples: What permits do I need? "
    "Is this a good area for residential development? "
    "What are the infrastructure gaps?"
)

if "con_chat" not in st.session_state:
    st.session_state["con_chat"] = []

for msg in st.session_state["con_chat"]:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if msg["role"] == "assistant":
            content = re.sub(
                r'\(Source: ([^)]+)\)',
                r'*(Source: \1)*',
                content
            )
        st.write(content)

user_input = st.chat_input(f"Ask about building in {area}...")

if user_input:
    st.session_state["con_chat"].append({
        "role":    "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_constructor_question(
                user_input,
                area_name=area,
            )
            print(f"\n[Constructor Chat] Answer:\n{answer}\n")
            # Italicize citations
            answer_display = re.sub(
                r'\(Source: ([^)]+)\)',
                r'*(Source: \1)*',
                answer
            )
            st.write(answer_display)

    st.session_state["con_chat"].append({
        "role":    "assistant",
        "content": answer,
    })
