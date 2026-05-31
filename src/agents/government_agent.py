# government_agent.py
# ---------------------------------------------
# GOVERNMENT AGENT
# ---------------------------------------------
#
# PURPOSE:
# Combines all available Toronto data to answer
# city planning questions and generate investment
# recommendations for city planners.
#
# INPUT (reads from stored files):
#   data/neighbourhood_transit.json  -> TTC scores for 158 neighbourhoods
#   data/urban_profiles_payload.json -> Ward stats (parks, businesses etc)
#   data/lookup_boundaries.geojson   -> Ward boundary polygons
#
# FUNCTIONS:
#   run_government_agent()
#     -> Full city-wide analysis combining all data
#     -> Returns structured investment priorities
#
#   get_area_profile(area_name)
#     -> Complete profile for one area
#     -> Combines TTC score + ward stats + boundary info
#
#   get_investment_priorities()
#     -> Ranked list of areas needing most investment
#     -> Scores each ward across all dimensions
#
#   ask_government_question(question)
#     -> Natural language Q&A for city planners
#     -> Uses all data as context
#
# OUTPUT:
#   {
#     "area": "Rexdale-Kipling",
#     "ttc_score": 7.0,
#     "ttc_stops": 35,
#     "parks": 45,
#     "businesses": 3200,
#     "bike_lanes": 12,
#     "transit_stops": 180,
#     "cultural_hotspots": 15,
#     "dev_applications": 2,
#     "investment_score": 6.2,
#     "priorities": ["transit", "parks", "bike_lanes"],
#     "summary": "..."
#   }
#
# USED BY:
#   - pages/1_Government.py
# ---------------------------------------------

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.agents.mobility_agent   import load_all_neighbourhood_scores
from src.services.openai_service import ask_ai, ask_ai_with_history

# -- File paths ---------------------------------------------------------------

DATA_DIR        = os.path.join(os.path.dirname(__file__), "../../data")
URBAN_PROFILES  = os.path.join(DATA_DIR, "urban_profiles_payload.json")
WARD_BOUNDARIES = os.path.join(DATA_DIR, "lookup_boundaries.geojson")

# -- System prompt ------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Toronto city planning advisor delivering factual briefings. "
    "STRICT RULES: "
    "1. Start with the answer immediately. No preamble. "
    "2. Never show reasoning, analysis steps, or self-correction. "
    "3. Never say 'From the data', 'I can see', 'Looking at', 'The data shows'. "
    "4. Never use rating labels like Good, Excellent, Poor, Critical, Adequate, Emerging. "
    "   Instead use real numbers and comparisons. "
    "   BAD: 'Willowdale has Good transit connectivity' "
    "   GOOD: 'Willowdale has 31 TTC stops, below the city average of 361' "
    "5. Never use score labels. Instead explain what the numbers mean. "
    "   BAD: 'score 7 out of 10' "
    "   GOOD: '37 stops serving the area, with subway and bus access' "
    "6. Always give actual numbers: stop counts, park counts, bike lane segments, businesses. "
    "7. Always compare to city averages when relevant. "
    "   City averages: 361 transit stops, 72 parks, 61 bike lanes, 5465 businesses per ward. "
    "8. Speak in clear professional sentences as if briefing a city councillor. "
    "9. Be concise. 3 to 5 sentences maximum unless the question requires more detail. "
)

# -- Data loaders -------------------------------------------------------------

def load_urban_profiles() -> dict:
    if not os.path.exists(URBAN_PROFILES):
        print(f"  Urban profiles not found: {URBAN_PROFILES}")
        return {}
    with open(URBAN_PROFILES, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ward_boundaries() -> dict:
    if not os.path.exists(WARD_BOUNDARIES):
        print(f"  Ward boundaries not found: {WARD_BOUNDARIES}")
        return {}
    with open(WARD_BOUNDARIES, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ward_number(boundaries: dict, ward_name: str) -> str:
    for feature in boundaries.get("features", []):
        props = feature.get("properties", {})
        if props.get("WARD_NAME", "").lower() == ward_name.lower():
            return props.get("WARD_NUMBER", "Unknown")
    return "Unknown"


# -- Core functions -----------------------------------------------------------

def get_area_profile(area_name: str) -> dict:
    """
    Get complete profile for any Toronto area.
    Combines TTC score + ward urban stats + boundary info.
    Supports partial name matching.

    Args:
        area_name: e.g. "Kipling", "Mimico", "Scarborough", "Humber"

    Returns:
        {
            "area":               "Rexdale-Kipling",
            "ward_number":        "1",
            "ttc_score":          7.0,
            "ttc_rating":         "Good",
            "ttc_stops":          35,
            "has_subway":         True,
            "has_streetcar":      True,
            "has_bus":            True,
            "sample_stops":       [...],
            "parks":              45,
            "businesses":         3200,
            "bike_lanes":         12,
            "transit_stops":      180,
            "road_segments":      1500,
            "cultural_hotspots":  15,
            "ice_rinks":          1,
            "dev_applications":   2,
            "recreation_score":   "Adequate",
            "vibrancy_score":     "Established Hub",
            "transit_label":      "High Connectivity",
            "economic_score":     "Commercial Center",
        }
    """
    print(f"\n[Government Agent] Getting profile for: {area_name}")

    ttc_scores   = load_all_neighbourhood_scores()
    profiles     = load_urban_profiles()
    boundaries   = load_ward_boundaries()

    search = area_name.lower().strip()
    result = {
        "area":            area_name,
        "search_term":     area_name,
        "matched_areas":   [],
    }

    # -- Match TTC scores (neighbourhood level) -------------------------------
    ttc_matches = []
    for name, data in ttc_scores.items():
        if search in name.lower() or any(
            word in name.lower()
            for word in search.split()
            if len(word) > 3
        ):
            ttc_matches.append({
                "neighbourhood":  name,
                "ttc_score":      data.get("connectivity_score", 0),
                "ttc_rating":     data.get("rating", "Unknown"),
                "ttc_stops":      data.get("stop_count", 0),
                "has_subway":     data.get("has_subway", False),
                "has_streetcar":  data.get("has_streetcar", False),
                "has_bus":        data.get("has_bus", False),
                "sample_stops":   data.get("sample_stops", [])[:5],
            })

    if ttc_matches:
        # Use best scoring match as primary
        best_ttc = max(ttc_matches, key=lambda x: x["ttc_score"])
        result.update(best_ttc)
        result["all_ttc_matches"] = ttc_matches

    # -- Match urban profiles (ward level) ------------------------------------
    ward_matches = []
    for ward, data in profiles.items():
        if search in ward.lower() or any(
            word in ward.lower()
            for word in search.split()
            if len(word) > 3
        ):
            ward_matches.append({
                "ward":              ward,
                "ward_number":       get_ward_number(boundaries, ward),
                "parks":             data.get("total_parks", 0),
                "businesses":        data.get("total_active_businesses", 0),
                "bike_lanes":        data.get("total_bike_lane_segments", 0),
                "road_segments":     data.get("total_road_segments", 0),
                "transit_stops":     data.get("total_active_transit_stops", 0),
                "cultural_hotspots": data.get("total_cultural_hotspots", 0),
                "ice_rinks":         data.get("total_outdoor_ice_rinks", 0),
                "dev_applications":  data.get("total_development_applications", 0),
                "recreation_score":  data.get("recreation_deficit_score", ""),
                "vibrancy_score":    data.get("community_vibrancy_score", ""),
                "transit_label":     data.get("transit_connectivity", ""),
                "economic_score":    data.get("economic_vitality_score", ""),
            })

    if ward_matches:
        best_ward = ward_matches[0]
        result.update(best_ward)
        result["all_ward_matches"] = ward_matches

    result["matched_areas"] = [m["neighbourhood"] for m in ttc_matches] + \
                               [m["ward"] for m in ward_matches]

    print(f"  TTC matches:  {[m['neighbourhood'] for m in ttc_matches]}")
    print(f"  Ward matches: {[m['ward'] for m in ward_matches]}")

    return result


def get_investment_priorities() -> list:
    """
    Score all wards across all dimensions and rank by investment need.
    Higher investment score = more urgent need.

    Scoring:
      Low transit stops     -> +2
      Few parks             -> +2
      Few bike lanes        -> +2
      Low businesses        -> +1
      Zero dev applications -> +1

    Returns:
        List of dicts sorted by investment_score descending
    """
    print("\n[Government Agent] Computing investment priorities...")

    profiles   = load_urban_profiles()
    ttc_scores = load_all_neighbourhood_scores()

    if not profiles:
        return []

    # Get averages for comparison
    all_transit  = [v.get("total_active_transit_stops", 0) for v in profiles.values()]
    all_parks    = [v.get("total_parks", 0) for v in profiles.values()]
    all_bikes    = [v.get("total_bike_lane_segments", 0) for v in profiles.values()]
    all_biz      = [v.get("total_active_businesses", 0) for v in profiles.values()]

    avg_transit  = sum(all_transit) / len(all_transit) if all_transit else 0
    avg_parks    = sum(all_parks)   / len(all_parks)   if all_parks   else 0
    avg_bikes    = sum(all_bikes)   / len(all_bikes)   if all_bikes   else 0
    avg_biz      = sum(all_biz)     / len(all_biz)     if all_biz     else 0

    priorities = []

    for ward, data in profiles.items():
        score     = 0
        reasons   = []
        transit   = data.get("total_active_transit_stops", 0)
        parks     = data.get("total_parks", 0)
        bikes     = data.get("total_bike_lane_segments", 0)
        biz       = data.get("total_active_businesses", 0)
        dev_apps  = data.get("total_development_applications", 0)

        if transit < avg_transit * 0.7:
            score += 2
            reasons.append(f"low transit stops ({transit} vs avg {avg_transit:.0f})")

        if parks < avg_parks * 0.7:
            score += 2
            reasons.append(f"few parks ({parks} vs avg {avg_parks:.0f})")

        if bikes < avg_bikes * 0.7:
            score += 2
            reasons.append(f"few bike lanes ({bikes} vs avg {avg_bikes:.0f})")

        if biz < avg_biz * 0.7:
            score += 1
            reasons.append(f"low business activity ({biz} vs avg {avg_biz:.0f})")

        if dev_apps == 0:
            score += 1
            reasons.append("no active development applications")

        priorities.append({
            "ward":              ward,
            "investment_score":  score,
            "transit_stops":     transit,
            "parks":             parks,
            "bike_lanes":        bikes,
            "businesses":        biz,
            "dev_applications":  dev_apps,
            "reasons":           reasons,
            "vibrancy":          data.get("community_vibrancy_score", ""),
            "economic":          data.get("economic_vitality_score", ""),
        })

    sorted_priorities = sorted(
        priorities,
        key=lambda x: x["investment_score"],
        reverse=True
    )

    print(f"  Top priority: {sorted_priorities[0]['ward']} "
          f"(score: {sorted_priorities[0]['investment_score']})")

    return sorted_priorities


def run_government_agent() -> dict:
    """
    Full city-wide government analysis.
    Combines transit + urban profiles + investment priorities.

    Returns:
        {
            "investment_priorities": [...top 10 wards...],
            "transit_summary":       {...},
            "city_averages":         {...},
            "total_wards":           int,
            "total_neighbourhoods":  int,
            "summary":               "..."
        }
    """
    print("\n[Government Agent] Running full city analysis...")

    profiles     = load_urban_profiles()
    ttc_scores   = load_all_neighbourhood_scores()
    priorities   = get_investment_priorities()

    # City averages
    if profiles:
        all_transit = [v.get("total_active_transit_stops", 0) for v in profiles.values()]
        all_parks   = [v.get("total_parks", 0) for v in profiles.values()]
        all_bikes   = [v.get("total_bike_lane_segments", 0) for v in profiles.values()]
        all_biz     = [v.get("total_active_businesses", 0) for v in profiles.values()]

        city_averages = {
            "avg_transit_stops": round(sum(all_transit) / len(all_transit), 1),
            "avg_parks":         round(sum(all_parks)   / len(all_parks),   1),
            "avg_bike_lanes":    round(sum(all_bikes)   / len(all_bikes),   1),
            "avg_businesses":    round(sum(all_biz)     / len(all_biz),     1),
        }
    else:
        city_averages = {}

    result = {
        "investment_priorities": priorities[:10],
        "city_averages":         city_averages,
        "total_wards":           len(profiles),
        "total_neighbourhoods":  len(ttc_scores),
        "top_priority_ward":     priorities[0]["ward"] if priorities else "Unknown",
        "summary": (
            f"Toronto has {len(profiles)} wards and "
            f"{len(ttc_scores)} neighbourhoods analyzed. "
            f"Top investment priority is {priorities[0]['ward'] if priorities else 'Unknown'} "
            f"with {priorities[0]['investment_score'] if priorities else 0} priority factors."
        ),
    }

    print("[Government Agent] Analysis complete.")
    return result
def get_chart_data(question: str) -> dict:
    """
    Return chart data relevant to the question asked.
    Used by the Government portal to show dynamic charts.

    Returns:
        {
            "chart_type": "bar",
            "title":      "Transit Stops by Ward",
            "data":       { "Ward A": 123, "Ward B": 456, ... },
            "highlight":  ["Ward A"],   <- wards mentioned in question
            "x_label":    "Ward",
            "y_label":    "Transit Stops",
        }
    """
    profiles = load_urban_profiles()
    scores   = load_all_neighbourhood_scores()

    if not profiles:
        return {}

    question_lower = question.lower()

    # Detect what the question is about
    is_transit  = any(w in question_lower for w in [
        "transit", "ttc", "bus", "subway", "stops", "streetcar", "connectivity"
    ])
    is_parks    = any(w in question_lower for w in [
        "park", "green", "recreation", "outdoor"
    ])
    is_bike     = any(w in question_lower for w in [
        "bike", "cycling", "cycle", "lane"
    ])
    is_business = any(w in question_lower for w in [
        "business", "economic", "commercial", "shop", "retail"
    ])
    is_invest   = any(w in question_lower for w in [
        "invest", "priorit", "develop", "focus", "improve", "need"
    ])
    is_compare  = any(w in question_lower for w in [
        "compare", "vs", "versus", "better", "between"
    ])

    # Find mentioned wards
    mentioned = []
    for ward in profiles:
        if any(word in ward.lower()
               for word in question_lower.split()
               if len(word) > 3):
            mentioned.append(ward)

    # Choose which metric to chart
    if is_invest:
        # Show investment priority scores
        prios = get_investment_priorities()
        data  = {p["ward"]: p["investment_score"] for p in prios[:10]}
        return {
            "chart_type": "bar",
            "title":      "Investment Priority Score by Ward (higher = more urgent)",
            "data":       data,
            "highlight":  mentioned,
            "x_label":    "Ward",
            "y_label":    "Priority Score",
        }

    elif is_transit:
        data = {
            ward: d.get("total_active_transit_stops", 0)
            for ward, d in profiles.items()
        }
        return {
            "chart_type": "bar",
            "title":      "Active Transit Stops by Ward",
            "data":       dict(sorted(data.items(), key=lambda x: x[1], reverse=True)),
            "highlight":  mentioned,
            "x_label":    "Ward",
            "y_label":    "Transit Stops",
        }

    elif is_parks:
        data = {
            ward: d.get("total_parks", 0)
            for ward, d in profiles.items()
        }
        return {
            "chart_type": "bar",
            "title":      "Total Parks by Ward",
            "data":       dict(sorted(data.items(), key=lambda x: x[1], reverse=True)),
            "highlight":  mentioned,
            "x_label":    "Ward",
            "y_label":    "Parks",
        }

    elif is_bike:
        data = {
            ward: d.get("total_bike_lane_segments", 0)
            for ward, d in profiles.items()
        }
        return {
            "chart_type": "bar",
            "title":      "Bike Lane Segments by Ward",
            "data":       dict(sorted(data.items(), key=lambda x: x[1], reverse=True)),
            "highlight":  mentioned,
            "x_label":    "Ward",
            "y_label":    "Bike Lane Segments",
        }

    elif is_business:
        data = {
            ward: d.get("total_active_businesses", 0)
            for ward, d in profiles.items()
        }
        return {
            "chart_type": "bar",
            "title":      "Active Businesses by Ward",
            "data":       dict(sorted(data.items(), key=lambda x: x[1], reverse=True)),
            "highlight":  mentioned,
            "x_label":    "Ward",
            "y_label":    "Active Businesses",
        }

    elif is_compare and mentioned:
        # Multi-metric comparison for mentioned wards
        metrics = ["total_active_transit_stops", "total_parks",
                   "total_bike_lane_segments", "total_active_businesses"]
        labels  = ["Transit Stops", "Parks", "Bike Lanes", "Businesses"]
        data    = {}
        for ward in mentioned:
            if ward in profiles:
                data[ward] = {
                    label: profiles[ward].get(metric, 0)
                    for metric, label in zip(metrics, labels)
                }
        return {
            "chart_type": "multi_bar",
            "title":      f"Comparison: {' vs '.join(mentioned)}",
            "data":       data,
            "highlight":  mentioned,
            "x_label":    "Metric",
            "y_label":    "Value",
        }

    else:
        # Default -- show transit stops
        data = {
            ward: d.get("total_active_transit_stops", 0)
            for ward, d in profiles.items()
        }
        return {
            "chart_type": "bar",
            "title":      "Active Transit Stops by Ward",
            "data":       dict(sorted(data.items(), key=lambda x: x[1], reverse=True)),
            "highlight":  mentioned,
            "x_label":    "Ward",
            "y_label":    "Transit Stops",
        }


def ask_government_question(question: str,
                             chat_history: list = None) -> str:
    return f"TEST ANSWER FOR: {question}"
    print(f"\n[Government Agent] Question: {question}")

    profiles   = load_urban_profiles()
    ttc_scores = load_all_neighbourhood_scores()
    priorities = get_investment_priorities()

    question_lower = question.lower()

    # City averages
    if profiles:
        all_transit = [v.get("total_active_transit_stops", 0) for v in profiles.values()]
        all_parks   = [v.get("total_parks", 0) for v in profiles.values()]
        all_bikes   = [v.get("total_bike_lane_segments", 0) for v in profiles.values()]
        all_biz     = [v.get("total_active_businesses", 0) for v in profiles.values()]
        avg_transit = round(sum(all_transit) / len(all_transit), 1)
        avg_parks   = round(sum(all_parks)   / len(all_parks),   1)
        avg_bikes   = round(sum(all_bikes)   / len(all_bikes),   1)
        avg_biz     = round(sum(all_biz)     / len(all_biz),     1)
    else:
        avg_transit = avg_parks = avg_bikes = avg_biz = 0

    # -- Detect question type and build a pre-written answer ------------------

    is_invest  = any(w in question_lower for w in [
        "invest", "priorit", "develop", "focus", "improve",
        "need", "which ward", "which area"
    ])
    is_compare = any(w in question_lower for w in [
        "compare", "vs", "versus", "between", "better"
    ])

    # Find mentioned areas
    matched_ward_data = {}
    for ward, data in profiles.items():
        if any(word in ward.lower()
               for word in question_lower.split() if len(word) > 3):
            matched_ward_data[ward] = data

    matched_ttc = {}
    for name, data in ttc_scores.items():
        if any(word in name.lower()
               for word in question_lower.split() if len(word) > 3):
            matched_ttc[name] = data

    # -- Build answer directly from data (no AI for logic) --------------------

    answer_lines = []

    if is_invest and not matched_ward_data:
        # General investment question -- list top priorities
        for p in priorities[:5]:
            reasons = " and ".join(p["reasons"])
            answer_lines.append(
                f"{p['ward']} needs investment: {reasons}."
            )

    elif matched_ward_data:
        # Specific area question
        for ward, data in matched_ward_data.items():
            parks   = data.get("total_parks", 0)
            bikes   = data.get("total_bike_lane_segments", 0)
            biz     = data.get("total_active_businesses", 0)
            transit = data.get("total_active_transit_stops", 0)
            cultural= data.get("total_cultural_hotspots", 0)
            dev     = data.get("total_development_applications", 0)
            roads   = data.get("total_road_segments", 0)

            answer_lines.append(
                f"{ward} ward has {transit} transit stops "
                f"({'above' if transit >= avg_transit else 'below'} city average of {avg_transit:.0f}), "
                f"{parks} parks "
                f"({'above' if parks >= avg_parks else 'below'} city average of {avg_parks:.0f}), "
                f"{bikes} bike lane segments "
                f"({'above' if bikes >= avg_bikes else 'below'} city average of {avg_bikes:.0f}), "
                f"and {biz} active businesses "
                f"({'above' if biz >= avg_biz else 'below'} city average of {avg_biz:.0f}). "
                f"There are {cultural} cultural hotspots, {roads} road segments, "
                f"and {dev} active development applications in the ward."
            )

        # Add matching TTC data
        for name, data in matched_ttc.items():
            stops     = data.get("stop_count", 0)
            subway    = "including subway" if data.get("has_subway") else "no subway"
            streetcar = ", streetcar" if data.get("has_streetcar") else ""
            bus       = ", bus" if data.get("has_bus") else ""
            samples   = ", ".join(data.get("sample_stops", [])[:3])
            answer_lines.append(
                f"{name} neighbourhood has {stops} TTC stops "
                f"({subway}{streetcar}{bus}). "
                f"Key stops include: {samples}."
            )

    elif matched_ttc and not matched_ward_data:
        # TTC-only match
        for name, data in matched_ttc.items():
            stops     = data.get("stop_count", 0)
            subway    = "including subway" if data.get("has_subway") else "no subway"
            streetcar = ", streetcar" if data.get("has_streetcar") else ""
            bus       = ", bus" if data.get("has_bus") else ""
            samples   = ", ".join(data.get("sample_stops", [])[:3])
            answer_lines.append(
                f"{name} has {stops} TTC stops "
                f"({subway}{streetcar}{bus}). "
                f"Key stops: {samples}."
            )

    else:
        # Fallback -- general city overview
        answer_lines.append(
            f"Toronto has {len(profiles)} wards with an average of "
            f"{avg_transit:.0f} transit stops, {avg_parks:.0f} parks, "
            f"{avg_bikes:.0f} bike lane segments, and {avg_biz:.0f} "
            f"active businesses per ward."
        )
        for p in priorities[:3]:
            answer_lines.append(
                f"{p['ward']} is a top priority: "
                + " and ".join(p["reasons"]) + "."
            )

    answer = " ".join(answer_lines)
    print(f"[Government Agent] Response ready.")
    return answer
# -- Test ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1 - Full Government Analysis")
    print("=" * 60)
    result = run_government_agent()
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2 - Area Profile")
    print("=" * 60)
    profile = get_area_profile("Kipling")
    print(json.dumps(profile, indent=2))

    print("\n" + "=" * 60)
    print("TEST 3 - Investment Priorities")
    print("=" * 60)
    priorities = get_investment_priorities()
    print("\nTop 5 wards needing investment:")
    for p in priorities[:5]:
        print(f"  {p['ward']}: score {p['investment_score']}")
        for r in p["reasons"]:
            print(f"    - {r}")

    print("\n" + "=" * 60)
    print("TEST 4 - Natural Language Q&A")
    print("=" * 60)
    questions = [
        "How is the park and TTC situation around Kipling?",
        "Which areas should I prioritize for development?",
        "I am planning to construct in Mimico. What should I know?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = ask_government_question(q)
        print(f"A: {answer}")
