# neighborhood_agent.py
# See comments above for full documentation

import json
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.services.toronto_data_service import get_311_requests, get_zoning
from src.services.openai_service import ask_ai

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a JSON API. "
    "Output only a single valid JSON object. "
    "Do not think. Do not reason. Do not explain. "
    "Start your response with { and end with }. "
    "Never output anything outside the JSON."
)

# ── Infrastructure keywords ───────────────────────────────────────────────────
# Used to categorize 311 complaints into infrastructure types

INFRASTRUCTURE_CATEGORIES = {
    "Potholes & Roads":    ["pothole", "road", "pavement", "asphalt", "crack"],
    "Streetlights":        ["light", "lamp", "streetlight", "illuminat"],
    "Graffiti":            ["graffiti", "vandal", "tagging"],
    "Parks & Green Space": ["park", "tree", "grass", "garden", "green"],
    "Sidewalks":           ["sidewalk", "curb", "walkway", "pedestrian"],
    "Garbage & Waste":     ["garbage", "litter", "waste", "trash", "bin"],
    "Flooding & Water":    ["flood", "water", "drain", "sewer", "leak"],
    "Noise":               ["noise", "sound", "loud"],
}

# ── Data summarizers ──────────────────────────────────────────────────────────

def summarize_complaints(df: pd.DataFrame) -> dict:
    """
    Count 311 complaints by category and ward.
    Returns top complaint types and most affected wards.
    """
    df.columns = [c.upper() for c in df.columns]

    # Find relevant columns
    type_col = next(
        (c for c in df.columns
         if "SERVICE_REQUEST_TYPE" in c or "TYPE" in c or "REQUEST" in c),
        None
    )
    ward_col   = next((c for c in df.columns if "WARD" in c), None)
    status_col = next((c for c in df.columns if "STATUS" in c), None)

    category_counts = {cat: 0 for cat in INFRASTRUCTURE_CATEGORIES}
    ward_counts      = {}
    open_complaints  = 0

    for _, row in df.iterrows():
        # Count open complaints
        if status_col:
            status = str(row[status_col]).lower()
            if any(s in status for s in ["open", "progress", "active", "new"]):
                open_complaints += 1

        # Count by category
        if type_col:
            complaint_text = str(row[type_col]).lower()
            for category, keywords in INFRASTRUCTURE_CATEGORIES.items():
                if any(kw in complaint_text for kw in keywords):
                    category_counts[category] += 1
                    break

        # Count by ward
        if ward_col:
            ward = str(row[ward_col]).strip()
            if ward and ward.lower() != "nan":
                ward_counts[ward] = ward_counts.get(ward, 0) + 1

    # Top 5 categories by count
    top_categories = dict(
        sorted(category_counts.items(),
               key=lambda x: x[1], reverse=True)[:5]
    )

    # Top 10 most affected wards
    top_wards = dict(
        sorted(ward_counts.items(),
               key=lambda x: x[1], reverse=True)[:10]
    )

    return {
        "top_categories":  top_categories,
        "top_wards":       top_wards,
        "open_complaints": open_complaints,
        "total":           len(df),
    }


def summarize_zoning(df: pd.DataFrame) -> dict:
    """
    Find vacant and underused land from zoning data.
    These are candidates for community spaces and bike lanes.
    """
    df.columns = [c.upper() for c in df.columns]

    zone_col = next(
        (c for c in df.columns
         if "ZONE" in c or "ZN" in c or "LABEL" in c),
        None
    )

    if not zone_col:
        return {"error": "No zone column found", "columns": list(df.columns)}

    zone_counts = {}
    vacant_keywords = [
        "vacant", "utility", "open space", "parking",
        "industrial", "employment", "mixed"
    ]

    for _, row in df.iterrows():
        zone = str(row[zone_col]).strip().lower()
        if not zone or zone == "nan":
            continue

        for kw in vacant_keywords:
            if kw in zone:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
                break

    top_zones = dict(
        sorted(zone_counts.items(),
               key=lambda x: x[1], reverse=True)[:10]
    )

    return {
        "underused_zone_types": top_zones,
        "total_parcels":        len(df),
    }


# ── Main agent function ───────────────────────────────────────────────────────

def run_neighborhood_agent(
    complaints_df: pd.DataFrame = None,
    zoning_df: pd.DataFrame = None,
    limit: int = 500
) -> dict:
    print("\n🏘️  [Neighborhood Agent] Starting analysis...")

    # Step 1 — Load data
    if complaints_df is None:
        print("  📦 Fetching 311 complaints...")
        complaints_df = get_311_requests(limit)

    if zoning_df is None:
        print("  📦 Fetching zoning data...")
        zoning_df = get_zoning(limit)

    print(f"  ✅ Complaints loaded: {len(complaints_df)} rows")
    print(f"  ✅ Zoning loaded:     {len(zoning_df)} rows")

    # Step 2 — Summarize complaints
    print("  🔍 Analyzing 311 complaints...")
    complaint_summary = summarize_complaints(complaints_df)

    # Step 3 — Summarize zoning
    print("  🗺️  Finding vacant and underused land...")
    zoning_summary = summarize_zoning(zoning_df)

    # Step 4 — Pre-build quick wins from top complaints
    quick_wins = [
        f"Address {count} '{category}' complaints urgently"
        for category, count in complaint_summary["top_categories"].items()
        if count > 0
    ][:5]

    # Step 5 — Build prompt
    user_message = f"""
TORONTO 311 COMPLAINTS DATA:

Top complaint categories:
{json.dumps(complaint_summary['top_categories'], indent=2)}

Most affected wards:
{json.dumps(complaint_summary['top_wards'], indent=2)}

Open/unresolved complaints: {complaint_summary['open_complaints']}
Total complaints analyzed:  {complaint_summary['total']}

ZONING DATA — Underused land types:
{json.dumps(zoning_summary.get('underused_zone_types', {}), indent=2)}

Total land parcels analyzed: {zoning_summary.get('total_parcels', 0)}

Pre-identified quick wins:
{json.dumps(quick_wins, indent=2)}

Return this exact JSON filled with real values from the data:

{{
    "top_complaint_clusters": [
        {{"type": "complaint type", "count": 0, "severity": "High/Medium/Low", "wards_affected": "ward info"}},
        {{"type": "complaint type", "count": 0, "severity": "High/Medium/Low", "wards_affected": "ward info"}},
        {{"type": "complaint type", "count": 0, "severity": "High/Medium/Low", "wards_affected": "ward info"}}
    ],
    "vacant_lot_opportunities": [
        {{"location": "area or ward", "current_use": "vacant/parking/industrial", "suggested_use": "community garden/park/bike lane", "reason": "one sentence"}},
        {{"location": "area or ward", "current_use": "vacant/parking/industrial", "suggested_use": "community garden/park/bike lane", "reason": "one sentence"}}
    ],
    "quick_win_fixes": [
        "quick fix 1",
        "quick fix 2",
        "quick fix 3",
        "quick fix 4",
        "quick fix 5"
    ],
    "community_transformations": [
        "longer term transformation 1",
        "longer term transformation 2",
        "longer term transformation 3"
    ],
    "summary": "2 sentence summary of Toronto neighborhood infrastructure situation"
}}
"""

    # Step 6 — Call AI
    print("  🤖 Sending to Mistral AI...")
    raw_response = ask_ai(SYSTEM_PROMPT, user_message, max_tokens=4000)

    # Step 7 — Parse JSON
    try:
        clean = raw_response.strip()
        if clean.startswith('"') and clean.endswith('"'):
            clean = clean[1:-1].replace('\\"', '"').replace('\\n', '\n')

        json_start = clean.find("{")
        json_end   = clean.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")

        json_str = clean[json_start:json_end]
        result   = json.loads(json_str)

    except Exception as e:
        print(f"  ⚠️ JSON parse failed: {e}")
        print(f"  ↳ Raw preview: {raw_response[:200]}")
        # Fallback — return pre-computed data
        result = {
            "top_complaint_clusters": [
                {"type": k, "count": v,
                 "severity": "High" if v > 10 else "Medium",
                 "wards_affected": "Multiple wards"}
                for k, v in complaint_summary["top_categories"].items()
                if v > 0
            ],
            "vacant_lot_opportunities": [
                {"location": "Toronto",
                 "current_use": zone,
                 "suggested_use": "Community space or bike lane",
                 "reason": f"{count} parcels available"}
                for zone, count in list(
                    zoning_summary.get("underused_zone_types", {}).items()
                )[:2]
            ],
            "quick_win_fixes": quick_wins,
            "community_transformations": [
                "Convert underused parking lots to pop-up parks",
                "Add protected bike lanes on high-complaint corridors",
                "Transform vacant industrial land into community gardens",
            ],
            "summary": (
                f"Toronto has {complaint_summary['open_complaints']} open "
                f"complaints out of {complaint_summary['total']} total. "
                "Key issues include roads, lighting and green space."
            ),
        }

    # Step 8 — Add metadata
    result["data_points"] = {
        "complaints_analyzed": len(complaints_df),
        "zoning_parcels":      len(zoning_df),
        "open_complaints":     complaint_summary["open_complaints"],
    }

    print("  ✅ Neighborhood analysis complete!")
    return result


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_neighborhood_agent(limit=400)
    print("\n📊 NEIGHBORHOOD AGENT RESULT:")
    print(json.dumps(result, indent=2))




























# neighborhood_agent.py
# ─────────────────────────────────────────────
# NEIGHBORHOOD AGENT
# ─────────────────────────────────────────────
#
# PURPOSE:
# Scans 311 service request complaints and
# zoning data across all of Toronto to find
# broken infrastructure and suggest fast
# real-world fixes like turning empty lots
# into community spaces or adding pop-up
# bike lanes.
#
# INPUT:
#   - complaints_311 DataFrame
#     (from toronto_data_service.get_311_requests)
#     comes from SR2026.csv inside ZIP
#     columns we use:
#       Service Request Type → type of complaint
#       Division             → city division handling it
#       Ward                 → which ward
#       Status               → open/closed/in progress
#       Creation Date        → when complaint was filed
#
#   - zoning DataFrame
#     (from toronto_data_service.get_zoning)
#     columns we use:
#       ZN_ZONE   → zoning category
#       LABEL     → zone label
#       AREA_DESC → area description
#
# WHAT IT DOES STEP BY STEP:
#   Step 1 → Load 311 complaints and zoning data
#   Step 2 → Count complaints by type
#            (potholes, graffiti, lights, parks etc)
#   Step 3 → Count complaints by ward
#            to find worst affected areas
#   Step 4 → Find open/unresolved complaints
#            these are the most urgent issues
#   Step 5 → Look for vacant/underused land
#            in zoning data for community spaces
#   Step 6 → Build summary for AI
#   Step 7 → Send to Mistral AI with question:
#            "What infrastructure is broken?
#             What quick fixes can we suggest?
#             Which vacant lots can become
#             community spaces or bike lanes?"
#   Step 8 → Parse AI response as structured JSON
#
# OUTPUT:
#   A dictionary with:
#   {
#     "top_complaint_clusters": [
#         {"type": "Pothole",
#          "count": 45,
#          "wards_affected": 3,
#          "severity": "High"},
#         ...
#     ],
#     "vacant_lot_opportunities": [
#         {"location": "Ward 4",
#          "current_use": "Vacant",
#          "suggested_use": "Community Garden",
#          "reason": "..."},
#         ...
#     ],
#     "quick_win_fixes": [
#         "Fill 45 potholes on Dufferin St",
#         "Replace broken streetlights in Ward 6",
#         ...
#     ],
#     "community_transformations": [
#         "Convert vacant lot at X to pop-up park",
#         "Add bike lane on Y Street",
#         ...
#     ],
#     "summary": "Overall summary text..."
#   }
#
# USED BY:
#   - government_agent.py → infrastructure decisions
#   - user_agent.py       → resident quality of life
# ─────────────────────────────────────────────