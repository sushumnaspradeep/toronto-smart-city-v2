# socioeconomic_agent.py
# ─────────────────────────────────────────────
# SOCIOECONOMIC AGENT
# ─────────────────────────────────────────────
#
# PURPOSE:
# Analyzes building permits and park facilities
# across all of Toronto to identify:
#   1. Which wards have too much new construction
#      that could spike local rents
#   2. Which zones have too few parks relative
#      to the amount of new construction
#
# INPUT:
#   - building_permits DataFrame
#     columns used:
#       WARD_GRID      → ward code e.g. N0629, E2029
#       STRUCTURE_TYPE → e.g. SFD-Detached, Apartment
#       EST_CONST_COST → estimated construction cost
#
#   - parks DataFrame
#     columns used:
#       geometry → JSON with lat/lon coordinates
#                  used to assign parks to zones
#
# OUTPUT:
#   {
#     "rent_risk_areas": [...],
#     "park_pressure_areas": [...],
#     "recommendations": [...],
#     "summary": "..."
#   }
#
# USED BY:
#   - government_agent.py  → city planner decisions
#   - constructor_agent.py → rent impact warnings
#   - user_agent.py        → resident rent questions
# ─────────────────────────────────────────────

import json
import json as json_lib
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.services.toronto_data_service import get_building_permits, get_parks
from src.services.openai_service import ask_ai

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a JSON API. "
    "Output only a single valid JSON object. "
    "Do not think. Do not reason. Do not explain. "
    "Start your response with { and end with }. "
    "Never output anything outside the JSON."
)

# ── Toronto zones for park assignment ─────────────────────────────────────────
# Parks dataset has no ward column — only lat/lon in geometry
# We use these bounding boxes to assign parks to zonesPARK_ZONES = {
    "Scarborough":   {"lat": (43.69, 43.82), "lon": (-79.30, -79.15)},
    "North York":    {"lat": (43.73, 43.82), "lon": (-79.55, -79.35)},
    "Etobicoke":     {"lat": (43.59, 43.78), "lon": (-79.60, -79.48)},
    "Downtown":      {"lat": (43.63, 43.67), "lon": (-79.42, -79.35)},
    "East York":     {"lat": (43.67, 43.73), "lon": (-79.35, -79.28)},
    "West End":      {"lat": (43.63, 43.69), "lon": (-79.49, -79.41)},
    "Midtown":       {"lat": (43.67, 43.73), "lon": (-79.42, -79.35)},
    "Humber Bay":    {"lat": (43.59, 43.64), "lon": (-79.50, -79.44)},
    "North Toronto": {"lat": (43.73, 43.78), "lon": (-79.42, -79.35)},
    "York":          {"lat": (43.68, 43.73), "lon": (-79.50, -79.44)},
}



# ── Data summarizers ──────────────────────────────────────────────────────────

def summarize_permits(df: pd.DataFrame) -> dict:
    """
    Summarize building permits by ward.
    Uses WARD_GRID column e.g. N0629, E2029.
    Uses STRUCTURE_TYPE to identify residential permits.
    Uses EST_CONST_COST to sum investment per ward.
    """
    # Use exact column names from dataset
    ward_col = "WARD_GRID"      if "WARD_GRID"      in df.columns else None
    type_col = "STRUCTURE_TYPE" if "STRUCTURE_TYPE" in df.columns else None
    cost_col = "EST_CONST_COST" if "EST_CONST_COST" in df.columns else None

    # Fallback to uppercase search
    if not ward_col:
        df.columns = [c.upper() for c in df.columns]
        ward_col = next((c for c in df.columns if "WARD" in c), None)
        type_col = next((c for c in df.columns if "STRUCTURE" in c or "PERMIT_TYPE" in c), None)
        cost_col = next((c for c in df.columns if "COST" in c), None)

    if not ward_col:
        return {}

    summary = {}
    for ward, group in df.groupby(ward_col):
        ward_str = str(ward).strip()
        if not ward_str or ward_str.lower() == "nan":
            continue

        # Count residential permits
        residential = 0
        if type_col:
            residential = group[type_col].str.contains(
                "sfd|detached|semi|apartment|townhouse|residential|dwelling|condo",
                case=False, na=False
            ).sum()

        # Sum construction cost
        total_cost = 0
        if cost_col:
            total_cost = pd.to_numeric(
                group[cost_col], errors="coerce"
            ).fillna(0).sum()

        summary[ward_str] = {
            "total_permits":       len(group),
            "residential_permits": int(residential),
            "total_cost_millions": round(float(total_cost) / 1_000_000, 2),
        }

    return summary


def summarize_parks(df: pd.DataFrame) -> dict:
    """
    Assign parks to Toronto zones using lat/lon
    from the geometry column.
    Format: {"type": "Point", "coordinates": [lon, lat]}
    """
    geo_col = "geometry" if "geometry" in df.columns else None

    if not geo_col:
        return {"error": "No geometry column", "total_parks": len(df)}

    zone_counts = {zone: 0 for zone in PARK_ZONES}
    unassigned  = 0

    for _, row in df.iterrows():
        try:
            geo    = json_lib.loads(str(row[geo_col]))
            coords = geo["coordinates"]  # [lon, lat] directly
            lon    = float(coords[0])
            lat    = float(coords[1])

            assigned = False
            for zone, bounds in PARK_ZONES.items():
                if (bounds["lat"][0] <= lat <= bounds["lat"][1] and
                        bounds["lon"][0] <= lon <= bounds["lon"][1]):
                    zone_counts[zone] += 1
                    assigned = True
                    break

            if not assigned:
                unassigned += 1

        except Exception:
            unassigned += 1

    zone_counts["unassigned"] = unassigned
    return zone_counts

# ── Main agent function ───────────────────────────────────────────────────────

def run_socioeconomic_agent(
    permits_df: pd.DataFrame = None,
    parks_df: pd.DataFrame = None,
    limit: int = 500
) -> dict:
    print("\n🏗️  [Socioeconomic Agent] Starting analysis...")

    # Step 1 — Load data
    if permits_df is None:
        print("  📦 Fetching building permits...")
        permits_df = get_building_permits(limit)

    if parks_df is None:
        print("  📦 Fetching parks data...")
        parks_df = get_parks(limit)

    print(f"  ✅ Permits loaded: {len(permits_df)} rows")
    print(f"  ✅ Parks loaded:   {len(parks_df)} rows")

    # Step 2 — Summarize permits by ward
    print("  🔍 Summarizing permits by ward...")
    permit_summary = summarize_permits(permits_df)

    # Step 3 — Summarize parks by zone
    print("  🌳 Assigning parks to zones...")
    park_summary = summarize_parks(parks_df)

    # Step 4 — Top 15 wards by permit volume
    top_permits = dict(
        sorted(permit_summary.items(),
               key=lambda x: x[1]["total_permits"],
               reverse=True)[:15]
    )

    # Step 5 — Zones with fewest parks
  # Step 5 — Zones with fewest parks
    # Filter out error keys, strings, and unassigned
    valid_parks = {
        z: c for z, c in park_summary.items()
        if isinstance(c, int)
        and z != "unassigned"
        and z != "error"
        and z != "total_parks"
    }

    park_gaps = dict(sorted(valid_parks.items(), key=lambda x: x[1]))

    # Debug — print what we got
    print(f"  🌳 Parks by zone: {park_gaps}")

    # Step 6 — Build prompt
    user_message = f"""
TORONTO BUILDING PERMITS (top 15 wards by volume):
{json.dumps(top_permits, indent=2)}

TORONTO PARKS BY ZONE (sorted fewest to most):
{json.dumps(park_gaps, indent=2)}

Total permits analyzed: {len(permits_df)}
Total parks analyzed:   {len(parks_df)}

Reply with this JSON and nothing else. Fill in real values:

{{
    "rent_risk_areas": [
        {{"ward": "top ward", "risk_level": "High", "total_permits": 0, "residential_permits": 0, "reason": "one sentence"}},
        {{"ward": "second ward", "risk_level": "High", "total_permits": 0, "residential_permits": 0, "reason": "one sentence"}},
        {{"ward": "third ward", "risk_level": "Medium", "total_permits": 0, "residential_permits": 0, "reason": "one sentence"}}
    ],
    "park_pressure_areas": [
        {{"zone": "zone name", "parks_count": 0, "pressure_level": "High", "gap": "one sentence"}},
        {{"zone": "zone name", "parks_count": 0, "pressure_level": "Medium", "gap": "one sentence"}}
    ],
    "recommendations": [
        "recommendation 1",
        "recommendation 2",
        "recommendation 3",
        "recommendation 4",
        "recommendation 5"
    ],
    "summary": "2 sentence summary of Toronto socioeconomic situation"
}}
"""

    # Step 7 — Call AI
    print("  🤖 Sending to Mistral AI...")
    raw_response = ask_ai(SYSTEM_PROMPT, user_message, max_tokens=3000)

    # Step 8 — Parse JSON
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
        print(f"  ↳ Raw preview: {raw_response[:300]}")
        # Fallback — return pre-computed data
        result = {
            "rent_risk_areas": [
                {
                    "ward":                w,
                    "risk_level":          "High" if d["total_permits"] > 5 else "Medium",
                    "total_permits":       d["total_permits"],
                    "residential_permits": d["residential_permits"],
                    "reason":              f"{d['total_permits']} active permits worth ${d['total_cost_millions']}M"
                }
                for w, d in list(top_permits.items())[:3]
            ],
            "park_pressure_areas": [
                {
                    "zone":           z,
                    "parks_count":    c,
                    "pressure_level": "High" if c < 5 else "Medium",
                    "gap":            f"Only {c} parks in this zone"
                }
                for z, c in list(park_gaps.items())[:2]
            ],
            "recommendations": [
                f"Increase green space in {list(park_gaps.keys())[0]}",
                f"Monitor rent pressure in {list(top_permits.keys())[0]}",
                "Add affordable housing requirements to high-permit wards",
                "Fast-track park development in zones with zero parks",
                "Review zoning bylaws in high-construction wards",
            ],
            "summary": (
                f"Toronto has {len(permits_df)} active permits. "
                f"Top ward {list(top_permits.keys())[0]} has "
                f"{list(top_permits.values())[0]['total_permits']} permits."
            ),
        }

    # Step 9 — Add metadata
    result["data_points"] = {
        "permits_analyzed": len(permits_df),
        "parks_analyzed":   len(parks_df),
    }

    print("  ✅ Socioeconomic analysis complete!")
    # DEBUG — check what geometry looks like
    print(f"  🔍 Parks columns: {list(parks_df.columns)}")
    print(f"  🔍 Sample geometry: {parks_df['geometry'].iloc[0] if 'geometry' in parks_df.columns else 'NO GEOMETRY COLUMN'}")
    return result



# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_socioeconomic_agent(limit=200)
    print("\n📊 SOCIOECONOMIC AGENT RESULT:")
    print(json.dumps(result, indent=2))
    