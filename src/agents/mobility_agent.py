# mobility_agent.py
# ─────────────────────────────────────────────
# MOBILITY AGENT
# ─────────────────────────────────────────────
#
# PURPOSE:
# Analyzes TTC routes and stop data across all
# of Toronto to identify transit deserts.
#
# INPUT:
#   - ttc_routes DataFrame → routes.txt from GTFS ZIP
#     columns used:
#       route_id        → unique route identifier
#       route_long_name → full name e.g. Dufferin North
#       route_type      → 1=subway, 3=bus, 0=streetcar
#
#   - ttc_stops DataFrame → stops.txt from GTFS ZIP
#     columns used:
#       stop_id   → unique stop identifier
#       stop_name → name e.g. Danforth Rd at Kennedy Rd
#       stop_lat  → latitude e.g. 43.714379
#       stop_lon  → longitude e.g. -79.260939
#
# WHAT IT DOES STEP BY STEP:
#   Step 1 → Load TTC routes and stops data
#   Step 2 → Count routes by type (bus/subway/streetcar)
#   Step 3 → Assign stops to zones using actual
#            stop_lat and stop_lon coordinates
#            Divides Toronto into a 3x3 grid
#            automatically from the data itself
#   Step 4 → Flag zones with fewer than 10 stops
#            as transit deserts
#   Step 5 → Send summary to Mistral AI
#   Step 6 → Parse and return JSON result
#
# OUTPUT:
#   {
#     "transit_deserts": [...],
#     "coverage_gaps": [...],
#     "recommendations": [...],
#     "route_summary": {...},
#     "summary": "..."
#   }
#
# USED BY:
#   - government_agent.py  → transit gap decisions
#   - constructor_agent.py → TTC score for a location
#   - user_agent.py        → resident transit questions
# ─────────────────────────────────────────────

import json
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.services.toronto_data_service import get_ttc_routes, get_ttc_stops
from src.services.openai_service import ask_ai

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a JSON API. "
    "Output only a single valid JSON object. "
    "Do not think. Do not reason. Do not explain. "
    "Start your response with { and end with }. "
    "Never output anything outside the JSON."
)

# ── Data summarizers ──────────────────────────────────────────────────────────

def summarize_routes(df: pd.DataFrame) -> dict:
    """
    Count TTC routes by type.
    Uses exact column route_type from routes.txt:
      0 = streetcar
      1 = subway
      3 = bus
    """
    df.columns = [c.lower() for c in df.columns]
    type_col   = "route_type" if "route_type" in df.columns else None
    route_types = {0: "streetcar", 1: "subway", 3: "bus"}
    counts = {"total": len(df), "bus": 0, "subway": 0, "streetcar": 0, "other": 0}

    if type_col:
        for _, row in df.iterrows():
            t     = int(row[type_col]) if str(row[type_col]).isdigit() else -1
            label = route_types.get(t, "other")
            counts[label] = counts.get(label, 0) + 1

    return counts


def assign_stops_to_zones(df: pd.DataFrame) -> dict:
    """
    Assign TTC stops to Toronto zones using
    actual stop_lat and stop_lon coordinates
    from stops.txt GTFS file.
    Divides Toronto into a 3x3 grid automatically
    based on actual coordinate ranges in the data.
    No hardcoded bounding boxes needed.

    Zone grid layout (South=lower lat, North=higher lat):
    ┌─────────────┬──────────────┬────────────┐
    │ North West  │ North Central│ North East │
    ├─────────────┼──────────────┼────────────┤
    │ Mid West    │   Central    │  Mid East  │
    ├─────────────┼──────────────┼────────────┤
    │ South West  │ South Central│ South East │
    └─────────────┴──────────────┴────────────┘
    """
    df.columns = [c.lower() for c in df.columns]

    lat_col = "stop_lat" if "stop_lat" in df.columns else None
    lon_col = "stop_lon" if "stop_lon" in df.columns else None


    if not lat_col or not lon_col:
        return {"error": "No stop_lat/stop_lon columns found"}
    

    # Extract valid coordinates
    coords = []
    for _, row in df.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            if lat > 0 and lon < 0:  # valid Toronto coords
                coords.append((lat, lon))
        except Exception:
            continue

    if not coords:
        return {"error": "No valid coordinates found"}

    # Get actual bounds from the data itself
    lats    = [c[0] for c in coords]
    lons    = [c[1] for c in coords]
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    

    print(f" Stop bounds: lat {min_lat:.3f}-{max_lat:.3f}, "
          f"lon {min_lon:.3f}-{max_lon:.3f}")

    # Divide into 3x3 grid
    lat_step = (max_lat - min_lat) / 3
    lon_step = (max_lon - min_lon) / 3

    zone_names = {
        (0, 0): "South West",    (0, 1): "South Central",  (0, 2): "South East",
        (1, 0): "Mid West",      (1, 1): "Central",         (1, 2): "Mid East",
        (2, 0): "North West",    (2, 1): "North Central",   (2, 2): "North East",
    }

    zone_counts = {name: 0 for name in zone_names.values()}
    unassigned  = 0

    for lat, lon in coords:
        try:
            col     = min(int((lon - min_lon) / lon_step), 2)
            row_idx = min(int((lat - min_lat) / lat_step), 2)
            zone    = zone_names[(row_idx, col)]
            zone_counts[zone] += 1
        except Exception:
            unassigned += 1

    zone_counts["unassigned"] = unassigned
    print(f"   Real bounds: lat {min_lat:.4f}-{max_lat:.4f}, lon {min_lon:.4f}-{max_lon:.4f}")
    print(f"   Grid steps: lat_step={lat_step:.4f}, lon_step={lon_step:.4f}")

    # Print first 5 stops with their assigned zones for verification
    debug_count = 0
    for lat, lon in coords[:5]:
        col     = min(int((lon - min_lon) / lon_step), 2)
        row_idx = min(int((lat - min_lat) / lat_step), 2)
        zone    = zone_names[(row_idx, col)]
        print(f"   lat={lat:.4f} lon={lon:.4f} → col={col} row={row_idx} → {zone}")
    return zone_counts


# ── Main agent function ───────────────────────────────────────────────────────

import math

def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    Check if a point (lat, lon) is inside a polygon.
    Uses ray casting algorithm.
    polygon is a list of [lon, lat] coordinate pairs.
    """
    n      = len(polygon)
    inside = False
    x, y   = lon, lat

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def get_neighbourhood_transit(neighbourhood_name: str) -> dict:
    """
    Get TTC connectivity score for any Toronto neighbourhood.
    
    Args:
        neighbourhood_name: e.g. "Lakeshore", "Etobicoke", "Danforth"
    
    Returns:
        {
            "neighbourhood": "Mimico-Lakeshore",
            "matched_name": "Mimico",
            "stop_count": 45,
            "connectivity_score": 6.5,
            "rating": "Good",
            "sample_stops": [...],
            "route_summary": {...}
        }
    """
    import json as json_lib
    from src.services.toronto_data_service import get_neighbourhoods, get_ttc_stops, get_ttc_routes

    print(f"\n🔍 Finding TTC connectivity for: {neighbourhood_name}")

    # Step 1 — Load neighbourhood boundaries
    neighbourhoods_df = get_neighbourhoods(limit=200)
    stops_df          = get_ttc_stops(limit=99999)
    routes_df         = get_ttc_routes(limit=99999)

    # Step 2 — Find matching neighbourhood by name
    name_col = "AREA_NAME" if "AREA_NAME" in neighbourhoods_df.columns else None
    if not name_col:
        return {"error": "No AREA_NAME column in neighbourhoods dataset"}

    # Fuzzy match — find neighbourhood whose name contains the search term
    search   = neighbourhood_name.lower()
    matches  = neighbourhoods_df[
        neighbourhoods_df[name_col].str.lower().str.contains(search, na=False)
    ]

    if matches.empty:
        return {
            "error": f"No neighbourhood found matching '{neighbourhood_name}'",
            "suggestion": "Try: Mimico, Lakeshore, Etobicoke, Danforth, Scarborough"
        }

    # Use first match
    match     = matches.iloc[0]
    matched_name = match[name_col]
    print(f"  ✅ Matched neighbourhood: {matched_name}")

    # Step 3 — Extract polygon from geometry
    # Step 3 — Extract polygon from geometry
    try:
        geo      = json_lib.loads(str(match["geometry"]))
        geo_type = geo["type"]

        if geo_type == "MultiPolygon":
            # MultiPolygon: coordinates[0][0] is the outer ring
            polygon = geo["coordinates"][0][0]
        elif geo_type == "Polygon":
            # Polygon: coordinates[0] is the outer ring
            polygon = geo["coordinates"][0]
        else:
            return {"error": f"Unknown geometry type: {geo_type}"}

        print(f"   Geometry type: {geo_type}, polygon points: {len(polygon)}")
        print(f"   Polygon bounds: lat {min(c[1] for c in polygon):.4f}-{max(c[1] for c in polygon):.4f}")

    except Exception as e:
        return {"error": f"Could not parse geometry: {e}"}

    # Step 4 — Find stops inside polygon
    stops_df.columns = [c.lower() for c in stops_df.columns]
    stops_inside     = []

    for _, stop in stops_df.iterrows():
        try:
            lat = float(stop["stop_lat"])
            lon = float(stop["stop_lon"])
            if point_in_polygon(lat, lon, polygon):
                stops_inside.append({
                    "stop_id":   stop["stop_id"],
                    "stop_name": stop["stop_name"],
                    "stop_lat":  lat,
                    "stop_lon":  lon,
                })
        except Exception:
            continue

    stop_count = len(stops_inside)
    print(f" Stops found inside {matched_name}: {stop_count}")
    if stop_count == 0:
        print("  ⚠️ Zero stops found — testing first 3 stops manually:")
        for _, stop in stops_df.head(3).iterrows():
            try:
                lat = float(stop["stop_lat"])
                lon = float(stop["stop_lon"])
                inside = point_in_polygon(lat, lon, polygon)
                print(f"    {stop['stop_name']}: lat={lat:.4f} lon={lon:.4f} inside={inside}")
            except Exception as e:
                print(f"    Error: {e}")

    # Step 5 — Calculate connectivity score (0-10)
    score = 0

    # Stop density score (0-4 points)
    if stop_count > 200:   score += 4
    elif stop_count > 100: score += 3
    elif stop_count > 50:  score += 2
    elif stop_count > 10:  score += 1

    # Route diversity score (0-3 points)
    routes_df.columns = [c.lower() for c in routes_df.columns]
    route_counts = routes_df["route_type"].value_counts().to_dict()

    has_subway    = 1 in route_counts
    has_streetcar = 0 in route_counts
    has_bus       = 3 in route_counts

    if has_subway:    score += 3
    if has_streetcar: score += 1
    if has_bus:       score += 1

    # Normalize to 10
    score = min(round(score, 1), 10)

    # Step 6 — Rating
    if score >= 8:   rating = "Excellent"
    elif score >= 6: rating = "Good"
    elif score >= 4: rating = "Fair"
    elif score >= 2: rating = "Poor"
    else:            rating = "Critical"

    return {
        "neighbourhood":    matched_name,
        "search_term":      neighbourhood_name,
        "stop_count":       stop_count,
        "connectivity_score": score,
        "rating":           rating,
        "has_subway":       has_subway,
        "has_streetcar":    has_streetcar,
        "has_bus":          has_bus,
        "sample_stops":     [s["stop_name"] for s in stops_inside[:10]],
        "all_stops":        stops_inside,
    }

def run_mobility_agent(
    routes_df: pd.DataFrame = None,
    stops_df: pd.DataFrame = None,
    limit: int = 500
) -> dict:
    print("\n  [Mobility Agent] Starting analysis...")

    # Step 1 — Load data
    if routes_df is None:
        print("   Fetching TTC routes...")
        routes_df = get_ttc_routes(limit)

    if stops_df is None:
        print("   Fetching TTC stops...")
        stops_df = get_ttc_stops(limit)

    print(f"   Routes loaded: {len(routes_df)} rows")
    print(f"   Stops loaded:  {len(stops_df)} rows")

    # Step 2 — Summarize routes by type
    print("   Summarizing routes by type...")
    route_summary = summarize_routes(routes_df)

    # Step 3 — Assign stops to zones using coordinates
    print("    Assigning stops to zones using coordinates...")
    zone_stop_counts = assign_stops_to_zones(stops_df)

    # Step 4 — Find transit deserts
    threshold = 10
    valid_zones = {
        z: c for z, c in zone_stop_counts.items()
        if isinstance(c, int)
        and z != "unassigned"
        and z != "error"
    }

    sorted_zones = dict(sorted(valid_zones.items(), key=lambda x: x[1]))

    deserts = {
        z: c for z, c in sorted_zones.items()
        if c < threshold
    }

    print(f"    Transit deserts found: {len(deserts)} zones")
    print(f"   Stops per zone: {sorted_zones}")

    # Step 5 — Pre-build desert list
    desert_list = [
        {
            "area":       zone,
            "stop_count": count,
            "severity":   "Critical" if count < 5 else "High",
            "reason":     f"Only {count} stops serving this zone"
        }
        for zone, count in deserts.items()
    ]

    # If no deserts found use lowest zones
    if not desert_list:
        for zone, count in list(sorted_zones.items())[:2]:
            desert_list.append({
                "area":       zone,
                "stop_count": count,
                "severity":   "Medium",
                "reason":     f"Lowest stop density with {count} stops"
            })

    # Step 6 — Build prompt
    user_message = f"""
TORONTO TTC DATA:
- Total routes: {len(routes_df)}
- Bus routes: {route_summary.get('bus', 0)}
- Subway routes: {route_summary.get('subway', 0)}
- Streetcar routes: {route_summary.get('streetcar', 0)}
- Total stops analyzed: {len(stops_df)}

STOPS PER ZONE (sorted lowest to highest):
{json.dumps(sorted_zones, indent=2)}

TRANSIT DESERTS (pre-identified, fewer than {threshold} stops):
{json.dumps(desert_list, indent=2)}

Return this exact JSON filled with real values:

{{
    "transit_deserts": {json.dumps(desert_list)},
    "coverage_gaps": [
        {{"location": "{list(sorted_zones.keys())[0] if sorted_zones else 'Unknown'}", "issue": "describe the transit gap", "severity": "High"}},
        {{"location": "{list(sorted_zones.keys())[1] if len(sorted_zones) > 1 else 'Unknown'}", "issue": "describe the transit gap", "severity": "Medium"}}
    ],
    "recommendations": [
        "recommendation 1",
        "recommendation 2",
        "recommendation 3",
        "recommendation 4",
        "recommendation 5"
    ],
    "route_summary": {{
        "total_routes": {len(routes_df)},
        "bus_routes": {route_summary.get('bus', 0)},
        "subway_routes": {route_summary.get('subway', 0)},
        "streetcar_routes": {route_summary.get('streetcar', 0)},
        "total_stops": {len(stops_df)}
    }},
    "summary": "2 sentences about Toronto transit coverage"
}}
"""

    # Step 7 — Call AI
    print("   Sending to Mistral AI...")
    raw_response = ask_ai(SYSTEM_PROMPT, user_message, max_tokens=4000)

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
        print(f"   JSON parse failed: {e}")
        print(f"  ↳ Raw preview: {raw_response[:200]}")
        result = {
            "transit_deserts": desert_list,
            "coverage_gaps": [
                {
                    "location": z,
                    "issue":    f"Low stop coverage with {c} stops",
                    "severity": "High"
                }
                for z, c in list(sorted_zones.items())[:2]
            ],
            "recommendations": [
                f"Increase bus frequency in {list(sorted_zones.keys())[0]}",
                "Extend subway lines to underserved areas",
                "Add new bus routes to transit deserts",
                "Improve stop infrastructure in low coverage zones",
                "Review TTC coverage gaps annually",
            ],
            "route_summary": {
                "total_routes":     len(routes_df),
                "bus_routes":       route_summary.get("bus", 0),
                "subway_routes":    route_summary.get("subway", 0),
                "streetcar_routes": route_summary.get("streetcar", 0),
                "total_stops":      len(stops_df),
            },
            "summary": "Toronto transit coverage varies across zones.",
        }

    # Step 9 — Add metadata
    result["data_points"] = {
        "routes_analyzed": len(routes_df),
        "stops_analyzed":  len(stops_df),
        "zones_analyzed":  9,
    }

    print(" Mobility analysis complete!")
    return result


# ── Test ──────────────────────────────────────────────────────────────────────

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test 1 — Full mobility analysis
    print("=" * 50)
    print("TEST 1 — Full Mobility Analysis")
    print("=" * 50)
    result = run_mobility_agent(limit=99999)
    print("\n MOBILITY AGENT RESULT:")
    print(json.dumps(result, indent=2))

    # Test 2 — Neighbourhood transit query
    print("\n" + "=" * 50)
    print("TEST 2 — Neighbourhood Transit Query")
    print("=" * 50)
    areas = ["Mimico", "Danforth", "Scarborough"]
    for area in areas:
        result = get_neighbourhood_transit(area)
        print(f"\n📊 {area}:")
        print(f"   Matched: {result.get('neighbourhood')}")
        print(f"   Stops:   {result.get('stop_count')}")
        print(f"   Score:   {result.get('connectivity_score')}/10 ({result.get('rating')})")
        if result.get('sample_stops'):
            print(f"   Sample:  {result['sample_stops'][:3]}")