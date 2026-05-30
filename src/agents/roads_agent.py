# roads_agent.py
# ---------------------------------------------
# ROADS AGENT
# ---------------------------------------------
#
# PURPOSE:
# Analyzes Toronto Centreline road network data
# to assess road connectivity for any neighbourhood.
#
# INPUT:
#   - roads DataFrame from get_roads()
#     columns used:
#       LINEAR_NAME_FULL  -> road name e.g. "Lake Shore Blvd W"
#       FEATURE_CODE_DESC -> road type e.g. "Major Arterial"
#       geometry          -> LineString coordinates
#
#   - neighbourhood_name -> e.g. "Mimico", "Scarborough"
#
# WHAT IT DOES STEP BY STEP:
#   Step 1 -> Load road centreline data (5000 segments)
#   Step 2 -> Load neighbourhood boundary polygon
#   Step 3 -> Get midpoint of each road segment
#   Step 4 -> Check if midpoint falls inside polygon
#   Step 5 -> Count roads by type
#   Step 6 -> Calculate road connectivity score 0-10
#   Step 7 -> Determine truck access rating
#   Step 8 -> Return structured result
#
# ROAD TYPES IN DATASET:
#   Expressway        -> highway level (401, 427, Gardiner)
#   Major Arterial    -> main city roads (Bloor, Eglinton)
#   Minor Arterial    -> secondary roads (Royal York, Kipling)
#   Collector         -> connects local to arterial
#   Local             -> residential streets
#   Laneway           -> back lanes
#
# SCORING (0-10):
#   Road density      -> 0-4 points (how many roads)
#   Road type quality -> 0-6 points (what types of roads)
#
# OUTPUT:
#   {
#     "neighbourhood":  "Mimico-Queensway",
#     "total_roads":    42,
#     "road_types":     {"Local": 28, "Major Arterial": 6},
#     "main_roads":     ["Lake Shore Blvd W", "Royal York Rd"],
#     "has_expressway": False,
#     "has_arterial":   True,
#     "road_score":     7.5,
#     "road_rating":    "Good",
#     "truck_access":   "Good"
#   }
#
# USED BY:
#   - connectivity_agent.py -> combined TTC + road score
#   - constructor_agent.py  -> can trucks reach the site?
#   - government_agent.py   -> which areas need new roads?
#   - user_agent.py         -> road quality questions
# ---------------------------------------------

import json
import json as json_lib
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.services.toronto_data_service import get_roads, get_neighbourhoods

# -- Road type weights for scoring --------------------------------------------

ROAD_WEIGHTS = {
    "Expressway":          4,
    "Major Arterial":      3,
    "Minor Arterial":      2,
    "Collector":           2,
    "Major Arterial Ramp": 1,
    "Expressway Ramp":     1,
    "Local":               0.5,
    "Laneway":             0.1,
}

# -- Helper functions ---------------------------------------------------------

def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    Ray casting algorithm to check if a point
    is inside a polygon.
    polygon is a list of [lon, lat] coordinate pairs.
    """
    n      = len(polygon)
    inside = False
    x, y   = lon, lat
    j      = n - 1

    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def get_road_midpoint(geometry_str: str):
    """
    Extract midpoint lat/lon from a road segment LineString.
    LineString format: {"type": "LineString",
                        "coordinates": [[lon,lat], [lon,lat]]}
    Returns (lat, lon) or None if parsing fails.
    """
    try:
        geo    = json_lib.loads(str(geometry_str))
        coords = geo["coordinates"]
        mid    = coords[len(coords) // 2]
        return float(mid[1]), float(mid[0])
    except Exception:
        return None


def extract_polygon(neighbourhood_name: str, nb_df: pd.DataFrame):
    """
    Find neighbourhood boundary polygon by name.
    Handles both Polygon and MultiPolygon geometry types.
    Returns (matched_name, polygon) or (None, None).
    """
    name_col = "AREA_NAME" if "AREA_NAME" in nb_df.columns else None
    if not name_col:
        return None, None

    matches = nb_df[
        nb_df[name_col].str.lower().str.contains(
            neighbourhood_name.lower(), na=False
        )
    ]

    if matches.empty:
        return None, None

    match = matches.iloc[0]

    try:
        geo      = json_lib.loads(str(match["geometry"]))
        geo_type = geo["type"]

        if geo_type == "MultiPolygon":
            polygon = geo["coordinates"][0][0]
        else:
            polygon = geo["coordinates"][0]

        return match[name_col], polygon

    except Exception:
        return None, None


# -- Main functions -----------------------------------------------------------

def get_neighbourhood_roads(neighbourhood_name: str,
                            roads_df: pd.DataFrame = None,
                            nb_df: pd.DataFrame = None) -> dict:
    """
    Get road connectivity score for any Toronto neighbourhood.

    Args:
        neighbourhood_name : e.g. "Mimico", "Scarborough"
        roads_df           : Optional pre-loaded roads DataFrame
        nb_df              : Optional pre-loaded neighbourhoods DataFrame

    Returns:
        {
            "neighbourhood":  "Mimico-Queensway",
            "total_roads":    42,
            "road_types":     {...},
            "main_roads":     [...],
            "has_expressway": False,
            "has_arterial":   True,
            "road_score":     7.5,
            "road_rating":    "Good",
            "truck_access":   "Good"
        }
    """
    print(f"Finding road connectivity for: {neighbourhood_name}")

    # Step 1 -- Load data if not passed in
    if roads_df is None:
        roads_df = get_roads(limit=5000)
    if nb_df is None:
        nb_df = get_neighbourhoods(limit=200)

    # Step 2 -- Find neighbourhood polygon
    matched_name, polygon = extract_polygon(neighbourhood_name, nb_df)

    if not matched_name:
        return {
            "error":      f"Neighbourhood '{neighbourhood_name}' not found",
            "suggestion": "Try: Mimico, Danforth, Scarborough, Etobicoke"
        }

    print(f"  Matched: {matched_name}")

    # Step 3 -- Find roads inside polygon
    road_types   = {}
    main_roads   = []
    roads_inside = []

    for _, road in roads_df.iterrows():
        midpoint = get_road_midpoint(road["geometry"])
        if not midpoint:
            continue

        lat, lon = midpoint
        if not point_in_polygon(lat, lon, polygon):
            continue

        road_name = str(road["LINEAR_NAME_FULL"])
        road_type = str(road["FEATURE_CODE_DESC"])

        roads_inside.append({
            "name": road_name,
            "type": road_type,
        })

        road_types[road_type] = road_types.get(road_type, 0) + 1

        if road_type in ["Expressway", "Major Arterial", "Minor Arterial"]:
            if road_name not in main_roads:
                main_roads.append(road_name)

    total_roads = len(roads_inside)
    print(f"  Roads found: {total_roads}")
    print(f"  Road types: {road_types}")

    # Step 4 -- Calculate road score (0-10)
    score = 0.0

    if total_roads > 100:  score += 4
    elif total_roads > 50: score += 3
    elif total_roads > 20: score += 2
    elif total_roads > 5:  score += 1

    for road_type, weight in ROAD_WEIGHTS.items():
        if road_types.get(road_type, 0) > 0:
            score += min(weight, 2)

    score = min(round(score, 1), 10)

    # Step 5 -- Ratings
    if score >= 8:   rating = "Excellent"
    elif score >= 6: rating = "Good"
    elif score >= 4: rating = "Fair"
    elif score >= 2: rating = "Poor"
    else:            rating = "Critical"

    # Step 6 -- Truck access
    has_expressway = "Expressway" in road_types
    has_arterial   = any(
        t in road_types
        for t in ["Major Arterial", "Minor Arterial"]
    )

    if has_expressway:     truck_access = "Excellent"
    elif has_arterial:     truck_access = "Good"
    elif total_roads > 10: truck_access = "Fair"
    else:                  truck_access = "Poor"

    return {
        "neighbourhood":  matched_name,
        "search_term":    neighbourhood_name,
        "total_roads":    total_roads,
        "road_types":     road_types,
        "main_roads":     main_roads[:10],
        "has_expressway": has_expressway,
        "has_arterial":   has_arterial,
        "road_score":     score,
        "road_rating":    rating,
        "truck_access":   truck_access,
    }


def run_roads_agent(limit: int = 5000) -> dict:
    """
    City-wide road network summary.
    Used by government_agent.py for city-wide analysis.
    """
    print("[Roads Agent] Starting city-wide analysis...")

    roads_df    = get_roads(limit)
    type_counts = roads_df["FEATURE_CODE_DESC"].value_counts().to_dict()

    expressways = roads_df[
        roads_df["FEATURE_CODE_DESC"] == "Expressway"
    ]["LINEAR_NAME_FULL"].unique().tolist()

    arterials = roads_df[
        roads_df["FEATURE_CODE_DESC"].isin(
            ["Major Arterial", "Minor Arterial"]
        )
    ]["LINEAR_NAME_FULL"].unique().tolist()

    print(f"  Total road segments: {len(roads_df)}")

    return {
        "total_segments": len(roads_df),
        "road_types":     type_counts,
        "expressways":    expressways[:20],
        "arterials":      arterials[:20],
        "data_points": {
            "segments_analyzed": len(roads_df),
        }
    }


# -- Test ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1 - City-wide Road Summary")
    print("=" * 50)
    summary = run_roads_agent()
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 50)
    print("TEST 2 - Neighbourhood Road Connectivity")
    print("=" * 50)
    areas = ["Mimico", "Danforth", "Scarborough"]
    for area in areas:
        result = get_neighbourhood_roads(area)
        print(f"\n{area}:")
        print(f"  Matched:    {result.get('neighbourhood')}")
        print(f"  Roads:      {result.get('total_roads')}")
        print(f"  Score:      {result.get('road_score')}/10 "
              f"({result.get('road_rating')})")
        print(f"  Truck:      {result.get('truck_access')}")
        print(f"  Main roads: {result.get('main_roads', [])[:3]}")
        print(f"  Types:      {result.get('road_types')}")