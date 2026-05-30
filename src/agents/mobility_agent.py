# mobility_agent.py
# ---------------------------------------------
# MOBILITY AGENT
# ---------------------------------------------
#
# PURPOSE:
# Analyzes TTC routes and stop data across all
# of Toronto to identify transit deserts and
# compute TTC connectivity scores for all 158
# Toronto neighbourhoods.
#
# HOW IT WORKS:
#   Run once with --store flag:
#     python src/agents/mobility_agent.py --store
#
#   This fetches TTC data from Toronto Open Data,
#   scores all 158 neighbourhoods, and saves to:
#     data/neighbourhood_transit.json
#
#   After that all agents read from that file.
#   No API calls needed.
#
# INPUT:
#   - TTC stops from Toronto Open Data (stops.txt)
#     columns: stop_id, stop_name, stop_lat, stop_lon
#
#   - TTC routes from Toronto Open Data (routes.txt)
#     columns: route_id, route_long_name, route_type
#       route_type: 0=streetcar, 1=subway, 3=bus
#
#   - Neighbourhood boundaries from Toronto Open Data
#     columns: AREA_NAME, geometry (polygon)
#
# OUTPUT FILE:
#   data/neighbourhood_transit.json
#   {
#     "Mimico-Queensway": {
#         "stop_count":         78,
#         "connectivity_score": 7.0,
#         "rating":             "Good",
#         "has_subway":         false,
#         "has_streetcar":      true,
#         "has_bus":            true,
#         "sample_stops":       ["stop name", ...],
#         "all_stops":          [{stop data}, ...]
#     },
#     ... all 158 neighbourhoods
#   }
#
# FUNCTIONS EXPORTED:
#   run_and_store_all_neighbourhoods()
#     -> fetch, score all 158, save to JSON
#
#   run_mobility_agent()
#     -> city-wide zone analysis
#
#   get_neighbourhood_transit(name)
#     -> score for one neighbourhood from JSON
#
#   load_all_neighbourhood_scores()
#     -> load all 158 scores from JSON
#     -> used by connectivity_agent, government_agent,
#        constructor_agent, user_agent
#
# SCORING (0-10):
#   Stop density    -> 0-4 points
#     > 200 stops  -> 4 pts
#     > 100 stops  -> 3 pts
#     > 50 stops   -> 2 pts
#     > 10 stops   -> 1 pt
#   Route diversity -> 0-6 points
#     subway       -> 3 pts
#     streetcar    -> 2 pts
#     bus          -> 1 pt
#
# USED BY:
#   - connectivity_agent.py -> combined TTC + road score
#   - government_agent.py   -> transit gap decisions
#   - constructor_agent.py  -> TTC score for build site
#   - user_agent.py         -> resident transit questions
# ---------------------------------------------

import json
import json as json_lib
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.services.toronto_data_service import (
    get_ttc_routes,
    get_ttc_stops,
    get_neighbourhoods,
)

# -- File paths ---------------------------------------------------------------

DATA_DIR        = os.path.join(os.path.dirname(__file__), "../../data")
NH_TRANSIT_FILE = os.path.join(DATA_DIR, "neighbourhood_transit.json")

# -- Helper functions ---------------------------------------------------------

def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    Ray casting algorithm.
    Check if a lat/lon point is inside a polygon.
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


def compute_score(stop_count: int,
                  has_subway: bool,
                  has_streetcar: bool,
                  has_bus: bool) -> tuple:
    """
    Calculate TTC connectivity score (0-10) and rating.
    Returns (score, rating).
    """
    score = 0

    if stop_count > 200:   score += 4
    elif stop_count > 100: score += 3
    elif stop_count > 50:  score += 2
    elif stop_count > 10:  score += 1

    if has_subway:    score += 3
    if has_streetcar: score += 2
    if has_bus:       score += 1

    score = min(round(score, 1), 10)

    if score >= 8:   rating = "Excellent"
    elif score >= 6: rating = "Good"
    elif score >= 4: rating = "Fair"
    elif score >= 2: rating = "Poor"
    else:            rating = "Critical"

    return score, rating


def extract_polygon(nb_row) -> list:
    """
    Extract outer ring polygon from a neighbourhood row.
    Handles both Polygon and MultiPolygon geometry types.
    Returns list of [lon, lat] pairs or None.
    """
    try:
        geo      = json_lib.loads(str(nb_row["geometry"]))
        geo_type = geo["type"]
        if geo_type == "MultiPolygon":
            return geo["coordinates"][0][0]
        else:
            return geo["coordinates"][0]
    except Exception:
        return None


# -- Data summarizers ---------------------------------------------------------

def summarize_routes(df: pd.DataFrame) -> dict:
    """
    Count TTC routes by type.
    route_type: 0=streetcar, 1=subway, 3=bus
    """
    df.columns  = [c.lower() for c in df.columns]
    type_col    = "route_type" if "route_type" in df.columns else None
    route_types = {0: "streetcar", 1: "subway", 3: "bus"}
    counts      = {
        "total":     len(df),
        "bus":       0,
        "subway":    0,
        "streetcar": 0,
        "other":     0,
    }
    if type_col:
        for _, row in df.iterrows():
            t     = int(row[type_col]) if str(row[type_col]).isdigit() else -1
            label = route_types.get(t, "other")
            counts[label] = counts.get(label, 0) + 1
    return counts


def assign_stops_to_zones(df: pd.DataFrame) -> dict:
    """
    Assign TTC stops to Toronto zones using
    stop_lat and stop_lon coordinates.
    Divides Toronto into a 3x3 grid automatically.

    Zone grid (South = lower lat, North = higher lat):
    +-----------+---------------+-----------+
    | North West | North Central | North East |
    +-----------+---------------+-----------+
    | Mid West   |    Central    |  Mid East  |
    +-----------+---------------+-----------+
    | South West | South Central | South East |
    +-----------+---------------+-----------+
    """
    df.columns = [c.lower() for c in df.columns]
    lat_col    = "stop_lat" if "stop_lat" in df.columns else None
    lon_col    = "stop_lon" if "stop_lon" in df.columns else None

    if not lat_col or not lon_col:
        return {"error": "No stop_lat/stop_lon columns found"}

    coords = []
    for _, row in df.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            if lat > 0 and lon < 0:
                coords.append((lat, lon))
        except Exception:
            continue

    if not coords:
        return {"error": "No valid coordinates found"}

    lats     = [c[0] for c in coords]
    lons     = [c[1] for c in coords]
    min_lat  = min(lats)
    max_lat  = max(lats)
    min_lon  = min(lons)
    max_lon  = max(lons)
    lat_step = (max_lat - min_lat) / 3
    lon_step = (max_lon - min_lon) / 3

    zone_names = {
        (0, 0): "South West",   (0, 1): "South Central", (0, 2): "South East",
        (1, 0): "Mid West",     (1, 1): "Central",        (1, 2): "Mid East",
        (2, 0): "North West",   (2, 1): "North Central",  (2, 2): "North East",
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
    return zone_counts


# -- Store output for all neighbourhoods --------------------------------------

def run_and_store_all_neighbourhoods():
    """
    Run mobility analysis for ALL 158 Toronto neighbourhoods.
    Fetches TTC data from Toronto Open Data API.
    Scores each neighbourhood using point-in-polygon.
    Saves all results to data/neighbourhood_transit.json.

    Run this ONCE. After that all agents read from the file.
    Takes about 2-3 minutes.

    Returns:
        dict of all 158 neighbourhood scores
    """
    print("=" * 60)
    print("MOBILITY AGENT -- SCORING ALL 158 NEIGHBOURHOODS")
    print("=" * 60)

    # Step 1 -- Fetch data
    print("\nStep 1 -- Fetching TTC stops...")
    stops_df = get_ttc_stops(limit=99999)
    print(f"  Stops: {len(stops_df)}")

    print("\nStep 2 -- Fetching TTC routes...")
    routes_df = get_ttc_routes(limit=99999)
    print(f"  Routes: {len(routes_df)}")

    print("\nStep 3 -- Fetching neighbourhood boundaries...")
    nb_df = get_neighbourhoods(limit=200)
    print(f"  Neighbourhoods: {len(nb_df)}")

    # Step 2 -- Prepare stops for fast iteration
    stops_df.columns  = [c.lower() for c in stops_df.columns]
    routes_df.columns = [c.lower() for c in routes_df.columns]

    stop_coords = []
    for _, stop in stops_df.iterrows():
        try:
            stop_coords.append((
                float(stop["stop_lat"]),
                float(stop["stop_lon"]),
                str(stop["stop_name"]),
                str(stop["stop_id"]),
            ))
        except Exception:
            continue

    print(f"\n  Stops prepared: {len(stop_coords)}")

    # Step 3 -- Route type flags
    route_type_counts = routes_df["route_type"].value_counts().to_dict()
    has_subway        = 1 in route_type_counts
    has_streetcar     = 0 in route_type_counts
    has_bus           = 3 in route_type_counts

    print(f"  Subway: {has_subway} | Streetcar: {has_streetcar} | Bus: {has_bus}")

    # Step 4 -- Score all neighbourhoods
    all_results = {}
    total       = len(nb_df)

    print(f"\nStep 4 -- Scoring {total} neighbourhoods...\n")

    for idx, (_, nb_row) in enumerate(nb_df.iterrows()):
        nb_name = str(nb_row["AREA_NAME"])
        print(f"  [{idx + 1}/{total}] {nb_name}")

        polygon = extract_polygon(nb_row)
        if polygon is None:
            all_results[nb_name] = {"error": "Could not parse geometry"}
            continue

        stops_inside = []
        for lat, lon, stop_name, stop_id in stop_coords:
            if point_in_polygon(lat, lon, polygon):
                stops_inside.append({
                    "stop_id":   stop_id,
                    "stop_name": stop_name,
                    "stop_lat":  lat,
                    "stop_lon":  lon,
                })

        stop_count    = len(stops_inside)
        score, rating = compute_score(
            stop_count, has_subway, has_streetcar, has_bus
        )

        all_results[nb_name] = {
            "stop_count":         stop_count,
            "connectivity_score": score,
            "rating":             rating,
            "has_subway":         has_subway,
            "has_streetcar":      has_streetcar,
            "has_bus":            has_bus,
            "sample_stops":       [s["stop_name"] for s in stops_inside[:10]],
            "all_stops":          stops_inside,
        }

        print(f"    Stops: {stop_count} | Score: {score}/10 ({rating})")

    # Step 5 -- Save to file
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(NH_TRANSIT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    size_kb = os.path.getsize(NH_TRANSIT_FILE) / 1024
    print(f"\nSaved neighbourhood_transit.json")
    print(f"  Neighbourhoods: {len(all_results)}")
    print(f"  File size:      {size_kb:.0f} KB")
    print(f"  Path:           {os.path.abspath(NH_TRANSIT_FILE)}")

    # Summary
    valid_scores = {k: v for k, v in all_results.items() if "error" not in v}
    sorted_scores = sorted(
        valid_scores.items(),
        key=lambda x: x[1]["connectivity_score"],
        reverse=True
    )

    print("\nTop 5 best connected:")
    for name, data in sorted_scores[:5]:
        print(f"  {name}: {data['connectivity_score']}/10 "
              f"({data['rating']}) -- {data['stop_count']} stops")

    print("\nBottom 5 worst connected:")
    for name, data in sorted_scores[-5:]:
        print(f"  {name}: {data['connectivity_score']}/10 "
              f"({data['rating']}) -- {data['stop_count']} stops")

    print("\nDone. All agents can now read from neighbourhood_transit.json")
    return all_results


# -- Load stored scores -------------------------------------------------------

def load_all_neighbourhood_scores() -> dict:
    """
    Load all 158 stored neighbourhood transit scores from JSON.

    Used by:
        connectivity_agent.py
        government_agent.py
        constructor_agent.py
        user_agent.py

    Returns:
        {
            "Mimico-Queensway":         { score data },
            "Playter Estates-Danforth": { score data },
            ...
        }

    Raises:
        FileNotFoundError if not stored yet.
        Run: python src/agents/mobility_agent.py --store
    """
    if not os.path.exists(NH_TRANSIT_FILE):
        raise FileNotFoundError(
            "neighbourhood_transit.json not found.\n"
            "Run: python src/agents/mobility_agent.py --store"
        )
    with open(NH_TRANSIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# -- Get single neighbourhood -------------------------------------------------

def get_neighbourhood_transit(neighbourhood_name: str) -> dict:
    """
    Get TTC connectivity score for one neighbourhood.
    Reads from stored JSON -- no API call.

    Args:
        neighbourhood_name: e.g. "Mimico", "Danforth"
                            Partial name match supported.

    Returns:
        {
            "neighbourhood":      "Mimico-Queensway",
            "search_term":        "Mimico",
            "stop_count":         78,
            "connectivity_score": 7.0,
            "rating":             "Good",
            "has_subway":         False,
            "has_streetcar":      True,
            "has_bus":            True,
            "sample_stops":       [...]
        }
    """
    print(f"Finding TTC connectivity for: {neighbourhood_name}")

    scores = load_all_neighbourhood_scores()
    search = neighbourhood_name.lower().strip()

    # Exact match
    for name, data in scores.items():
        if name.lower() == search:
            print(f"  Matched (exact): {name}")
            return {
                "neighbourhood": name,
                "search_term":   neighbourhood_name,
                **{k: v for k, v in data.items() if k != "all_stops"},
            }

    # Partial match
    for name, data in scores.items():
        if search in name.lower():
            print(f"  Matched (partial): {name}")
            return {
                "neighbourhood": name,
                "search_term":   neighbourhood_name,
                **{k: v for k, v in data.items() if k != "all_stops"},
            }

    available = list(scores.keys())
    return {
        "error":      f"No neighbourhood matching '{neighbourhood_name}'",
        "suggestion": "Check available names below",
        "available":  available[:20],
    }


# -- City-wide analysis -------------------------------------------------------

def run_mobility_agent() -> dict:
    """
    City-wide TTC analysis.
    Identifies transit deserts across all of Toronto.
    Reads neighbourhood scores from stored JSON.

    Returns:
        {
            "transit_deserts":        [...],
            "coverage_gaps":          [...],
            "recommendations":        [...],
            "route_summary":          {...},
            "zone_breakdown":         {...},
            "neighbourhood_rankings": {...},
            "summary":                "..."
        }
    """
    print("[Mobility Agent] Starting city-wide analysis...")

    # Load stored neighbourhood scores
    scores = load_all_neighbourhood_scores()
    print(f"  Neighbourhood scores loaded: {len(scores)}")

    # Fetch raw data for zone grid analysis
    print("  Fetching TTC data for zone analysis...")
    stops_df  = get_ttc_stops(limit=99999)
    routes_df = get_ttc_routes(limit=99999)
    print(f"  Stops: {len(stops_df)} | Routes: {len(routes_df)}")

    route_summary    = summarize_routes(routes_df)
    zone_stop_counts = assign_stops_to_zones(stops_df)

    threshold   = 10
    valid_zones = {
        z: c for z, c in zone_stop_counts.items()
        if isinstance(c, int) and z not in ("unassigned", "error")
    }
    sorted_zones = dict(sorted(valid_zones.items(), key=lambda x: x[1]))
    deserts      = {z: c for z, c in sorted_zones.items() if c < threshold}

    print(f"  Transit deserts: {len(deserts)}")
    print(f"  Stops per zone:  {sorted_zones}")

    desert_list = [
        {
            "area":       zone,
            "stop_count": count,
            "severity":   "Critical" if count < 5 else "High",
            "reason":     f"Only {count} stops serving this zone",
        }
        for zone, count in deserts.items()
    ]

    if not desert_list:
        for zone, count in list(sorted_zones.items())[:2]:
            desert_list.append({
                "area":       zone,
                "stop_count": count,
                "severity":   "Medium",
                "reason":     f"Lowest stop density with {count} stops",
            })

    # Neighbourhood rankings
    valid_scores = {k: v for k, v in scores.items() if "error" not in v}
    sorted_nb    = sorted(
        valid_scores.items(),
        key=lambda x: x[1].get("connectivity_score", 0),
        reverse=True
    )

    best_5  = [
        {"neighbourhood": k, "score": v["connectivity_score"],
         "rating": v["rating"], "stops": v["stop_count"]}
        for k, v in sorted_nb[:5]
    ]
    worst_5 = [
        {"neighbourhood": k, "score": v["connectivity_score"],
         "rating": v["rating"], "stops": v["stop_count"]}
        for k, v in sorted_nb[-5:]
    ]

    worst_zone  = list(sorted_zones.keys())[0] if sorted_zones else "Unknown"
    second_zone = (
        list(sorted_zones.keys())[1] if len(sorted_zones) > 1 else "Unknown"
    )

    result = {
        "transit_deserts": desert_list,
        "coverage_gaps": [
            {
                "location": worst_zone,
                "issue":    f"Only {sorted_zones.get(worst_zone, 0)} stops "
                            f"significantly below city average",
                "severity": "High",
            },
            {
                "location": second_zone,
                "issue":    f"Only {sorted_zones.get(second_zone, 0)} stops "
                            f"second lowest in Toronto",
                "severity": "Medium",
            },
        ],
        "recommendations": [
            f"Expand bus routes to {worst_zone} -- only "
            f"{sorted_zones.get(worst_zone, 0)} stops currently",
            f"Increase stop density in {second_zone}",
            "Extend subway lines to underserved outer areas",
            "Implement on-demand microtransit in low-density zones",
            "Review TTC coverage gaps annually",
        ],
        "route_summary": {
            "total_routes":     len(routes_df),
            "bus_routes":       route_summary.get("bus", 0),
            "subway_routes":    route_summary.get("subway", 0),
            "streetcar_routes": route_summary.get("streetcar", 0),
            "total_stops":      len(stops_df),
        },
        "zone_breakdown": sorted_zones,
        "neighbourhood_rankings": {
            "best_connected":  best_5,
            "worst_connected": worst_5,
        },
        "summary": (
            f"Toronto TTC operates {len(routes_df)} routes with "
            f"{len(stops_df)} stops. "
            f"{worst_zone} zone has the lowest coverage with "
            f"{sorted_zones.get(worst_zone, 0)} stops."
        ),
        "data_points": {
            "routes_analyzed":       len(routes_df),
            "stops_analyzed":        len(stops_df),
            "zones_analyzed":        9,
            "neighbourhoods_scored": len(scores),
        },
    }

    print("[Mobility Agent] Analysis complete.")
    return result


# -- Run ----------------------------------------------------------------------

if __name__ == "__main__":

    if "--store" in sys.argv:
        run_and_store_all_neighbourhoods()

    else:
        print("=" * 60)
        print("TEST 1 - City-wide Mobility Analysis")
        print("=" * 60)
        result = run_mobility_agent()
        print(json.dumps(result, indent=2))

        print("\n" + "=" * 60)
        print("TEST 2 - Specific Neighbourhood Queries")
        print("=" * 60)
        areas = ["Mimico", "Danforth", "Scarborough", "Downtown", "Etobicoke"]
        for area in areas:
            r = get_neighbourhood_transit(area)
            print(f"\n{area}:")
            print(f"  Matched: {r.get('neighbourhood')}")
            print(f"  Stops:   {r.get('stop_count')}")
            print(f"  Score:   {r.get('connectivity_score')}/10 "
                  f"({r.get('rating')})")
            if r.get("sample_stops"):
                print(f"  Sample:  {r['sample_stops'][:2]}")

        print("\n" + "=" * 60)
        print("TEST 3 - All Neighbourhood Rankings")
        print("=" * 60)
        all_scores = load_all_neighbourhood_scores()
        sorted_all = sorted(
            {k: v for k, v in all_scores.items() if "error" not in v}.items(),
            key=lambda x: x[1].get("connectivity_score", 0),
            reverse=True
        )
        print(f"Total neighbourhoods: {len(all_scores)}")
        print("\nTop 5 best connected:")
        for name, data in sorted_all[:5]:
            print(f"  {name}: {data['connectivity_score']}/10 "
                  f"({data['rating']}) -- {data['stop_count']} stops")
        print("\nBottom 5 worst connected:")
        for name, data in sorted_all[-5:]:
            print(f"  {name}: {data['connectivity_score']}/10 "
                  f"({data['rating']}) -- {data['stop_count']} stops")
