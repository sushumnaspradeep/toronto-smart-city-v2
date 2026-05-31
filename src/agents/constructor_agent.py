# constructor_agent.py
# ---------------------------------------------
# CONSTRUCTOR AGENT
# ---------------------------------------------
#
# Provides site assessment for developers and
# constructors planning to build in Toronto.
#
# INPUT:
#   data/building_permits_summary.json
#   data/building_permits_coords.json
#   data/cleared_permits_summary.json
#   data/cleared_permits_coords.json
#   data/neighbourhood_transit.json
#   data/urban_profiles_payload.json
#   data/lookup_boundaries.geojson
#
# KEY FUNCTIONS:
#   get_construction_profile(area_name)
#   get_ward_for_address(address)
#   ask_constructor_question(question)
#   get_cost_estimate(structure_type)
#   get_builder_performance()
# ---------------------------------------------

import os
import sys
import json
import math
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.agents.mobility_agent   import load_all_neighbourhood_scores
from src.agents.government_agent import (
    load_urban_profiles,
    get_investment_priorities,
)
from src.services.openai_service import ask_ai

try:
    from shapely.geometry import shape, Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("  WARNING: shapely not installed. pip install shapely")

# =============================================================================
# FILE PATHS
# =============================================================================

DATA_DIR        = os.path.join(os.path.dirname(__file__), "../../data")
PERMITS_SUMMARY = os.path.join(DATA_DIR, "building_permits_summary.json")
PERMITS_COORDS  = os.path.join(DATA_DIR, "building_permits_coords.json")
CLEARED_SUMMARY = os.path.join(DATA_DIR, "cleared_permits_summary.json")
CLEARED_COORDS  = os.path.join(DATA_DIR, "cleared_permits_coords.json")
WARD_BOUNDARIES = os.path.join(DATA_DIR, "lookup_boundaries.geojson")

# =============================================================================
# NEIGHBOURHOOD -> FSA MAPPING
# Used when area name does not appear in permit addresses
# =============================================================================

NEIGHBOURHOOD_FSA = {
    "mimico":          ["M8V"],
    "new toronto":     ["M8V", "M8Z"],
    "long branch":     ["M8W"],
    "alderwood":       ["M8W", "M8Z"],
    "lakeshore":       ["M8V", "M8W", "M8X", "M8Y", "M8Z"],
    "etobicoke":       ["M8V", "M8W", "M8X", "M8Y", "M8Z",
                        "M9A", "M9B", "M9C", "M9P", "M9R", "M9V", "M9W"],
    "rexdale":         ["M9V", "M9W"],
    "kipling":         ["M8Z", "M9A"],
    "islington":       ["M9A", "M9B"],
    "bloor west":      ["M6S", "M6R"],
    "junction":        ["M6N", "M6P"],
    "swansea":         ["M6S"],
    "high park":       ["M6P", "M6R"],
    "parkdale":        ["M6K", "M6R"],
    "liberty village": ["M6K"],
    "little portugal": ["M6J", "M6K"],
    "little italy":    ["M6G", "M6H"],
    "corso italia":    ["M6E", "M6H"],
    "kensington":      ["M5T"],
    "annex":           ["M5R", "M5S"],
    "forest hill":     ["M5P"],
    "midtown":         ["M4S", "M4T", "M4V", "M5N"],
    "davisville":      ["M4S"],
    "rosedale":        ["M4W"],
    "moore park":      ["M4T"],
    "lawrence":        ["M3H", "M3L", "M4N"],
    "willowdale":      ["M2M", "M2N"],
    "north york":      ["M2M", "M2N", "M3A", "M3B", "M3C"],
    "don mills":       ["M3A", "M3B", "M3C"],
    "flemingdon":      ["M3C"],
    "east york":       ["M4B", "M4C", "M4J"],
    "leslieville":     ["M4M"],
    "beaches":         ["M4E", "M4L"],
    "danforth":        ["M4J", "M4K"],
    "riverdale":       ["M4J", "M4K"],
    "scarborough":     ["M1B", "M1C", "M1E", "M1G", "M1H",
                        "M1J", "M1K", "M1L", "M1M", "M1N",
                        "M1P", "M1R", "M1S", "M1T", "M1V",
                        "M1W", "M1X"],
    "agincourt":       ["M1S", "M1T", "M1V"],
    "malvern":         ["M1B"],
    "rouge":           ["M1X"],
    "weston":          ["M9L", "M9M", "M9N"],
    "mount dennis":    ["M6M", "M9M"],
    "york":            ["M6E", "M6M"],
    "downtown":        ["M5A", "M5B", "M5C", "M5E", "M5G",
                        "M5H", "M5J", "M5K", "M5L", "M5V", "M5X"],
}

# =============================================================================
# NEIGHBOURHOOD -> WARD MAPPING
# =============================================================================

NEIGHBOURHOOD_TO_WARD = {
    "mimico":          "etobicoke-lakeshore",
    "lakeshore":       "etobicoke-lakeshore",
    "long branch":     "etobicoke-lakeshore",
    "new toronto":     "etobicoke-lakeshore",
    "alderwood":       "etobicoke-lakeshore",
    "swansea":         "etobicoke-lakeshore",
    "islington":       "etobicoke-centre",
    "kipling":         "etobicoke-centre",
    "bloor west":      "etobicoke-centre",
    "rexdale":         "etobicoke-north",
    "willowdale":      "willowdale",
    "north york":      "willowdale",
    "scarborough":     "scarborough-centre",
    "agincourt":       "scarborough-agincourt",
    "malvern":         "scarborough-rouge park",
    "beaches":         "beaches-east york",
    "leslieville":     "toronto-danforth",
    "danforth":        "toronto-danforth",
    "riverdale":       "toronto-danforth",
    "parkdale":        "parkdale-high park",
    "high park":       "parkdale-high park",
    "junction":        "davenport",
    "davenport":       "davenport",
    "little italy":    "davenport",
    "corso italia":    "york south-weston",
    "york":            "york south-weston",
    "weston":          "york south-weston",
    "mount dennis":    "humber river-black creek",
    "humber":          "humber river-black creek",
    "annex":           "university-rosedale",
    "rosedale":        "university-rosedale",
    "kensington":      "spadina-fort york",
    "liberty village": "spadina-fort york",
    "downtown":        "spadina-fort york",
    "forest hill":     "toronto-st pauls",
    "midtown":         "toronto-st pauls",
    "lawrence":        "eglinton-lawrence",
    "don mills":       "don valley east",
    "flemingdon":      "don valley east",
    "east york":       "beaches-east york",
}

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_permits_summary() -> dict:
    if not os.path.exists(PERMITS_SUMMARY):
        return {}
    with open(PERMITS_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)

def load_permits_coords() -> dict:
    if not os.path.exists(PERMITS_COORDS):
        return {}
    with open(PERMITS_COORDS, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cleared_summary() -> dict:
    if not os.path.exists(CLEARED_SUMMARY):
        return {}
    with open(CLEARED_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cleared_coords() -> dict:
    if not os.path.exists(CLEARED_COORDS):
        return {}
    with open(CLEARED_COORDS, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ward_boundaries() -> dict:
    if not os.path.exists(WARD_BOUNDARIES):
        return {}
    with open(WARD_BOUNDARIES, "r", encoding="utf-8") as f:
        return json.load(f)

# =============================================================================
# SPATIAL HELPERS
# =============================================================================

def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def get_ward_for_coords(lat: float, lon: float) -> dict:
    """
    Spatial lookup: find which ward polygon contains (lat, lon).
    Requires shapely.
    """
    if not SHAPELY_AVAILABLE:
        return {"ward_name": "Unknown", "ward_number": "Unknown", "found": False}

    boundaries = load_ward_boundaries()
    point      = Point(lon, lat)  # shapely: (lon, lat)

    for feature in boundaries.get("features", []):
        props    = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        try:
            polygon = shape(geometry)
            if polygon.contains(point):
                return {
                    "ward_name":   props.get("WARD_NAME", "Unknown"),
                    "ward_number": props.get("WARD_NUMBER", "Unknown"),
                    "found":       True,
                }
        except Exception:
            continue

    return {"ward_name": "Unknown", "ward_number": "Unknown", "found": False}


def get_area_centre(area_name: str) -> tuple:
    """
    Calculate geographic centre of an area from permit coordinates.

    Strategy:
    1. Search permit addresses for area name substring
    2. If no address match, use NEIGHBOURHOOD_FSA mapping
    3. Calculate centroid with IQR outlier removal
    4. Fallback: Toronto city centre
    """
    all_coords: dict = {}
    all_coords.update(load_cleared_coords())
    all_coords.update(load_permits_coords())

    area_lower = area_name.lower().strip()
    matching   = []

    # Pass 1 -- area name in address
    for data in all_coords.values():
        address = data.get("address", "").lower()
        lat     = data.get("lat")
        lon     = data.get("lon")
        if lat and lon and area_lower in address:
            matching.append((lat, lon))

    # Pass 2 -- FSA mapping
    if not matching:
        target_fsas: set = set()
        for key, fsas in NEIGHBOURHOOD_FSA.items():
            if key in area_lower or area_lower in key:
                target_fsas.update(fsas)

        if target_fsas:
            for data in all_coords.values():
                fsa = data.get("fsa", "")
                lat = data.get("lat")
                lon = data.get("lon")
                if lat and lon and fsa in target_fsas:
                    matching.append((lat, lon))

    if matching:
        # IQR outlier removal
        lats = sorted(p[0] for p in matching)
        lons = sorted(p[1] for p in matching)
        n    = len(lats)

        if n >= 4:
            q1_lat, q3_lat = lats[n // 4], lats[3 * n // 4]
            q1_lon, q3_lon = lons[n // 4], lons[3 * n // 4]
            iqr_lat = q3_lat - q1_lat
            iqr_lon = q3_lon - q1_lon
            filtered = [
                p for p in matching
                if (q1_lat - 1.5 * iqr_lat) <= p[0] <= (q3_lat + 1.5 * iqr_lat)
                and (q1_lon - 1.5 * iqr_lon) <= p[1] <= (q3_lon + 1.5 * iqr_lon)
            ]
            matching = filtered if filtered else matching

        avg_lat = sum(p[0] for p in matching) / len(matching)
        avg_lon = sum(p[1] for p in matching) / len(matching)
        print(f"  Centre '{area_name}': {avg_lat:.4f}N, {avg_lon:.4f}W "
              f"({len(matching)} permits)")
        return avg_lat, avg_lon

    print(f"  WARNING: No data for '{area_name}'. Using Toronto centre.")
    return 43.6532, -79.3832

# =============================================================================
# PERMIT QUERIES
# =============================================================================

def get_nearby_permits(lat: float, lon: float,
                       radius_km: float = 2.0) -> list:
    """Active permits within radius_km."""
    coords = load_permits_coords()
    nearby = []
    for permit_id, data in coords.items():
        p_lat = data.get("lat")
        p_lon = data.get("lon")
        if not p_lat or not p_lon:
            continue
        dist = haversine_km(lat, lon, p_lat, p_lon)
        if dist <= radius_km:
            nearby.append({
                "permit_id":   permit_id,
                "address":     data.get("address", ""),
                "fsa":         data.get("fsa", ""),
                "lat":         p_lat,
                "lon":         p_lon,
                "distance_km": round(dist, 2),
                "status":      "Active",
            })
    return sorted(nearby, key=lambda x: x["distance_km"])


def get_nearby_cleared_permits(lat: float, lon: float,
                                radius_km: float = 2.0) -> list:
    """Cleared/completed permits within radius_km."""
    coords = load_cleared_coords()
    nearby = []
    for permit_id, data in coords.items():
        p_lat = data.get("lat")
        p_lon = data.get("lon")
        if not p_lat or not p_lon:
            continue
        dist = haversine_km(lat, lon, p_lat, p_lon)
        if dist <= radius_km:
            nearby.append({
                "permit_id":   permit_id,
                "address":     data.get("address", ""),
                "fsa":         data.get("fsa", ""),
                "lat":         p_lat,
                "lon":         p_lon,
                "distance_km": round(dist, 2),
                "status":      "Cleared/Completed",
            })
    return sorted(nearby, key=lambda x: x["distance_km"])

# =============================================================================
# ADDRESS LOOKUP
# =============================================================================

def get_ward_for_address(address: str) -> dict:
    """
    Find which ward an address is in:
    1. Search cleared + active permit coords for matching address
    2. Use lat/lon for spatial ward polygon lookup
    """
    print(f"\n[Constructor Agent] Address lookup: {address}")

    all_coords: dict = {}
    all_coords.update(load_cleared_coords())
    all_coords.update(load_permits_coords())

    addr_lower  = address.lower().strip()
    parts       = addr_lower.split()
    street_name = " ".join(parts[1:]) if parts and parts[0].isdigit() else addr_lower

    best_match = None
    best_score = 0

    for permit_id, data in all_coords.items():
        db_addr = data.get("address", "").lower()
        if addr_lower in db_addr or db_addr in addr_lower:
            best_match = data
            break
        if street_name and len(street_name) > 4 and street_name in db_addr:
            score = len(street_name)
            if score > best_score:
                best_score = score
                best_match = data

    if not best_match:
        return {
            "address": address,
            "error":   "Address not found in permit database",
            "note":    "No matching address in active or cleared permit records.",
        }

    lat = best_match.get("lat")
    lon = best_match.get("lon")
    fsa = best_match.get("fsa", "")

    ward_info      = get_ward_for_coords(lat, lon) if lat and lon else {}
    nearby_active  = len(get_nearby_permits(lat, lon, radius_km=1.0))  if lat else 0
    nearby_cleared = len(get_nearby_cleared_permits(lat, lon, radius_km=1.0)) if lat else 0

    result = {
        "address":         address,
        "matched_address": best_match.get("address", ""),
        "lat":             lat,
        "lon":             lon,
        "fsa":             fsa,
        "ward_name":       ward_info.get("ward_name", "Unknown"),
        "ward_number":     ward_info.get("ward_number", "Unknown"),
        "ward_found":      ward_info.get("found", False),
        "nearby_active":   nearby_active,
        "nearby_cleared":  nearby_cleared,
    }

    print(f"  Matched: {result['matched_address']}")
    print(f"  Coords:  {lat}, {lon}")
    print(f"  FSA:     {fsa}")
    print(f"  Ward:    {result['ward_name']} (#{result['ward_number']})")
    return result

# =============================================================================
# STATS FUNCTIONS
# =============================================================================

def get_area_permit_stats(area_name: str) -> dict:
    print(f"\n[Constructor Agent] Permit stats: {area_name}")
    coords     = load_permits_coords()
    area_lower = area_name.lower()
    matched    = []
    fsa_codes  = set()
    samples    = []

    for permit_id, data in coords.items():
        address = data.get("address", "").lower()
        fsa     = data.get("fsa", "")
        if area_lower in address:
            matched.append(permit_id)
            fsa_codes.add(fsa)
            if len(samples) < 5:
                samples.append(data.get("address", ""))

    return {
        "area":             area_name,
        "total_permits":    len(matched),
        "fsa_codes":        list(fsa_codes),
        "sample_addresses": samples,
    }


def get_cost_estimate(structure_type: str) -> dict:
    """Construction cost stats for a structure type."""
    summary  = load_cleared_summary()
    cost_map = summary.get("cost", {}).get("by_structure_type", {})

    if structure_type in cost_map:
        d = cost_map[structure_type]
        return {
            "structure_type": structure_type,
            "median_cost":    d.get("median", 0),
            "mean_cost":      round(d.get("mean", 0), 0),
            "p90_cost":       d.get("p90", 0),
            "permit_count":   d.get("n", 0),
        }

    search = structure_type.lower()
    for key, d in cost_map.items():
        if search in key.lower() or key.lower() in search:
            return {
                "structure_type": key,
                "median_cost":    d.get("median", 0),
                "mean_cost":      round(d.get("mean", 0), 0),
                "p90_cost":       d.get("p90", 0),
                "permit_count":   d.get("n", 0),
            }

    overall = summary.get("cost", {}).get("overall", {})
    return {
        "structure_type": "Overall",
        "median_cost":    overall.get("median", 0),
        "mean_cost":      round(overall.get("mean", 0), 0),
        "p90_cost":       overall.get("p90", 0),
        "permit_count":   overall.get("n", 0),
    }


def get_approval_time_by_type(permit_type: str) -> dict:
    summary = load_cleared_summary()
    times   = summary.get("approval_time_by_permit_type", [])
    search  = permit_type.lower()
    for entry in times:
        if search in entry.get("key", "").lower():
            return {
                "permit_type":  entry["key"],
                "median_days":  round(entry["median_days"], 0),
                "mean_days":    round(entry["mean_days"], 0),
                "p90_days":     round(entry["p90_days"], 0),
                "permit_count": entry["n"],
            }
    return {}


def get_builder_performance() -> dict:
    summary = load_cleared_summary()
    perf    = summary.get("builder_performance", {})
    top_vol = summary.get("top_builders_by_volume", {})

    fastest = sorted(
        perf.get("fastest_builders", []),
        key=lambda x: x.get("median_days", 999)
    )[:5]
    slowest = sorted(
        perf.get("slowest_builders", []),
        key=lambda x: x.get("median_days", 0),
        reverse=True
    )[:5]

    return {
        "fastest":          fastest,
        "slowest":          slowest,
        "city_median_days": perf.get("city_median_days", 28),
        "top_by_volume":    dict(list(top_vol.items())[:5]),
    }


def get_construction_volume_trend() -> dict:
    summary = load_cleared_summary()
    vol_raw = summary.get("volume_by_year", {})
    by_year = {}

    for k, v in vol_raw.items():
        try:
            year = int(float(k))
            if 1990 <= year <= 2030:
                by_year[str(year)] = v
        except (ValueError, TypeError):
            continue

    if not by_year:
        return {}

    peak_year = max(by_year, key=lambda x: by_year[x])
    recent    = {k: by_year[k] for k in sorted(by_year)[-3:]}

    return {
        "by_year":      by_year,
        "peak_year":    peak_year,
        "recent_years": recent,
    }

# =============================================================================
# MAIN PROFILE FUNCTION
# =============================================================================

def get_construction_profile(area_name: str) -> dict:
    """
    Full construction site assessment for a Toronto area.
    Scoped to the specific neighbourhood using FSA + radius,
    with broader ward context for planning data.
    """
    print(f"\n[Constructor Agent] Building profile for: {area_name}")

    ttc_scores = load_all_neighbourhood_scores()
    profiles   = load_urban_profiles()
    priorities = get_investment_priorities()

    search     = area_name.lower().strip()
    result: dict = {"area": area_name}

    # -- Ward search term (use mapping if available) --------------------------
    ward_search = NEIGHBOURHOOD_TO_WARD.get(search, search)

    # -- TTC (neighbourhood level) --------------------------------------------
    ttc_matches = []
    for name, data in ttc_scores.items():
        name_lower = name.lower()
        words      = [w for w in search.split() if len(w) > 4]
        if search in name_lower or (
            len(words) > 1 and all(w in name_lower for w in words)
        ):
            ttc_matches.append({
                "neighbourhood":  name,
                "stops":          data.get("stop_count", 0),
                "has_subway":     data.get("has_subway", False),
                "has_streetcar":  data.get("has_streetcar", False),
                "has_bus":        data.get("has_bus", False),
                "sample_stops":   data.get("sample_stops", [])[:5],
            })

    if ttc_matches:
        exact = [m for m in ttc_matches if search in m["neighbourhood"].lower()]
        best  = max(exact if exact else ttc_matches, key=lambda x: x["stops"])
        result["ttc"]             = best
        result["all_ttc_matches"] = exact if exact else ttc_matches
    else:
        result["ttc"]             = None
        result["all_ttc_matches"] = []

    # -- Ward profile (use ward mapping for correct ward) ---------------------
    ward_matches = []
    for ward, data in profiles.items():
        ward_lower = ward.lower()
        if ward_search in ward_lower or search in ward_lower:
            ward_matches.append({
                "name":           ward,
                "parks":          data.get("total_parks", 0),
                "businesses":     data.get("total_active_businesses", 0),
                "bike_lanes":     data.get("total_bike_lane_segments", 0),
                "transit_stops":  data.get("total_active_transit_stops", 0),
                "cultural_spots": data.get("total_cultural_hotspots", 0),
                "road_segments":  data.get("total_road_segments", 0),
                "ice_rinks":      data.get("total_outdoor_ice_rinks", 0),
                "dev_apps":       data.get("total_development_applications", 0),
            })

    result["ward"]             = ward_matches[0] if ward_matches else None
    result["all_ward_matches"] = ward_matches

    # -- Area centre (calculated from real permit data) -----------------------
    centre_lat, centre_lon = get_area_centre(area_name)

    # -- Active + cleared permits within 2km of neighbourhood centre ----------
    nearby_active  = get_nearby_permits(centre_lat, centre_lon, radius_km=2.0)
    nearby_cleared = get_nearby_cleared_permits(centre_lat, centre_lon, radius_km=2.0)

    # -- FSA codes for this specific area ------------------------------------
    target_fsas: set = set()
    for key, fsas in NEIGHBOURHOOD_FSA.items():
        if key in search or search in key:
            target_fsas.update(fsas)

    # Also collect from nearby permits
    for permit in nearby_active + nearby_cleared:
        fsa = permit.get("fsa", "")
        if fsa:
            target_fsas.add(fsa)

    # -- Extract unique streets from nearby permits ---------------------------
    street_counts: dict = {}
    for permit in nearby_active + nearby_cleared:
        address = permit.get("address", "")
        if not address:
            continue
        parts = address.strip().split()
        if len(parts) >= 2:
            street = " ".join(
                parts[1:] if parts[0].isdigit() else parts
            ).upper()
            street_counts[street] = street_counts.get(street, 0) + 1

    top_streets = [
        s[0] for s in sorted(
            street_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    ]

    result["permits"] = {
        "total_nearby":         len(nearby_active),
        "nearby_list":          nearby_active[:10],
        "total_cleared_nearby": len(nearby_cleared),
        "cleared_nearby_list":  nearby_cleared[:10],
        "centre_lat":           centre_lat,
        "centre_lon":           centre_lon,
        "search_radius":        "2km",
    }

    result["neighbourhood_stats"] = {
        "area":           area_name,
        "centre":         f"{centre_lat:.4f}N, {abs(centre_lon):.4f}W",
        "fsa_codes":      sorted(target_fsas),
        "unique_streets": len(street_counts),
        "top_streets":    top_streets,
        "active_permits": len(nearby_active),
        "cleared_permits":len(nearby_cleared),
    }

    # -- FSA approval times ---------------------------------------------------
    summary       = load_permits_summary()
    times         = summary.get("approval_time_by_fsa", [])
    approval_data = []
    for entry in times:
        if entry.get("key") in target_fsas:
            approval_data.append({
                "fsa":          entry["key"],
                "median_days":  round(entry["median_days"], 0),
                "mean_days":    round(entry["mean_days"], 0),
                "p90_days":     round(entry["p90_days"], 0),
                "permit_count": entry["n"],
            })

    result["approval_times"] = approval_data
    result["area_fsas"]      = sorted(target_fsas)

    # -- Volume trend ---------------------------------------------------------
    result["volume_trend"] = get_construction_volume_trend()

    # -- Investment risk ------------------------------------------------------
    investment_rank = None
    for i, p in enumerate(priorities):
        p_ward_lower = p["ward"].lower()
        if ward_search in p_ward_lower or search in p_ward_lower:
            investment_rank = i + 1
            break

    if investment_rank is None:
        result["investment_risk"] = "Low"
        result["investment_rank"] = None
        result["investment_note"] = (
            "This ward is not in the top investment priority list, "
            "indicating adequate existing infrastructure."
        )
    elif investment_rank <= 3:
        result["investment_risk"] = "High"
        result["investment_rank"] = investment_rank
        result["investment_note"] = (
            f"This ward ranks #{investment_rank} in investment priority, "
            "indicating significant infrastructure gaps."
        )
    else:
        result["investment_risk"] = "Medium"
        result["investment_rank"] = investment_rank
        result["investment_note"] = (
            f"This ward ranks #{investment_rank} in investment priority, "
            "with some infrastructure gaps but generally adequate services."
        )

    # -- City averages --------------------------------------------------------
    if profiles:
        all_transit = [v.get("total_active_transit_stops", 0) for v in profiles.values()]
        all_parks   = [v.get("total_parks", 0) for v in profiles.values()]
        all_bikes   = [v.get("total_bike_lane_segments", 0) for v in profiles.values()]
        all_biz     = [v.get("total_active_businesses", 0) for v in profiles.values()]
        result["city_averages"] = {
            "transit":    round(sum(all_transit) / len(all_transit), 1),
            "parks":      round(sum(all_parks)   / len(all_parks),   1),
            "bike_lanes": round(sum(all_bikes)   / len(all_bikes),   1),
            "businesses": round(sum(all_biz)     / len(all_biz),     1),
        }
    else:
        result["city_averages"] = {
            "transit": 360.7, "parks": 71.6,
            "bike_lanes": 61.4, "businesses": 5465.4,
        }

    print(f"  TTC matches:     {[m['neighbourhood'] for m in ttc_matches]}")
    print(f"  Ward:            {result['ward']['name'] if result['ward'] else 'None'}")
    print(f"  Active nearby:   {len(nearby_active)}")
    print(f"  Cleared nearby:  {len(nearby_cleared)}")
    print(f"  FSA codes:       {sorted(target_fsas)}")
    print(f"  Investment risk: {result['investment_risk']}")

    return result

# =============================================================================
# Q&A FUNCTION
# =============================================================================

def ask_constructor_question(question: str,
                              area_name: str = None) -> str:
    """
    Answer construction/development questions with cited facts.
    Supports street address lookup via spatial ward boundary matching.
    """
    print(f"\n[Constructor Agent] Question: {question}")

    search_term    = area_name
    address_lookup = None

    # -- Detect street address ------------------------------------------------
    address_match = re.search(
        r'\d+\s+[a-zA-Z][a-zA-Z\s]+'
        r'(rd|st|ave|blvd|dr|cres|way|ln|pl|ct|road|street|avenue|drive|crescent)\b',
        question,
        re.IGNORECASE
    )

    if address_match:
        detected_address = address_match.group(0).strip()
        address_lookup   = get_ward_for_address(detected_address)
        if not search_term and address_lookup.get("ward_found"):
            search_term = address_lookup["ward_name"]
            print(f"  Ward from spatial lookup: {search_term}")

    # -- Detect area from question text ---------------------------------------
    if not search_term:
        ttc_scores = load_all_neighbourhood_scores()
        profiles   = load_urban_profiles()
        q_lower    = question.lower()

        # Check neighbourhood-to-ward mapping keys first
        for key in NEIGHBOURHOOD_TO_WARD:
            if key in q_lower:
                search_term = key
                break

        # Then ward names
        if not search_term:
            for name in profiles.keys():
                if name.lower() in q_lower:
                    search_term = name
                    break

        # Then TTC neighbourhood names
        if not search_term:
            for name in ttc_scores.keys():
                if name.lower() in q_lower:
                    search_term = name
                    break

        # Partial word match on ward names
        if not search_term:
            for name in profiles.keys():
                if any(word in name.lower()
                       for word in q_lower.split() if len(word) > 5):
                    search_term = name
                    break

    profile = get_construction_profile(search_term) if search_term else None

    avgs = (profile or {}).get("city_averages", {
        "transit": 360.7, "parks": 71.6,
        "bike_lanes": 61.4, "businesses": 5465.4,
    })

    # -- Build cited facts ----------------------------------------------------
    facts = [
        "CRITICAL RULE: Only use facts from the data below. "
        "Never invent school names, residence counts, capacities, "
        "distances to specific buildings, or statistics not provided here."
    ]

    # Address lookup fact
    if address_lookup:
        if address_lookup.get("error"):
            facts.append(
                f"Address lookup: '{address_lookup['address']}' — "
                f"{address_lookup['error']}. {address_lookup.get('note', '')}"
            )
        else:
            facts.append(
                f"[Source: Toronto Building Permits + Ward Boundary Polygons] "
                f"Address '{address_lookup['address']}' is in "
                f"{address_lookup['ward_name']} ward "
                f"(Ward #{address_lookup['ward_number']}), "
                f"FSA {address_lookup['fsa']}. "
                f"Nearest permit record: {address_lookup['matched_address']}. "
                f"{address_lookup['nearby_active']} active permits and "
                f"{address_lookup['nearby_cleared']} cleared permits within 1km."
            )

    if profile:
        # Neighbourhood-specific stats
        nb = profile.get("neighbourhood_stats", {})
        if nb:
            facts.append(
                f"[Source: Toronto Building Permits Open Dataset] "
                f"{nb['area']} area (centre: {nb['centre']}, 2km radius): "
                f"{nb['active_permits']} active permits, "
                f"{nb['cleared_permits']} cleared/completed permits, "
                f"{nb['unique_streets']} unique streets with permit activity. "
                f"FSA codes: {', '.join(nb['fsa_codes'][:6])}. "
                f"Most active streets: {', '.join(nb['top_streets'][:5])}."
            )

        # TTC
        ttc = profile.get("ttc")
        if ttc:
            subway    = "subway"    if ttc.get("has_subway")    else "no subway"
            streetcar = "streetcar" if ttc.get("has_streetcar") else "no streetcar"
            bus       = "bus"       if ttc.get("has_bus")       else "no bus"
            samples   = ", ".join(ttc.get("sample_stops", [])[:3])
            facts.append(
                f"[Source: Toronto TTC GTFS Schedule Data] "
                f"{ttc['neighbourhood']}: {ttc['stops']} TTC stops "
                f"({subway}, {streetcar}, {bus}). "
                f"Key stops: {samples}."
            )

        # Ward context
        ward = profile.get("ward")
        if ward:
            a = avgs
            facts.append(
                f"[Source: Toronto Urban Profile Dataset] "
                f"Broader ward — {ward['name']}: "
                f"{ward['transit_stops']} transit stops "
                f"({'above' if ward['transit_stops'] >= a['transit'] else 'below'} "
                f"city avg {a['transit']}), "
                f"{ward['parks']} parks "
                f"({'above' if ward['parks'] >= a['parks'] else 'below'} "
                f"city avg {a['parks']}), "
                f"{ward['bike_lanes']} bike lanes "
                f"({'above' if ward['bike_lanes'] >= a['bike_lanes'] else 'below'} "
                f"city avg {a['bike_lanes']}), "
                f"{ward['businesses']} businesses "
                f"({'above' if ward['businesses'] >= a['businesses'] else 'below'} "
                f"city avg {a['businesses']}), "
                f"{ward['cultural_spots']} cultural hotspots, "
                f"{ward['road_segments']} road segments, "
                f"{ward['dev_apps']} active development applications."
            )

        # Approval times
        approval_times = profile.get("approval_times", [])
        if approval_times:
            best   = min(approval_times, key=lambda x: x["median_days"])
            others = ", ".join(
                f"FSA {a['fsa']} {a['median_days']:.0f}d"
                for a in approval_times[1:3]
            )
            facts.append(
                f"[Source: Toronto Building Permits Approval Time Analysis] "
                f"Permit approval — FSA {best['fsa']}: "
                f"median {best['median_days']:.0f} days, "
                f"90th percentile {best['p90_days']:.0f} days "
                f"({best['permit_count']:,} permits). "
                + (f"Other FSAs: {others}." if others else "")
            )

        # Construction costs by structure type
        cleared_data = load_cleared_summary()
        cost_map     = cleared_data.get("cost", {}).get("by_structure_type", {})
        key_types    = [
            "Apartment Building", "SFD - Detached", "Office",
            "Retail Store", "Multiple Unit Building", "Stacked Townhouses",
        ]
        cost_lines = []
        for stype in key_types:
            if stype in cost_map:
                d = cost_map[stype]
                cost_lines.append(
                    f"{stype}: median ${d.get('median', 0):,.0f}, "
                    f"mean ${d.get('mean', 0):,.0f}, "
                    f"90th pct ${d.get('p90', 0):,.0f} "
                    f"({d.get('n', 0):,} permits)"
                )
        if cost_lines:
            facts.append(
                f"[Source: Toronto Building Permits Cost Analysis] "
                f"Construction costs by structure type: "
                + " | ".join(cost_lines)
            )

        # Investment risk
        facts.append(
            f"[Source: City Investment Priority Analysis] "
            f"Investment risk: {profile.get('investment_risk', 'Unknown')}. "
            f"{profile.get('investment_note', '')}"
        )

        # Volume trend
        trend = profile.get("volume_trend", {})
        if trend.get("recent_years"):
            recent = trend["recent_years"]
            facts.append(
                f"[Source: Toronto Building Permits Volume Analysis] "
                f"City-wide recent volume: "
                + ", ".join(f"{yr}: {cnt:,}" for yr, cnt in recent.items())
                + f". Peak year: {trend.get('peak_year', 'Unknown')}."
            )

    # City-wide summary
    summary = load_permits_summary()
    if summary:
        status    = summary.get("status_breakdown", {})
        p_types   = summary.get("permit_type_counts", {})
        top_types = sorted(p_types.items(), key=lambda x: x[1], reverse=True)[:4]
        total     = summary.get("dataset", {}).get("rows", 0)
        facts.append(
            f"[Source: Toronto Building Permits Summary Report] "
            f"City-wide: {total:,} total permits. "
            f"Top types: "
            + ", ".join(f"{t[0]} ({t[1]:,})" for t in top_types) + "."
        )

    context = "\n\n".join(facts)

    few_shot = """
Example Q: I want to plan new construction in Mimico area. What do I need to know?
Example A: Mimico (FSA M8V, centre ~43.6167N, 79.4833W) has 23 active permits and 45 cleared permits within 2km showing consistent construction activity along streets including LAKE SHORE BLVD W and ROYAL YORK RD (Source: Toronto Building Permits Open Dataset). The broader Etobicoke-Lakeshore ward has 524 transit stops above the city average of 361, 114 parks above the city average of 72, and 105 bike lanes above the city average of 61 (Source: Toronto Urban Profile Dataset). Permit approval times in FSA M8V have a median of 32 days with a 90th percentile of 183 days (Source: Toronto Building Permits Approval Time Analysis). Apartment Building construction has a median cost of $120,000 with a 90th percentile of $1,500,000 (Source: Toronto Building Permits Cost Analysis). The investment risk is Low indicating adequate infrastructure with no significant gaps (Source: City Investment Priority Analysis).

Example Q: I want to build a school near 30 Bevdale Rd. Are there enough residences?
Example A: 30 Bevdale Rd is located in Willowdale ward (Ward #18), FSA M2N, confirmed via ward boundary spatial lookup (Source: Toronto Building Permits + Ward Boundary Polygons). There are 12 active permits and 34 cleared permits within 1km of this address (Source: Toronto Building Permits Open Dataset). Willowdale ward has 277 transit stops below the city average of 361 and 78 parks above the city average of 72 (Source: Toronto Urban Profile Dataset). Residence counts and school capacity data are not available in our datasets — for demographic analysis consult Statistics Canada and the Toronto District School Board directly.

Example Q: How much does it cost to build an apartment building in Toronto?
Example A: Based on cleared permit records, Apartment Building construction has a median estimated cost of $120,000, a mean of $850,000, and a 90th percentile of $1,500,000 across 16,100 permits (Source: Toronto Building Permits Cost Analysis). For comparison, Office construction has a median of $80,000 and Retail Store has a median of $50,000. These figures represent permit-declared estimated construction costs and actual costs may vary significantly based on size, finishes and site conditions.
"""

    prompt = (
        f"Data:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = ask_ai(
        "You are a Toronto construction and development advisor. "
        "CRITICAL: Only use facts from the Data section. "
        "Never invent school names, residence counts, distances to buildings, "
        "capacities, or statistics not in the data. "
        "If data is unavailable, say so and direct to the right authority. "
        "Cover: specific neighbourhood permit activity, TTC access, "
        "ward infrastructure context, approval times, construction costs, "
        "investment risk. "
        "Always present neighbourhood-specific data first, ward context second. "
        "Compare metrics to city averages using exact above/below labels. "
        "Write 5-7 sentences. Start immediately with the answer. "
        "Cite each fact like (Source: Toronto TTC GTFS Schedule Data). "
        "Never show reasoning.\n"
        + few_shot,
        prompt,
        max_tokens=700,
    )

    print(f"[Constructor Agent] Response: {response[:80]}")
    return response

# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1 - Area Centre: Mimico")
    print("=" * 60)
    lat, lon = get_area_centre("Mimico")
    print(f"Centre: {lat:.4f}N, {lon:.4f}W")

    print("\n" + "=" * 60)
    print("TEST 2 - Construction Profile: Mimico")
    print("=" * 60)
    profile = get_construction_profile("Mimico")
    print(f"Ward:          {profile['ward']['name'] if profile['ward'] else 'None'}")
    print(f"Active nearby: {profile['permits']['total_nearby']}")
    print(f"Cleared nearby:{profile['permits']['total_cleared_nearby']}")
    print(f"FSA codes:     {profile['area_fsas']}")
    print(f"Top streets:   {profile['neighbourhood_stats']['top_streets'][:3]}")
    print(f"Risk:          {profile['investment_risk']}")

    print("\n" + "=" * 60)
    print("TEST 3 - Address Lookup: 30 Bevdale Rd")
    print("=" * 60)
    result = get_ward_for_address("30 Bevdale Rd")
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 4 - Cost Estimate: Apartment Building")
    print("=" * 60)
    print(json.dumps(get_cost_estimate("Apartment Building"), indent=2))

    print("\n" + "=" * 60)
    print("TEST 5 - Q&A")
    print("=" * 60)
    questions = [
        "I want to plan new construction in Mimico area. What do I need to know?",
        "How much does it cost to build an apartment building in Toronto?",
        "I want to build a school near 30 Bevdale Rd.",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = ask_constructor_question(q)
        print(f"A: {answer}")