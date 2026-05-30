#!/usr/bin/env python3
"""
toronto_permit_agent.py
─────────────────────────────────────────────────────────────────────────────
Toronto Building Permit Intelligence Agent

A data service that:
  1. Loads building permits and zoning from Toronto Open Data
  2. Geocodes every address ONCE and caches the result (geocode_cache.json)
  3. Classifies each building (residential / commercial / mixed_use / etc.)
  4. Accepts a structured INPUT from either a constructor or a government user
  5. Returns a tailored JSON output for the requesting agent

────── INPUT ────────────────────────────────────────────────────────────────
  {
    "proposed_category": "residential",       # residential/commercial/mixed_use/...
    "structure_type": "apartment",            # free-text descriptor
    "address": "123 King St W",
    "builder": "ABC Developers Inc"           # optional
  }

────── OUTPUT (unified — every consumer gets the full picture) ──────────────
  {
    "predicted_category": ...,
    "geocoded_location": ...,
    "spatial_context": ...,
    "approval_timeline": ...,             # how long similar projects take
    "revisions_expected": ...,            # revisions to expect
    "value_benchmarks_cad": ...,          # cost range of similar projects
    "scale_observed_nearby": ...,         # max units / scale proxy
    "structure_types_nearby": ...,
    "ward_signal": ...,
    "nearest_precedents": ...,            # 3 closest comparable permits
    "historical_approval": ...,           # city-wide approval rate
    "historical_approval_time": ...,      # city-wide approval times
    "socio_economic_impact": ...,         # construction jobs, housing supply,
                                          # property tax revenue (2025 rates)
    "saturation_analysis": ...,           # is this area oversaturated?
    "recommendation": ...,                # approve / standard review / reconsider
    "alternate_locations": ...,           # underserved wards if saturated
  }

Usage:
    python toronto_permit_agent.py --input request.json --pretty
    python toronto_permit_agent.py --input request.json --output result.json
    cat request.json | python toronto_permit_agent.py --stdin

    from toronto_permit_agent import handle_request
    payload = handle_request(request_dict)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from toronto_data_service import get_building_permits, get_zoning
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    DATA_SERVICE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GEOCODE_CACHE_FILE = Path("geocode_cache.json")
NOMINATIM_URL      = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS  = {"User-Agent": "TorontoPermitAgent/1.0 (urban-dev-research)"}
GEOCODE_DELAY_SEC  = 1.0   # Nominatim usage policy: 1 req/sec
SEARCH_RADIUS_M    = 500   # "same area" = within 500m
ALTERNATE_RADIUS_M = 2000  # search radius for alternate location suggestions


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION RULES
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_RULES = {
    "residential": [
        "sfd", "single family", "semi", "duplex", "triplex", "townhouse",
        "townhome", "row house", "rowhouse", "apartment", "condominium",
        "condo", "residential", "dwelling", "house", "multiplex", "multi-unit",
        "garden suite", "laneway suite", "secondary suite",
    ],
    "commercial": [
        "commercial", "retail", "store", "shop", "office", "restaurant",
        "hotel", "motel", "warehouse retail", "mercantile", "personal service",
        "business and personal services", "bank",
    ],
    "industrial": [
        "industrial", "factory", "manufacturing", "warehouse", "plant",
        "logistics", "distribution", "production",
    ],
    "institutional": [
        "school", "hospital", "church", "religious", "community centre",
        "community center", "library", "government", "institutional",
        "place of worship", "daycare", "assembly", "university", "college",
    ],
    "mixed_use": [
        "mixed use", "mixed-use", "live/work", "live-work",
        "residential commercial", "commercial residential", "mixed residential",
    ],
}


def _classify_text(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    t = text.lower()
    for kw in KEYWORD_RULES["mixed_use"]:
        if kw in t:
            return "mixed_use"
    scores = {}
    for cat, keywords in KEYWORD_RULES.items():
        if cat == "mixed_use":
            continue
        hits = sum(1 for kw in keywords if kw in t)
        if hits:
            scores[cat] = hits
    return max(scores, key=scores.get) if scores else None


def classify_building(row: dict) -> dict:
    structure = str(row.get("STRUCTURE_TYPE") or "").strip()
    work      = str(row.get("WORK")           or "").strip()
    current   = str(row.get("CURRENT_USE")    or "").strip()
    proposed  = str(row.get("PROPOSED_USE")   or "").strip()
    description = str(row.get("DESCRIPTION")  or "").strip()

    cat = _classify_text(structure)
    if cat:
        return {"category": cat, "confidence": "high",
                "signal_used": "structure_type", "evidence": structure}

    if current and proposed and current != proposed:
        cur_cat, pro_cat = _classify_text(current), _classify_text(proposed)
        if cur_cat and pro_cat and cur_cat != pro_cat:
            return {"category": "mixed_use", "confidence": "medium",
                    "signal_used": "use_transition",
                    "evidence": f"{current} → {proposed}"}

    cat = _classify_text(proposed)
    if cat:
        return {"category": cat, "confidence": "high",
                "signal_used": "proposed_use", "evidence": proposed}

    cat = _classify_text(current)
    if cat:
        return {"category": cat, "confidence": "medium",
                "signal_used": "current_use", "evidence": current}

    cat = _classify_text(work + " " + description)
    if cat:
        return {"category": cat, "confidence": "low",
                "signal_used": "work_description",
                "evidence": work or description[:80]}

    return {"category": "other", "confidence": "none",
            "signal_used": "unclassified", "evidence": None}


# ─────────────────────────────────────────────────────────────────────────────
# GEOCODING
# ─────────────────────────────────────────────────────────────────────────────

def _load_geocode_cache() -> dict:
    if GEOCODE_CACHE_FILE.exists():
        try:
            return json.loads(GEOCODE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_geocode_cache(cache: dict):
    GEOCODE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _normalize_address(addr: str) -> str:
    if not addr:
        return ""
    addr = re.sub(r"\s+", " ", str(addr).strip()).lower()
    return addr


def geocode_address(address: str, city: str = "Toronto", cache: dict = None,
                    rate_limit: bool = True) -> dict | None:
    """
    Geocode a single address via Nominatim. Returns {lat, lon, display_name}
    or None on failure. Uses persistent JSON cache.
    """
    if not address:
        return None

    if cache is None:
        cache = _load_geocode_cache()

    key = _normalize_address(f"{address}, {city}")
    if key in cache:
        return cache[key]

    try:
        params = {
            "q": f"{address}, {city}, Ontario, Canada",
            "format": "json",
            "limit": 1,
        }
        if rate_limit:
            time.sleep(GEOCODE_DELAY_SEC)
        resp = requests.get(NOMINATIM_URL, params=params,
                            headers=NOMINATIM_HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            cache[key] = None
            return None

        r = results[0]
        record = {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "display_name": r.get("display_name"),
        }
        cache[key] = record
        return record
    except Exception as e:
        print(f"  geocode error for '{address}': {e}", file=sys.stderr)
        cache[key] = None
        return None


def assemble_address(row: pd.Series, cols: list) -> str:
    """Build a clean address string from the permit row."""
    f_addr = _find_col(cols, "ADDRESS")
    if f_addr and pd.notna(row.get(f_addr)):
        return str(row[f_addr]).strip()

    f_num = _find_col(cols, "STREET_NUM", "STREET NUM")
    f_name = _find_col(cols, "STREET_NAME", "STREET NAME")
    f_type = _find_col(cols, "STREET_TYPE", "STREET TYPE")
    f_dir  = _find_col(cols, "STREET_DIRECTION", "STREET DIR")

    parts = []
    for f in (f_num, f_name, f_type, f_dir):
        if f and pd.notna(row.get(f)):
            val = str(row[f]).strip()
            if val and val.lower() not in ("nan", "none"):
                parts.append(val)
    return " ".join(parts)


def geocode_dataframe(df: pd.DataFrame, max_geocode: int = None,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Add LAT and LON columns. Uses cache aggressively to avoid re-geocoding.
    max_geocode limits NEW geocoding calls (cache hits are unlimited).
    """
    df = df.copy()
    cols = list(df.columns)
    cache = _load_geocode_cache()

    addresses = df.apply(lambda r: assemble_address(r, cols), axis=1)
    lats, lons = [], []
    new_geocodes = 0
    cache_hits = 0

    for i, addr in enumerate(addresses):
        if not addr:
            lats.append(None); lons.append(None)
            continue

        key = _normalize_address(f"{addr}, Toronto")
        if key in cache:
            cache_hits += 1
            rec = cache[key]
            if rec:
                lats.append(rec["lat"]); lons.append(rec["lon"])
            else:
                lats.append(None); lons.append(None)
            continue

        if max_geocode is not None and new_geocodes >= max_geocode:
            lats.append(None); lons.append(None)
            continue

        if verbose and new_geocodes % 10 == 0:
            print(f"  geocoding {new_geocodes + 1}... '{addr}'", file=sys.stderr)
        rec = geocode_address(addr, cache=cache)
        new_geocodes += 1
        if rec:
            lats.append(rec["lat"]); lons.append(rec["lon"])
        else:
            lats.append(None); lons.append(None)

    df["LAT"] = lats
    df["LON"] = lons

    _save_geocode_cache(cache)
    if verbose:
        print(f"  geocoding done: {cache_hits} cached, {new_geocodes} new, "
              f"{len(df) - cache_hits - new_geocodes} skipped",
              file=sys.stderr)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearby_permits(df: pd.DataFrame, lat: float, lon: float,
                        radius_m: float, category: str | None = None) -> pd.DataFrame:
    """Filter permits within radius_m of a point, optionally same category."""
    if "LAT" not in df.columns or lat is None or lon is None:
        return df.iloc[0:0]
    valid = df.dropna(subset=["LAT", "LON"]).copy()
    valid["_DIST_M"] = valid.apply(
        lambda r: haversine_m(lat, lon, r["LAT"], r["LON"]), axis=1
    )
    nearby = valid[valid["_DIST_M"] <= radius_m]
    if category:
        nearby = nearby[nearby["_CATEGORY"] == category]
    return nearby.sort_values("_DIST_M")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(columns, *fragments) -> str | None:
    cols_upper = [c.upper() for c in columns]
    for frag in fragments:
        for i, c in enumerate(cols_upper):
            if frag in c:
                return columns[i]
    return None


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _approval_days(row: pd.Series, app_col: str, issued_col: str) -> float | None:
    """Days from application to issue."""
    if not app_col or not issued_col:
        return None
    a = pd.to_datetime(row.get(app_col), errors="coerce")
    i = pd.to_datetime(row.get(issued_col), errors="coerce")
    if pd.isna(a) or pd.isna(i):
        return None
    delta = (i - a).days
    return float(delta) if delta >= 0 else None


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    classifications = df.apply(lambda row: classify_building(row.to_dict()), axis=1)
    df["_CATEGORY"]   = classifications.apply(lambda c: c["category"])
    df["_CONFIDENCE"] = classifications.apply(lambda c: c["confidence"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED OUTPUT — same payload for every consumer
# ─────────────────────────────────────────────────────────────────────────────

# Rough socio-economic multipliers (documented in output for transparency).
# Construction-job multiplier: ~$200k of construction value = 1 person-year of
# direct construction employment in Ontario (Statistics Canada I-O multipliers).
EMPLOYMENT_PER_DOLLAR_CONSTRUCTION = 1.0 / 200_000
OPERATIONAL_JOBS_PER_M2 = {
    "commercial":    0.04,   # ~25 m² per worker (office/retail)
    "industrial":    0.015,  # ~65 m² per worker
    "institutional": 0.025,
    "mixed_use":     0.03,
    "residential":   0.0,    # operational employment is minimal
    "other":         0.01,
}

# ── Toronto property tax rates for 2025 ──────────────────────────────────────
# Source: City of Toronto, "Property Tax Rates & Fees" + 2025 Budget docs.
# Rates apply to the property's Current Value Assessment (CVA) per MPAC.
# Each rate = municipal rate + City Building Fund + education rate (combined).
# Values verified against City reporting and independent guides, May 2025.
TORONTO_TAX_RATES_2025 = {
    "residential":         0.00754087,  # 0.754087%
    "new_multi_res":       0.00754087,  # same as residential for 35 years
    "multi_residential":   0.01197305,  # 1.197305% (traditional apartment buildings)
    "commercial":          0.02275478,  # 2.275478% (retail/office/warehouse)
    "industrial":          0.02500000,  # ~2.5% (approximate combined; industrial
                                        # has separate sub-classes; this is an
                                        # average across small/large industrial)
}

# Category → tax class mapping. The agent's internal category labels don't map
# 1-to-1 to Toronto's tax classes, so we route them here.
CATEGORY_TO_TAX_CLASS = {
    "residential":   "residential",
    "commercial":    "commercial",
    "industrial":    "industrial",
    "institutional": "commercial",       # institutions usually taxed as commercial
    "mixed_use":     "commercial",       # mixed-use commonly assessed at commercial rate
    "other":         "residential",      # safe default
}

# Construction-cost-to-assessed-value ratio. MPAC's CVA is generally lower than
# construction cost — assessments are still anchored to a 2016 valuation date
# per Ontario's frozen reassessment policy. A 0.75 ratio is a conservative
# proxy until the agent has access to per-property MPAC data.
CONSTRUCTION_COST_TO_CVA_RATIO = 0.75


def estimate_property_tax_revenue(category: str,
                                  construction_value_cad: float) -> dict:
    """
    Estimate annual municipal property tax revenue for a project, using:
      assessed value ≈ construction cost × ratio
      tax revenue   = assessed value × applicable Toronto rate
    """
    tax_class = CATEGORY_TO_TAX_CLASS.get(category, "residential")
    rate = TORONTO_TAX_RATES_2025[tax_class]
    estimated_cva = construction_value_cad * CONSTRUCTION_COST_TO_CVA_RATIO
    annual_revenue = estimated_cva * rate

    return {
        "tax_class_applied":          tax_class,
        "tax_rate_2025":              rate,
        "tax_rate_pct":               round(rate * 100, 4),
        "estimated_cva_cad":          round(estimated_cva),
        "cva_to_construction_ratio":  CONSTRUCTION_COST_TO_CVA_RATIO,
        "annual_property_tax_cad":    round(annual_revenue),
        "ten_year_revenue_cad":       round(annual_revenue * 10),
        "source": "City of Toronto 2025 property tax rates "
                  "(municipal + City Building Fund + education, combined). "
                  "Assessed value approximated from construction cost; "
                  "actual CVA is set by MPAC per Ontario Assessment Act.",
    }


def build_unified_output(request: dict, df: pd.DataFrame,
                         coord: dict | None) -> dict:
    """
    Single unified payload. Same shape regardless of who's asking.
    Contains every metric the agent can compute about the request.
    """
    # Resolve predicted category from multiple possible input keys
    proposed_text = " ".join([
        request.get("proposed_category", ""),
        request.get("building_category", ""),
        request.get("structure_type", ""),
    ]).strip()
    predicted = (
        _classify_text(proposed_text)
        or request.get("proposed_category")
        or request.get("building_category")
        or "other"
    )

    out = {
        "request_received":   request,
        "predicted_category": predicted,
        "geocoded_location":  coord,
    }

    if not coord:
        out["error"] = "Could not geocode the provided address. Cannot generate spatial insights."
        return out

    cols = list(df.columns)
    f_status = _find_col(cols, "STATUS")
    f_app    = _find_col(cols, "APPLICATION_DATE", "APPLICATION DATE")
    f_iss    = _find_col(cols, "ISSUED_DATE", "ISSUED DATE")
    f_rev    = _find_col(cols, "REVISION_NUM", "REVISION")
    f_units  = _find_col(cols, "UNITS_CREATED", "DWELLING_UNITS_CREATED")
    f_val    = _find_col(cols, "ESTIMATED_CONSTRUCTION_COST", "EST_CONST_COST",
                                "CONSTRUCTION_VALUE", "VALUE", "COST")
    f_ward   = _find_col(cols, "WARD")
    f_struct = _find_col(cols, "STRUCTURE_TYPE", "STRUCTURE")
    f_gfa    = _find_col(cols, "RESIDENTIAL_GFA", "GFA")

    nearby_same = find_nearby_permits(df, coord["lat"], coord["lon"],
                                      SEARCH_RADIUS_M, predicted)
    nearby_any  = find_nearby_permits(df, coord["lat"], coord["lon"],
                                      SEARCH_RADIUS_M)
    same_cat_all = df[df["_CATEGORY"] == predicted]

    out["spatial_context"] = {
        "search_radius_m":              SEARCH_RADIUS_M,
        "permits_in_radius_total":      len(nearby_any),
        "permits_in_radius_same_type":  len(nearby_same),
        "city_wide_same_category":      len(same_cat_all),
    }

    # ── Approval timeline (nearby similar permits) ───────────────────────────
    if f_app and f_iss and len(nearby_same) > 0:
        days = nearby_same.apply(
            lambda r: _approval_days(r, f_app, f_iss), axis=1
        ).dropna()
        if not days.empty:
            out["approval_timeline"] = {
                "scope":          "nearby_similar_permits",
                "based_on_n":     int(len(days)),
                "median_days":    round(float(days.median())),
                "mean_days":      round(float(days.mean())),
                "p25_days":       round(float(days.quantile(0.25))),
                "p75_days":       round(float(days.quantile(0.75))),
                "fastest_days":   int(days.min()),
                "slowest_days":   int(days.max()),
            }

    # ── Revisions expected ────────────────────────────────────────────────────
    if f_rev and len(nearby_same) > 0:
        revs = pd.to_numeric(nearby_same[f_rev], errors="coerce").dropna()
        if not revs.empty:
            out["revisions_expected"] = {
                "based_on_n": int(len(revs)),
                "mean":       round(float(revs.mean()), 1),
                "median":     int(revs.median()),
                "max":        int(revs.max()),
                "pct_with_at_least_one_revision":
                    round(float((revs > 0).mean() * 100), 1),
            }

    # ── Construction value benchmarks (nearby) ────────────────────────────────
    if f_val and len(nearby_same) > 0:
        vals = _safe_num(nearby_same[f_val])
        if not vals.empty:
            out["value_benchmarks_cad"] = {
                "scope":      "nearby_similar_permits",
                "based_on_n": int(len(vals)),
                "median":     round(float(vals.median())),
                "mean":       round(float(vals.mean())),
                "p25":        round(float(vals.quantile(0.25))),
                "p75":        round(float(vals.quantile(0.75))),
            }

    # ── Scale observed nearby (dwelling units as floor-count proxy) ──────────
    if f_units and len(nearby_same) > 0:
        units = pd.to_numeric(nearby_same[f_units], errors="coerce").dropna()
        units = units[units > 0]
        if not units.empty:
            out["scale_observed_nearby"] = {
                "based_on_n":            int(len(units)),
                "max_dwelling_units":    int(units.max()),
                "median_dwelling_units": int(units.median()),
                "note": "Toronto permit data does not expose floor count directly. "
                        "Dwelling units serve as a scale proxy for residential projects. "
                        "Actual height limits are governed by zoning by-law and Official Plan.",
            }

    # ── Structure types observed nearby ───────────────────────────────────────
    if f_struct and len(nearby_same) > 0:
        out["structure_types_nearby"] = [
            {"type": str(k), "count": int(v)}
            for k, v in nearby_same[f_struct].value_counts().head(6).items()
        ]

    # ── Ward signal ───────────────────────────────────────────────────────────
    if f_ward and len(nearby_any) > 0:
        ward_mode = nearby_any[f_ward].mode()
        if not ward_mode.empty:
            ward = str(ward_mode.iloc[0])
            ward_permits = df[df[f_ward].astype(str) == ward]
            ward_same_cat = ward_permits[ward_permits["_CATEGORY"] == predicted]
            out["ward_signal"] = {
                "ward": ward,
                "total_active_permits_in_ward": int(len(ward_permits)),
                "same_category_in_ward":        int(len(ward_same_cat)),
            }

    # ── Nearest precedents (3 closest comparable permits) ────────────────────
    if len(nearby_same) > 0:
        precedents = []
        for _, r in nearby_same.head(3).iterrows():
            precedents.append({
                "distance_m":     round(float(r["_DIST_M"])),
                "address":        assemble_address(r, cols),
                "structure_type": str(r.get(f_struct)) if f_struct else None,
                "status":         str(r.get(f_status)) if f_status else None,
                "applied":        str(pd.to_datetime(r.get(f_app), errors="coerce").date())
                                  if f_app and pd.notna(r.get(f_app)) else None,
            })
        out["nearest_precedents"] = precedents

    # ── Historical approval rate (city-wide for this category) ───────────────
    if f_status and len(same_cat_all) > 0:
        statuses = same_cat_all[f_status].astype(str).str.lower()
        issued_keywords    = ["issued", "closed", "completed", "cleared", "active"]
        cancelled_keywords = ["cancel", "withdrawn", "rejected", "denied", "refused"]

        n_approved  = int(statuses.apply(
            lambda s: any(k in s for k in issued_keywords)).sum())
        n_cancelled = int(statuses.apply(
            lambda s: any(k in s for k in cancelled_keywords)).sum())
        n_total = len(statuses)

        out["historical_approval"] = {
            "scope":                  "city_wide_same_category",
            "category":               predicted,
            "total_permits_analyzed": n_total,
            "approved_or_active":     n_approved,
            "cancelled_or_rejected":  n_cancelled,
            "approval_rate_pct":      round(n_approved / n_total * 100, 1) if n_total else None,
            "status_breakdown":       same_cat_all[f_status].value_counts().head(6).to_dict(),
        }

    # ── Historical approval time (city-wide) ──────────────────────────────────
    if f_app and f_iss and len(same_cat_all) > 0:
        days = same_cat_all.apply(
            lambda r: _approval_days(r, f_app, f_iss), axis=1
        ).dropna()
        if not days.empty:
            out["historical_approval_time"] = {
                "scope":        "city_wide_same_category",
                "based_on_n":   int(len(days)),
                "median_days":  round(float(days.median())),
                "mean_days":    round(float(days.mean())),
                "p25_days":     round(float(days.quantile(0.25))),
                "p75_days":     round(float(days.quantile(0.75))),
            }

    # ── Socio-economic impact ─────────────────────────────────────────────────
    socio = {"methodology_notes": []}

    if f_val and len(same_cat_all) > 0:
        vals = _safe_num(same_cat_all[f_val])
        vals = vals[vals > 0]
        if not vals.empty:
            socio["construction_value_per_project_cad"] = {
                "median": round(float(vals.median())),
                "mean":   round(float(vals.mean())),
                "total_across_category_in_sample": round(float(vals.sum())),
            }
            est_proj_value = float(vals.median())
            est_const_jobs = round(est_proj_value * EMPLOYMENT_PER_DOLLAR_CONSTRUCTION, 1)
            socio["construction_employment_proxy"] = {
                "estimated_construction_person_years_for_one_project": est_const_jobs,
                "based_on_median_project_value_cad": round(est_proj_value),
                "multiplier": "1 person-year per $200k construction value (Ontario I-O proxy)",
            }

            # ── Property tax revenue estimate ─────────────────────────────────
            # Real, defensible number — uses City of Toronto 2025 published rates.
            tax_estimate = estimate_property_tax_revenue(predicted, est_proj_value)
            socio["property_tax_revenue_for_one_project"] = tax_estimate

            # Aggregate revenue if all similar permits in sample were built:
            total_proj_value = float(vals.sum())
            agg_tax = estimate_property_tax_revenue(predicted, total_proj_value)
            socio["property_tax_revenue_if_all_built_in_sample"] = {
                "annual_property_tax_cad": agg_tax["annual_property_tax_cad"],
                "ten_year_revenue_cad":    agg_tax["ten_year_revenue_cad"],
                "based_on_n_projects":     int(len(vals)),
                "total_construction_value_cad": round(total_proj_value),
            }

            socio["methodology_notes"].append(
                "Property tax revenue estimated from 2025 City of Toronto tax rates "
                "applied to assessed value (approximated as 75% of construction cost). "
                "Real CVA is set by MPAC per Ontario Assessment Act. "
                "Estimate excludes one-time development charges and education portion variations."
            )

    if f_gfa and predicted in OPERATIONAL_JOBS_PER_M2 and OPERATIONAL_JOBS_PER_M2[predicted] > 0:
        gfas = _safe_num(same_cat_all[f_gfa])
        gfas = gfas[gfas > 0]
        if not gfas.empty:
            median_gfa = float(gfas.median())
            jobs_per_m2 = OPERATIONAL_JOBS_PER_M2[predicted]
            socio["operational_employment_proxy"] = {
                "median_gfa_m2_for_category":         round(median_gfa),
                "jobs_per_m2_assumption":             jobs_per_m2,
                "estimated_ongoing_jobs_per_project": round(median_gfa * jobs_per_m2, 1),
            }
            socio["methodology_notes"].append(
                "Operational employment estimated from gross floor area and "
                "category-specific job density assumptions."
            )

    if f_units and predicted in ("residential", "mixed_use") and len(same_cat_all) > 0:
        units = pd.to_numeric(same_cat_all[f_units], errors="coerce").dropna()
        units = units[units > 0]
        if not units.empty:
            socio["housing_supply"] = {
                "median_units_per_project": int(units.median()),
                "total_units_in_category":  int(units.sum()),
                "n_projects_analyzed":      int(len(units)),
            }

    if len(socio) > 1:
        out["socio_economic_impact"] = socio

    # ── Saturation analysis + recommendation ─────────────────────────────────
    saturation_score = None
    same_in_radius   = len(nearby_same)
    total_in_radius  = len(nearby_any)

    if total_in_radius > 0:
        density = same_in_radius / total_in_radius
        if same_in_radius >= 10 and density > 0.5:
            saturation_level, saturation_score = "high", round(density * 100, 1)
        elif same_in_radius >= 5 and density > 0.3:
            saturation_level, saturation_score = "moderate", round(density * 100, 1)
        elif same_in_radius >= 2:
            saturation_level, saturation_score = "low", round(density * 100, 1)
        else:
            saturation_level, saturation_score = "minimal", round(density * 100, 1)

        out["saturation_analysis"] = {
            "radius_m":                     SEARCH_RADIUS_M,
            "same_category_in_radius":      same_in_radius,
            "all_permits_in_radius":        total_in_radius,
            "density_pct_of_same_category": saturation_score,
            "saturation_level":             saturation_level,
            "is_oversaturated":             saturation_level == "high",
        }

        if saturation_level == "high":
            out["recommendation"] = {
                "verdict": "consider_alternate_location",
                "reason":  f"{same_in_radius} similar {predicted} permits already exist "
                           f"within {SEARCH_RADIUS_M}m. New development may further "
                           f"concentrate this use type when other areas are underserved.",
            }
        elif saturation_level == "minimal":
            out["recommendation"] = {
                "verdict": "approve_recommended",
                "reason":  f"Only {same_in_radius} similar permit(s) in the area. "
                           f"This development would diversify the local building mix.",
            }
        else:
            out["recommendation"] = {
                "verdict": "approve_with_standard_review",
                "reason":  f"Existing {predicted} density in area is {saturation_level}. "
                           f"No saturation concerns.",
            }

    # ── Alternate location suggestions when saturated ─────────────────────────
    if saturation_score is not None and saturation_score > 50 and f_ward:
        ward_counts = df.groupby([f_ward, "_CATEGORY"]).size().unstack(fill_value=0)
        if predicted in ward_counts.columns:
            underserved = ward_counts.sort_values(predicted).head(5)
            alternates = []
            for ward, row in underserved.iterrows():
                if pd.isna(ward) or str(ward).lower() in ("nan", "none"):
                    continue
                alternates.append({
                    "ward":                  str(ward),
                    "same_category_permits": int(row.get(predicted, 0)),
                    "total_permits_in_ward": int(row.sum()),
                })
            out["alternate_locations"] = {
                "rationale":         f"These wards have fewer {predicted} developments, "
                                     f"suggesting capacity for additional projects.",
                "underserved_wards": alternates[:5],
            }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached at module level for repeated calls)
# ─────────────────────────────────────────────────────────────────────────────

_data_cache = {"df": None, "limit": None}


def load_and_prepare_data(limit: int = 500, geocode_limit: int = None,
                          force_reload: bool = False,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Load permits, classify, geocode (cached), return ready-to-query DataFrame.
    Cached across calls in the same process.
    """
    if (not force_reload and _data_cache["df"] is not None
            and _data_cache["limit"] == limit):
        return _data_cache["df"]

    if verbose:
        print(f"Loading {limit} permits from Toronto Open Data...", file=sys.stderr)
    try:
        df = get_building_permits(limit)
    except NameError:
        raise RuntimeError("toronto_data_service.py not found in same directory")

    if verbose:
        print("Classifying buildings...", file=sys.stderr)
    df = classify_all(df)

    if verbose:
        print("Geocoding addresses (using cache)...", file=sys.stderr)
    df = geocode_dataframe(df, max_geocode=geocode_limit, verbose=verbose)

    _data_cache["df"]    = df
    _data_cache["limit"] = limit
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST HANDLER (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def handle_request(request: dict, df: pd.DataFrame = None,
                   limit: int = 500, geocode_limit: int = None) -> dict:
    """
    Main agent entry point. Returns the same unified payload regardless of
    who is asking (constructor, government, or any downstream consumer).
    Loads data on first call (cached afterwards).
    """
    address = (request.get("address") or "").strip()
    if not address:
        return {
            "error":    "address is required",
            "received": request,
        }

    if df is None:
        df = load_and_prepare_data(limit=limit, geocode_limit=geocode_limit)

    coord = geocode_address(address)
    result = build_unified_output(request, df, coord)

    return {
        "agent":        "toronto_building_permit_agent",
        "version":      "3.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "result":       result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not JSON serializable")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Toronto Building Permit Intelligence Agent"
    )
    parser.add_argument("--input",         type=str,
                        help="Path to JSON file containing the request")
    parser.add_argument("--stdin",         action="store_true",
                        help="Read request JSON from stdin")
    parser.add_argument("--output", "-o",  type=str,
                        help="Write result JSON to this file (default: stdout)")
    parser.add_argument("--limit",         type=int, default=500,
                        help="Records to fetch from Toronto Open Data")
    parser.add_argument("--geocode-limit", type=int, default=None,
                        help="Max NEW geocoding calls per run (cache hits unlimited)")
    parser.add_argument("--pretty",        action="store_true",
                        help="Pretty-print JSON")
    args = parser.parse_args()

    # ── Load request ──────────────────────────────────────────────────────────
    if args.stdin:
        request = json.loads(sys.stdin.read())
    elif args.input:
        request = json.loads(Path(args.input).read_text())
    else:
        parser.error("Provide --input <file.json> or --stdin")

    # ── Run agent ─────────────────────────────────────────────────────────────
    result = handle_request(request, limit=args.limit,
                            geocode_limit=args.geocode_limit)

    # ── Output ────────────────────────────────────────────────────────────────
    output_str = json.dumps(
        result, default=_json_safe,
        indent=2 if args.pretty else None,
        ensure_ascii=False,
    )

    if args.output:
        Path(args.output).write_text(output_str)
        print(f"Wrote {len(output_str):,} chars → {args.output}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()