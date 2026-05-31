"""
Toronto building-permit sub-agent.

Modes:
  python building_permit.py build <insights.json> [--max-new N]
      Fetches live permit data + address points from Toronto Open Data,
      builds a fast geocoding index, geocodes every permit address, and
      writes insights. Produces four files:
        • address_index.json  (civic key → lat/lon, ~500k entries)
        • geo_cache.json      (universal address → coords cache)
        • permit_coords.json  (per-permit coords, used by spatial search)
        • <insights.json>     (analytics, the path you provide)
      Use --max-new to cap NEW Nominatim fallback calls per run.

      First run builds the address index (~30s for 500k addresses); after
      that, geocoding is a hash lookup — ~1000× faster than Nominatim.

  echo '{"lat":43.65,"lon":-79.38,"user_question":"..."}' | python building_permit.py run

Programmatic:
  from building_permit import run
  run({"lat":43.65,"lon":-79.38,"user_question":"permits within 500m?"})
"""
import os, sys, json, math, time, re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.services.toronto_data_service import (
    get_building_permits,
    get_address_points,
)

# ============================================================
# CONFIG
# ============================================================
INSIGHTS_PATH    = os.environ.get("INSIGHTS_PATH",    "insights.json")
GEO_CACHE_PATH   = os.environ.get("GEO_CACHE_PATH",   "geo_cache.json")
PERMIT_COORDS_PATH = os.environ.get("PERMIT_COORDS_PATH", "permit_coords.json")
ADDRESS_INDEX_PATH  = os.environ.get("ADDRESS_INDEX_PATH",
                                     "address_index.json")
MODEL            = os.environ.get("PERMIT_AGENT_MODEL", "gpt-4o-mini")

# Nominatim geocoding (free, no key, 1 req/sec policy) — fallback only
NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "TorontoPermitAgent/1.0 (urban-dev-research)"}
NOMINATIM_DELAY_SEC = 1.0

DATE_COLS = ["APPLICATION_DATE", "ISSUED_DATE", "COMPLETED_DATE"]
NUM_COLS  = ["DWELLING_UNITS_CREATED", "DWELLING_UNITS_LOST", "EST_CONST_COST",
             "ASSEMBLY", "INSTITUTIONAL", "RESIDENTIAL",
             "BUSINESS_AND_PERSONAL_SERVICES", "MERCANTILE", "INDUSTRIAL",
             "INTERIOR_ALTERATIONS", "DEMOLITION"]
MIN_PERMITS_BUILDER = 10
DEFAULT_RADIUS_M    = 500    # default spatial-search radius for nearby permits

# ============================================================
# CITY OF TORONTO 2025 PROPERTY TAX RATES
# Source: City of Toronto, "Property Tax Rates & Fees" + 2025 Budget docs.
# Each rate = combined municipal + City Building Fund + education.
# Applied against a property's MPAC Current Value Assessment (CVA).
# ============================================================
TORONTO_TAX_RATES_2025 = {
    "residential":         0.00754087,  # 0.754087%
    "new_multi_res":       0.00754087,  # same as residential for 35 years
    "multi_residential":   0.01197305,  # 1.197305%
    "commercial":          0.02275478,  # 2.275478%
    "industrial":          0.02500000,  # ~2.5% (avg across sub-classes)
}

# Routes a permit's structure / use category to a Toronto tax class.
# These are the most common labels you'll see in STRUCTURE_TYPE or PERMIT_TYPE.
STRUCTURE_TO_TAX_CLASS = {
    # residential
    "sfd":                          "residential",
    "single family dwelling":       "residential",
    "semi-detached":                "residential",
    "townhouse":                    "residential",
    "row house":                    "residential",
    "house":                        "residential",
    "condominium":                  "residential",
    "condo":                        "residential",
    # multi-residential
    "apartment":                    "multi_residential",
    "apartment building":           "multi_residential",
    "multi-residential":            "multi_residential",
    # commercial
    "commercial":                   "commercial",
    "retail":                       "commercial",
    "office":                       "commercial",
    "mercantile":                   "commercial",
    "mixed use":                    "commercial",
    "mixed-use":                    "commercial",
    "institutional":                "commercial",
    "assembly":                     "commercial",
    # industrial
    "industrial":                   "industrial",
    "warehouse":                    "industrial",
    "factory":                      "industrial",
}

# MPAC's CVA is anchored to 2016 valuations; current construction cost is
# typically higher than assessed value. 0.75 is a conservative proxy until
# the agent has per-property MPAC access.
CONSTRUCTION_COST_TO_CVA_RATIO = 0.75


def _classify_tax_class(structure_type: str = None,
                        permit_type: str = None) -> str:
    """Map a free-text structure or permit type to a Toronto tax class."""
    for source in (structure_type, permit_type):
        if not source:
            continue
        s = str(source).lower().strip()
        # Try exact match first, then substring
        if s in STRUCTURE_TO_TAX_CLASS:
            return STRUCTURE_TO_TAX_CLASS[s]
        for key, tax_class in STRUCTURE_TO_TAX_CLASS.items():
            if key in s:
                return tax_class
    return "residential"  # safe default


# ============================================================
# GEOCODING
# ============================================================
# Two-file cache strategy:
#   geo_cache.json     — normalized_address → {lat, lon, display_name}
#                        (key/value store, survives runs, sharable across scripts)
#   permit_coords.json — permit_number     → {lat, lon, fsa, address}
#                        (per-permit lookup, used by spatial search at runtime)
# ============================================================

def _normalize_address(addr: str, city: str = "Toronto") -> str:
    if not addr:
        return ""
    addr = re.sub(r"\s+", " ", str(addr).strip()).lower()
    return f"{addr}, {city.lower()}"


# ─────────────── Civic-key normalization for the address index ──────────────
# Toronto's Address Points dataset has ADDRESS_FULL like "1871 Davenport Rd",
# while permit rows have STREET_NUM + STREET_NAME + STREET_TYPE in separate
# columns. We normalize both to a common "civic key" so a hash lookup works:
#   "1871 davenport rd"  →  (lat, lon)

# Common street-type variants to canonicalize (avoid spurious misses)
_STREET_TYPE_MAP = {
    "street":  "st",   "st.":  "st",
    "avenue":  "ave",  "ave.": "ave",
    "road":    "rd",   "rd.":  "rd",
    "drive":   "dr",   "dr.":  "dr",
    "boulevard": "blvd","blvd.":"blvd",
    "court":   "crt",  "crt.": "crt",
    "crescent":"cres", "cres.":"cres",
    "place":   "pl",   "pl.":  "pl",
    "parkway": "pkwy", "pkwy.":"pkwy",
    "lane":    "lane",
    "trail":   "trl",  "trl.": "trl",
    "way":     "way",
    "circle":  "circ", "circ.":"circ",
    "terrace": "ter",  "ter.": "ter",
}

# Common street-direction variants
_STREET_DIR_MAP = {
    "north": "n", "south": "s", "east": "e", "west": "w",
    "n.": "n", "s.": "s", "e.": "e", "w.": "w",
}


def _civic_key(number, name, stype="", direction="") -> str:
    """Build a normalized 'number street type direction' key."""
    def clean(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return re.sub(r"\s+", " ", str(x).strip().lower())

    number = clean(number)
    name   = clean(name)
    stype  = _STREET_TYPE_MAP.get(clean(stype), clean(stype))
    direction = _STREET_DIR_MAP.get(clean(direction), clean(direction))

    if not number or not name:
        return ""

    parts = [number, name]
    if stype:     parts.append(stype)
    if direction: parts.append(direction)
    return " ".join(parts)


def _civic_key_from_address_full(addr_full: str) -> str:
    """
    Parse an ADDRESS_FULL string like '1871 Davenport Rd' into the same
    normalized civic key. Splits "<number> <rest>" and normalizes street types.
    """
    if not addr_full:
        return ""
    s = re.sub(r"\s+", " ", str(addr_full).strip().lower())
    m = re.match(r"^(\d+\S*)\s+(.+)$", s)
    if not m:
        return ""
    number, rest = m.group(1), m.group(2)
    tokens = rest.split()

    # Canonicalize each token if it matches a known type/direction
    out = []
    for t in tokens:
        if t in _STREET_TYPE_MAP:
            out.append(_STREET_TYPE_MAP[t])
        elif t in _STREET_DIR_MAP:
            out.append(_STREET_DIR_MAP[t])
        else:
            out.append(t)
    return number + " " + " ".join(out)


def build_address_index(df: pd.DataFrame = None,
                        index_out_path: str = None,
                        limit: int = 600_000,
                        verbose: bool = True) -> dict:
    """
    Build a fast civic-key → (lat, lon) index from Toronto's Address Points
    dataset. Saved as JSON for portability and inspectability.

    Source of address points (in order):
      • df       → use a pre-loaded DataFrame
      • neither  → fetch via get_address_points(limit=limit) from
                   src.services.toronto_data_service

    Expected columns: ADDRESS_FULL, geometry (GeoJSON Point dict or string)
    """
    index_out_path = index_out_path or ADDRESS_INDEX_PATH

    if df is None:
        if verbose:
            print(f"  fetching address points from "
                  f"get_address_points(limit={limit:,})…", file=sys.stderr)
        df = get_address_points(limit=limit)

    if verbose:
        print(f"  {len(df):,} address points loaded", file=sys.stderr)

    # Identify columns case-insensitively
    cols = {c.upper(): c for c in df.columns}
    addr_col = cols.get("ADDRESS_FULL")
    geom_col = cols.get("GEOMETRY")
    if not addr_col or not geom_col:
        raise ValueError(
            f"Expected ADDRESS_FULL and geometry columns. Found: {list(df.columns)}"
        )

    index = {}
    n_skipped = 0

    for _, row in df.iterrows():
        addr_full = row[addr_col]
        key = _civic_key_from_address_full(addr_full)
        if not key:
            n_skipped += 1
            continue

        # Parse geometry — may be GeoJSON dict, JSON string, or list
        geom = row[geom_col]
        lat, lon = None, None
        try:
            if isinstance(geom, str):
                geom = json.loads(geom)
            if isinstance(geom, dict) and geom.get("type") == "Point":
                lon, lat = geom["coordinates"]
            elif isinstance(geom, (list, tuple)) and len(geom) == 2:
                lon, lat = geom
        except Exception:
            n_skipped += 1
            continue

        if lat is None or lon is None:
            n_skipped += 1
            continue

        # Last write wins for duplicates (usually identical or near-identical)
        index[key] = {"lat": float(lat), "lon": float(lon)}

    _save_json(index_out_path, index)

    if verbose:
        print(f"  built address index: {len(index):,} entries "
              f"({n_skipped:,} skipped)", file=sys.stderr)
        print(f"  wrote {index_out_path}", file=sys.stderr)

    return {"entries": len(index), "skipped": n_skipped,
            "written_to": index_out_path}


@lru_cache(maxsize=1)
def _address_index():
    """Cached load of the address index — used by geocode_one fast path."""
    return _load_json(ADDRESS_INDEX_PATH, default={})


def lookup_civic(number, name, stype="", direction="") -> dict | None:
    """O(1) lookup against the address index. Returns {lat, lon} or None."""
    idx = _address_index()
    if not idx:
        return None
    key = _civic_key(number, name, stype, direction)
    if not key:
        return None
    hit = idx.get(key)
    if hit:
        return {"lat": hit["lat"], "lon": hit["lon"],
                "display_name": f"{number} {name}".strip(),
                "source": "address_points_index"}
    return None


def _load_json(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return default if default is not None else {}


def _save_json(path: str, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def _assemble_address(row: pd.Series) -> str:
    """Build a clean address string from permit columns."""
    parts = []
    for col in ("STREET_NUM", "STREET_NAME", "STREET_TYPE", "STREET_DIRECTION"):
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none"):
                parts.append(val)
    return " ".join(parts)


def geocode_one(address: str, city: str = "Toronto",
                cache: dict = None, rate_limit: bool = True) -> dict | None:
    """
    Geocode a single address via Nominatim. Returns {lat, lon, display_name}
    or None on failure. Uses provided cache dict (mutated in place).
    """
    if not address:
        return None
    key = _normalize_address(address, city)

    if cache is not None and key in cache:
        return cache[key]

    try:
        if rate_limit:
            time.sleep(NOMINATIM_DELAY_SEC)
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": f"{address}, {city}, Ontario, Canada",
                    "format": "json", "limit": 1},
            headers=NOMINATIM_HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            record = None
        else:
            r = results[0]
            record = {"lat": float(r["lat"]),
                      "lon": float(r["lon"]),
                      "display_name": r.get("display_name")}
    except Exception as e:
        print(f"  geocode error for '{address}': {e}", file=sys.stderr)
        record = None

    if cache is not None:
        cache[key] = record
    return record


def geocode_permits(df: pd.DataFrame = None,
                    geo_cache_path: str = None,
                    coords_out_path: str = None,
                    max_new: int = None,
                    verbose: bool = True) -> dict:
    """
    Offline: walk the permit data, geocode each address once, save two files:
      • geo_cache.json     — universal address → coords map
      • permit_coords.json — permit_number → coords (used at runtime)

    Source of permits (pick one):
      • csv_path → load via pd.read_csv
      • df       → use a pre-loaded DataFrame (lets the caller share work
                   with build_insights without re-loading)

    Resumable: re-runs only geocode addresses not already in the cache.
    Honors Nominatim's 1 req/sec rate limit.
    """
    if df is None and csv_path is None:
        raise ValueError("Provide either csv_path or df")
    geo_cache_path   = geo_cache_path   or GEO_CACHE_PATH
    coords_out_path  = coords_out_path  or PERMIT_COORDS_PATH

    if df is None:
        df = pd.read_csv(csv_path, low_memory=False)

    # Find columns case-insensitively
    cols = {c.upper(): c for c in df.columns}
    permit_col = cols.get("PERMIT_NUM") or cols.get("PERMIT_NUMBER")
    postal_col = cols.get("POSTAL")
    num_col    = cols.get("STREET_NUM")
    name_col   = cols.get("STREET_NAME")
    type_col   = cols.get("STREET_TYPE")
    dir_col    = cols.get("STREET_DIRECTION")

    cache         = _load_json(geo_cache_path, default={})
    permit_coords = _load_json(coords_out_path, default={})
    addr_index    = _address_index()    # cached, fast lookup

    n_total       = len(df)
    n_skipped     = 0
    n_cached      = 0
    n_index_hit   = 0      # NEW: hits against Address Points index
    n_new         = 0      # only counts Nominatim calls
    n_failed      = 0

    if verbose:
        print(f"  geocoding {n_total} permits → cache: {geo_cache_path}",
              file=sys.stderr)
        print(f"  address index has {len(addr_index):,} entries; "
              f"cache has {len(cache):,} addresses, "
              f"{len(permit_coords):,} permits already mapped", file=sys.stderr)

    for idx, row in df.iterrows():
        permit_id = str(row.get(permit_col, "")).strip() if permit_col else None
        print(permit_id)
        # Skip if we already have this permit
        if permit_id and permit_id in permit_coords:
            n_skipped += 1
            continue

        address = _assemble_address(row)
        if not address:
            n_failed += 1
            continue

        # ── FAST PATH: Toronto Address Points index ──────────────────────────
        record = None
        if addr_index and num_col and name_col:
            record = lookup_civic(
                row.get(num_col),
                row.get(name_col),
                row.get(type_col) if type_col else "",
                row.get(dir_col)  if dir_col  else "",
            )
            if record:
                n_index_hit += 1
                # Also seed the address-string cache so future runs
                # that take the slow path benefit
                cache[_normalize_address(address)] = {
                    "lat": record["lat"], "lon": record["lon"],
                    "display_name": record["display_name"],
                }
                print(record)

        # ── FALLBACK: in-memory address cache or Nominatim ───────────────────
        # if record is None:
        #     key = _normalize_address(address)
        #     in_cache = key in cache

        #     # Respect max_new cap — only Nominatim calls count
        #     if not in_cache and max_new is not None and n_new >= max_new:
        #         n_skipped += 1
        #         continue

        #     record = geocode_one(address, cache=cache,
        #                          rate_limit=not in_cache)
        #     if in_cache:
        #         n_cached += 1
        #     else:
        #         n_new += 1
        #         if verbose and n_new % 25 == 0:
        #             print(f"  …Nominatim calls: {n_new} "
        #                   f"(index hits so far: {n_index_hit})", file=sys.stderr)

        if record and permit_id:
            permit_coords[permit_id] = {
                "lat":     record["lat"],
                "lon":     record["lon"],
                "address": address,
                "fsa":     (str(row[postal_col]).strip().upper()[:3]
                            if postal_col and pd.notna(row.get(postal_col)) else None),
            }
        elif not record:
            n_failed += 1

        # Periodic flush so a crash doesn't lose progress
        if (n_new + n_index_hit) and (n_new + n_index_hit) % 100 == 0:
            _save_json(geo_cache_path,  cache)
            _save_json(coords_out_path, permit_coords)

    _save_json(geo_cache_path,  cache)
    _save_json(coords_out_path, permit_coords)

    summary = {
        "total_permits":       n_total,
        "already_mapped":      n_skipped,
        "index_hits":          n_index_hit,
        "string_cache_hits":   n_cached,
        "nominatim_calls":     n_new,
        "failed":              n_failed,
        "final_cache_size":    len(cache),
        "final_coords_size":   len(permit_coords),
    }
    if verbose:
        print(f"  done. {json.dumps(summary, indent=2)}", file=sys.stderr)
    return summary


# ─────────────── Spatial utilities (used by the runtime tool) ────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _permit_coords():
    """Cached load of permit_coords.json — used by runtime spatial search."""
    return _load_json(PERMIT_COORDS_PATH, default={})


# ============================================================
# OFFLINE: BUILD insights.json
# ============================================================
def _pct(s, p):
    s = s.dropna()
    return float(np.nanpercentile(s, p)) if len(s) else None

def _stats(s):
    s = s.dropna()
    if s.empty: return None
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
            "p25": _pct(s,25), "p75": _pct(s,75),
            "p90": _pct(s,90), "p99": _pct(s,99),
            "min": float(s.min()), "max": float(s.max())}

def _group_stats(frame, by, value="approval_days", min_n=20, top=25):
    out = []
    for key, vals in frame.groupby(by)[value]:
        v = vals.dropna()
        if len(v) < min_n: continue
        out.append({"key": key if not isinstance(key, tuple) else list(key),
                    "n": int(len(v)), "median_days": float(v.median()),
                    "p90_days": _pct(v,90), "mean_days": float(v.mean())})
    out.sort(key=lambda x: -x["n"])
    return out[:top]

def _builder_table(frame, min_n=MIN_PERMITS_BUILDER, top=50, ascending=True):
    rows = []
    for name, vals in frame.groupby("BUILDER_NAME")["approval_days"]:
        v = vals.dropna()
        if len(v) < min_n: continue
        rows.append({"builder": name, "n_permits": int(len(v)),
                     "median_days": float(v.median()),
                     "p90_days": _pct(v,90), "mean_days": float(v.mean())})
    rows.sort(key=lambda r: r["median_days"], reverse=not ascending)
    return rows[:top]

def _builder_delta_table(frame, min_n=MIN_PERMITS_BUILDER, top=25, ascending=True):
    rows = []
    for name, sub in frame.groupby("BUILDER_NAME"):
        v = sub["speed_delta"].dropna()
        if len(v) < min_n: continue
        rows.append({"builder": name, "n_permits": int(len(v)),
                     "median_delta_days": float(v.median()),
                     "median_days_absolute": float(sub["approval_days"].median())})
    rows.sort(key=lambda r: r["median_delta_days"], reverse=not ascending)
    return rows[:top]

def build_insights(csv_path: str = None, df: pd.DataFrame = None) -> dict:
    """
    Build the analytics insights bundle.

    Source of permits (pick one):
      • csv_path → load via pd.read_csv
      • df       → use a pre-loaded DataFrame
    """
    if df is None and csv_path is None:
        raise ValueError("Provide either csv_path or df")
    if df is None:
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        df = df.copy()  # don't mutate the caller's frame
    for c in DATE_COLS: df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUM_COLS:  df[c] = pd.to_numeric(df[c], errors="coerce")
    df["FSA"]             = df["POSTAL"].astype(str).str.strip().str.upper().str[:3]
    df["approval_days"]   = (df["ISSUED_DATE"]    - df["APPLICATION_DATE"]).dt.days
    df["completion_days"] = (df["COMPLETED_DATE"] - df["ISSUED_DATE"]).dt.days
    df["app_year"]        = df["APPLICATION_DATE"].dt.year
    issued = df[df["ISSUED_DATE"].notna() & (df["approval_days"] >= 0)].copy()
    b = issued.copy()
    b["BUILDER_NAME"] = b["BUILDER_NAME"].fillna("").str.strip().str.upper()
    b = b[b["BUILDER_NAME"] != ""]
    fsa_median = issued.groupby("FSA")["approval_days"].median()
    b["fsa_baseline"] = b["FSA"].map(fsa_median)
    b["speed_delta"]  = b["approval_days"] - b["fsa_baseline"]
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": csv_path,
        "dataset": {
            "rows": int(len(df)),
            "date_range": {
                "application_min": str(df["APPLICATION_DATE"].min()),
                "application_max": str(df["APPLICATION_DATE"].max()),
            },
            "null_rate": {c: float(df[c].isna().mean()) for c in df.columns},
        },
        "status_breakdown":      df["STATUS"].value_counts(dropna=False).to_dict(),
        "permit_type_counts":    df["PERMIT_TYPE"].value_counts().head(30).to_dict(),
        "structure_type_counts": df["STRUCTURE_TYPE"].value_counts().head(30).to_dict(),
        "work_counts":           df["WORK"].value_counts().head(30).to_dict(),
        "approval_time_overall":   _stats(issued["approval_days"]),
        "completion_time_overall": _stats(df["completion_days"]),
        "approval_time_by_permit_type":    _group_stats(issued, "PERMIT_TYPE"),
        "approval_time_by_structure_type": _group_stats(issued, "STRUCTURE_TYPE"),
        "approval_time_by_work":           _group_stats(issued, "WORK"),
        "approval_time_by_fsa":            _group_stats(issued, "FSA", top=200),
        "approval_time_by_ward":           _group_stats(issued, "WARD_GRID", top=50),
        "approval_time_by_year":           _group_stats(issued, "app_year",
                                                         min_n=50, top=50),
        "approval_time_by_type_and_fsa":   _group_stats(issued,
                                                        ["PERMIT_TYPE","FSA"],
                                                         min_n=20, top=500),
        "volume_by_year":  df.groupby("app_year").size().dropna().astype(int).to_dict(),
        "cost": {
            "overall": _stats(df["EST_CONST_COST"]),
            "by_structure_type": {
                k: _stats(g["EST_CONST_COST"])
                for k, g in df.groupby("STRUCTURE_TYPE") if len(g) >= 50
            },
        },
        "dwelling_units": {
            "total_created": int(df["DWELLING_UNITS_CREATED"].sum(skipna=True)),
            "total_lost":    int(df["DWELLING_UNITS_LOST"].sum(skipna=True)),
            "net_added":     int((df["DWELLING_UNITS_CREATED"]
                                  - df["DWELLING_UNITS_LOST"]).sum(skipna=True)),
        },
        "top_builders_by_volume": (
            df["BUILDER_NAME"].dropna().str.strip().str.upper()
              .value_counts().head(25).to_dict()
        ),
        "builder_performance": {
            "min_permits_threshold": MIN_PERMITS_BUILDER,
            "city_median_days": float(issued["approval_days"].median()),
            "fastest_builders":         _builder_table(b, ascending=True,  top=25),
            "slowest_builders":         _builder_table(b, ascending=False, top=25),
            "most_active_builders":     sorted(
                _builder_table(b, top=10_000),
                key=lambda r: -r["n_permits"])[:25],
            "fastest_builders_vs_area": _builder_delta_table(b, ascending=True,  top=25),
            "slowest_builders_vs_area": _builder_delta_table(b, ascending=False, top=25),
            "fastest_by_permit_type": {
                p: _builder_table(b[b["PERMIT_TYPE"]==p], min_n=10, top=10)
                for p in b["PERMIT_TYPE"].value_counts().head(8).index
            },
            "fastest_by_structure_type": {
                s: _builder_table(b[b["STRUCTURE_TYPE"]==s], min_n=10, top=10)
                for s in b["STRUCTURE_TYPE"].value_counts().head(8).index
            },
            "fastest_by_fsa": {
                f: _builder_table(b[b["FSA"]==f], min_n=5, top=10)
                for f in b["FSA"].value_counts().head(25).index
            },
        },
        "pending_backlog": {
            "still_open":  int(df["ISSUED_DATE"].isna().sum()),
            "by_status":   df[df["ISSUED_DATE"].isna()]["STATUS"]
                             .value_counts().to_dict(),
        },
        "data_quality_flags": {
            "negative_approval_days":   int((df["approval_days"] < 0).sum()),
            "missing_postal":           int(df["POSTAL"].isna().sum()),
            "missing_application_date": int(df["APPLICATION_DATE"].isna().sum()),
        },
    }


# ============================================================
# RUNTIME: TOOLS over insights.json
# ============================================================
@lru_cache(maxsize=1)
def _insights():
    with open(INSIGHTS_PATH) as f:
        return json.load(f)

def lookup_approval_time(permit_type=None, fsa=None, ward=None,
                         structure_type=None, min_n=20):
    I = _insights()
    if permit_type and fsa:
        for r in I["approval_time_by_type_and_fsa"]:
            if r["key"] == [permit_type, fsa] and r["n"] >= min_n:
                return {**r, "fallback_level": "permit_type+fsa"}
    if fsa:
        for r in I["approval_time_by_fsa"]:
            if r["key"] == fsa and r["n"] >= min_n:
                return {**r, "fallback_level": "fsa"}
    if ward:
        for r in I["approval_time_by_ward"]:
            if r["key"] == ward and r["n"] >= min_n:
                return {**r, "fallback_level": "ward"}
    if structure_type:
        for r in I["approval_time_by_structure_type"]:
            if r["key"] == structure_type and r["n"] >= min_n:
                return {**r, "fallback_level": "structure_type"}
    if permit_type:
        for r in I["approval_time_by_permit_type"]:
            if r["key"] == permit_type and r["n"] >= min_n:
                return {**r, "fallback_level": "permit_type"}
    o = I["approval_time_overall"]
    return {"n": o["n"], "median_days": o["median"], "p90_days": o["p90"],
            "mean_days": o["mean"], "fallback_level": "overall"}

def lookup_builders(fsa=None, permit_type=None, structure_type=None,
                    mode="vs_area", top=5):
    bp = _insights()["builder_performance"]
    if fsa and fsa in bp["fastest_by_fsa"]:
        return bp["fastest_by_fsa"][fsa][:top]
    if structure_type and structure_type in bp["fastest_by_structure_type"]:
        return bp["fastest_by_structure_type"][structure_type][:top]
    if permit_type and permit_type in bp["fastest_by_permit_type"]:
        return bp["fastest_by_permit_type"][permit_type][:top]
    key = "fastest_builders_vs_area" if mode == "vs_area" else "fastest_builders"
    return bp[key][:top]

def dataset_summary():
    I = _insights()
    return {
        "rows": I["dataset"]["rows"],
        "date_range": I["dataset"]["date_range"],
        "city_median_approval_days": I["builder_performance"]["city_median_days"],
        "open_permits": I["pending_backlog"]["still_open"],
    }


def nearby_permits(lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M,
                   top: int = 25) -> dict:
    """
    Spatial lookup: find permits within `radius_m` metres of (lat, lon).
    Reads from the offline-built permit_coords.json cache.
    Returns aggregate stats + up to `top` nearest permits.
    """
    coords = _permit_coords()
    if not coords:
        return {
            "error": "permit_coords.json is empty. "
                     "Run: python building_permit.py geocode <permits.csv>",
            "n_permits_indexed": 0,
        }
    if lat is None or lon is None:
        return {"error": "lat and lon required"}

    matches = []
    for permit_id, rec in coords.items():
        d = haversine_m(lat, lon, rec["lat"], rec["lon"])
        if d <= radius_m:
            matches.append({"permit_num": permit_id,
                            "distance_m": round(d, 1),
                            "address":    rec.get("address"),
                            "fsa":        rec.get("fsa")})

    matches.sort(key=lambda x: x["distance_m"])
    fsa_counts = {}
    for m in matches:
        if m["fsa"]:
            fsa_counts[m["fsa"]] = fsa_counts.get(m["fsa"], 0) + 1

    return {
        "query":             {"lat": lat, "lon": lon, "radius_m": radius_m},
        "n_permits_indexed": len(coords),
        "n_in_radius":       len(matches),
        "fsa_breakdown":     dict(sorted(fsa_counts.items(),
                                          key=lambda kv: -kv[1])),
        "nearest":           matches[:top],
    }


def estimate_property_tax(structure_type: str = None,
                          permit_type: str = None,
                          construction_value_cad: float = None,
                          tax_class_override: str = None) -> dict:
    """
    Estimate annual + 10-year property tax for a project using City of Toronto
    2025 rates. If construction_value_cad is omitted, falls back to the median
    construction cost for the matching structure_type from insights.json.

    Routing:
      - explicit `tax_class_override` wins (must be one of TORONTO_TAX_RATES_2025 keys)
      - otherwise infers from structure_type, then permit_type
    """
    # ── Determine tax class ───────────────────────────────────────────────────
    if tax_class_override and tax_class_override in TORONTO_TAX_RATES_2025:
        tax_class = tax_class_override
        routing = "explicit_override"
    else:
        tax_class = _classify_tax_class(structure_type, permit_type)
        routing = f"inferred_from_{structure_type or permit_type or 'default'}"

    rate = TORONTO_TAX_RATES_2025[tax_class]

    # ── Determine construction value ──────────────────────────────────────────
    value_source = "user_provided"
    if construction_value_cad is None:
        I = _insights()
        cost_by_struct = I.get("cost", {}).get("by_structure_type", {})
        if structure_type and structure_type in cost_by_struct \
                and cost_by_struct[structure_type] \
                and cost_by_struct[structure_type].get("median"):
            construction_value_cad = cost_by_struct[structure_type]["median"]
            value_source = f"median_for_{structure_type}_in_dataset"
        elif I.get("cost", {}).get("overall", {}).get("median"):
            construction_value_cad = I["cost"]["overall"]["median"]
            value_source = "city_wide_median_cost"
        else:
            return {
                "error": "construction_value_cad required and no fallback available",
                "tax_class_applied": tax_class,
            }

    # ── Compute ───────────────────────────────────────────────────────────────
    estimated_cva   = construction_value_cad * CONSTRUCTION_COST_TO_CVA_RATIO
    annual_tax      = estimated_cva * rate
    ten_year_tax    = annual_tax * 10

    return {
        "tax_class_applied":          tax_class,
        "tax_class_routing":          routing,
        "tax_rate_2025":              rate,
        "tax_rate_pct":               round(rate * 100, 4),
        "construction_value_cad":     round(construction_value_cad),
        "construction_value_source":  value_source,
        "estimated_cva_cad":          round(estimated_cva),
        "cva_to_construction_ratio":  CONSTRUCTION_COST_TO_CVA_RATIO,
        "annual_property_tax_cad":    round(annual_tax),
        "ten_year_revenue_cad":       round(ten_year_tax),
        "source": "City of Toronto 2025 property tax rates "
                  "(municipal + City Building Fund + education, combined). "
                  "CVA approximated from construction cost; actual CVA is set "
                  "by MPAC per Ontario Assessment Act.",
        "caveats": [
            "Excludes one-time development charges (DC) and parkland levies",
            "MPAC assessments are anchored to 2016 valuation date",
            "Industrial sub-classes vary; rate shown is an average",
        ],
    }


# ============================================================
# RUNTIME: LLM AGENT
# ============================================================
SYSTEM = """You answer Toronto building-permit questions using ONLY the tools provided.

The user's location arrives as structured fields. Use them as follows:
  • lat/lon  → call nearby_permits for spatial searches (within Xm)
  • fsa/ward → call lookup_approval_time / lookup_builders for area stats
  • address  → already resolved upstream; do NOT geocode yourself

Rules:
1. Never invent numbers — every figure must come from a tool result.
2. Always include sample size (n) and fallback_level when citing approval times.
3. Prefer 'vs_area' mode when ranking builders; always cite n_permits.
4. If n < 20 for any stat, warn the user it is low-confidence.
5. When asked about revenue, taxes, or fiscal impact, call estimate_property_tax.
   Always quote the tax_class_applied and source so the user can verify.
6. For "near this location" or "within X metres" questions, call nearby_permits
   with the user's lat/lon. Cite n_in_radius and the actual radius used.
7. Return a concise answer focused on what the user asked.
"""

TOOLS_SCHEMA = [
    {"type":"function","function":{
        "name":"lookup_approval_time",
        "description":"Historical permit approval-time stats for a bucket.",
        "parameters":{"type":"object","properties":{
            "permit_type":{"type":"string"},
            "structure_type":{"type":"string"},
            "fsa":{"type":"string"},
            "ward":{"type":"string"}}}}},
    {"type":"function","function":{
        "name":"lookup_builders",
        "description":"Ranked builders by approval speed for a bucket.",
        "parameters":{"type":"object","properties":{
            "fsa":{"type":"string"},
            "permit_type":{"type":"string"},
            "structure_type":{"type":"string"},
            "mode":{"type":"string","enum":["absolute","vs_area"]},
            "top":{"type":"integer"}}}}},
    {"type":"function","function":{
        "name":"dataset_summary",
        "description":"Dataset coverage and city baselines.",
        "parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{
        "name":"nearby_permits",
        "description":"Spatial search: find permits within a radius (metres) "
                      "of a lat/lon point. Returns count, FSA breakdown, and "
                      "nearest permits sorted by distance. Use this whenever "
                      "the user asks about a specific location or area.",
        "parameters":{"type":"object","properties":{
            "lat":      {"type":"number"},
            "lon":      {"type":"number"},
            "radius_m": {"type":"number",
                "description":f"Search radius in metres (default {DEFAULT_RADIUS_M})"},
            "top":      {"type":"integer",
                "description":"Max nearest permits to return (default 25)"}},
            "required": ["lat","lon"]}}},
    {"type":"function","function":{
        "name":"estimate_property_tax",
        "description":"Estimate annual and 10-year property tax revenue for a "
                      "project using City of Toronto 2025 rates. If "
                      "construction_value_cad is omitted, the median cost for "
                      "the matching structure type is used. Use this for any "
                      "question about fiscal impact, revenue, or property tax.",
        "parameters":{"type":"object","properties":{
            "structure_type":         {"type":"string",
                "description":"e.g. 'Apartment Building', 'SFD', 'Commercial / Retail'"},
            "permit_type":            {"type":"string"},
            "construction_value_cad": {"type":"number",
                "description":"Optional. If omitted, falls back to dataset median."},
            "tax_class_override":     {"type":"string",
                "enum":["residential","new_multi_res","multi_residential",
                        "commercial","industrial"],
                "description":"Force a specific tax class instead of inferring."}}}}},
]

DISPATCH = {
    "lookup_approval_time":    lambda **kw: lookup_approval_time(**kw),
    "lookup_builders":         lambda **kw: lookup_builders(**kw),
    "dataset_summary":         lambda **kw: dataset_summary(),
    "nearby_permits":          lambda **kw: nearby_permits(**kw),
    "estimate_property_tax":   lambda **kw: estimate_property_tax(**kw),
}

def run(payload: dict, max_iters: int = 5) -> dict:
    """
    payload keys (all optional except user_question + at least fsa or ward):
      address, fsa, ward, lat, lon,
      permit_type, structure_type, intent, user_question,
      construction_value_cad
    """
    from openai import OpenAI
    client = OpenAI()
    context = {k: payload.get(k) for k in
               ["address","fsa","ward","lat","lon",
                "permit_type","structure_type","intent",
                "construction_value_cad"]
               if payload.get(k) is not None}
    user_msg = (
        f"Resolved location and hints from upstream:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        f"User question: {payload.get('user_question','(none)')}"
    )
    messages = [{"role":"system","content":SYSTEM},
                {"role":"user","content":user_msg}]
    trace = []
    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS_SCHEMA, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return {"answer": msg.content, "trace": trace,
                    "input_context": payload}
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            # Auto-fill from payload, but only for tools that accept that arg
            if call.function.name in ("lookup_approval_time", "lookup_builders"):
                for k in ("fsa","ward","permit_type","structure_type"):
                    args.setdefault(k, payload.get(k))
            elif call.function.name == "nearby_permits":
                for k in ("lat", "lon"):
                    args.setdefault(k, payload.get(k))
            elif call.function.name == "estimate_property_tax":
                for k in ("structure_type","permit_type","construction_value_cad"):
                    args.setdefault(k, payload.get(k))
            args = {k:v for k,v in args.items() if v is not None}
            result = DISPATCH[call.function.name](**args)
            trace.append({"tool": call.function.name,
                          "args": args, "result": result})
            messages.append({"role":"tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})
    return {"answer":"iteration limit reached", "trace": trace}

# ============================================================
# CLI
# ============================================================
def _main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build":
        # python building_permit.py build <insights.json> [--max-new N]
        # Pulls live data via get_building_permits(), geocodes every address
        # (resumable via cache), and writes insights.json. All three outputs
        # —geo_cache.json, permit_coords.json, insights.json— are produced
        # by this one command.
        #
        # On first run, fetch Toronto's Address Points dataset via the data
        # service and build address_index.json. This makes geocoding ~1000×
        # faster (hash lookup instead of Nominatim HTTP calls).
        if len(sys.argv) < 3:
            print("usage: python building_permit.py build <insights.json> [--max-new N]",
                  file=sys.stderr)
            sys.exit(1)
        out_path = sys.argv[2]

        # Auto-build the address index on first run (fetches address points
        # via get_address_points from the data service)
        if not Path(ADDRESS_INDEX_PATH).exists():
            print(f"\n  ─── building address index from Toronto Open Data ───",
                  file=sys.stderr)
            try:
                build_address_index()
                _address_index.cache_clear()  # force reload of the new index
            except Exception as e:
                print(f"  WARNING: could not build address index ({e}). "
                      f"Falling back to Nominatim-only geocoding.",
                      file=sys.stderr)

        # Load data ONCE, share the DataFrame between both steps
        print("  fetching permits from get_building_permits(limit=300)", file=sys.stderr)
        df = get_building_permits(limit=100000)
        print(f"  source has {len(df)} permits", file=sys.stderr)

        print("\n  ─── geocoding addresses ───", file=sys.stderr)
        geo_summary = geocode_permits(df=df)

        print("\n  ─── building insights ───", file=sys.stderr)
        insights = build_insights(df=df)
        with open(out_path, "w") as f:
            json.dump(insights, f, indent=2, default=str)
        print(f"  wrote {out_path}", file=sys.stderr)

        print(json.dumps({
            "rows_loaded":      len(df),
            "geocoding":        geo_summary,
            "insights_written": out_path,
        }, indent=2))
    elif cmd == "run":
        payload = json.load(sys.stdin)
        print(json.dumps(run(payload), indent=2, default=str))
    else:
        print(f"unknown command: {cmd}"); sys.exit(1)

if __name__ == "__main__":
    _main()