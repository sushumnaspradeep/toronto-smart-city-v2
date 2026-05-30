# toronto_data_service.py
# ─────────────────────────────────────────────
# Fetches all datasets from Toronto Open Data
# CKAN API for all of Toronto
# ─────────────────────────────────────────────

import requests
import pandas as pd
import zipfile
import io
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv(
    "TORONTO_BASE_URL",
    "https://ckan0.cf.opendata.inter.prod-toronto.ca"
)

# ── Confirmed working resource IDs ────────────────────────────────────────────
RESOURCE_IDS = {
    "building_permits":  "6d0229af-bc54-46de-9c2b-26759b01dd05",
    "parks":             "e8cd0f4d-4910-42a0-81f9-cf8c2218753a",
    "zoning":            "76a2620f-a6b4-495d-8e41-c0ede1f8a928",
}



DATASET_SLUGS = {
    "ttc":  "ttc-routes-and-schedules",
    "s311": "311-service-requests-customer-initiated",
}

# ── Core helpers ──────────────────────────────────────────────────────────────

def datastore_search(resource_id: str, limit: int = 500) -> pd.DataFrame:
    """
    Query a CKAN datastore resource and return a DataFrame.
    """
    url = f"{BASE_URL}/api/3/action/datastore_search"
    params = {"resource_id": resource_id, "limit": limit}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    records = data["result"]["records"]
    return pd.DataFrame(records)


def get_package(slug: str) -> dict:
    """
    Fetch package metadata from CKAN by dataset slug.
    Returns the full package dict including all resources.
    """
    url = f"{BASE_URL}/api/3/action/package_show"
    response = requests.get(url, params={"id": slug}, timeout=15)
    response.raise_for_status()
    return response.json()["result"]


def fetch_zip_csv(slug: str, target_file: str, limit: int = 500) -> pd.DataFrame:
    """
    Download the first ZIP resource from a package,
    extract a specific file inside it, and return as DataFrame.
    Works for TTC GTFS ZIPs and 311 ZIPs.
    """
    package = get_package(slug)

    # Find the first ZIP resource
    zip_resource = next(
        (r for r in package["resources"]
         if r.get("format", "").upper() == "ZIP"),
        None
    )

    if not zip_resource:
        raise ValueError(f"No ZIP resource found in package: {slug}")

    print(f"  Downloading: {zip_resource['name']}")
    response = requests.get(zip_resource["url"], timeout=60)
    response.raise_for_status()

    # Open ZIP from memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print(f" ZIP contents: {z.namelist()}")

        # Find target file inside ZIP
        match = next(
            (f for f in z.namelist() if target_file.lower() in f.lower()),
            None
        )

        if not match:
            raise ValueError(
                f"'{target_file}' not found in ZIP. "
                f"Contents: {z.namelist()}"
            )

        print(f" Parsing: {match}")
        with z.open(match) as f:
            df = pd.read_csv(f, nrows=limit)

    return df

# ── Dataset fetchers ──────────────────────────────────────────────────────────

def get_building_permits(limit: int = 500) -> pd.DataFrame:
    """
    Active building permits across all of Toronto.
    Columns: address, ward, permit_type, work_type,
             estimated cost, status
    """
    print(" Fetching building permits...")
    return datastore_search(RESOURCE_IDS["building_permits"], limit)


def get_parks(limit: int = 500) -> pd.DataFrame:
    """
    Parks and recreation facilities across all of Toronto.
    Columns: name, ward, type, address
    """
    print("Fetching parks and recreation facilities...")
    return datastore_search(RESOURCE_IDS["parks"], limit)


# Cache for TTC ZIP so it only downloads once
_ttc_zip_cache = {}

def _get_ttc_zip() -> bytes:
    """Download TTC ZIP once and cache it in memory."""
    if "data" in _ttc_zip_cache:
        return _ttc_zip_cache["data"]

    package     = get_package(DATASET_SLUGS["ttc"])
    zip_resource = next(
        (r for r in package["resources"]
         if r.get("format", "").upper() == "ZIP"),
        None
    )
    if not zip_resource:
        raise ValueError("No ZIP resource found for TTC")

    print(f"  Downloading: {zip_resource['name']}")
    response = requests.get(zip_resource["url"], timeout=60)
    response.raise_for_status()
    _ttc_zip_cache["data"] = response.content
    return _ttc_zip_cache["data"]


def get_ttc_routes(limit: int = 500) -> pd.DataFrame:
    """TTC routes from GTFS ZIP — routes.txt"""
    print(" Fetching TTC routes...")
    zip_bytes = _get_ttc_zip()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open("routes.txt") as f:
            return pd.read_csv(f, nrows=limit)


def get_ttc_stops(limit: int = 5000) -> pd.DataFrame:
    """TTC stops from GTFS ZIP — stops.txt"""
    print("Fetching TTC stops...")
    zip_bytes = _get_ttc_zip()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open("stops.txt") as f:
            return pd.read_csv(f, nrows=limit)


def get_311_requests(limit: int = 500) -> pd.DataFrame:
    """
    311 service requests for 2026 across all of Toronto.
    Columns: service_request_type, division,
             ward, status, creation_date
    """
    print(" Fetching 311 service requests...")
    return fetch_zip_csv(DATASET_SLUGS["s311"], "SR2026.csv", limit)

def get_neighbourhoods(limit: int = 200) -> pd.DataFrame:
    """
    Official Toronto neighbourhood boundaries.
    158 neighbourhoods with exact polygon boundaries.
    """
    print("📦 Fetching neighbourhood boundaries...")
    # Use slug instead of hardcoded resource ID
    pkg = get_package("neighbourhoods")
    resource = next(
        (r for r in pkg["resources"] if r.get("datastore_active")),
        None
    )
    if not resource:
        # Try CSV fallback
        resource = next(
            (r for r in pkg["resources"]
             if r.get("format", "").upper() == "CSV"),
            None
        )
    if not resource:
        raise ValueError(f"No usable resource. Available: {[r['name'] for r in pkg['resources']]}")

    print(f"  ↳ Using resource: {resource['name']} ({resource['id']})")
    return datastore_search(resource["id"], limit)

def get_zoning(limit: int = 500) -> pd.DataFrame:
    """
    Zoning by-law data across all of Toronto.
    Used to find vacant lots and land use rules.
    """
    print(" Fetching zoning data...")
    return datastore_search(RESOURCE_IDS["zoning"], limit)
# Centreline resource ID confirmed working
CENTRELINE_RESOURCE_ID = "ad296ebf-fca6-4e67-b3ce-48040a20e6cd"

def get_roads(limit: int = 5000) -> pd.DataFrame:
    """
    Toronto Centreline road network.
    Contains every road segment in Toronto.
    Columns used:
      LINEAR_NAME_FULL  → road name
      FEATURE_CODE_DESC → road type (Local, Arterial, Expressway etc)
      geometry          → LineString coordinates
    """
    print("📦 Fetching road centreline data...")
    return datastore_search(CENTRELINE_RESOURCE_ID, limit)

# ── Combined loader ───────────────────────────────────────────────────────────

def load_all_data(limit: int = 500) -> dict:
    """
    Loads all 6 datasets at once.
    Returns a dict of DataFrames.
    Call this once and pass the result to all agents.

    Returns:
    {
        "building_permits": DataFrame,
        "parks":            DataFrame,
        "ttc_routes":       DataFrame,
        "ttc_stops":        DataFrame,
        "complaints_311":   DataFrame,
        "zoning":           DataFrame,
    }
    """
    print("\n  Loading all Toronto Open Data...\n")
    return {
        "building_permits": get_building_permits(limit),
        "parks":            get_parks(limit),
        "ttc_routes":       get_ttc_routes(limit),
        "ttc_stops":        get_ttc_stops(limit),
        "complaints_311":   get_311_requests(limit),
        "zoning":           get_zoning(limit),
    }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all_data(limit=10)
    for name, df in data.items():
        print(f"\n {name}: {len(df)} rows, columns: {list(df.columns)}")
