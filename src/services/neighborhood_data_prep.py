"""
===============================================================================
Urban Neighborhood Data Pipeline
===============================================================================

Description:
This pipeline automates the extraction, spatial integration, and aggregation of 
civic datasets from the City of Toronto Open Data Portal (CKAN). It is strictly 
designed to build a highly structured, flat JSON knowledge base for Large Language 
Model (LLM) agents assisting government agencies with urban planning.

Core Actions & Pipeline Flow:
1. Extraction: Paginates through CKAN datastore APIs and downloads zipped assets 
   to bypass hard data limits and file extension obfuscation.
2. Spatial Integration (Hybrid): Parses raw coordinates into geographic Points 
   and performs Point-in-Polygon mapping against Toronto City Ward boundaries. 
   Relies on tabular string-matching for datasets lacking coordinate geometries.
3. Aggregation: Harmonizes the mapped data into localized ward-level statistics 
   (e.g., total transit stops, active businesses, cultural hotspots).
4. Serialization: Calculates analytic scores (e.g., 'recreation_deficit_score') 
   based on parameterizable thresholds and exports a flat JSON payload optimized 
   for O(1) runtime agent retrieval.

Primary Data Sources (City of Toronto):
- City Wards (Base Boundaries)
- Parks and Recreation Facilities
- TTC Routes and Schedules (GTFS)
- Municipal Licensing and Standards (Business Licences)
- Development Applications
- Cultural Hotspot Points of Interest
- Outdoor Artificial Ice Rinks
- Cycling Network & Toronto Centreline (Roads)

Generated Artifacts:
- `lookup_boundaries.geojson`: Unprojected coordinate map for runtime geocoding.
- `urban_profiles_payload.json`: The aggregated metric knowledge base.

Dependencies:
- pandas, geopandas, shapely, requests
===============================================================================
"""

import os
import io
import json
import zipfile
import logging
import requests
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Any
from shapely.geometry import shape

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*')
BASE_URL = os.getenv("TORONTO_BASE_URL", "https://ckan0.cf.opendata.inter.prod-toronto.ca")

# ── Data Extraction Helpers ───────────────────────────────────────────────────

def datastore_search(resource_id: str, limit: int = 32000) -> pd.DataFrame:
    all_records = []
    offset = 0
    while True:
        url = f"{BASE_URL}/api/3/action/datastore_search"
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        chunk = response.json()["result"]["records"]
        if not chunk: 
            break
        all_records.extend(chunk)
        if len(chunk) < limit: 
            break
        offset += limit
    return pd.DataFrame(all_records)

def get_package(slug: str) -> Dict[str, Any]:
    response = requests.get(f"{BASE_URL}/api/3/action/package_show", params={"id": slug}, timeout=15)
    response.raise_for_status()
    return response.json()["result"]

def get_datastore_resource_id(slug: str) -> str:
    pkg = get_package(slug)
    res = next((r for r in pkg["resources"] if r.get("datastore_active")), None)
    if not res: 
        raise ValueError(f"No datastore found for {slug}")
    return res["id"]

def fetch_zip_csv(slug: str, target_file: str, limit: Optional[int] = None) -> pd.DataFrame:
    package = get_package(slug)
    zip_res = next((r for r in package["resources"] if r.get("format", "").upper() == "ZIP"), None)
    if not zip_res: 
        raise ValueError(f"No ZIP found for {slug}")
    
    response = requests.get(zip_res["url"], timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        match = next((f for f in z.namelist() if target_file.lower() in f.lower()), None)
        if not match:
            raise ValueError(f"File {target_file} not found in zip.")
        with z.open(match) as f:
            return pd.read_csv(f, nrows=limit)

# ── Processing Pipeline ───────────────────────────────────────────────────────

class UrbanRevitalizationJSONPipeline:
    def __init__(self, output_dir: str = "./data_payloads", custom_thresholds: Optional[Dict[str, int]] = None) -> None:
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_crs: str = "EPSG:32617" 
        
        # Explicit type hinting for class attributes to satisfy strict linting
        self.base_boundaries: Optional[gpd.GeoDataFrame] = None
        self.spatial_counts: Dict[str, pd.DataFrame] = {}
        self.tabular_counts: Dict[str, pd.DataFrame] = {}
        self.aggregated_df: Optional[pd.DataFrame] = None

        self.thresholds: Dict[str, int] = {
            "parks_adequate": 10,
            "vibrancy_established": 15,
            "transit_high": 50,
            "bike_robust": 500,
            "biz_center": 1000
        }
        
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def _parse_geometry(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: shape(json.loads(x)) if pd.notnull(x) else None)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            col_map = {c.upper(): c for c in df.columns}
            lon_col = col_map.get('LONGITUDE') or col_map.get('STOP_LON')
            lat_col = col_map.get('LATITUDE') or col_map.get('STOP_LAT')
            
            if lon_col and lat_col:
                df = df.dropna(subset=[lon_col, lat_col])
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
            else:
                raise ValueError("No valid geometry or coordinates found.")
        return gdf.to_crs(self.target_crs)

    def step2_spatial_integration(self) -> None:
            logger.info("Step 2: Hybrid Spatial & Tabular Integration...")

            logger.info("  -> Loading Base Boundaries...")
            wards_df = datastore_search(get_datastore_resource_id("city-wards"))
            self.base_boundaries = self._parse_geometry(wards_df)

            name_col = next(c for c in self.base_boundaries.columns if "AREA_NAME" in c.upper())
            num_col = next(c for c in self.base_boundaries.columns if "AREA_SHORT_CODE" in c.upper())

            self.base_boundaries.rename(columns={name_col: "WARD_NAME", num_col: "WARD_NUMBER"}, inplace=True)
            self.base_boundaries['WARD_NUMBER'] = self.base_boundaries['WARD_NUMBER'].astype(str).str.replace(r'\.0$', '', regex=True).str.lstrip('0')
            self.base_boundaries = self.base_boundaries[['WARD_NAME', 'WARD_NUMBER', 'geometry']]

            lookup_map = self.base_boundaries.to_crs("EPSG:4326")
            lookup_map.to_file(self.output_dir / "lookup_boundaries.geojson", driver='GeoJSON')

            # Updated the toronto-centreline slug to 'toronto-centreline-tcl'
            datasets_spatial = [
                ("parks-and-recreation-facilities", "parks", True), 
                ("cultural-hotspot-points-of-interest", "culture", False),
                ("outdoor-artificial-ice-rinks", "rinks", True),
                ("development-applications", "dev_apps", False),
                ("cycling-network", "bike_lanes", True),      
                ("toronto-centreline-tcl", "roads", True)         
            ]

            for slug, key, convert_to_centroid in datasets_spatial:
                logger.info(f"  -> Mapping Spatial Dataset: {slug}...")
                try:
                    df = datastore_search(get_datastore_resource_id(slug))
                    gdf = self._parse_geometry(df)

                    if convert_to_centroid and not (gdf.geom_type == 'Point').all():
                        gdf['geometry'] = gdf.geometry.centroid

                    mapped = gpd.sjoin(gdf, self.base_boundaries, how='inner', predicate='within')
                    self.spatial_counts[key] = mapped.groupby('WARD_NAME').size().reset_index(name=f'total_{key}')
                except Exception as e:
                    logger.error(f"Failed to map {slug}: {e}")

            logger.info("  -> Mapping Mobility (TTC Stops)...")
            stops_df = fetch_zip_csv("ttc-routes-and-schedules", "stops.txt")
            stops_gdf = self._parse_geometry(stops_df)
            mapped_stops = gpd.sjoin(stops_gdf, self.base_boundaries, how='inner', predicate='within')
            self.spatial_counts["transit"] = mapped_stops.groupby('WARD_NAME').size().reset_index(name='total_transit')

            logger.info("  -> Mapping Commercial Establishments (Relational Match)...")
            biz_df = datastore_search(get_datastore_resource_id("municipal-licensing-and-standards-business-licences-and-permits"))
            if 'Licence Status' in biz_df.columns:
                biz_df = biz_df[biz_df['Licence Status'].str.contains('Issued', case=False, na=False)]
            if 'Ward' in biz_df.columns:
                biz_df['Clean_Ward'] = biz_df['Ward'].astype(str).str.extract(r'(\d+)')[0].str.lstrip('0')
                self.tabular_counts["biz"] = biz_df.groupby('Clean_Ward').size().reset_index(name='total_biz')

    def step3_aggregation(self) -> None:
        logger.info("Step 3: Harmonized Aggregation...")
        
        if self.base_boundaries is None:
            raise RuntimeError("Base boundaries are missing. Run step2 first.")
            
        profiles = self.base_boundaries.drop(columns='geometry').drop_duplicates()
        
        for key, count_df in self.spatial_counts.items():
            profiles = profiles.merge(count_df, on='WARD_NAME', how='left')
            
        if "biz" in self.tabular_counts:
            profiles = profiles.merge(self.tabular_counts["biz"], left_on='WARD_NUMBER', right_on='Clean_Ward', how='left')
            
        self.aggregated_df = profiles.fillna(0)

    def export_json_payload(self) -> None:
        logger.info("Structuring Flat JSON Payload...")
        
        if self.aggregated_df is None:
            raise RuntimeError("Aggregated dataframe is missing. Run step3 first.")
            
        payload: Dict[str, Dict[str, Any]] = {}
        t = self.thresholds 
        
        for _, row in self.aggregated_df.iterrows():
            ward_name = str(row['WARD_NAME'])
            
            parks_count = int(row.get('total_parks', 0))
            transit_count = int(row.get('total_transit', 0))
            biz_count = int(row.get('total_biz', 0))
            dev_count = int(row.get('total_dev_apps', 0))
            culture_count = int(row.get('total_culture', 0))
            rinks_count = int(row.get('total_rinks', 0))
            bike_count = int(row.get('total_bike_lanes', 0))
            road_count = int(row.get('total_roads', 0))
            
            payload[ward_name] = {
                "total_parks": parks_count,
                "total_cultural_hotspots": culture_count,
                "total_outdoor_ice_rinks": rinks_count,
                "total_active_transit_stops": transit_count,
                "total_bike_lane_segments": bike_count,
                "total_road_segments": road_count,
                "total_active_businesses": biz_count,
                "total_development_applications": dev_count,
                
                "recreation_deficit_score": "High Deficit" if parks_count < t["parks_adequate"] else "Adequate",
                "community_vibrancy_score": "Emerging" if (culture_count + rinks_count) < t["vibrancy_established"] else "Established Hub",
                "transit_connectivity": "Needs Improvement" if transit_count < t["transit_high"] else "High Connectivity",
                "active_transit_infrastructure": "Robust" if bike_count > t["bike_robust"] else "Developing",
                "economic_vitality_score": "Developing" if biz_count < t["biz_center"] else "Commercial Center"
            }
            
        output_file = self.output_dir / "urban_profiles_payload.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully exported comprehensive JSON payload to {output_file}")

# --- Execution ---
if __name__ == "__main__":
    pipeline = UrbanRevitalizationJSONPipeline()
    try:
        pipeline.step2_spatial_integration()
        pipeline.step3_aggregation()
        pipeline.export_json_payload()
        logger.info("Data preparation complete. Ready for agent ingestion.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")