# core_engine.py
import json
from pathlib import Path
from shapely.geometry import shape, Point

class SpatialContextEngine:
    def __init__(self, geojson_path: str, profiles_path: str):
        self.geojson_path = Path(geojson_path)
        self.profiles_path = Path(profiles_path)
        self.spatial_index = []
        self._load_urban_profiles()
        self._build_spatial_index()

    def _load_urban_profiles(self):
        with open(self.profiles_path, 'r', encoding='utf-8') as f:
            self.urban_profiles = json.load(f)

    def _build_spatial_index(self):
        with open(self.geojson_path, 'r', encoding='utf-8') as f:
            self.geojson_data = json.load(f)
            
        for feature in self.geojson_data.get("features", []):
            properties = feature.get("properties", {})
            ward_name = properties.get("WARD_NAME")
            ward_num = properties.get("WARD_NUMBER", "N/A")
            
            if not ward_name:
                continue
                
            polygon_geom = shape(feature["geometry"])
            profile_metrics = self.urban_profiles.get(ward_name, {})
            
            self.spatial_index.append({
                "ward_name": ward_name,
                "ward_number": ward_num,
                "polygon": polygon_geom,
                "profile": profile_metrics
            })

    def get_context_by_coordinates(self, lng: float, lat: float) -> dict:
        point = Point(lng, lat)
        for record in self.spatial_index:
            if record["polygon"].contains(point):
                return {
                    "status": "success",
                    "ward_name": record["ward_name"],
                    "ward_number": record["ward_number"],
                    "urban_profile": record["profile"]
                }
        return {
            "status": "outside_boundaries",
            "ward_name": "Unknown Zone",
            "ward_number": "N/A",
            "urban_profile": {}
        }