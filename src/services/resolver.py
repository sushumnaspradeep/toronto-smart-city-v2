import json
import logging
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class TorontoUrbanLocationResolver:
    """
    Service class to load generated data and instantly map text addresses/streets
    to the pre-calculated metrics JSON context blocks.
    """
    def __init__(self, data_dir: str = "../../data_payloads") -> None:
        self.data_dir = Path(data_dir)
        
        # 1. Load the structured payload metrics
        payload_path = self.data_dir / "urban_profiles_payload.json"
        if not payload_path.exists():
            raise FileNotFoundError(f"Missing payload file at: {payload_path}")
            
        with open(payload_path, "r", encoding="utf-8") as f:
            self.metrics_db: Dict[str, Any] = json.load(f)
            
        # 2. Load the geojson boundary geometry for spatial lookups
        boundary_path = self.data_dir / "lookup_boundaries.geojson"
        if not boundary_path.exists():
            raise FileNotFoundError(f"Missing boundary file at: {boundary_path}")
            
        self.boundaries = gpd.read_file(boundary_path)
        
        # 3. Initialize the free OpenStreetMap geocoder
        self.geolocator = Nominatim(user_agent="toronto_urban_resolver")
        logger.info("Resolver initialized successfully with local data payloads.")

    def resolve_profile_by_address(self, input_address: str) -> Dict[str, Any]:
        """
        Converts an address string to coordinates and matches it to a metric payload.
        """
        logger.info(f"Geocoding location text: '{input_address}'")
        full_query = f"{input_address}, Toronto, ON, Canada"
        
        try:
            # Step 1: Text to Coordinates
            location = self.geolocator.geocode(full_query, timeout=10)
            if not location:
                return {"error": f"Address text '{input_address}' could not be resolved in Toronto."}
            
            # Step 2: Construct a shapely Point from geocoded coordinates
            point = Point(location.longitude, location.latitude)
            
            # Step 3: Locate the containing polygon using an in-memory spatial check
            matched_ward = None
            for _, row in self.boundaries.iterrows():
                if row['geometry'].contains(point):
                    matched_ward = row['WARD_NAME']
                    break
            
            if not matched_ward:
                return {"error": f"Coordinates for '{input_address}' fall outside mapped Toronto boundaries."}
                
            # Step 4: Retrieve the profile from the JSON dictionary payload
            profile_data = self.metrics_db.get(matched_ward)
            
            if not profile_data:
                return {"error": f"Ward '{matched_ward}' found, but no metric data exists in the JSON payload."}
            
            return {
                "input_query": input_address,
                "resolved_coordinates": {"lat": location.latitude, "lon": location.longitude},
                "matched_neighborhood": matched_ward,
                "data_payload": profile_data
            }
            
        except Exception as e:
            return {"error": f"Failed resolving address due to runtime exception: {str(e)}"}


# --- Independent Testing ---
# You can run this file directly to test the resolver without triggering the LLM.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Assuming you run this from the 'src' directory
    try:
        resolver = TorontoUrbanLocationResolver(data_dir="../data_payloads")
        
        test_address = "100 Queen St W"
        print(f"\nTesting Resolver with address: {test_address}")
        print("-" * 50)
        
        result = resolver.resolve_profile_by_address(test_address)
        print(json.dumps(result, indent=2))
        
    except FileNotFoundError as e:
        print(f"Test failed: {e}")
        print("Please ensure you have run pipeline.py first to generate the data payloads.")