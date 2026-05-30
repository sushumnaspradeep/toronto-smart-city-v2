import sys
import json
sys.path.append('.')
from src.services.toronto_data_service import get_neighbourhoods

nb = get_neighbourhoods(limit=200)

# Show all neighbourhoods with Mimico or Lakeshore in name
matches = nb[nb['AREA_NAME'].str.contains(
    'Mimico|Lake|Shore|Long Branch|New Toronto',
    case=False, na=False
)]

print('Matching neighbourhoods:')
for _, row in matches.iterrows():
    try:
        geo  = json.loads(str(row['geometry']))
        geo_type = geo['type']

        if geo_type == 'MultiPolygon':
            poly = geo['coordinates'][0][0]
        else:
            poly = geo['coordinates'][0]

        lats = [c[1] for c in poly]
        lons = [c[0] for c in poly]
        print(f"  {row['AREA_NAME']}")
        print(f"    lat: {min(lats):.4f} to {max(lats):.4f}")
        print(f"    lon: {min(lons):.4f} to {max(lons):.4f}")
    except Exception as e:
        print(f"  {row['AREA_NAME']} - ERROR: {e}")