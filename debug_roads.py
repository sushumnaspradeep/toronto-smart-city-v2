import sys
sys.path.append('.')
from src.services.toronto_data_service import datastore_search

RESOURCE_ID = "ad296ebf-fca6-4e67-b3ce-48040a20e6cd"

df = datastore_search(RESOURCE_ID, limit=2000)

print("Road types (FEATURE_CODE_DESC):")
print(df["FEATURE_CODE_DESC"].value_counts().to_string())