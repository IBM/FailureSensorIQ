import requests
import time
import json


def get_crossref_count(keyword):
    """Query Crossref API and return the number of articles containing the keyword."""
    url = "https://api.crossref.org/works"
    params = {"query": keyword, "rows": 0}  # We only need metadata, not full results
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data["message"]["total-results"]
    return 0


# List of industrial asset keywords
asset_keywords = [
    "centrifugal pump",
    "chiller",
    "boiler",
    "gearbox",
    "wind turbine",
    "cooling tower",
    "heat exchanger",
    "fan coil unit",
    "motor drive",
    "steam turbine",
    "pump",
    "electric motor",
    "compressor",
    "aero gas turbine",
    "fan",
    "power transformer",
    "industrial gas turbine",
    "electic generator",
    "reciprocating internal combustion engine"
]

# Dictionary to store keyword counts
results = {}

# Query Crossref for each keyword
for keyword in asset_keywords:
    count = get_crossref_count(keyword)
    results[keyword] = count
    print(f"{keyword}: {count} articles")
    time.sleep(1)  # Be polite to the API

# Save results to JSON
with open("crossref_asset_counts.json", "w") as f:
    json.dump(results, f, indent=2)
