from serpapi import GoogleSearch
import time
import json

SERP_API_KEY = "0f12716a005352b28de527b43bfc39977cc3dcd1b4d66327f1634af031e50c63"

def get_google_result_count(keyword):
    params = {"q": keyword, "api_key": SERP_API_KEY, "engine": "google", "num": 10}
    search = GoogleSearch(params)
    results = search.get_dict()
    # print (results)
    try:
        return results["search_information"]["total_results"]
    except KeyError:
        print(f"No result count found for '{keyword}'")
        return 0


# Asset keywords to search
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

# Collect counts
results = {}
for asset in asset_keywords:
    count = get_google_result_count(asset)
    results[asset] = count
    print(f"{asset}: {count:,} results")
    time.sleep(1)  # Be kind to API

# Save to file (optional)
with open("serpapi_google_asset_counts.json", "w") as f:
    json.dump(results, f, indent=2)
