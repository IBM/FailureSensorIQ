import requests


def search_wikipedia(keyword):
    """Search Wikipedia and return number of matching pages."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "format": "json",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["query"]["searchinfo"]["totalhits"]


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

# Count and display results
results = {}
for asset in asset_keywords:
    count = search_wikipedia(asset)
    results[asset] = count
    print(f"{asset}: {count} Wikipedia pages")

# Optional: Write to CSV or JSON
import json

with open("wiki_asset_counts.json", "w") as f:
    json.dump(results, f, indent=2)
