import requests
import time


def get_arxiv_count(keyword):
    """
    Query the arXiv API for the given keyword and return the total number of matching articles.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
        "max_results": 0,  # We only need the count
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Parse the total results from the response
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.text)
        total_results = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        if total_results is not None:
            return int(total_results.text)
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

# Dictionary to store the results
results = {}

# Query arXiv for each keyword
for keyword in asset_keywords:
    count = get_arxiv_count(keyword)
    results[keyword] = count
    print(f"{keyword}: {count} articles")
    time.sleep(3)  # Respect arXiv's rate limit of 1 request every 3 seconds

# Optional: Save results to a JSON file
import json

with open("arxiv_asset_counts.json", "w") as f:
    json.dump(results, f, indent=2)
