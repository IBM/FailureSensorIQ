import requests
import xml.etree.ElementTree as ET
import time
import json

def search_arxiv(keyword, max_results=3):
    """Query arXiv API with a keyword and return parsed results."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
        "max_results": max_results
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            link = entry.find('{http://www.w3.org/2005/Atom}id').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            authors = [
                author.find('{http://www.w3.org/2005/Atom}name').text.strip()
                for author in entry.findall('{http://www.w3.org/2005/Atom}author')
            ]
            results.append({
                "title": title,
                "link": link,
                "summary": summary,
                "authors": authors
            })

        return results
    except Exception as e:
        print(f"Error fetching for '{keyword}': {e}")
        return []

def main():
    input_file = "unique_keywords.txt"
    output_file = "arxiv_results.json"
    max_results_per_keyword = 3
    sleep_between_requests = 1.5  # seconds

    all_results = {}

    with open(input_file, "r") as f:
        keywords = [line.strip() for line in f if line.strip()]

    for idx, keyword in enumerate(keywords):
        print(f"[{idx+1}/{len(keywords)}] Searching for: {keyword}")
        results = search_arxiv(keyword, max_results=max_results_per_keyword)
        all_results[keyword] = results
        time.sleep(sleep_between_requests)

    with open(output_file, "w") as out_f:
        json.dump(all_results, out_f, indent=2)

    print(f"✅ Saved results for {len(keywords)} keywords to '{output_file}'")

if __name__ == "__main__":
    main()
