import os
import json
import requests
import re
from time import sleep

def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '_', name)

def download_pdf(pdf_url, save_path):
    try:
        response = requests.get(pdf_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Downloaded: {save_path}")
        else:
            print(f"❌ Failed to download {pdf_url} (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading {pdf_url}: {e}")

def main():
    input_json = "arxiv_results.json"
    output_dir = "downloaded_pdfs"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r") as f:
        data = json.load(f)

    seen_ids = set()

    for keyword, papers in data.items():
        for paper in papers:
            arxiv_id = paper["link"].split("/")[-1]
            if arxiv_id in seen_ids:
                continue  # Skip already downloaded papers
            seen_ids.add(arxiv_id)

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            title = paper.get("title", f"{arxiv_id}")
            filename = sanitize_filename(f"{title[:80]}_{arxiv_id}.pdf")
            save_path = os.path.join(output_dir, filename)

            if not os.path.exists(save_path):
                download_pdf(pdf_url, save_path)
                sleep(1.2)  # Respect arXiv's rate limits

if __name__ == "__main__":
    main()
