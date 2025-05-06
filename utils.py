from huggingface_hub import snapshot_download
from huggingface_hub import login
import os
import json
# login()
def get_requests(repo_name):
    request_path = snapshot_download(repo_name, repo_type="dataset")
    print(request_path)
    all_request_models = []
    for org in os.listdir(request_path):
        full_path = os.path.join(request_path, org)
        if not os.path.isdir(full_path):
            continue
        for fname in os.listdir(full_path):
            full_fname = os.path.join(full_path, fname)
            if not full_fname.endswith('json'):
                continue
            with open(full_fname, 'r') as f:
                data = json.load(f)
            all_request_models.append(data['model'].lower())
            # print(full_fname)
    return set(all_request_models)

def get_results(repo_name):
    result_path = snapshot_download(repo_name, repo_type="dataset")
    result_path = os.path.join(result_path, 'demo-leaderboard')
    all_result_models = []
    for org in os.listdir(result_path):
        full_path = os.path.join(result_path, org)
        print(full_path)
        if not os.path.isdir(full_path):
            continue
        for fname in os.listdir(full_path):
            full_fname = os.path.join(full_path, fname)
            if not full_fname.endswith('json'):
                continue
            with open(full_fname, 'r') as f:
                data = json.load(f)
            all_result_models.append(data['config']['model_name'].lower())
            # print(full_fname)
    return set(all_result_models)