import json

# Load JSONL into a dict keyed by ID
def load_jsonl_as_dict(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            data[item['id']] = item
    return data

# Load list from JSONL
def load_jsonl_as_list(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

# Save list to JSONL
def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# File paths
file1 = 'fmsr_filtered_perturbed_data_all_llama.jsonl'      # The file with updated questions
file2 = 'filtered_data_all_Mar_30_2025_10options_all_simple.jsonl'      # The file to update

# Load files
source_dict = load_jsonl_as_dict(file1)
target_list = load_jsonl_as_list(file2)

# Replace questions in target
for item in target_list:
    if item['id'] in source_dict:
        item['question'] = source_dict[item['id']]['question']

# Save updated data
save_jsonl(target_list, 'filtered_data_all_Mar_30_2025_10options_all_complex.jsonl')
