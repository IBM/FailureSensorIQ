import json
from collections import defaultdict

# Initialize containers
option_lengths = defaultdict(list)

# Load the .jsonl file
file_path = '../eval_data/fmsr_processed/filtered_data_all_Mar_30_2025.jsonl'
output_path = 'asset_option_counts_sorted.txt'

with open(file_path, 'r') as file:
    for line in file:
        item = json.loads(line)
        asset = item.get("asset_name")
        option_ids = item.get("option_ids", [])
        
        # Filter for option_ids length == 2
        if len(option_ids) == 2:
            option_lengths[asset].append(len(option_ids))

# Compute total count of option_ids per asset_name
total_counts = {asset: sum(lengths) for asset, lengths in option_lengths.items()}

# Sort results by count (descending order)
sorted_counts = sorted(total_counts.items(), key=lambda x: x[1], reverse=False)

# Print and save the results
with open(output_path, 'w') as output_file:
    for asset, count in sorted_counts:
        result_line = f"{asset}: total option_ids count = {count}"
        print(result_line)
        output_file.write(result_line + '\n')

print(f"\nResults saved to {output_path}")
