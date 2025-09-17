import json
import random


random.seed(42)

input_jsonl_file = "./sdg_chain/ft_dataset/formatting_mixtrallarge_cot_inductive_all.json"
output_jsonl_file = input_jsonl_file.replace('_all', '_shuffle')

with open(input_jsonl_file, "r", encoding="utf-8") as infile:
    lines = [json.loads(line) for line in infile]

random.shuffle(lines)

with open(output_jsonl_file, "w", encoding="utf-8") as outfile:
    for entry in lines:
        json.dump(entry, outfile)
        outfile.write("\n")

print(f"Shuffled JSONL file saved to {output_jsonl_file}")