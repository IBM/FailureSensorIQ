import json

# Input and output file paths
input_jsonl_file = "./sdg_chain/ft_dataset/formatting_mixtrallarge_cot_inductive.jsonl" 
output_json_file = "./sdg_chain/ft_dataset/formatting_mixtrallarge_cot_inductive_all.json"

# Function to convert JSONL to JSON format
def convert_jsonl_to_json(input_file, output_file):
    messages_list = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            input_text = data["input"]
            output_text = data["output"]
            
            # Reformat the data into the desired structure
            messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
            messages_list.append({"messages": messages})
    
    # Save to output JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in messages_list:
            json.dump(entry, outfile)
            outfile.write('\n')

# Run the conversion
convert_jsonl_to_json(input_jsonl_file, output_json_file)

print(f"Conversion completed. Output saved to {output_json_file}")
