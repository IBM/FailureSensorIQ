import json
import os
import jsonlines
from transformers import AutoTokenizer


def read_json_files(directory):
    json_data = {}

    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):  # Check for JSON files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)  # Read JSON content
                        json_data[file_path] = data  # Add to the list
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return json_data


def prep_chat_example(system_prompt, user_prompt, answer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]
    template_q = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=False
    )
    template_a = tokenizer.apply_chat_template(
        [messages[2]], tokenize=False, add_generation_prompt=False
    )
    return {"input": template_q, "output": template_a}


def save_json_line(data, savefilename):
    with jsonlines.open(savefilename, "w") as writer:
        writer.write_all(data)



directory_path = "./generations"
all_json_data = read_json_files(directory_path)


print('number of files:')
print(len(all_json_data))
print('number of data samples:')
print(sum(len(v) for v in all_json_data.values() if isinstance(v, list)))

model_path = "ibm-granite/granite-3.0-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='/dccstor/cbm/cache')

sys_prompt = '''You are an expert in industrial asset management who specializes in Failure Mode and Effect Analysis. Your task is to understand the quesion and select the most appropriate answer. 
Failure Mode and Effect Analysis (FMEA) is conducted to identify anticipated faults, symptoms, and potential parameters that indicate the presence or occurrence of faults.
'''

all_processed_data = []
for key, json_data in all_json_data.items():
    for sample in json_data:
        processed = prep_chat_example(system_prompt=sys_prompt, user_prompt=sample['prompt'], answer=sample['answer'])
        all_processed_data.append(processed)

save_json_line(all_processed_data, './ft_dataset/fine_tune_Dec_10.jsonl')
save_json_line(all_processed_data[:8], './ft_dataset/fine_tune_Dec_10_sample.jsonl')