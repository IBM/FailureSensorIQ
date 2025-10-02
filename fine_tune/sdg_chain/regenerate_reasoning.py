import json
import os
import jsonlines
from transformers import AutoTokenizer
from tqdm.contrib.concurrent import process_map  # or thread_map

from industrialqa_fmsr.baseline.function.chain_of_thought.self_cot import generate_final_answer as cot_generate_final_answer


llm = "watsonx/meta-llama/llama-3-1-70b-instruct"
# llm = "watsonx/mistralai/mistral-large"  
MAX_RETRIES = 5

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


def save_json_line(data, savefilename):
    with jsonlines.open(savefilename, "w") as writer:
        writer.write_all(data)

def regenerate(inputs):
    prompt, answer_letter = inputs[1], inputs[2]
    retry = 0
    temperature = 0
    while retry < MAX_RETRIES:
        try:
            print('prompt:\n'+prompt)
            response = cot_generate_final_answer(
                question_prompt=prompt,
                model_name=llm,
                params={"temperature": temperature}, few_shot=True
                )
            generated_option = response['generated_option_1']
            model_output =response['model_output']
            if generated_option == answer_letter:
                print("answer_letter", answer_letter)
                print('model_output:\n'+model_output)
                return inputs[0], {"input": prompt, "output": model_output}
            else:
                temperature += 0.1
                retry += 1
        except Exception as e:
            retry += 1
    return inputs[0], None



directory_path = "./generations"
all_json_data = read_json_files(directory_path)


print('number of files:')
print(len(all_json_data))
print('number of data samples:')
print(sum(len(v) for v in all_json_data.values() if isinstance(v, list)))

all_processed_data = []
for key, json_data in all_json_data.items():
    for sample in json_data:
        original_prompt = sample['prompt']
        # expert_prompt = original_prompt.replace('\nAnswer:', '\nAnswer: Let me think step by step as a reliability engineer.\n')
        # inductive_prompt = original_prompt.replace('\nAnswer:', "\nAnswer: Let's use step by step inductive reasoning, given the domain specific nature of the question.\n")
        # cot_standard_prompt = original_prompt.replace('\nAnswer:', "\nAnswer: Let me think step by step.\n")
        original_prompt = original_prompt.replace("Answer:", "").replace('\nA. ', "\nOptions:\nA. ")
        prefix = "Please select the correct option from the following options given the question. To solve the problem, follow the Let's think Step by Step as Reliability Enginneer Expert reasoning strategy.\n"
        expert_prompt = prefix + original_prompt + "\nYour output must strictly follow this format:\n{\"reasoning\": <\"Your reasoning step-by-step\">, \"answer\": <the list of selected option, e.g., [\"A\", \"B\", \"C\", \"D\", \"E\"]>}\nYour output:"
        prefix = "Please select the correct option from the following options given the question. To solve the problem, follow the Let's use step by step inductive reasoning, given the domain specific nature of the question reasoning strategy.\n"
        inductive_prompt = prefix + original_prompt + "\nYour output must strictly follow this format:\n{\"reasoning\": <\"Your reasoning step-by-step\">, \"answer\": <the list of selected option, e.g., [\"A\", \"B\", \"C\", \"D\", \"E\"]>}\nYour output:"
        prefix = "Please select the correct option from the following options given the question. To solve the problem, follow the Let's think Step by Step reasoning strategy.\n"
        cot_standard_prompt = prefix + original_prompt + "\nYour output must strictly follow this format:\n{\"reasoning\": <\"Your reasoning step-by-step\">, \"answer\": <the list of selected option, e.g., [\"A\", \"B\", \"C\", \"D\", \"E\"]>}\nYour output:"
        
        all_processed_data.extend([
            ('cot_standard', cot_standard_prompt, sample['letter']), 
            ('cot_expert', expert_prompt, sample['letter']),
            ('cot_inductive', inductive_prompt, sample['letter'])
        ])
print("number of samples")
print(len(all_processed_data))
# all_processed_data = all_processed_data[:10]
all_processed_result = {}


r = process_map(regenerate, all_processed_data, max_workers=8)

for k, processed in r:
    if processed:
        print(processed)
        if k in all_processed_result:
            all_processed_result[k].append(processed)
        else:
            all_processed_result[k] = [processed]

with open('tmp_all.json', 'w') as f:
    json.dump(all_processed_result, f, indent=4)

for k in all_processed_result:
    save_json_line(all_processed_result[k], f'./ft_dataset/ft_formatting70b_{k}.jsonl')
    save_json_line(all_processed_result[k][:8], f'./ft_dataset/ft_formatting70b_sample_{k}.jsonl')


'''

{
        "prompt": "Question: What sensor is suitable for monitoring Aero gas turbine to detect Air inlet blockage?\nA. Excitation current\nB. Cylinder pressure\nC. Air flow\nD. Compressor efficiency\nE. Axial flux\nAnswer:",
        "answer": "An air flow sensor is the most suitable for monitoring an aero gas turbine to detect air inlet blockage. This type of sensor measures the rate of air movement through the system, which is directly affected by any obstructions in the air inlet.\nAir inlet blockage would result in a reduction in air flow, which can be accurately detected by an air flow sensor. Other options such as excitation current, cylinder pressure, compressor efficiency, and axial flux are not directly related to measuring air flow and would not provide a clear indication of an air inlet blockage.\nSo the final answer is C. Air flow.",
        "unique_id": "a5346c44",
        "letter": "C",
        "letter_with_choice": "C. Air flow"
    }
    
    '''

'''
Please select the correct option(s) from the following options given the question. To solve the problem, follow the Let's think Step by Step as Reliability Enginneer Expert reasoning strategy.\n
'''