import transition_analysis as tas
import json
import os

log_directory = "./log"

prefer_result = [
    '2025-03-21'
]

# prefer_result = ['llama-3-1-8b-instruct']

# Loop through all JSONL files in the directory
for filename in os.listdir(log_directory):
    contains_keyword = any(keyword in filename for keyword in prefer_result)
    if (
        filename.endswith(".jsonl")
        and contains_keyword
        and "all_simple" not in filename
    ):
        filepath = os.path.join(log_directory, filename)
        print(f"Processing file: {filepath}")
        tas.response_pattern_analysis(filepath)

"""
print ('o1-preview')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_o1-preview-2024-09-12_2025-01-19 09:45:18.273176.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_o1-preview-2024-09-12_2025-01-19 18:33:42.130113.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)

print ('llama-large')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_llama-3-405b-instruct_2025-01-19 09:12:05.443454.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_llama-3-405b-instruct_2025-01-19 17:07:21.396309.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)

print ('mistral-large')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_mistral-large_2025-01-19 09:32:26.408976.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_mistral-large_2025-01-19 18:25:44.862923.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)

print ('mixtral-8x22B')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_mixtral-8x22B-instruct-v0.1_2025-01-18 22:43:38.808079.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_mixtral-8x22B-instruct-v0.1_2025-01-19 15:48:36.195856.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)

print ('llama-3-3-70b')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_llama-3-3-70b-instruct_2025-01-18 22:34:09.446967.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_llama-3-3-70b-instruct_2025-01-19 15:42:09.022452.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)

print ('all_mixtral-8x7B')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_mixtral-8x7B-instruct-v0.1_2025-01-20 09:33:33.125437.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_mixtral-8x7B-instruct-v0.1_2025-01-20 08:34:10.971725.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)

print ('Llama-3.1-8B')
tas.transition_analysis(
    "./log/fmsr_filtered_data_all_Llama-3.1-8B-Instruct_2025-01-19 05:59:13.046874.jsonl",
    "./log/fmsr_filtered_perturbed_data_all_llama_Llama-3.1-8B-Instruct_2025-01-19 17:53:25.111295.jsonl",
    subjects=["failure_mode_sensor_analysis"],
)


tas.response_pattern_analysis(
    "./log/fmsr_filtered_perturbed_data_all_llama_o1-preview-2024-09-12_2025-01-19 18:33:42.130113.jsonl",
)

tas.response_pattern_analysis(
    "./log/fmsr_filtered_data_all_o1-preview-2024-09-12_2025-01-19 09:45:18.273176.jsonl"
)
"""
