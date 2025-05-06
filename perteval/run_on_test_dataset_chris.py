from functools import partial
from run_benchmark import test_dataset, parallel_test_dataset
from GeneralLLM import WatsonxModel

test_subjects = ['failure_mode_sensor_analysis']
API_KEY = '' # Use your own OPENAI_API_KEY

#test_data_path = "./eval_data/fmsr_processed/fmsr_501_data.jsonl"
#log_path_prefix = "./log/fmsr_501_data_mistral-large"

# this is for original test dataset
test_data_path = "./eval_data/fmsr_processed/fmsr_filtered_data_all.jsonl"
log_path_prefix = "./log/fmsr_filtered_data_all_"

# this is a llama generated questions and all perturbation
# test_data_path = "./eval_data/fmsr_processed/fmsr_filtered_perturbed_data_all_llama.jsonl"
# log_path_prefix = "./log/fmsr_filtered_perturbed_data_all_llama_"

#test_data_path = "./eval_data/fmsr_processed/fmsr_filtered_perturbed_data_all_simple.jsonl"
#log_path_prefix = "./log/fmsr_filtered_perturbed_data_all_simple_"




modelset = [
    'watsonx-rits/meta-llama/llama-3-3-70b-instruct'
]

trigger_statements = {
    # 'direct': None, 
    'cot_standard': "Let's think Step by Step",
    # 'cot_expert': "Let me solve this step by step as a reliability engineer",
    # 'cot_inductive': "Let's use step by step inductive reasoning, given the domain specific nature of the question"
}
for model in modelset:
    for cot in trigger_statements:
        if cot == 'direct':
            continue
        parallel_test_dataset(
            file_path = test_data_path,
            log_path_prefix = log_path_prefix + cot + "_" + model.split('/')[-1],
            simple_question_path = None,
            subjects = test_subjects,
            model_class = partial(WatsonxModel, api_key = API_KEY),
            model_selection = model,
            temperature = 0.0,
            thread_func = test_dataset,
            n_thread = 8,
            start_id = None,
            end_id = None,
            trigger_statement=trigger_statements[cot]
        )

# for model, cot in modelset_ft:
#     parallel_test_dataset(
#         file_path = test_data_path,
#         log_path_prefix = log_path_prefix + cot + "_" + model.split('/')[-1],
#         simple_question_path = None,
#         subjects = test_subjects,
#         model_class = partial(WatsonxModel, api_key = API_KEY),
#         model_selection = model,
#         temperature = 0.0,
#         thread_func = test_dataset,
#         n_thread = 8,
#         start_id = None,
#         end_id = None,
#         trigger_statement=trigger_statements[cot]
#     )