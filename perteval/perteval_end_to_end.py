from run_benchmark import test_dataset, parallel_test_dataset
from GeneralLLM import TransformersModel, ChatGPT
from openai import OpenAI
from datetime import datetime
import os
import subprocess
import time
import signal
import requests
def get_perteval_results(model_name, mode='original', cot='cot_standard', dataset='full'):
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    if not os.path.exists('log'):
        os.mkdir('log')
    proc = subprocess.Popen(["vllm", "serve", model_name, "--port", "8003"], preexec_fn=os.setsid)
    n_retries = 150
    while n_retries > 0:
        try:
            response = requests.get('http://localhost:8003/v1/models')
            if response.status_code == 200:
                break
        except:
            pass
        print(f'waiting for vllm to launch. Retrying in 10 seconds')
        time.sleep(10)
        n_retries -= 1
    model = ChatGPT(base_url="http://localhost:8003/v1", model=model_name, api_key='empty')
    test_subjects = ['failure_mode_sensor_analysis']
    if mode == 'original':
        if dataset == 'sample':
            # 50 mcp
            test_data_path = "./eval_data/industrial_mcp_original.jsonl"
        else:
            # full
            test_data_path = "./eval_data/fmsr_processed/fmsr_filtered_data_all.jsonl"
        log_path_prefix = f"./log/fmsr_filtered_data_{dataset}"
    elif mode == 'perturb':
        if dataset == 'sample':
            test_data_path = "./eval_data/industrial_mcp_perturbed.jsonl"
        else:
            test_data_path = "./eval_data/fmsr_processed/fmsr_filtered_perturbed_data_all_simple.jsonl"
        log_path_prefix = f"./log/fmsr_filtered_perturbed_data_all_{dataset}"
    else:
        raise ValueError(f'Invalid perteval mode {mode}')
    trigger_statements = {
        'direct': None, 
        'cot_standard': "Let's think Step by Step",
        'cot_expert': "Let me solve this step by step as a reliability engineer",
        'cot_inductive': "Let's use step by step inductive reasoning, given the domain specific nature of the question"
    }
    log_path = f"{log_path_prefix}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.jsonl"
    parallel_test_dataset(
        file_path = test_data_path,
        log_path=log_path,
        simple_question_path = None,
        subjects = test_subjects,
        model_class = model,
        model_selection = model_name,
        temperature = 0.0,
        thread_func = test_dataset,
        n_thread = 1,
        start_id = None,
        end_id = None,
        trigger_statement=trigger_statements[cot]
    )
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    return log_path
