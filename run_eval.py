#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[1]:


import sys
import os
sys.path.append('perteval')
sys.path.append('uq_score')


# In[2]:


import json
from utils import get_requests, get_results
from perteval.perteval_end_to_end import get_perteval_results
# from uq_score.uq_end_to_end import run_uq
from llm_uncertainty_bench.uq_end_to_end import run_uq_benchmark
import datetime
import perteval.transition_analysis as tas
import time
import subprocess


# In[3]:


# try killing old processes that utilize the gpu
try:
    subprocess.run(["nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9"], shell=True)
except:
    pass


# In[4]:


if len(sys.argv) == 1 or sys.argv[1] == '-f':
    # -f is to avoid the case that this runs as ipynb
    all_request_models = get_requests('cc4718/requests')
    all_result_models = get_results('cc4718/results')
    models_todo = set(all_request_models) - set(all_result_models)
    # dataset can be full/sample
    dataset = 'full'
else:
    # huggingface models
    models_todo = [sys.argv[1]]
    # dataset full/sample
    dataset = sys.argv[2]
print('models to evaluate:')
print(models_todo)
print('dataset:')
print(dataset)


# In[ ]:


for model_name in models_todo:
    # perteval
    original_log_path, test_path = get_perteval_results(model_name, 
                                                        mode='original', 
                                                        cot='cot_standard', 
                                                        dataset=dataset)
    print('Finished perteval on original data')
    # wait a few seconds to kill vllm and spin up a new
    time.sleep(10)
    perturb_log_path, _ = get_perteval_results(model_name, 
                                               mode='perturb', 
                                               cot='cot_standard', 
                                               dataset=dataset)
    print('Finished perteval on perturbed data')
    original, perturb, consist = tas.transition_analysis(original_log_path, 
                                                         perturb_log_path, 
                                                         subjects=["failure_mode_sensor_analysis"])
    asset_scores = tas.get_record_id_for_correct_answer(original_log_path, 
                                                        dimention='asset_name', 
                                                        fdata_path=test_path)
    relevancy_scores = tas.get_record_id_for_correct_answer(original_log_path, 
                                                            dimention='relevancy', 
                                                            fdata_path=test_path)
    # uq bench
    print('Running LLM Uncertainty Bench')
    uq_scores = run_uq_benchmark(model_name, prompt_type='chat', dataset=dataset)
    if dataset == 'sample':
        acc_sel_key = 'relevant_sensor_for_failure_mode'
        acc_el_key = 'irrelevant_sensor_for_failure_mode'
    else:
        acc_sel_key = 'relevant_sensors_for_failure_mode'
        acc_el_key = 'irrelevant_sensors_for_failure_mode'
    result_dict = {
        "config": {
            "model_dtype": "torch.bfloat16", 
            "model_name": model_name,
            "model_sha": "main"
        },
        "results": {
            "acc_overall": {
                "acc": original
            },
            "acc_sel": {
                "acc_sel": relevancy_scores[acc_sel_key]
            },
            "acc_el": {
                "acc_el": relevancy_scores[acc_el_key]
            },
            "acc_perturb": {
                "perturb_score": perturb
            },
            "score_consistency": {
                "consist_score": consist
            }
        },
    }
    print(uq_scores)
    for k, v in uq_scores.items():
        result_dict['results'][k] = {k: v}
    for asset in asset_scores:
        if not asset:
            asset_lower = 'other'
        else:
            asset_lower = asset.lower().replace(' ', '_')
        result_dict['results'][f'acc_{asset_lower}'] = {f'acc_{asset_lower}': asset_scores[asset]}
    out_model_name = model_name.replace('/', '--')
    out_fname = f'results/demo-leaderboard/gpt2-demo/results_{out_model_name}.json'
    with open(out_fname, 'w') as f:
        f.write(json.dumps(result_dict))
    print(f'all results written to {out_fname}')


# In[ ]:


# !nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9


# In[ ]:




