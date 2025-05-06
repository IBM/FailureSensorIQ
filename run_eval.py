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


all_request_models = get_requests('cc4718/requests')
all_result_models = get_results('cc4718/results')


# In[4]:


models_todo = set(all_request_models) - set(all_result_models)
print(models_todo)


# In[ ]:


for model_name in models_todo:
    # perteval
    original_log_path = get_perteval_results(model_name, 'original')
    time.sleep(10)
    perturb_log_path = get_perteval_results(model_name, 'perturb')
    # stats = tas.transition_analysis(original_log_path, perturb_log_path, subjects=["failure_mode_sensor_analysis"])
    original, perturb, consist = tas.transition_analysis(original_log_path, perturb_log_path, subjects=["failure_mode_sensor_analysis"])
    asset_scores = tas.get_record_id_for_correct_answer(original_log_path, dimention='asset_name')
    relevancy_scores = tas.get_record_id_for_correct_answer(original_log_path, dimention='relevancy')
    # uq on the original dataset
    # dataset_path = 'data/fmsr'
    # accuracy, nll_loss_avg, ece_score_avg = run_uq(model_name=model_name, dataset_path=dataset_path)
    # uq bench
    uq_score = run_uq_benchmark(model_name)
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
                "acc_sel": relevancy_scores['relevant_sensors_for_failure_mode']
            },
            "acc_el": {
                "acc_el": relevancy_scores['irrelevant_sensors_for_failure_mode']
            },
            "acc_perturb": {
                "perturb_score": perturb
            },
            "score_consistency": {
                "consist_score": consist
            },
            "uncertainty": {
                "uncertainty_score": uq_score
            }
        },
    }
    for asset in asset_scores:
        result_dict['results'][f'acc_{asset}'] = asset_scores[asset]
    # time_now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_model_name = model_name.replace('/', '--')
    out_fname = f'results/demo-leaderboard/gpt2-demo/results_{out_model_name}.json'
    with open(out_fname, 'w') as f:
        f.write(json.dumps(result_dict))


# In[1]:


# !nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9


# In[ ]:




