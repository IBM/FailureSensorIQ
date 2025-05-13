import copy
import json
import logging
import Levenshtein
import os
import numpy as np
import pandas as pd
import re
import spacy
from datetime import datetime
import sys
from collections import Counter
from functools import partial
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import recall_score, precision_score, accuracy_score
from scipy.stats import wilcoxon
from tqdm import tqdm
from typing import List

from dataset_preprocessing import load_from_jsonl, write_to_jsonl
from GeneralLLM import LargeLanguageModel, Qwen, ChatGPT
from KPPerturbation import MultipleChoiceQuestion, get_mcq_llm_answer, QuestionGenerator

logging.basicConfig(level=logging.INFO)

def generate_prompt_for_data(data:list):
    result = copy.deepcopy(data)
    for i in range(len(data)):
        mcq = MultipleChoiceQuestion()
        mcq.load_dict(data[i])
        result[i]['prompt'] = mcq.get_prompt()
    return result

def jaccard_similarity(x:set, y:set):
    result = len(x.intersection(y)) / len(x.union(y))
    return result

def consist_similarity(x:set, y:set):
    result = len(x.intersection(y)) / len(x)
    return result

def consist_score(x:np.array, y:np.array):
    assert(len(x) == len(y))
    n_both_correct = 0
    for a, b in zip(x,y):
        n_both_correct += (a == 1 and b == 1)
    result = n_both_correct / len(x)
    return result

def joint_analysis(left_data: List[dict], right_data: List[dict]):
    left_df = pd.DataFrame(left_data)
    right_df = pd.DataFrame(right_data)

    # merge two data
    merge_df = pd.merge(left_df, right_df, on = ['subject', 'id'], how = 'inner')

    merge_df['score_x'] = (merge_df['true_answer_x'] == merge_df['model_output_x']).astype(int)
    merge_df['score_y'] = (merge_df['true_answer_y'] == merge_df['model_output_y']).astype(int)

    score_x = merge_df['score_x'].mean()
    score_y = merge_df['score_y'].mean()
    # wilcoxon_result = wilcoxon(
    #     merge_df['score_x'].values,
    #     merge_df['score_y'].values,
    #     alternative = 'greater'
    # )

    print('MCQ Benchmark Performance:')
    print(f"score_1 = {score_x.round(4)} ({merge_df['score_x'].sum()}/{merge_df.shape[0]})")
    print(f"score_2 = {score_y.round(4)} ({merge_df['score_y'].sum()}/{merge_df.shape[0]})")
    print(f"performance_drop_rate = {((score_y-score_x)).round(4)}")
    # print(f"Wilcoxon hypothesis test = {wilcoxon_result}")
    
    recall_c = recall_score(merge_df['score_x'], merge_df['score_y'])
    consist_c = consist_score(merge_df['score_x'], merge_df['score_y'])

    # print('Performance Consistency:')
    print(f"recall_c = {round(recall_c,4)}")
    print(f"acc@consist = {round(consist_c,4)}")
    
    return merge_df

def llm_based_knowledge_invariance_analysis(
    left_data: List[dict], right_data: List[dict],
    judge: ChatGPT, log_save_path:str, start_id:int = 0):
    left_df = pd.DataFrame(left_data)
    right_df = pd.DataFrame(right_data)
    scores = []
    results = []
    # merge two data
    merge_df = pd.merge(left_df, right_df, on = ['subject', 'id'], how = 'inner')
    current_id = 0
    for _, row in merge_df.iterrows():
        if current_id < start_id:
            current_id += 1
            continue
        print(f'current id = {current_id} start.')
        prompt_x = row['prompt_x']
        prompt_y = row['prompt_y']
        judge_prompt = '''Your task is to grade the knowledge invariance degree of a perturbed multiple choice question against the original question.
You clearly know that if a perturbed question is knowledge-invariant, the question has to satisfy the following requirements:
[Perturbation Requirements Start]
1. Semantic Information Invariance. The perturbed question must have the same semantic information as the original question, which cannot change the name of entities, logic of statements and meaning of equations.
2. Reasoning Invariance. A human test-taker's reasoning process to obtain his/her response in the perturbed question should be consistent with that in the original question.
3. Answer Invariance. The answer of a perturbed question should be semantically equivalent to the answer of the original question.
4. Statement Clarity. The perturbed question should clearly present contexts, conditions and the target of the question without ambiguous statement.
[Perturbation Requirements End]

The grading score is from 1 to 5. Grading criteria are given in the following:
[Grading Criteria Start]
1.0 - There are fatal flaws in the perturbed question that makes it entirely unacceptable.
2.0 - There are major flaws in the perturbed question that makes it unacceptable.
3.0 - Only some parts of the perturbation is acceptable. As a whole, the perturbed question is less acceptable.
4.0 - There are only minor flaws in the perturbed question. As a whole, the perturbed question is acceptable.
5.0 - The perturbation perfectly satisfies all the requirements and is entirely acceptable.
[Grading Criteria End]

[Original Question Start]: 
%s
[Original Question End]

[Perturbed Question Start]:
%s
[Perturbed Question End]

You should grade the perturbation following these steps:
1. Recall the perturbation requirements and grading criteria, and read the original and the perturbed questions in detail.
2. For each of perturbation requirements, carefully judge its satisfaction degree of the perturbed question.
3. Based on step 1 and step 2, give a total grading score for the perturbed question.
4. Analyze strengths and weakness of the perturbed question from the view of perturbation requirements based on step 1,2,3.
Think carefully for a while, then propose your conclusion. Your output template is given as follows:

[Template Start]
{
  "score": <numeric score from 1 to 5>,
  "strength": <"xxx", strengths of the perturbation>,
  "weakness": <"xxx", weaknesses of the perturbation>
}
[Template End]

Your conclusion:
'''%(prompt_x, prompt_y)
        judge_ok = False
        n_retry = 0
        while n_retry < 3:
            try:
                judge.refresh()
                response = judge.listen_and_response(judge_prompt)
                judgement = re.findall(r'[{][^{]*[}]', response)[0]
                judgement = json.loads(judgement)
                assert(judgement.get('score', None) is not None)
                assert(judgement.get('strength', None) is not None)
                assert(judgement.get('weakness', None) is not None)
                judge_ok = True
                break
            except:
                n_retry += 1
                logging.error('Judgement output error. Try again.')
        if judge_ok is False:
            logging.error('Judgement output fatal error. Exit.')
            return -1
        judgement['subject'] = row['subject']
        judgement['id'] = row['id']
        judgement['prompt_x'] = prompt_x
        judgement['prompt_y'] = prompt_y
        scores.append(judgement['score'])
        results.append(judgement)
        write_to_jsonl([results[-1]], log_save_path, 'a')
        current_id += 1
    scores = np.array(scores)
    print(f"Knowledge Invariance Score = {np.mean(scores).round(4)} +- {np.std(scores).round(4)}")
    
    return results

def edit_distance_analysis(left_data: List[dict], right_data: List[dict]):
    left_df = pd.DataFrame(left_data)
    right_df = pd.DataFrame(right_data)
    distances = []
    results = []
    # merge two data
    merge_df = pd.merge(left_df, right_df, on = ['subject', 'id'], how = 'inner')
    for _, row in merge_df.iterrows():
        prompt_x = row['prompt_x']
        prompt_y = row['prompt_y']
        edit_distance = Levenshtein.distance(prompt_x, prompt_y)
        distances.append(edit_distance)
        results.append({
            'subject': row['subject'],
            'id': row['id'],
            'prompt_x': prompt_x,
            'prompt_y': prompt_y,
            'edit_distance': edit_distance
        })
    print(f'Edit distance = {np.mean(distances).round(2)} +- {np.std(distances).round(2)}')
    return results

def question_sentence_analysis(data:List[dict]):
    nlp = spacy.load('en_core_web_sm')
    n_sents = 0
    for item in tqdm(data):
        doc = nlp(item["question_text"])
        n_sents += len(list(doc.sents))
    print(f"Avg # of sentences = {n_sents}/{len(data)} = {n_sents/len(data)}")

def transition_analysis(log_path_1:str, log_path_2:str, 
                        subjects:list,
                        question_id_path:str=None):
    raw_log1 = load_from_jsonl(log_path_1)
    raw_log2 = load_from_jsonl(log_path_2)
    score_xs = []
    score_ys = []
    rops = []
    css = []
    if question_id_path is not None:
        df_qid = pd.read_csv(question_id_path)
    all_log1 = []
    all_log2 = []
    for subject in subjects:
        print(f'========{subject}========')
        log1 = []
        log2 = []
        if question_id_path is None:
            for item in raw_log1:
                if item["subject"] == subject:
                    log1.append(item)
            for item in raw_log2:
                if item["subject"] == subject:
                    log2.append(item)
        else:
            for item in raw_log1:
                if item["subject"] == subject and df_qid[(df_qid['subject'] == item['subject']) * (df_qid['id'] == item['id'])].shape[0] > 0:
                    log1.append(item)
            for item in raw_log2:
                if item["subject"] == subject and df_qid[(df_qid['subject'] == item['subject']) * (df_qid['id'] == item['id'])].shape[0] > 0:
                    log2.append(item)
        all_log1 += log1
        all_log2 += log2
        result = joint_analysis(log1, log2)
        score_xs.append(result['score_x'].mean())
        score_ys.append(result['score_y'].mean())
        rops.append(recall_score(result['score_x'], result['score_y']))
        css.append(consist_score(result['score_x'], result['score_y']))

    print('========All (micro)========')
    joint_analysis(all_log1, all_log2)
    print('========All (macro)========')
    score_xs = np.array(score_xs)
    score_ys = np.array(score_ys)
    macro_pdr = np.mean(score_ys - score_xs)
    macro_rop = np.mean(rops)
    macro_css = np.mean(css)
    print(f"macro_acc (before) = {np.mean(score_xs)}")
    print(f"macro_acc (after) = {np.mean(score_ys)}")
    print(f"macro_pdr = {macro_pdr}")
    print(f"macro_rop = {macro_rop}")
    print(f"macro_acc@consist = {macro_css}")
    return score_xs[0], score_ys[0], css[0]

def knowledge_invariance_analysis(original_data_path, perturbed_data_path,
                                  subjects: list,
                                  referee: LargeLanguageModel,
                                  llm_ki_to_save_path = None,
                                  systematic_gap = 10):
    raw_original_data = load_from_jsonl(original_data_path)
    raw_perturbed_data = load_from_jsonl(perturbed_data_path)
    raw_original_data = generate_prompt_for_data(raw_original_data)
    raw_perturbed_data = generate_prompt_for_data(raw_perturbed_data)
    original_data = []
    perturbed_data = []
    for elem in raw_original_data:
        if elem['id'] % systematic_gap == 0:
            original_data.append(elem)
    for elem in raw_perturbed_data:
        if elem['id'] % systematic_gap == 0:
            perturbed_data.append(elem)
    print(f"n_original_data (after sampling) = {len(original_data)}")
    print(f"n_perturbed_data (after sampling) = {len(perturbed_data)}")
    print("Get the intersaction of the two dataset for analysis.")
    
    score_results = llm_based_knowledge_invariance_analysis(
        left_data = original_data,
        right_data = perturbed_data,
        log_save_path = llm_ki_to_save_path,
        judge = referee,
        start_id = 0
    )
    
    edit_distance_results = edit_distance_analysis(
        left_data = original_data,
        right_data = perturbed_data
    )
    for subject in subjects:
        subject_scores = []
        subject_edit_distances = []
        for elem in score_results:
            if elem['subject'] == subject:
                subject_scores.append(elem['score'])
        for elem in edit_distance_results:
            if elem['subject'] == subject:
                subject_edit_distances.append(elem['edit_distance'])
        print(f'======{subject}======')
        print(f'Edit distance = {np.median(subject_edit_distances).round(2)} +- {np.std(subject_edit_distances).round(2)}')
        print(f'Knowledge invariance score = {np.mean(subject_scores).round(2)} +- {np.std(subject_scores).round(2)}\n')

def get_record_id_for_correct_answer(log_path: str, dimention="asset_name", fdata_path="eval_data/industrial_mcp_original.jsonl"):
    data = load_from_jsonl(log_path)
    data = sorted(data, key=lambda x: x["id"])
    fdata = load_from_jsonl(fdata_path)
    n_correct = 0
    n_incorrect = 0
    n_invalid = 0
    n_multiple = 0
    n_incorrect_single = 0
    n_incorrect_multiple = 0
    correct_record_ids = []
    correct_asset = []
    total_asset = []
    for elem_index, elem in enumerate(data):
        total_asset.append(fdata[elem_index][dimention])
        if elem["model_output"] == elem["true_answer"]:
            correct_asset.append(fdata[elem_index][dimention])
            n_correct += 1
            correct_record_ids.append(elem_index)
            continue
        model_output = np.array(elem["model_output"])
        true_answer = np.array(elem["true_answer"])
        n_incorrect += 1
        # Invalid response
        if np.sum(model_output) == 0:
            n_invalid += 1
        # Select extra options
        elif np.sum(model_output * true_answer) == np.sum(true_answer):
            n_multiple += 1
        elif np.sum(model_output) == 1:
            n_incorrect_single += 1
        else:
            n_incorrect_multiple += 1
    print("======Incorrect Response Analaysis======")

    asset_counter = Counter(total_asset)
    c_asset_counter = Counter(correct_asset)
    print("======")
    res = {}
    for asset, count in asset_counter.items():
        print(f"{asset}: {count}: {c_asset_counter[asset]*1.0/count}")
        res[asset] = c_asset_counter[asset]*1.0/count
    print("======")
    return res

def response_pattern_analysis(log_path:str):
    data = load_from_jsonl(log_path)
    n_correct = 0
    n_incorrect = 0
    n_invalid = 0
    n_multiple = 0
    n_incorrect_single = 0
    n_incorrect_multiple = 0
    for elem in data:
        if elem['model_output'] == elem['true_answer']:
            n_correct += 1
            continue
        model_output = np.array(elem['model_output'])
        true_answer = np.array(elem['true_answer'])
        n_incorrect += 1
        # Invalid response
        if np.sum(model_output) == 0:
            n_invalid += 1
        # Select extra options
        elif np.sum(model_output * true_answer) == np.sum(true_answer):
            n_multiple += 1
        elif np.sum(model_output) == 1:
            n_incorrect_single += 1
        else:
            n_incorrect_multiple += 1
    print('======Incorrect Response Analaysis======')
    print(f"% of correct responses: {round(n_correct / len(data),4)}")
    print(f"% of invalid responses: {round(n_invalid / len(data),4)}")
    print(f"% of too many options: {round(n_multiple / len(data),4)}")
    print(f"% of incorrect single option: {round(n_incorrect_single / len(data),4)}")
    print(f"% of incorrect multiple options: {round(n_incorrect_multiple / len(data),4)}")
    return {
        'accuracy': round(n_correct / len(data),4),
        'pct_invalid': round(n_invalid / len(data),4),
        'pct_too_many': round(n_multiple / len(data),4),
        'pct_incorrect_single': round(n_incorrect_single / len(data),4),
        'pct_incorrect_multi': round(n_incorrect_multiple / len(data),4)
    }
    
