import copy
import json
import math
import os
import pandas as pd
import random
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
import sys

from GeneralLLM import LargeLanguageModel, Qwen, ChatGPT
from KPPerturbation import MultipleChoiceQuestion, KPPerturbation, OptionFormatPerturbation, CaesarPerturbation, OptionPermutationPerturbation, ChangeQuestionPosPerturbation, ChangeTypePerturbation, QuestionGenerator, MixedPerturbation, ParaphrasingPerturbation

logging.basicConfig(level=logging.INFO)

def write_to_jsonl(x:list, file_name:str, mode:str = 'w'):
    with open(file_name, mode) as f:
        for item in x:
            json.dump(item, f)
            f.write('\n')
    return

def load_from_jsonl(file_name:str)->list:
    result = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            result.append(json.loads(line))
    return result

def preprocess_mmlu(folder_path:str, save_path:str):
    ''' Transform the original MMLU dataset to the format that PertEval can handle.
        All questions in the same set (dev/val/test) will be aggregated in to a single file.
    Args:
        folder_path: the folder path of the original MMLU dataset.
        save_path: the path to save processed data.
    Return:
        dev_data:List[dict]
        val_data:List[dict]
        test_data:List[dict]
        Each element of the data list contains the following keys and values
            that PertEval can handle:
            subject:str, the subject of current question.
            id:str, the id of current question. A question can be uniquely
                dentified using its subject and id.
            question:str, the question text.
            options:List[str], the list of option contents.
            option_ids:List[str], the list of option ids. Default = ['A', 'B', 'C', 'D'].
            question_first:bool, indicating whether the question appears before options.
                It is intervened by the knowledge-invariant perturbation SwapPos.
            correct:List[bool], the list that indicates wheather each option is the correct
                answer or not. For example, correct = [True, False, False, True] means that 
                only the first and the last options are correct answers.
            text_type: str, either 'choice' (multiple-choice question) or 'judgement' (judgement
                question), which controls the question type, and is intervened by
                the knowledge-invariant perturbation ChangeType.
    '''
    option_ids = ['A','B','C','D']
    dev_data = []
    val_data = []
    test_data = []
    # visit all files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # root = path, file = file_name
            if "test" in file:
                current_data = test_data
            elif "val" in root:
                current_data = val_data
            else:
                current_data = dev_data
            file_path = os.path.join(root, file)
            data = pd.read_csv(file_path, names = ['question','A','B','C','D','answer'])
            for id, row in data.iterrows():
                # generate a question
                mcq = {'subject': file.replace('.csv',''),
                       'id': id,
                       'question': row['question'],
                       'options': [row[elem] for elem in ['A', 'B', 'C', 'D']],
                       'option_ids': option_ids, 
                       'question_first': True,
                       'correct': [oid == row['answer'] for oid in option_ids],
                       'text_type': 'choice'}
                current_data.append(mcq)
            print(os.path.join(root,file), 'processed') 
    # write to file
    write_to_jsonl(dev_data, os.path.join(save_path, "dev.jsonl"))
    write_to_jsonl(val_data, os.path.join(save_path, "val.jsonl"))
    write_to_jsonl(test_data, os.path.join(save_path, "test.jsonl"))

    return dev_data, val_data, test_data

def perturb_dataset(ptb: KPPerturbation,
                    file_name:str,
                    target_name:str,
                    subjects:list = None):
    '''Perturb multiple-choice questions with knowledge-invariant perutrbations then save to file.
    Args:
        ptb:KPPerturbation, the perturbation method.
        file_name:str, the path to the .jsonl file of original multiple-choice questions.
            each line of the .jsonl file is a dictionary of a multiple-choice question including:
            
            subject:str, the subject of current question.
            id:str, the id of current question. A question can be uniquely
                dentified using its subject and id.
            question:str, the question text.
            options:List[str], the list of option contents.
            option_ids:List[str], the list of option ids. Default = ['A', 'B', 'C', 'D'].
            question_first:bool, indicating whether the question appears before options.
                It is intervened by the knowledge-invariant perturbation SwapPos.
            correct:List[bool], the list that indicates wheather each option is the correct
                answer or not. For example, correct = [True, False, False, True] means that 
                only the first and the last options are correct answers.
            text_type: str, either 'choice' (multiple-choice question) or 'judgement' (judgement
                question), which controls the question type, and is intervened by
                the knowledge-invariant perturbation ChangeType.
        target_name:str, the name to save perturbed question to a new .jsonl file.
        subjects:List[str], the list of subjects to be selected and perturbed,
            e.g., ['college_mathematics_test', 'high_school_world_history_test']
    Return:
        ptbd_ques: List[dict]. Each element is a perturbed question. The result is also written
            to the .jsonl file target_name.
    '''
    questions = load_from_jsonl(file_name)
    valid_questions = []
    if subjects is not None:
        for elem in questions:
            if elem['subject'] in subjects:
                valid_questions.append(elem)
    else:
        valid_questions = questions
    ptbd_ques = []
    for elem in tqdm(valid_questions):
        mcq = MultipleChoiceQuestion()
        mcq.load_dict(elem)
        mcq_dict = ptb.perturb(mcq).to_dict()
        mcq_dict['subject'] = elem['subject']
        mcq_dict['id'] = elem['id']
        ptbd_ques.append(mcq_dict)
    write_to_jsonl(ptbd_ques, target_name)
    return ptbd_ques

def generate_qa_data_from_perturbation(ptb: KPPerturbation,
                                       file_name:str,
                                       target_name:str,
                                       subjects:list = None):
    '''Generate the question answering data using PertEval. The result could
        be used for fine-tuning LLMs to enhance their knowledge capacity. 
    Args:
        ptb: KPPerturbation, the perturbation method.
        file_name:str, the path to the .jsonl file of original multiple-choice questions.
            each line of the .jsonl file is a dictionary of a multiple-choice question including:
            
            subject:str, the subject of current question.
            id:str, the id of current question. A question can be uniquely
                dentified using its subject and id.
            question:str, the question text.
            options:List[str], the list of option contents.
            option_ids:List[str], the list of option ids. Default = ['A', 'B', 'C', 'D'].
            question_first:bool, indicating whether the question appears before options.
                It is intervened by the knowledge-invariant perturbation SwapPos.
            correct:List[bool], the list that indicates wheather each option is the correct
                answer or not. For example, correct = [True, False, False, True] means that 
                only the first and the last options are correct answers.
            text_type: str, either 'choice' (multiple-choice question) or 'judgement' (judgement
                question), which controls the question type, and is intervened by
                the knowledge-invariant perturbation ChangeType.
        target_name:str, the name to save perturbed question to a new .jsonl file.
        subjects:List[str], the list of subjects to be selected and perturbed,
            e.g., ['college_mathematics_test', 'high_school_world_history_test']
    Return:
        results: List[dict]. Each element is a question-answer pair.
            The result format is aligned with LLaMA-Factory for convenience:

            instruction:str, the question prompt.
            input: str, an empty string.
            output: str, the formatted true answer of the question.
        
    '''
    questions = load_from_jsonl(file_name)
    valid_questions = []
    if subjects is not None:
        for elem in questions:
            if elem['subject'] in subjects:
                valid_questions.append(elem)
    else:
        valid_questions = questions
    results = []
    for elem in tqdm(valid_questions):
        mcq = MultipleChoiceQuestion()
        mcq.load_dict(elem)
        if ptb is None:
            mcq_ptbd = copy.deepcopy(mcq)
        else:
            mcq_ptbd = ptb.perturb(mcq)
        qa = {
            'instruction': mcq_ptbd.get_prompt(),
            'input':'',
            'output': mcq_ptbd.get_formatted_answer()
        }
        results.append(qa)

    data_string = json.dumps(results, indent = 4)
    with open(target_name, 'w') as fp:
        fp.write(data_string)
    print(f'Data already saved at {target_name}.')
    return results



