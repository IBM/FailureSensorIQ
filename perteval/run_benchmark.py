import json
import logging
import math
import os
import pandas as pd
import time
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Optional

from dataset_preprocessing import load_from_jsonl, write_to_jsonl
from GeneralLLM import LargeLanguageModel, ChatGPT
from KPPerturbation import MultipleChoiceQuestion, get_mcq_llm_answer, QuestionRewriter

logging.basicConfig(level=logging.INFO)


def test_dataset(data: list, model: LargeLanguageModel, trigger_statement: Optional[str] = None) -> list:
    """
    Args:
        data:List[dict]. Each element contains a question
            keys = ['subject', 'id', 'question', 'options', 'option_ids', 'correct', 'question_first']
        model:LargeLanguageModel. The LLM API
    Return:
        List[dict]. The list containing all output logs.
            keys = ['subject', 'id', 'prompt', 'true_answer', 'model', 'model_output']
    """
    results = []
    n_complete = 0
    for i, elem in enumerate(data):
        print(i)
        mcq = MultipleChoiceQuestion()
        mcq.load_dict(elem)
        if trigger_statement:
            mcq.add_trigger_statement(trigger_statement=trigger_statement)

        # --- START: to delete ---
        # mcq.question = mcq.question[:int(len(mcq.question)/2)]
        # --- END: to delete --
        prompt = mcq.get_prompt()
        logging.info(f"prompt: {prompt}")

        response, response_text = get_mcq_llm_answer(mcq, model)
        # logging.info(f"true_answer: {mcq.correct}")
        # logging.info(f"response: {response}")
        # logging.info(f"response_text: {response_text}")
        a = {
                "subject": elem["subject"],
                "id": elem["id"],
                "prompt": prompt,
                "question_text": mcq.question,
                "options_text": mcq.options,
                "true_answer": mcq.correct,
                "model": model.model,
                "model_output": response,
                "model_original_output": response_text,
            }
        logging.info(a)
        results.append(a)
        n_complete += 1
        # --- START: Control response time ---
        # time.sleep(0.2)
        # --- END: Control response time ---
        if n_complete % 10 == 0:
            logging.info(f"One thread {n_complete} / {len(data)} samples completed.")
        if n_complete == 2:
            break
    return results


def parallel_test_dataset(
    file_path: str,
    log_path: str,
    model_class: type,
    model_selection: str,
    temperature: float,
    subjects: list,
    thread_func=test_dataset,
    n_thread=4,
    start_id=None,
    end_id=None,
    simple_question_path=None,
    trigger_statement=None
) -> str:
    # 1. read data
    raw_data = load_from_jsonl(file_path)

    # data = raw_data
    data = []
    if simple_question_path is None:
        for item in raw_data:
            if item["subject"] in subjects:
                data.append(item)
    else:
        logging.info("Question selection strategy: simple_questions")
        df_sq = pd.read_csv(simple_question_path)
        for item in raw_data:
            if (
                df_sq[
                    (df_sq["subject"] == item["subject"]) * (df_sq["id"] == item["id"])
                ].shape[0]
                > 0
            ):
                data.append(item)

    if start_id is None:
        start_id = 0
    if end_id is None:
        end_id = len(data)
    data = data[start_id:end_id]

    print(f"# of samples = {len(data)}")

    # 2. split data into disjoint chunks
    chunk_size = math.ceil(len(data) / n_thread)
    data_chunks = []
    for id in range(n_thread):
        data_chunks.append(
            data[(id * chunk_size) : min((id + 1) * chunk_size, len(data))]
        )

    # 3. concurrently test the response of the LLM on each chunk
    logging.info(f"Log path: {log_path}")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                thread_func,
                data=chunk,
                model=model_class,
                trigger_statement=trigger_statement,
            )
            for chunk in data_chunks
        ]

        # obtain and aggregate results
        n_complete = 1
        for future in as_completed(futures):
            try:
                result = future.result()
                write_to_jsonl(result, log_path, "a")
                logging.info(f"{n_complete} / {n_thread} thread(s) completed.")
                n_complete += 1

            except Exception as exc:
                logging.info(f"Task generated an exception: {exc}")

    return log_path


if __name__ == "__main__":
    file_paths = ["./eval_data/test.jsonl"]
    log_path_prefices = ["./log/gpt-3.5-turbo-test"]
    for file_path, log_path_prefix in zip(file_paths, log_path_prefices):
        print(f"file_path = {file_path}")
        print(f"log_path_prefix = {log_path_prefix}")
        start_time = time.time()
        parallel_test_dataset(
            file_path=file_path,
            log_path_prefix=log_path_prefix,
            simple_question_path=None,
            model_class=ChatGPT,
            model_selection="gpt-3.5-turbo",
            temperature=0.2,
            thread_func=test_dataset,
            n_thread=8,
            start_id=None,
            end_id=None,
        )
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Running time = {run_time} s")
