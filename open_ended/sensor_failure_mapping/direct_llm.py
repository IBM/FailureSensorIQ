import json
from pathlib import Path
from typing import List
import re
from reactxen.utils.model_inference import watsonx_llm as call_llm
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import os


def call_vllm(prompt, model_id=1):
    from openai import OpenAI

    client = OpenAI(
        base_url="http://cccxc605.pok.ibm.com:8004/v1",
        api_key="EMPTY",  # Required, even if not validated
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )

    return {'generated_text': response.choices[0].message.content}


# --- System Prompts ---
question = ""
model_answer = ""
model_id = 16
item_count = 5

GENERATION_SYSTEM_PROMPT = """
You are a reliability engineering expert specializing in industrial equipment diagnostics.

Your task is to respond to a question about the relationship between sensors and failure modes for a given industrial asset. Depending on the question’s intent, return either:

- A flat list of the most relevant **sensors** (if the question is about detecting a failure mode), or  
- A flat list of the most relevant **failure modes** (if the question is about what a sensor can detect).

✱ Your response must follow these rules:
- Only include **1 to {item_count} items** in a **Python-style list**, e.g., ["answer 1", "answer 2"]
- Do **not** include explanations, descriptions, or extra formatting
- Each item must be clear and specific

Here is the question: {question}

Now generate your answer.
"""

EXTRACTION_SYSTEM_PROMPT = """
You are a data extraction agent.

Given a model-generated response containing a Python-style list of strings, extract this list and return a valid JSON object with a single key `"answer"` whose value is that list.

Your output must:
- Contain only the JSON object with the exact format below (no extra text or explanation):
{{
  "answer": [...]
}}
- Ensure valid JSON syntax.
- The list should contain strings exactly as extracted from the input.

Do not include any other text.

Here is the model answer: {model_answer}

Now generate your answer.
"""


# --- LLM Calls (Mock Implementation) ---
def call_llm_generation(question: str) -> str:
    prompt_1 = GENERATION_SYSTEM_PROMPT.format(question=question, item_count=item_count)
    ans = call_vllm(prompt_1, model_id=model_id)
    print(model_id)
    return ans


def call_llm_extraction(model_answer: str) -> dict:
    prompt_1 = EXTRACTION_SYSTEM_PROMPT.format(model_answer=model_answer)
    ans = call_vllm(prompt_1, model_id=model_id)
    return ans


# --- Main Processing Function ---
import json
import re

NUM_WORKERS = 8  # Tune this based on your API rate limit and CPU cores

import json
import os
import re
import multiprocessing as mp


def process_question(args) -> str:
    line, model_id, item_count = args
    item = json.loads(line)
    question = item["text"]
    print(f"🔍 Processing: {question} (Model {model_id})")

    GENERATION_SYSTEM_PROMPT = f"""
You are a reliability engineering expert specializing in industrial equipment diagnostics.

Your task is to respond to a question about the relationship between sensors and failure modes for a given industrial asset. Depending on the question’s intent, return either:

- A flat list of the most relevant **sensors** (if the question is about detecting a failure mode), or  
- A flat list of the most relevant **failure modes** (if the question is about what a sensor can detect).

✱ Your response must follow these rules:
- Only include **1 to {item_count} items** in a **Python-style list**, e.g., ["answer 1", "answer 2"]
- Do **not** include explanations, descriptions, or extra formatting
- Each item must be clear and specific

Here is the question: {question}

Now generate your answer.
"""

    EXTRACTION_SYSTEM_PROMPT = """
You are a data extraction agent.

Given a model-generated response containing a Python-style list of strings, extract this list and return a valid JSON object with a single key answer whose value is that list.

Your output must:
- Contain only the JSON object with the exact format below (no extra text or explanation):
{{
    "answer": [...]
}}
- Ensure valid JSON syntax.
- The list should contain strings exactly as extracted from the input.

Do not include any other text.

Here is the model answer: {model_answer}

Now generate your answer.
"""

    def call_llm_generation() -> str:
        prompt = GENERATION_SYSTEM_PROMPT
        ans = call_vllm(prompt, model_id=model_id)
        return ans

    def call_llm_extraction(model_answer: str) -> dict:
        prompt = EXTRACTION_SYSTEM_PROMPT.format(model_answer=model_answer)
        ans = call_vllm(prompt, model_id=model_id)
        return ans

    try:
        answer_1 = call_llm_generation()["generated_text"]
        print(answer_1)

        final_answer = call_llm_extraction(answer_1)["generated_text"]
        print(final_answer)

        match = re.search(r"```json\s*(\{.*?\})\s*```", final_answer, re.DOTALL)
        if not match:
            match = re.search(
                r"({\s*\"answer\"\s*:\s*\[.*?\]\s*})", final_answer, re.DOTALL
            )

        if match:
            json_text = match.group(1)
            try:
                data = json.loads(json_text)
                item["canswer"] = data["answer"]
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse JSON for: {question}")
                item["canswer"] = []
        else:
            print(f"⚠️ No JSON block found for: {question}")
            item["canswer"] = []

    except Exception as e:
        print(f"❌ Error: {e} in question: {question}")
        item["canswer"] = []

    return json.dumps(item)


def writer_process(queue: mp.Queue, output_file: str):
    with open(output_file, "a") as fout:
        while True:
            line = queue.get()
            if line == "DONE":
                break
            fout.write(line + "\n")
            fout.flush()  # flush after every write
            os.fsync(fout.fileno())


def parallel_process_file(
    input_file: str,
    output_file: str,
    model_id: int,
    item_count: int,
    num_workers: int = 8,
):
    manager = mp.Manager()
    queue = manager.Queue()

    writer = mp.Process(target=writer_process, args=(queue, output_file))
    writer.start()

    with open(input_file, "r") as fin:
        lines = fin.readlines()

    # Package args for each task
    args_list = [(line, model_id, item_count) for line in lines]

    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(process_question, args_list, chunksize=1):
            queue.put(result)

    queue.put("DONE")
    writer.join()
    print(f"✅ Parallel processing complete → {output_file}")


if __name__ == "__main__":
    for model_id in [2]:
        for item_count in [5, 10]:
            input_path = "senarios_10_asset_class.jsonl"
            output_path = f"senarios_10_asset_class_with_answers_model_{model_id}_control_{item_count}.jsonl"
            parallel_process_file(
                input_path, output_path, model_id=model_id, item_count=item_count
            )
