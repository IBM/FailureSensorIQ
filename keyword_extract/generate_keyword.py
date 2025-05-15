import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate
from industrialqa_fmsr.baseline.PertEval.keyword_extract.prep_mcqa import format_mcqa_from_jsonl
from industrialqa_fmsr.wrapper.watsonx_llm import get_completion_response
from industrialqa_fmsr.baseline.PertEval.keyword_extract.prompt import system_promopt
from json_repair import repair_json

# Step 1: Load all formatted MCQA questions
all_q = format_mcqa_from_jsonl('../eval_data/fmsr_processed/filtered_data_all_Mar_30_2025.jsonl')

# Step 2: Prepare the LangChain prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template=system_promopt
)

# Step 3: Define a function to run the LLM prompt for a single question
def run_prompt(question_text):
    try:
        formatted = prompt.format(question=question_text)
        response = get_completion_response(formatted, model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8')
        return repair_json(response)
    except Exception as e:
        return {"error": str(e), "input": question_text}

# Step 4: Run all prompts in parallel
def run_all_prompts_parallel(mcqs, max_workers=8):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_prompt, q): i for i, q in enumerate(mcqs)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as exc:
                results.append((idx, {"error": str(exc)}))
    # Sort by original question order
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]

# Step 5: Save to .jsonl
def save_results_to_jsonl(results, filepath="fmsr_llm_responses.jsonl"):
    with open(filepath, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"Running LLM on {len(all_q)} questions...")
    all_results = run_all_prompts_parallel(all_q, max_workers=16)
    save_results_to_jsonl(all_results, "fmsr_llm_responses.jsonl")
    print("Results saved to fmsr_llm_responses.jsonl")
