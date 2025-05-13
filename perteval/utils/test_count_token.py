from industrialqa_fmsr.wrapper.watsonx_llm import count_tokens
from industrialqa_fmsr.wrapper.ccc_llm import count_tokens as ccc_count_tokens
from industrialqa_fmsr.wrapper.rits_llm import count_tokens as rits_count_tokens
from industrialqa_fmsr.wrapper.lite_llm import count_tokens as litellm_count_tokens

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

file_path = "../../baseline/function/example_store/sdg_curated/qa_exam_0.json"
import json
try:
    with open(file_path, "r") as file:
        data = json.load(file)
        print("JSON data loaded successfully!")
        print(data)  # This will print the contents of the JSON file
except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")

text1 = data['question'] + " Option: " + " ".join(data['options'])
text2 = data['question'] + " Option: " + " ".join(data['options']) + " Rational: " + data['answer_reason']

for text in [text1, text2]:
    print (text)

    total_token_count = count_tokens(text)
    print(total_token_count)

    total_token_count = ccc_count_tokens(text)
    print(total_token_count)

    total_token_count = ccc_count_tokens(
        text, model_id="ccc/mistralai/Ministral-8B-Instruct-2410"
    )
    print (total_token_count)

    total_token_count = rits_count_tokens(text)
    print (total_token_count)

    total_token_count = litellm_count_tokens(text)
    print (total_token_count)