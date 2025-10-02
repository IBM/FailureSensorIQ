import os
import openai
from dotenv import load_dotenv
from utils.utils.prepare_chat_message import get_chat_message


DIR_PREFIX = '/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/'
API_PREFIX = "http://{server}.pok.ibm.com:5997/v1/"

load_dotenv()
ccc_ids = {
    # Deepseek
    "ccc/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "ccc/deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "ccc/mistralai/Ministral-8B-Instruct-2410": "mistralai/Ministral-8B-Instruct-2410",
    "ccc/Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "ccc/google/gemma-2-9b-it": "google/gemma-2-9b-it",

    # granite ft COT (3)
    "ccc/granite-3.1-8b-instruct-cot-standard": "/dccstor/cbm/cache/dmf_models/granite-3.1-8b-instruct-r241212a.20250128T044224-20250128T044228",
    "ccc/granite-3.1-8b-instruct-cot-expert": "/dccstor/cbm/cache/dmf_models/granite-3.1-8b-instruct-r241212a.20250123T051500-20250123T051505",
    "ccc/granite-3.1-8b-instruct-cot-inductive": "/dccstor/cbm/cache/dmf_models/granite-3.1-8b-instruct-r241212a.20250123T234031-20250123T234034",
    
    # ministral 8b ft COT (3)
    "ccc/Ministral-8B-Instruct-2410-cot-standard": "/dccstor/cbm/cbm-benchmark/hosting/Ministral-8B-Instruct-2410-cot_standard",
    "ccc/Ministral-8B-Instruct-2410-cot-inductive": "/dccstor/cbm/cbm-benchmark/hosting/Ministral-8B-Instruct-2410-cot_inductive",
    "ccc/Ministral-8B-Instruct-2410-cot-expert": "/dccstor/cbm/cbm-benchmark/hosting/Ministral-8B-Instruct-2410-cot_expert",

    # simple
    "ccc/ibm-granite/granite-3.2-8b-instruct-preview": "ibm-granite/granite-3.2-8b-instruct-preview",

    # llama 3.1 8b ft COT (3)
    "ccc/Meta-Llama-3.1-8B-Instruct-cot-expert": "/dccstor/cbm/cbm-benchmark/hosting/Meta-Llama-3.1-8B-Instruct-cot_expert",
    "ccc/Meta-Llama-3.1-8B-Instruct-cot-inductive": "/dccstor/cbm/cbm-benchmark/hosting/Meta-Llama-3.1-8B-Instruct-cot_inductive",
    "ccc/Meta-Llama-3.1-8B-Instruct-cot-standard": "/dccstor/cbm/cbm-benchmark/hosting/Meta-Llama-3.1-8B-Instruct-cot_standard",
    
    'ccc/llama-3-1-8b-cot-expert-qlora-2048-ep1':'/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-cot-expert-qlora-2048-ep1',
    'ccc/llama-3-1-8b-cot-expert-qlora-2048-ep5':'/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-cot-expert-qlora-2048-ep5',
    'ccc/llama-3-1-8b-instruct-cot-expert-qlora-2048-ep5':'/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-instruct-cot-expert-qlora-2048-ep5',
    'ccc/llama-3-1-8b-instruct-cot-expert-qlora-2048-ep1':'/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-instruct-cot-expert-qlora-2048-ep1',
    
    'ccc/llama-3-1-8b-formatting_mixtrallarge-cot-expert-spectrum-2048-ep1':'/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-formatting_mixtrallarge-cot-expert-spectrum-2048-ep1',
    'ccc/llama-3-1-8b-formatting_mixtrallarge-cot-expert-qlora-2048-ep1':'/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-formatting_mixtrallarge-cot-expert-qlora-2048-ep1',
    "ccc/llama-3-1-8b-formatting_mixtrallarge-cot-expert-qlora-2048-ep3":"/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-formatting_mixtrallarge-cot-expert-qlora-2048-ep3",
    "ccc/llama-3-1-8b-instruct-formatting_mixtrallarge-cot-expert-qlora-2048-ep1":"/dccstor/cbm/cbm-benchmark/benchmark/finetuning/merged/llama-3-1-8b-instruct-formatting_mixtrallarge-cot-expert-qlora-2048-ep1",
    
    "ccc/llama-3-1-8b_formatting-mixtrallarge-cot-expert-shuffle_spectrum_2048_ep1": DIR_PREFIX+"llama-3-1-8b_formatting-mixtrallarge-cot-expert-shuffle_spectrum_2048_ep1/",
    "ccc/llama-3-1-8b_formatting-mixtrallarge-cot-inductive-shuffle_spectrum_2048_ep1": DIR_PREFIX+"llama-3-1-8b_formatting-mixtrallarge-cot-inductive-shuffle_spectrum_2048_ep1/",
    "ccc/llama-3-1-8b_formatting-mixtrallarge-cot-standard-shuffle_spectrum_2048_ep1": DIR_PREFIX+"llama-3-1-8b_formatting-mixtrallarge-cot-standard-shuffle_spectrum_2048_ep1/",
    "ccc//llama-3-1-8b-instruct_formatting-mixtrallarge-cot-expert-shuffle_spectrum_2048_ep1": DIR_PREFIX+"/llama-3-1-8b-instruct_formatting-mixtrallarge-cot-expert-shuffle_spectrum_2048_ep1/",
    "ccc/llama-3-1-8b-instruct_formatting-mixtrallarge-cot-inductive-shuffle_spectrum_2048_ep1": DIR_PREFIX+"llama-3-1-8b-instruct_formatting-mixtrallarge-cot-inductive-shuffle_spectrum_2048_ep1/",
    "ccc/llama-3-1-8b-instruct_formatting-mixtrallarge-cot-standard-shuffle_spectrum_2048_ep1": DIR_PREFIX+"llama-3-1-8b-instruct_formatting-mixtrallarge-cot-standard-shuffle_spectrum_2048_ep1/",

}

model_base_urls = {
    "ccc/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "http://cccxc588.pok.ibm.com:5991/v1",
    "ccc/deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "http://cccxc588.pok.ibm.com:5992/v1",
    "ccc/Qwen/Qwen2.5-7B-Instruct": "http://cccxc591.pok.ibm.com:5993/v1",
    "ccc/ibm-granite/granite-3.2-8b-instruct-preview": "http://cccxc592.pok.ibm.com:5999/v1",
    "ccc/mistralai/Ministral-8B-Instruct-2410": "http://cccxc583.pok.ibm.com:5999/v1/",
    "ccc/google/gemma-2-9b-it": "http://cccxc610.pok.ibm.com:5993/v1",

    "ccc/granite-3.1-8b-instruct-cot-standard": "http://cccxc580.pok.ibm.com:5998/v1/",
    "ccc/granite-3.1-8b-instruct-cot-expert": "http://cccxc610.pok.ibm.com:5998/v1/",
    "ccc/granite-3.1-8b-instruct-cot-inductive": "http://cccxc612.pok.ibm.com:5998/v1/",

    "ccc/Meta-Llama-3.1-8B-Instruct-cot-expert": "http://cccxc608.pok.ibm.com:5999/v1/",
    "ccc/Meta-Llama-3.1-8B-Instruct-cot-standard": "http://cccxc594.pok.ibm.com:5999/v1/",
    "ccc/Meta-Llama-3.1-8B-Instruct-cot-inductive": "http://cccxc604.pok.ibm.com:5999/v1/",

    "ccc/Ministral-8B-Instruct-2410-cot-standard": "http://cccxc610.pok.ibm.com:5998/v1/",
    "ccc/Ministral-8B-Instruct-2410-cot-inductive": "http://cccxc560.pok.ibm.com:5998/v1/",
    "ccc/Ministral-8B-Instruct-2410-cot-expert": "http://cccxc598.pok.ibm.com:5998/v1/",

    'ccc/llama-3-1-8b-cot-expert-qlora-2048-ep1':"http://cccxc586.pok.ibm.com:5998/v1/",
    'ccc/llama-3-1-8b-cot-expert-qlora-2048-ep5':"http://cccxc596.pok.ibm.com:5998/v1/",
    'ccc/llama-3-1-8b-instruct-cot-expert-qlora-2048-ep5':"http://cccxc594.pok.ibm.com:5998/v1/",
    'ccc/llama-3-1-8b-instruct-cot-expert-qlora-2048-ep1':"http://cccxc612.pok.ibm.com:5998/v1/",

    'ccc/llama-3-1-8b-formatting_mixtrallarge-cot-expert-spectrum-2048-ep1':'http://cccxc596.pok.ibm.com:5998/v1/',
    'ccc/llama-3-1-8b-formatting_mixtrallarge-cot-expert-qlora-2048-ep1':'http://cccxc508.pok.ibm.com:5998/v1/',
    "ccc/llama-3-1-8b-formatting_mixtrallarge-cot-expert-qlora-2048-ep3": 'http://cccxc506.pok.ibm.com:5998/v1/',
    "ccc/llama-3-1-8b-instruct-formatting_mixtrallarge-cot-expert-qlora-2048-ep1": "http://cccxc587.pok.ibm.com:5998/v1/",


    "ccc/llama-3-1-8b_formatting-mixtrallarge-cot-expert-shuffle_spectrum_2048_ep1": API_PREFIX.format(server='cccxc594'),
    "ccc/llama-3-1-8b_formatting-mixtrallarge-cot-inductive-shuffle_spectrum_2048_ep1": API_PREFIX.format(server='cccxc614'),
    "ccc/llama-3-1-8b_formatting-mixtrallarge-cot-standard-shuffle_spectrum_2048_ep1": API_PREFIX.format(server='cccxc600'),
    "ccc//llama-3-1-8b-instruct_formatting-mixtrallarge-cot-expert-shuffle_spectrum_2048_ep1": API_PREFIX.format(server='ccxc509'),
    "ccc/llama-3-1-8b-instruct_formatting-mixtrallarge-cot-inductive-shuffle_spectrum_2048_ep1": API_PREFIX.format(server='cccxc508'),
    "ccc/llama-3-1-8b-instruct_formatting-mixtrallarge-cot-standard-shuffle_spectrum_2048_ep1": API_PREFIX.format(server='cccxc603'),
}


def update_model_base_url(model_id: str, new_base_url: str):
    if model_id not in ccc_ids:
        raise ValueError(f"Model ID '{model_id}' not found in ccc_ids.")
    if not new_base_url:
        raise ValueError("The new base URL cannot be empty.")

    model_base_urls[model_id] = new_base_url
    print(f"Base URL for model '{model_id}' updated to: {new_base_url}")


def add_model_to_ccc_ids(model_id: str, model_path: str):
    if model_id in ccc_ids:
        raise ValueError(f"Model ID '{model_id}' already exists in ccc_ids.")
    ccc_ids[model_id] = model_path
    print(f"Model '{model_id}' added successfully to ccc_ids.")


def get_model_base_url(model_id: str):
    base_url = model_base_urls.get(model_id)
    if not base_url:
        raise ValueError(f"Base URL for model ID '{model_id}' not found.")
    return base_url


def get_ccc_client(model_id: str, num_retries=3):
    """Returns an OpenAI client instance configured with the CCC API base."""
    api_base = os.environ.get("CCC_API_BASE")
    api_key = os.getenv(
        "CCC_API_KEY", "1111"
    )  # It's better to use an environment variable for the API key

    api_base = get_model_base_url(model_id)
    #print (api_base)

    # Initialize and return the client
    client = openai.Client(
        api_key=api_key,
        base_url=api_base,
        timeout=120.0,
        max_retries=num_retries,
    )
    return client


def get_chat_response(
    messages,
    model_id="granite-3.1-8b-instruct-r241212a",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=2,
    seed=20,
    is_system_prompt=False,
    **kwargs, # Accept any additional parameters
):

    if stop is None:
        stop = ["<>", "Note:"]

    c_messages = get_chat_message(messages, is_system_prompt)
    client = get_ccc_client(model_id, num_retries=num_retries)
    model_name = ccc_ids.get(model_id)
    # print("c_messages", c_messages)
    chat_completion = client.chat.completions.create(
        messages=c_messages,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
        **kwargs, # Accept any additional parameters
    )
    # print("chat_completion.choices[0].message.content", chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content


def get_completion_response(
    message,
    model_id="granite-3.1-8b-instruct-r241212a",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=2,
    seed=20,
    is_system_prompt=False,
    **kwargs, # Accept any additional parameters
):

    if stop is None:
        stop = ["<>", "Note:", "<|end of text|>"]
    elif isinstance(stop, str):
        stop = [stop]
    else:
        pass

    client = get_ccc_client(model_id, num_retries=num_retries)
    model_name = ccc_ids.get(model_id)
    completion = client.completions.create(
        model=model_name,
        prompt=message,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
        **kwargs, # Accept any additional parameters
    )
    return completion.choices[0].text


def count_tokens(text, model_id="ccc/Qwen/Qwen2.5-7B"):
    c_messages = get_chat_message(text, False)
    client = get_ccc_client(model_id, num_retries=1)
    model_name = ccc_ids.get(model_id)
    response = client.chat.completions.create(
        messages=c_messages,
        model=model_name,
        max_tokens=1,
        temperature=0,
    )
    return response.usage.prompt_tokens