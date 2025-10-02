import os
from litellm import completion
from litellm import batch_completion
from dotenv import load_dotenv
from litellm import token_counter
from utils.prepare_chat_message import get_chat_message

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
keys = os.getenv("WATSONX_APIKEY", "")
urls = os.getenv("WATSONX_URL", "")
pid = os.getenv("WATSONX_PROJECT_ID", "")
openai_keys = os.getenv("OPENAI_API_KEY", "")
openai_orgs = os.getenv("OPENAI_ORGANIZATION", "")


def get_batch_chat_response(
    messages,
    model_id="watsonx/ibm/granite-3-8b-instruct",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=2,
    seed=20,
    is_system_prompt=False,
    **kwargs,  # Accept any additional parameters
):
    if stop is None:
        stop = ["<>", "Note:"]
    elif isinstance(stop, str):
        stop = [stop]
    else:
        pass

    decorated_messages = []
    for message in messages:
        c_messages = get_chat_message(message, is_system_prompt)
        decorated_messages.append(c_messages)

    batch_response = batch_completion(
        model=model_id,
        messages=decorated_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        num_retries=num_retries,
        seed=seed,
        **kwargs,  # Pass any additional parameters
    )
    final_content = []
    for response in batch_response:
        final_content.append(response["choices"][0]["message"]["content"])
    return final_content


def get_chat_response(
    messages,
    model_id="watsonx/ibm/granite-3-8b-instruct",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=2,
    seed=20,
    is_system_prompt=False,
    **kwargs,  # Accept any additional parameters
):
    if stop is None:
        stop = ["<>", "Note:"]
    elif isinstance(stop, str):
        stop = [stop]
    else:
        pass

    if "mixtral-8x7B-instruct-v0.1" in model_id:
        if is_system_prompt:
            is_system_prompt = False
            messages = messages[1:]

    c_messages = get_chat_message(
        messages, is_system_prompt
    )

    trial = 3
    while trial > 0:
        try:
            response = completion(
                model=model_id,
                messages=c_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                num_retries=num_retries,
                seed=seed,
                **kwargs,
            )
            break
        except Exception as ex:
            print (ex)
        trial = trial - 1

    try:
        return response["choices"][0]["message"]["content"]
    except:
        return "Error"


def count_tokens(text, model_id="watsonx/ibm/granite-3-8b-instruct"):
    messages = [{"user": "role", "content": text}]
    return token_counter(model=model_id, messages=messages)
