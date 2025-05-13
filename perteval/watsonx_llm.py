from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as TextParams
from ibm_watsonx_ai.metanames import GenChatParamsMetaNames as ChatParams
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
import os
from perteval.utils.prepare_chat_message import get_chat_message

load_dotenv()


def get_completion_response(
    message,
    model_id="mistralai/mistral-large",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=3,
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

    if isinstance(stop, str):
        stop = [stop]

    keys = os.environ.get("WATSONX_APIKEY", "")
    urls = os.environ.get("WATSONX_URL", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    print ('000001112121222')

    # ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()
    parameters = {
        TextParams.RANDOM_SEED: seed,
        TextParams.TEMPERATURE: temperature,
        TextParams.MAX_NEW_TOKENS: max_tokens,
        TextParams.MIN_NEW_TOKENS: 1,
        TextParams.STOP_SEQUENCES: stop,
    }

    # additional parameters
    all_supported_params = TextParams().get()
    for key, value in kwargs.items():
        if key.upper() in all_supported_params:
            parameters["key"] = value

    credentials = Credentials(
        url=urls,
        api_key=keys,
    )

    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
    )

    print (message)
    print('*****1111****')
    try:
        generated_response = model.generate(prompt=message)
        print('done---------')
    except Exception as ex:
        print (ex)
    print (generated_response)
    return generated_response["results"][0]["generated_text"]


def get_chat_response(
    messages,
    model_id="mistralai/mistral-large",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=3,
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

    keys = os.environ.get("WATSONX_APIKEY", "")
    urls = os.environ.get("WATSONX_URL", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")

    # ibm_watsonx_ai.metanames.GenChatParamsMetaNames().show()
    parameters = {
        # ChatParams.RANDOM_SEED: seed, # Not supported
        ChatParams.TEMPERATURE: temperature,
        ChatParams.MAX_TOKENS: max_tokens,
        # ChatParams.STOP_SEQUENCES: stop, # Not supported
    }

    # pass additional parameters
    all_supported_params = ChatParams().get()
    for key, value in kwargs.items():
        if key.upper() in all_supported_params:
            parameters["key"] = value

    credentials = Credentials(
        url=urls,
        api_key=keys,
    )

    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
    )

    replace_system_by_assistant = False
    if "mixtral-8x7B-instruct-v0.1".lower() in model_id.lower():
        replace_system_by_assistant = True

    messages = get_chat_message(messages, is_system_prompt, replace_system_by_assistant)
    generated_response = model.chat(messages=messages)

    # Print only content
    return generated_response["choices"][0]["message"]["content"]


def count_tokens(text, model_id="mistralai/mistral-large"):

    keys = os.environ.get("WATSONX_APIKEY", "")
    urls = os.environ.get("WATSONX_URL", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")

    credentials = Credentials(
        url=urls,
        api_key=keys,
    )

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
    )

    tokenized_response = model.tokenize(prompt=text, return_tokens=True)
    return tokenized_response["result"]["token_count"]
