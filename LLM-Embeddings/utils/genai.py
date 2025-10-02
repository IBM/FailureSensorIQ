from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    TextGenerationParameters,
)
from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import TextTokenizationParameters, TextTokenizationReturnOptions

completion_tokens = prompt_tokens = api_calls = 0
MAX_TOKENS = 4000

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
load_dotenv()
import os
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = Client(credentials=Credentials.from_env())

modelset = [
    "meta-llama/llama-3-70b-instruct",
    "ibm/granite-13b-chat-v2",
    "mistralai/mixtral-8x7b-instruct-v01",
    "ibm-meta/llama-2-70b-chat-q",
    "openai",
]

selected_model = modelset[0]


def genai_llm(
    prompt,
    model_id=selected_model,
    decoding_method=DecodingMethod.GREEDY,
    temperature=1.0,
    max_tokens=2000,
    n=1,
    stop=['<>'],
) -> list:
    messages = prompt
    if selected_model != "openai":
        messages = [HumanMessage(content=prompt)]
    if isinstance(stop, str):
        stop = [stop]
    return chatmodel(
        messages,
        model_id=model_id,
        decoding_method=decoding_method,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )


def chatmodel(
    messages,
    model_id=selected_model,
    decoding_method=DecodingMethod.GREEDY,
    temperature=1.0,
    max_tokens=2000,
    n=1,
    stop=[],
) -> list:

    if selected_model == "openai":
        return openaicall(messages=messages, stop=stop)

    # print (messages)
    global completion_tokens, prompt_tokens, api_calls
    outputs = []
    parameters = TextGenerationParameters(
        decoding_method=decoding_method,
        max_new_tokens=max_tokens,
        min_new_tokens=10,
        temperature=temperature,
        stop_sequences=stop,
    )

    while n > 0:
        n -= 1
        response = client.text.chat.create(
            model_id=model_id,
            messages=messages,
            parameters=parameters,
        )
        outputs.extend([response.results[0].generated_text])
        completion_tokens += response.results[0].generated_token_count
        prompt_tokens += response.results[0].input_token_count
        api_calls = api_calls + 1

        # print(completion_tokens, prompt_tokens, api_calls)

        # print (messages)
        # print (response)

    # print (outputs[0])
    return outputs[0]


def openaicall(
    messages,
    temperature=1.0,
    max_tokens=2000,
    n=1,
    stop=[],
):
    global completion_tokens, prompt_tokens, api_calls
    if isinstance(stop, str):
        stop = [stop]
    from openai import OpenAI
    client = OpenAI()
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop,
    )
    completion_text = response.choices[0].text
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    api_calls = api_calls + 1
    return completion_text


def return_usage():
    global completion_tokens, prompt_tokens, api_calls
    return completion_tokens, prompt_tokens, api_calls


"""
ans = genai_llm(
    "What is NLP?",
    model_id="meta-llama/llama-3-70b-instruct",
    decoding_method=DecodingMethod.SAMPLE,
    temperature=1.0,
    max_tokens=100,
    n=3,
    stop=["Observation"],
)
"""


def gpt_usage(backend="bam"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "bam":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }


def count_tokens(input, model_id=selected_model):
    if selected_model == "openai":
        return count_tokens(input)

    lots_of_greetings = [input]
    response = next(
        client.text.tokenization.create(
            model_id=model_id,
            input=lots_of_greetings,
            parameters=TextTokenizationParameters(
                return_options=TextTokenizationReturnOptions(
                    tokens=False,  # return tokens
                )
            ),
        )
    )
    return response.results[0].token_count


def count_tokens(text, model="gpt-3.5-turbo-instruct"):
    # Load the tokenizer
    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")
    import tiktoken

    enc = tiktoken.encoding_for_model(model)

    # Encode the text to get tokens
    tokens = enc.encode(text)

    # Count the tokens
    token_count = len(tokens)
    return token_count
