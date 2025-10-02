from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
import os

load_dotenv()

keys = os.environ.get("WATSONX_APIKEY", "")
urls = os.environ.get("WATSONX_URL", "")

os.environ["WATSONX_APIKEY"] = keys
os.environ["WATSONX_URL"] = urls


def get_response(
    prompt,
    model_id="mistralai/mistral-large",
    parameters=None,
    stop=[
        "<>",
        "Note:",
    ],
):
    if not parameters:
        parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 2000,
            "min_new_tokens": 1,
        }

    if isinstance(stop, str):
        stop = [stop]

    if "stop_sequences" in parameters.keys():
        parameters["stop_sequences"] = parameters["stop_sequences"].extend(stop)
    else:
        parameters["stop_sequences"] = stop

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=keys,
    )

    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id="c4bfae5a-377f-44d6-b37a-68435a056744",
    )

    generated_response = model.generate(prompt=prompt)
    return generated_response["results"][0]["generated_text"]


def count_tokens(text, model_id="mistralai/mistral-large"):

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=keys,
    )

    model = Model(
        model_id=model_id,
        credentials=credentials,
        project_id="c4bfae5a-377f-44d6-b37a-68435a056744",
    )

    tokenized_response = model.tokenize(prompt=text, return_tokens=True)
    return tokenized_response["result"]["token_count"]
