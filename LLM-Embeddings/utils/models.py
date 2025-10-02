import os
from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

keys = os.environ.get("WATSONX_APIKEY", "")
os.environ["WATSONX_APIKEY"] = keys


def get_llm(model_id="granite", parameters=None):
    if parameters is None:
        parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 2000,
            "min_new_tokens": 1,
        }

    chat = ChatWatsonx(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="c4bfae5a-377f-44d6-b37a-68435a056744",  # CBM project
        params=parameters,
    )
    return chat


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

    chat = ChatWatsonx(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="c4bfae5a-377f-44d6-b37a-68435a056744",  # CBM project
        params=parameters,
    )

    messages = [
        ("human", prompt),
    ]

    # print (prompt)
    # print (parameters)

    ans = chat.invoke(messages)
    # print (ans)
    # exit(0)
    # return ans
    return ans.content
