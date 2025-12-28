import os
import base64
import mimetypes
from typing import Optional, Literal
import openai

# Set your API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_available_models = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4o"
]


def encode_image(image_path: str, mime_type: Optional[str] = None) -> str:
    file_name = os.path.basename(image_path)
    mime_type = mime_type if mime_type else mimetypes.guess_type(file_name)[0]

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    if not mime_type or not mime_type.startswith("image/"):
        print("Warning: mime_type not specified or not an image. Defaulting to image/png.")
        mime_type = "image/png"

    return f"data:{mime_type};base64,{encoded_image}"


def get_chat_completion(
    model: Optional[openai_available_models] = "gpt-4o",
    *args,
    **kwargs,
):
    """
    Use OpenAI's v1.0+ Python SDK for Chat Completions
    """
    if model is None:
        raise ValueError("model name must be specified")

    response = client.chat.completions.create(
        model=model,
        *args,
        **kwargs
    )
    return response


def test_get_chat_completion():
    def test_call(*args, **kwargs):
        test_message = "What is the content?"
        test_chat_message = [{"role": "user", "content": test_message}]

        response = get_chat_completion(
            model="gpt-4o",
            messages=test_chat_message,
            temperature=0.7,
            max_tokens=100,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            *args,
            **kwargs,
        )

        print(response.choices[0].message.content)

    print("Test using OpenAI official SDK v1.x")
    test_call()


if __name__ == "__main__":
    test_get_chat_completion()
