# `get_response` Function Documentation

The `get_response` function is designed to interact with an API for text generation. Below is a table that describes the function parameters and how they relate to the `watsonx` model provider.

## Parameters:

| Parameter        | Description                                                                                 | watsonx Model Provider Value                                      |
|------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `messages`       | A list of input messages that the model will process and respond to.                         | `["Hello, how are you?", "Can you help me with this task?"]`      |
| `model_id`       | The identifier for the model to use. Defaults to `"watsonx/ibm/granite-3-8b-instruct"`.       | `"watsonx/ibm/granite-3-8b-instruct"`                             |
| `max_tokens`     | The maximum number of tokens (words, subwords, etc.) to generate in the response.            | `2000`                                                            |
| `temperature`    | A float that controls the randomness of the generated response. Lower values make it more deterministic. | `0`                                                               |
| `stop`           | A list of stop sequences that will halt the response generation. Defaults to `["<>", "Note:"]`. | `["<>", "Note:"]`                                                 |
| `num_retries`    | The number of retries in case of a failed API request (e.g., network issues).                | `2`                                                               |
| `seed`           | A random seed to ensure reproducibility of the results.                                      | `20`                                                              |
| `is_system_prompt` | A boolean flag indicating whether the messages are system prompts (True/False).               | `False`                                                           |

## Example:

```python
response = get_response(
    messages=["Hello, how are you?", "Can you help me with this task?"]
)
print(response)
