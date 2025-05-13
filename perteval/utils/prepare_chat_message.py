from datetime import datetime


def get_chat_message(
    messages, is_system_prompt=False, replace_system_by_assistant=False
):
    c_messages = []
    if isinstance(messages, str):  # Handle the autoregressive nature
        c_messages.append({"content": messages, "role": "user"})
    elif isinstance(messages, list) and len(messages) == 1:
        c_messages.append({"content": messages[0], "role": "user"})
    elif isinstance(messages, list) and is_system_prompt:
        if replace_system_by_assistant:
            c_messages.append({"content": messages[0], "role": "assistant"})
        else:
            c_messages.append({"content": messages[0], "role": "system"})
        if len(messages) > 1:
            c_messages.append({"content": messages[1], "role": "user"})
            for i in range(2, len(messages), 2):
                c_messages.append({"content": messages[i], "role": "assistant"})
                c_messages.append({"content": messages[i + 1], "role": "user"})
    elif isinstance(messages, list):
        c_messages.append({"content": messages[0], "role": "user"})
        for i in range(1, len(messages), 2):
            c_messages.append({"content": messages[i], "role": "assistant"})
            c_messages.append({"content": messages[i + 1], "role": "user"})
    else:
        pass
    print(c_messages)
    return c_messages


def get_decorated_chat_template(model_path, user_message):
    from transformers import AutoTokenizer

    conv = [
        {"role": "user", "content": user_message}
    ]  # Update the value of content as needed.
    if model_path == 'ibm/granite-3-2-8b-instruct':
        model_path = 'ibm-granite/granite-3.3-8b-instruct'
    print (model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    decorated_text = tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        #return_tensors="pt",
        thinking=True,
        #return_dict=True,
        add_generation_prompt=True,
    )
    return decorated_text
