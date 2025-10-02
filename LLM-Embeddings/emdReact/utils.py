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
                if i+1 < len(messages):
                    c_messages.append({"content": messages[i + 1], "role": "user"})
                else:
                    c_messages.append({"content": '\nYour output:', "role": "user"})
    elif isinstance(messages, list):
        c_messages.append({"content": messages[0], "role": "user"})
        for i in range(1, len(messages), 2):
            c_messages.append({"content": messages[i], "role": "assistant"})
            c_messages.append({"content": messages[i + 1], "role": "user"})
    else:
        pass
    return c_messages
    
def get_decorated_chat_template(model_path, user_message):
    from transformers import AutoTokenizer

    conv = [
        {"role": "user", "content": user_message}
    ]  # Update the value of content as needed.
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



def get_llm_response(model_name, prompt, params):
    """
    Dynamically selects and calls the appropriate LLM model's `get_response` function.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The question or prompt for the LLM model.
        params (dict): Additional parameters to pass to the model's response function.

    Returns:
        str: The answer from the selected LLM model.
    """

    print('test------------', model_name)
    if params:
        params_copy = params.copy()
    else:
        params_copy = {}

    text_generation_choice = params_copy.pop("text_generation_choice", "chat")
    # print (text_generation_choice)

    # this is a temporary code
    if text_generation_choice == "template":
        if isinstance(prompt, str):
            prompt = get_decorated_chat_template(model_name.split("/", 1)[-1], prompt)
        else:
            prompt = get_decorated_chat_template(model_name.split("/", 1)[-1], prompt[-1])

    # Check if model is for RITS
    if model_name.startswith("rits/"):
        from industrialqa_fmsr.wrapper.rits_llm import (
            get_chat_response,
            get_completion_response,
        )
        print (prompt, model_name)
        if text_generation_choice == "chat":
            ans_exp = get_chat_response(prompt, model_id=model_name, **params_copy)
        elif text_generation_choice == "text" or text_generation_choice == "template":
            ans_exp = get_completion_response(
                prompt, model_id=model_name, **params_copy
            )
        else:
            raise ValueError("Invalid text_generation_choice for RITS.")
        #print(f"Using RITS model: {model_name}")

    # Check if model is for WatsonX
    elif model_name.startswith("watsonx/"):
        from industrialqa_fmsr.wrapper.lite_llm import get_chat_response

        if text_generation_choice == "chat":
            ans_exp = get_chat_response(prompt, model_id=model_name, **params_copy)
        elif text_generation_choice == 'template':
            from industrialqa_fmsr.wrapper.watsonx_llm import get_completion_response
            ans_exp = get_completion_response(prompt, model_id=model_name.split("/", 1)[-1], **params_copy)
        elif text_generation_choice == "text":
            ans_exp = get_chat_response(prompt, model_id=model_name, **params_copy)
            """            
            raise NotImplementedError(
                "Text generation for Lite LLM is not implemented yet."
            )
            """
        
        else:
            raise ValueError("Invalid text_generation_choice for Litellm.")
        #print(f"Using Lite LLM model: {model_name}")

    # Check if model is for CCC
    elif model_name.startswith("ccc/"):
        from industrialqa_fmsr.wrapper.ccc_llm import (
            get_chat_response,
            get_completion_response,
        )
        if text_generation_choice == "chat":
            ans_exp = get_chat_response(prompt, model_id=model_name, **params_copy)
        elif text_generation_choice == "text":
            ans_exp = get_completion_response(
                prompt, model_id=model_name, **params_copy
            )
        else:
            raise ValueError("Invalid text_generation_choice for CCC.")
        #print(f"Using CCC model: {model_name}")
    # Default: Using Lite LLM
    # Azure
    elif model_name.startswith("azureopenai/"):
        from industrialqa_fmsr.wrapper.azure_llm import get_chat_response
        if text_generation_choice == "chat":
            ans_exp = get_chat_response(prompt, model_id=model_name, **params_copy)
    else:
        from industrialqa_fmsr.wrapper.watsonx_llm import (
            get_chat_response,
            get_completion_response,
        )

        if text_generation_choice == "chat":
            ans_exp = get_chat_response(prompt, model_id=model_name, **params_copy)
        elif text_generation_choice == "text":
            ans_exp = get_completion_response(
                prompt, model_id=model_name, **params_copy
            )
        else:
            raise ValueError("Invalid text_generation_choice for Watsonx.")
        #print(f"Using WatsonX model: {model_name}")

    return ans_exp





