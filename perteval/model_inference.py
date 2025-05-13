from perteval.utils.prepare_chat_message import (
    get_decorated_chat_template,
)


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
            raise NotImplementedError(
                "Text generation for Lite LLM is not implemented yet."
            )
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
