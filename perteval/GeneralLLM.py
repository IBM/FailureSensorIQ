import random
import dashscope
import time
import google.generativeai as genai

from http import HTTPStatus
from openai import OpenAI
from zhipuai import ZhipuAI
from industrialqa_fmsr.wrapper.watsonx_llm import get_chat_response
from industrialqa_fmsr.wrapper.model_inference import get_llm_response
from transformers import AutoModelForCausalLM, AutoTokenizer


class LargeLanguageModel(object):
    """
    The interface for large language models (LLMs).
    """

    def __init__(self, name: str, description: str, temperature: float):
        """
        name:str, the user-defined name of the LLM.
        description:str, the user-defined description of the LLM.
        temperature:float, the temperature parameter of close-sourced LLMs.
        """
        self.name = name
        self.description = name
        self.context = []
        self.temperature = temperature

    def listen_and_response(self, message: str):
        """
        Args:
            message:str, the current message to be sent to the LLM, given the chat history.
        Return:
            result:str, the output of the LLM given the current message and the chat history.
        """
        pass

    def refresh(self):
        """Refresh self.context (the chat history) to initialze the chat environment."""
        pass

    def get_history(self):
        pass

    def append_feedback(self, feedback_error):
        pass

class TransformersModel(LargeLanguageModel):
    def __init__(
        self,
        model_name: str = "ibm-granite/granite-3.2-8b-instruct",
        system_prompt: str = "You are a helpful assistant.",
        description: str = "",
        temperature: float = 0.0
    ):
        super().__init__(
            model_name, description, temperature
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.context.append(system_prompt)

    def append_feedback(self, feedback_error):
        self.context.append(feedback_error)

    def get_llm_response(self):
        messages = [
            {'role': 'system', 'content': self.context[0]},
            {'role': 'user', 'content': '\n'.join(self.context[1:])}
        ]
        tokenized = self.tokenizer.apply_chat_template(messages, return_tensors='pt')
        # tokenized = self.tokenizer(chat_template, return_tensors='pt')
        print('before generate')
        generated_tokens = self.model.generate(tokenized, max_new_tokens=1024)
        print('after generate')
        # response = self.tokenizer.decode(generated_tokens[0][len(tokenized[0]):])
        response = self.tokenizer.decode(generated_tokens[0][len(tokenized[0]):], skip_special_tokens=True)
        # response = response.replace('<|start_header_id|>assistant<|end_header_id|>', '').strip()
        return response
    
    def listen_and_response(
        self,
        message: str = "How to cook Chinese stir-fried eggs and tomatoes?",
        n_outputs=1,
    ):
        self.context.append(message)

        response_ok = False

        response = self.get_llm_response()
        self.context.append(response)

        result = None
        if n_outputs == 1:
            result = response
        else:
            result = [response]
        return result

    def refresh(self, system_prompt: str = "You are a helpful assistant."):
        self.context = [system_prompt]
        return True

    def get_history(self):
        return self.context.copy()


class WatsonxModel(LargeLanguageModel):
    def __init__(
        self,
        name: str = "watsonx-rits/meta-llama/llama-3-3-70b-instruct",
        description: str = "",
        api_key: str = None,
        model: str = "watsonx-rits/meta-llama/llama-3-3-70b-instruct",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 1.0,
    ):
        super(WatsonxModel, self).__init__(
            name.split("watsonx-")[-1], description, temperature
        )
        self.client = None
        self.model = model.split("watsonx-")[-1]
        self.context.append(system_prompt)

    def append_feedback(self, feedback_error):
        self.context.append(feedback_error)

    def listen_and_response(
        self,
        message: str = "How to cook Chinese stir-fried eggs and tomatoes?",
        n_outputs=1,
    ):
        self.context.append(message)

        response_ok = False
        # sleep_time = 0.1
        # time.sleep(sleep_time)

        response = get_llm_response(
            model_name=self.model,
            prompt=self.context,
            params={"temperature": self.temperature,
                    "is_system_prompt": True},
        )
        self.context.append(response)

        result = None
        if n_outputs == 1:
            result = response
        else:
            result = [response]
        return result

    def refresh(self, system_prompt: str = "You are a helpful assistant."):
        self.context = [system_prompt]
        return True

    def get_history(self):
        return self.context.copy()


class Qwen(LargeLanguageModel):
    def __init__(
        self,
        name: str = "qwen-max",
        description: str = "",
        api_key: str = None,
        model: str = "qwen-max",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
    ):
        """
        Args:
            api_key:str, the api key of Qwen models.
            model:str, the model name of the Qwen model to be used.
            system_prompt:str, the system prompt for the LLM.
            for other parameters, see the interface LargeLanguageModel.
        """
        super(Qwen, self).__init__(name, description, temperature)
        dashscope.api_key = api_key
        self.model = model
        self.context.append({"role": "system", "content": system_prompt})

    def _update_sleep_time(self, sleep_time: float):
        result = 0
        if sleep_time <= 4:
            result *= 2
        else:
            result = 0.5
        return result

    def listen_and_response(
        self, message: str = "How to cook Chinese stir-fried eggs and tomatoes?"
    ):
        self.context.append({"role": "user", "content": message})

        response_ok = False
        sleep_time = 0.25
        max_retry = 10
        n_retry = 0
        while response_ok != HTTPStatus.OK and n_retry < max_retry:
            time.sleep(sleep_time)
            response = dashscope.Generation.call(
                model=self.model,
                messages=self.context,
                seed=random.randint(1, 10000),
                temperature=self.temperature,
                result_format="message",  # set the result to be "message" format.
            )
            response_ok = response.status_code
            if response.status_code == HTTPStatus.OK:
                self.context.append(
                    {
                        "role": response.output.choices[0]["message"]["role"],
                        "content": response.output.choices[0]["message"]["content"],
                    }
                )
                result = response.output.choices[0]["message"]["content"]
            else:
                sleep_time = self._update_sleep_time(sleep_time)
            n_retry += 1

        if response_ok != HTTPStatus.OK:
            self.context.append(
                {"role": "assistant", "content": "Error. Exceed max_retry."}
            )
            result = "Error. Exceed max_retry."
            logging.error(result)

        return result

    def refresh(self, system_prompt: str = "You are a helpful assistant."):
        self.context = [{"role": "system", "content": system_prompt}]
        return True

    def get_history(self):
        return self.context.copy()


class ChatGPT(LargeLanguageModel):
    def __init__(
        self,
        name: str = "gpt-3.5-turbo",
        description: str = "",
        api_key: str = None,
        model: str = "gpt-3.5-turbo",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 1.0,
        base_url=None
    ):
        super().__init__(name, description, temperature)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.context.append({"role": "system", "content": system_prompt})

    def listen_and_response(
        self,
        message: str = "How to cook Chinese stir-fried eggs and tomatoes?",
        n_outputs=1,
    ):
        self.context.append({"role": "user", "content": message})

        response_ok = False
        sleep_time = 0.1
        time.sleep(sleep_time)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.context,
            n=n_outputs,
            temperature=self.temperature,
        )
        self.context.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        result = None
        if n_outputs == 1:
            result = response.choices[0].message.content
        else:
            result = [response.choices[i].message.content for i in range(n_outputs)]
        return result

    def refresh(self, system_prompt: str = "You are a helpful assistant."):
        self.context = [{"role": "system", "content": system_prompt}]
        return True

    def get_history(self):
        return self.context.copy()


class GLM(LargeLanguageModel):
    def __init__(
        self,
        name: str = "ChatGLM",
        description: str = "The ChatGLM assistant.",
        api_key: str = None,
        model: str = "glm-3-turbo",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
    ):
        super(GLM, self).__init__(name, description, temperature)
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.context.append({"role": "system", "content": system_prompt})

    def listen_and_response(
        self, message: str = "How to cook Chinese stir-fried eggs and tomatoes?"
    ):
        self.context.append({"role": "user", "content": message})

        response_ok = False
        sleep_time = 0.1
        time.sleep(sleep_time)

        response = self.client.chat.completions.create(
            model=self.model, messages=self.context, temperature=self.temperature
        )

        self.context.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        result = response.choices[0].message.content
        return result

    def refresh(self, system_prompt: str = "You are a helpful assistant."):
        self.context = [{"role": "system", "content": system_prompt}]
        return True

    def get_history(self):
        return self.context.copy()


class Gemini(LargeLanguageModel):
    def __init__(
        self,
        name: str = "",
        description: str = "",
        api_key: str = None,
        model: str = "gemini-1.0-pro",
        temperature: float = 0.5,
    ):
        super(Gemini, self).__init__(name, description, temperature)
        self.model = model
        self.temperature = temperature
        genai.configure(api_key=api_key)
        self.generationConfig = genai.GenerationConfig(temperature=self.temperature)
        self.gemini = genai.GenerativeModel(self.model)

    def listen_and_response(self, message: str):
        """
        Please notice that the history-based chat is still not implemented in this method.
        """
        sleep_time = 0.25
        time.sleep(sleep_time)
        response = self.gemini.generate_content(
            message, generation_config=self.generationConfig
        )
        return response.text

    def refresh(self):
        pass

    def get_history(self):
        pass
