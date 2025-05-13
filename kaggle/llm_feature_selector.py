import numpy as np
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.base import _fit_context
from sklearn.utils.validation import validate_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from accelerate import Accelerator

class LLMFeatureSelector:
    def __init__(self, model_name, feature_names, target_variable, prompt_template=None, topk=None):
        self.model_name = model_name
        self.feature_names = feature_names
        self.prompt_template = prompt_template
        self.target_variable = target_variable
        self.topk = topk
        if not self.prompt_template:
            self.prompt_template = 'Select the variables from the list that are most relevant for predicting <target_variable>. ' +\
                  'Provide the variables sorted starting with the one with the highest priority. ' +\
                  'All variables: <all_variables>\n' + \
                  '```json\n{"selected_variables": ["variable 1", "variable 2", ..., "variable n"]}\n```'
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = transformers.GenerationConfig(max_length=512, stop_strings=['}'])
        
    
    def fit(self, X, y=None):
        """Use an LLM given feature names and optionally descriptions to select variables.
    
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Data from which to compute variances, where `n_samples` is
                the number of samples and `n_features` is the number of features.
    
            y : any, default=None
                Ignored. This parameter exists only for compatibility with
                sklearn.pipeline.Pipeline.
    
            Returns
            -------
            self : object
                Returns the instance itself.
        """
        X = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_all_finite="allow-nan",
        )
        prompt = self.prompt_template.replace('<all_variables>', ', '.join(self.feature_names))
        prompt = prompt.replace('<target_variable>', self.target_variable)
        # generate response
        messages = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': '{"selected_variables": ["'}
        ]
        accelerator = Accelerator()
        tokens = self.tokenizer.apply_chat_template(messages, continue_final_message=True, return_tensors='pt').to(accelerator.device)
        output = self.model.generate(tokens, generation_config=self.generation_config, tokenizer=self.tokenizer)
        output = self.tokenizer.decode(output[0][tokens.shape[1]:], skip_special_tokens=True)
        output = '{"selected_variables": ["' + output
        print(output)
        try:
            output = eval(output)
        except Exception as e:
            print(e)
            return X
        valid_idxs = []
        for col in output['selected_variables'][:self.topk]:
            try:
                idx = self.feature_names.index(col)
                valid_idxs.append(idx)
            except:
                pass
        return X[valid_idxs]
        
    def transform(self):
        pass