#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import sys


# In[3]:


model_id = 'mistralai/mistral-large'
use_tools = True
if sys.argv[1] != '-f':
    model_id = sys.argv[1]
    use_tools = eval(sys.argv[2])


# In[4]:


model_id


# In[5]:


import os
from mcqa_tools import fm2s, s2fm
import json


# In[6]:


from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from langchain_ibm.chat_models import convert_to_openai_tool


# In[7]:


load_dotenv()


# In[8]:


ds = load_dataset('cc4718/FailureSensorIQ', data_files='all.jsonl')['train']
ds = ds.filter(lambda x: x['asset_name'] in ['industrial gas turbine', 'electric motor'])


# In[9]:


params = {
    "temperature": 0.0,
    "max_tokens": 2000,
}
names_to_functions = {
    "fm2s": fm2s,
    "s2fm": s2fm,
}
tools = [convert_to_openai_tool(tool) for tool in names_to_functions.values()]

credentials = Credentials(
    url=os.environ['WATSONX_URL'],
    api_key=os.environ['WATSONX_APIKEY']
)

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=os.environ['WATSONX_PROJECT_ID'],
    params=params
)
json_format = '''```json
{"reasoning": "<your reasoning>", "answer": ["<answer letter>"]}
```'''
kwargs = {} if not use_tools else {'tools': tools, 'tool_choice': 'auto'}


# In[ ]:


n_correct = 0
n_invalid = 0
for item in tqdm(ds):
    prompt = item['question'] + '\n'
    for c, opt in zip(item['option_ids'], item['options']):
        prompt += c + '. ' + opt + '\n'
    messages = [
        {
            "role": "user", 
            "content": prompt if use_tools else prompt + f'\n{json_format}'
        }
    ]
    response = model.chat(messages, **kwargs)
    if use_tools:
        try:
            tool_call = response["choices"][0]["message"]["tool_calls"]
        except:
            n_invalid += 1
            continue
        function_name = tool_call[0]["function"]["name"]
        function_params = json.loads(tool_call[0]["function"]["arguments"])
        # print(f"Executing function: `{function_name}`, with parameters: {function_params}")
        function_result = names_to_functions[function_name](**function_params)
        ans = item['options'].index(function_result) if function_result in item['options'] else -1
        n_correct += int(ans == item['correct'].index(True))
        if ans == -1:
            n_invalid += 1
    else:
        raw = response['choices'][0]['message']['content']
        start_idx = raw.find('{')
        end_idx = raw.find('}') + 1
        try:
            cleaned = json.loads(raw[start_idx:end_idx])
            pred_letter = cleaned['answer'][0]
            ans = ord(pred_letter) - ord('A')
            n_correct += int(ans == item['correct'].index(True))    
        except:
            n_invalid += 1
acc = round(n_correct / (len(ds) - n_invalid) * 100, 2)
print(acc)
print(round(n_invalid / len(ds) * 100, 2))


# In[ ]:




