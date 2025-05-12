This contains the 3 kaggle datasets where we run the experiments. We curated the datasets, prepared the prompt 

## Electrical Transformer Predictive Maintenance
This [dataset](https://www.kaggle.com/datasets/sreshta140/ai-transformer-monitoring/data) consists of 48 timeseries variables collected from IoT devices placed in an electrical transformer from June 25th, 2019 to April 14th, 2020 which was updated every 15 minutes. We focused in predicting magnetic oil gauge faults.

## Air Compressor Predictive Maintenance
The Air Compressor [dataset](https://www.kaggle.com/datasets/afumetto/predictive-maintenance-dataset-air-compressor/code) consists of 19 sensor variables and 5 different failure modes: bearing, water pump, outlet valve, motor, and radiator failure indicators. For each failure mode, we provide the top-5 recommendations along with their correlations. We omit motor failure, because the dataset doesn’t include any failure instance.

## Wind Mill Power Production Forecasting
The aim of this [dataset](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting) is to predict the wind power that could be generated from the windmill for the next 15 days across 20 sensor variables.


## How to use SKLearn Feature Selector
```
from llm_feature_selector import LLMFeatureSelector
feat_sel = LLMFeatureSelector(model_name=model_name, # HF model
                              feature_names=all_cols, # all available feature names
                              target_variable='magnetic oil gauge fault', # target variable name/description
                              prompt_template=prompt_template, # optional prompt template, can be custom and add variable descriptions.
                              topk=3 # optional if you want to return topk results, otherwise set to None
                             )
output = feat_sel.fit(df)
```
Currently, it only supports Hugging Face compatible models. Example usage in transformer/transformer.ipynb 