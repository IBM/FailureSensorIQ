# Towards Building General Purpose Embedding Models for Industry 4.0 Agents - EMNLP 2025 Industry Track

## Preprocessing
### Create entity descriptions using LLM
`preprocess/create_descriptions.ipynb`
### Preprocess all tasks
`preprocess/preprocess_all_tasks.ipynb` contains the script to preprocess all tasks into a uniform format

## Training
Training is based on `sentence-transformers` library  
`training/multi_task_embedding.ipynb`  
If you want to compare it with the default in batch negative loss:
`training/multi_task_inbatch_neg.ipynb`  

## Evaluation
To evaluate bm25: `evaluation/bm25.ipynb`


## Ablation Study
1. Prepare all data for ablation:  ```./preprocess_tasks_ablation.sh```  
2. Train all for ablation of varying probability of LLM augmented description:  ```./train_bert_varyingseed.sh```  
3. Train all for ablation of varying probability of LLM augmented description for in batch negatives loss:  ```./train_inbatch.sh```  
4. Train all for ablation of varying ratio of in-batch positives to negatives:  ```./train_varying_positive_ratio.sh```  
5. Visualize results:  ```evaluation/eval_emb_models_ablation.ipynb```  
There are also logs in wandb.

## ReAct Agent combined with LLM embedding
all the code is in `emdReact` folder
