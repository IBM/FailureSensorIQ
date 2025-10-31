<div align="center">

# FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes

![Question Answering](https://img.shields.io/badge/Task-Question_Answering-red) 
![MCQA](https://img.shields.io/badge/Task-Multi--Choice--QA-red) 
![Industrial](https://img.shields.io/badge/Domain-Industrial--Assets-red) 
![OpenAI](https://img.shields.io/badge/Model-OpenAI-21C2A4)
![Llama](https://img.shields.io/badge/Model-Llama-21C2A4)
![Gemma](https://img.shields.io/badge/Model-Gemma-21C2A4)
![Phi](https://img.shields.io/badge/Model-Phi-21C2A4)
![Mistral](https://img.shields.io/badge/Model-Mistral-21C2A4) 
![Granite](https://img.shields.io/badge/Model-Granite-21C2A4)
![Qwen](https://img.shields.io/badge/Model-Qwen-21C2A4)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek-21C2A4)

📰 [Paper](https://arxiv.org/abs/2506.03278), 🤗 [Leaderboard](https://huggingface.co/spaces/cc4718/FailureSensorIQ)

</div>


## 🚀 Highlights

- **Dataset – FailureSensorIQ**  
  A curated dataset of 8,296 multiple-choice QA on sensor-failure reasoning, hosted on [HuggingFace](https://huggingface.co/datasets/ibm-research/FailureSensorIQ).

- **Leaderboard**  
  Comparative results of 27 frontier and open-source LLMs evaluated on FailureSensorIQ, hosted on [HuggingFace](https://huggingface.co/spaces/cc4718/FailureSensorIQ).

- **Extended Research I – Knowledge Distillation & Fine-Tuning**  
  Distilling reasoning capabilities from LLMs into smaller SLMs and assessing performance on FailureSensorIQ. [[Code](https://github.com/IBM/FailureSensorIQ/tree/main/fine_tune)] [[Paper](https://arxiv.org/abs/2510.18817)] 

- **Extended Research II – Embedding Models**  
  Developing generalized embeddings for Industry 4.0 applications and benchmarking on FailureSensorIQ. [[Code](https://github.com/IBM/FailureSensorIQ/tree/main/LLM-Embeddings)] [[Paper](https://arxiv.org/abs/2506.12607)] 

## 🆕 What's New?
- [9/25/2025] Embedding paper has been accepted in EMNLP 2025 Industry Track!
- [9/18/2025] FailureSensorIQ has been accepted in NeurIPS 2025 Datasets & Benchmarks Track!
- [8/20/2025] Fine-Tuning paper has been accepted to EMNLP 2025 Findings!

## 1. Introduction
As industries increasingly adopt autonomous AI agents, the need for models that can not only recall facts but also demonstrate a deep understanding of operational contexts—such as sensor relevance, fault prediction, and diagnostic reasoning—is paramount. Unlike traditional QA datasets, our dataset focuses on multiple aspects of reasoning through failure modes, sensor data, and the relationships between them across various industrial
assets. Failure modes, rooted in the theoretical framework of reliability engineering, represent potential points of failure within an asset or system. In contrast, sensors are physical manifestations that collect real-time data from operational systems. By combining these two concepts, our proposed dataset offers an opportunity to assess an LLM’s ability to reason across both theoretical and real-world domains, providing insights into their capacity to understand complex industrial processes. 

## 2. Dataset
We introduce FailureSensorIQ, a Multi-Choice QA (MCQA) dataset that explores the relationships between sensors and failure modes for 10 industrial assets. By only leveraging the information found in ISO documents, we developed a data generation pipeline that creates questions in two formats: (i) row-centric (FM2Sensor) and (ii) column-centric (Sensor2FM). Additionally, we designed questions
in a selection vs. elimination format, taking advantage of the fact that the absence of an ✓ in a cell (as shown in Table 1) indicates irrelevant information. The FailureSensorIQ dataset consists of 8,296
questions across 10 assets, with 2,667 single-correct-answer questions and 5,629 multi-correct-answer questions.

### Asset Distribution: 
Electric Motor (234), Steam Turbine (171), Aero Gas Turbine (336), Industrial Gas Turbine (240), Pump (152), Compressor (220), Reciprocating IC Engine (336), Electric Generator (234), Fan (200), Power Transformer (544)

### Option Distribution: 
Option A: 752, Option B: 729, Option C: 491, Option D: 408, Option E: 208

### Distribution of Questions
2-options: 487, 3-options: 266, 4-options: 389, 5-options: 1525

### Eval and test set
We release the 2667 SC-MCQA and 5629 MC-MCQA dataset and the pipeline to update the scores on Hugging Face leaderboard. We also release another 50 expert curated MCQA for different assets as a validation set to quickly be able to run the pipeline and get some results.

## 3. PertEval
Recent literature highlights concerns about LLMs’ ability to reliably select the correct answer in multiple-choice questions, raising the question of whether models select an answer first and then generate reasoning or reason before choosing. The tendency to favor specific options introduces biases that vary across models and are hard to quantify. To address these challenges, we evaluate model performance on both the original (ST-MCQA) and a perturbed dataset, which underwent rigorous modifications. We adopted the [PertEval toolkit](https://github.com/aigc-apps/PertEval) enabled us to create a copy of the perturbed dataset. We developed two versions of the perturbed dataset: (i) SimplePert, which modifies the formatting of the questions by reordering the options, adding a right parenthesis to each option, and changing the option labels from A, B, C, etc., to P, Q, R, and so on. (ii) ComplexPert, apply all the question permutation as well as use LLM (llama-3-70b in this case) to change the questions also.

## 4. Uncertainty Quantification
We adopt the [LLM Uncertainty Bench framework](https://github.com/smartyfh/LLM-Uncertainty-Bench) to assess model uncertainty in Multi-Choice Question Answering. Each LLM is prompted with their Base Prompting method to output prediction probabilities for all answer options. To calibrate uncertainty estimates, we partition the dataset by asset type into a calibration set and a test set. Using the calibration set, we compute conformal scores that define a confidence threshold ˆq. For the test set, any answer option with a probability exceeding ˆq is selected as a prediction. This approach allows a variable number of predicted options per question, ranging from zero to all available choices.


## LLMFeatureSelect and Kaggle experiments
`kaggle` folder contains the 3 dataset experiments we performed to evaluate the model's feature suggestions. It also contains `LLMFeatureSelector`, an sklearn pipeline for feature selection which uses and supports Hugging Face models. 

## Loading the dataset from Hugging Face 🤗
```
from datasets import load_dataset
load_dataset('ibm-research/FailureSensorIQ', data_files='failuresensoriq_standard/all.jsonl')
```
For loading the perturbed or an extra sample of the dataset check out `load_dataset.ipynb`  

## Installing and running our evaluation pipeline
Tested with `python 3.10.4`  
Clone repo and submodules
```
git clone --recurse-submodules https://github.com/IBM/FailureSensorIQ.git
cd FailureSensorIQ
```
Optional: create a new conda env
```
conda create -n failuresensoriq python=3.10.4
conda activate failuresensoriq
```
Install requirements
```
pip install vllm==0.8.5.post1
pip install -r requirements.txt
```
```
python run_eval.py <hf-model-id> full
```
`full` refers to evaluating on the full dataset. You can first try `sample` instead to try on a few samples of the dataset and make sure that everything runs as intended before running on the full dataset.
If no argument is given, the code will fetch all the pending models for evaluation from huggingface and run them on the full dataset.  

If everything ran successfully you should be able to see the performance metrics under `results/demo-leaderboard/gpt2-demo/results_<model-name>.json`

## Troubleshooting
```all CUDA-capable devices are busy or unavailable```
Sometimes if the execution crashes/interrupts before it finishes, the vllm child process may be still running occupying memory in the GPU. We use the following to kill any jobs occupying GPU memory:
```
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
```

## Hardware  
For running the evaluation pipeline we tested this on an A100 80GB. The hardware requirements depend on what model you choose to evaluate.
