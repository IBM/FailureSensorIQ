<div align="center">

# Industrial AI Benchmark: MCQA Dataset and Specialized LLMs for Domain-Aware Reasoning

![Question Answering](https://img.shields.io/badge/Task-Question_Answering-red) 
![MCQA](https://img.shields.io/badge/Task-Multi--Choice--QA-red) 
![Industrial](https://img.shields.io/badge/Domain-Industrial--Assets-red)

![EMNLP 2025](https://img.shields.io/badge/EMNLP--2025-Accepted-blueviolet)
![NeurIPS 2025](https://img.shields.io/badge/NeurIPS--2025-Accepted-blueviolet)
![IBM Research](https://img.shields.io/badge/Org-IBM--Research-black)

📰 [Paper](https://arxiv.org/abs/2506.03278), 🤗 [Dataset](https://huggingface.co/datasets/ibm-research/FailureSensorIQ), 🪧[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/121805.png?t=1762411263.6030943), 📹 [Video](https://recorder-v3.slideslive.com/#/share?share=104505&s=d208150a-78d8-4139-94e4-f8b4c88d2456)

</div>


## 🚀 Highlights

- **Dataset – FailureSensorIQ**  
  A curated dataset of 8,296 multiple-choice QA on sensor-failure reasoning, hosted on [HuggingFace](https://huggingface.co/datasets/ibm-research/FailureSensorIQ).

- **Leaderboard**  
  Comparative results of 27 frontier and open-source LLMs evaluated on FailureSensorIQ, hosted on [HuggingFace](https://huggingface.co/spaces/cc4718/FailureSensorIQ).

- **Extended Research I – Knowledge Distillation & Fine-Tuning**  
  Distilling reasoning capabilities from LLMs into smaller SLMs and assessing performance on FailureSensorIQ. [[Code](https://github.com/IBM/FailureSensorIQ/tree/main/fine_tune)] [[Paper](https://arxiv.org/abs/2510.18817)] [[Poster](https://github.com/IBM/FailureSensorIQ/blob/main/resources/Fine-Tuned-Thoughts-poster.pdf)]

- **Extended Research II – Embedding Models**  
  Developing generalized embeddings for Industry 4.0 applications and benchmarking on FailureSensorIQ. [[Code](https://github.com/IBM/FailureSensorIQ/tree/main/LLM-Embeddings)] [[Paper](https://arxiv.org/abs/2506.12607)] [[Poster](https://github.com/IBM/FailureSensorIQ/blob/50c46106a50a83642ac1da453e5086afad15a214/resources/Embeddings-poster.pdf)]

## 🆕 What's New?
- [9/25/2025] Embedding paper has been accepted in EMNLP 2025 Industry Track!
- [9/18/2025] FailureSensorIQ has been accepted in NeurIPS 2025 Datasets & Benchmarks Track!
- [8/20/2025] Fine-Tuning paper has been accepted to EMNLP 2025 Findings!


## 1. Dataset
We introduce [FailureSensorIQ](https://huggingface.co/datasets/ibm-research/FailureSensorIQ), a Multi-Choice QA (MCQA) dataset that explores the relationships between sensors and failure modes for 10 industrial assets. By only leveraging the information found in ISO documents, we developed a data generation pipeline that creates questions of two types: 
- Failure Mode to Sensors (FM2Sensor): what sensor should be monitored to detect a failure early?
- Sensor to Failure Mode (Sensor2FM): predict failures when anormal sensor behavior is detected.

The FailureSensorIQ dataset consists of 8,296
questions across 10 assets, with 2,667 single-correct-answer questions and 5,629 multi-correct-answer questions.

## 2. Evaluation: Perturbation-Uncertainty-Complexity analysis
We evaluate 27 frontier and open-source LLMs through our Perturbation–Uncertainty-Complexity evaluation pipeline, which systematically measures model robustness to question reformulation, model confidence, and ability to handle increasing distractor options, respectively.

**Perturbation**. We adopted the [PertEval toolkit](https://github.com/aigc-apps/PertEval) enabled us to create a copy of the perturbed dataset. We developed two versions of the perturbed dataset: (i) _SimplePert_, which modifies the formatting of the questions by reordering the options, adding a right parenthesis to each option, and changing the option labels from A, B, C, etc., to P, Q, R, and so on. (ii) _ComplexPert_, apply all the question permutation as well as use LLM (llama-3-70b in this case) to change the questions also.

**Uncertainty**. We adopted the [LLM Uncertainty Bench framework](https://github.com/smartyfh/LLM-Uncertainty-Bench) to assess model uncertainty in Multi-Choice Question Answering. Each LLM is prompted with their Base Prompting method to output prediction probabilities for all answer options. To calibrate uncertainty estimates, we partition the dataset by asset type into a calibration set and a test set. Using the calibration set, we compute conformal scores that define a confidence threshold ˆq. For the test set, any answer option with a probability exceeding ˆq is selected as a prediction. 

**Complexity**. We created _OptionsPert_, which extend each question in single-correct-answer MCQA to have 10 options. New choices are all distractors and we randomized the options. The
purpose of this dataset is to reduce the likelihood of random guessing and enable systematic evaluation of model robustness under increased ambiguity.

## 3. LLMFeatureSelect and Kaggle experiments
`kaggle` folder contains the 3 dataset experiments we performed to evaluate the model's feature suggestions. It also contains `LLMFeatureSelector`, an sklearn pipeline for feature selection which uses and supports Hugging Face models. 

## 4. Loading the dataset from HuggingFace 🤗
To load `single_true_multi_choice_qa` and `multi_true_multi_choice_qa` subsets from HuggingFace,
```
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds_sc = load_dataset("ibm-research/FailureSensorIQ", "single_true_multi_choice_qa")
ds_mc = load_dataset("ibm-research/FailureSensorIQ", "multi_true_multi_choice_qa")
```
To load the _SimplePert_, _ComplexPert_, _OptionsPert_ of the dataset, check out [load_dataset.ipynb](https://github.com/IBM/FailureSensorIQ/blob/main/load_dataset.ipynb).

## 5. Install and run our evaluation pipeline
Tested with `python 3.10.4`  
Clone repo and submodules, create a conda env
```
git clone --recurse-submodules https://github.com/IBM/FailureSensorIQ.git
cd FailureSensorIQ
# Optional
conda create -n failuresensoriq python=3.10.4
conda activate failuresensoriq
```
Install requirements
```
pip install vllm==0.8.5.post1
pip install -r requirements.txt
```
Run evaluation pipeline
```
python run_eval.py <hf-model-id> full
```
`full` refers to evaluating on the full dataset. You can first try `sample` instead to try on a few samples of the dataset and make sure that everything runs as intended before running on the full dataset.
If no argument is given, the code will fetch all the pending models for evaluation from huggingface and run them on the full dataset.  

If everything ran successfully you should be able to see the performance metrics under `results/demo-leaderboard/gpt2-demo/results_<model-name>.json`

## 6. Hardware and troubleshooting
For running the evaluation pipeline we tested this on an A100 80GB. The hardware requirements depend on what model you choose to evaluate.
```all CUDA-capable devices are busy or unavailable```
Sometimes if the execution crashes/interrupts before it finishes, the vllm child process may be still running occupying memory in the GPU. We use the following to kill any jobs occupying GPU memory:
```
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
```
## 7. Citation
If you use our dataset in your paper, please cite our dataset by
```
@inproceedings{
constantinides2025failuresensoriq,
title={FailureSensor{IQ}: A Multi-Choice {QA} Dataset for Understanding Sensor Relationships and Failure Modes},
author={Christodoulos Constantinides and Dhaval C Patel and Shuxin Lin and Claudio Guerrero and SUNIL DAGAJIRAO PATIL and Jayant Kalagnanam},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=9KfkMAy2ut}
}
```
