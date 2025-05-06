<div align="center">

# FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes

![Question Answering](https://img.shields.io/badge/Task-Question_Answering-red) 
![MCQA](https://img.shields.io/badge/Task-Multi--Choice--QA-red) 
![Industrial](https://img.shields.io/badge/Domain-Industrial--Assets-red) 
![OpenAI](https://img.shields.io/badge/Model-Llama-21C2A4)
![Llama](https://img.shields.io/badge/Model-Llama-21C2A4)
![Mistral](https://img.shields.io/badge/Model-Mistral-21C2A4) 
![Granite](https://img.shields.io/badge/Model-Granite-21C2A4)
![Qwen](https://img.shields.io/badge/Model-Qwen-21C2A4)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek-21C2A4)

📰 [Paper](https://arxiv.org/abs/), 🤗 [Leaderboard](https://huggingface.co/spaces/cc4718/FailureSensorIQ)

</div>

## 1. Introduction
As industries increasingly adopt autonomous AI agents, the need for models that can not only recall facts but also demonstrate a deep understanding of operational contexts—such as sensor relevance, fault prediction, and diagnostic reasoning—is paramount. Unlike traditional QA datasets, our dataset focuses on multiple aspects of reasoning through failure modes, sensor data, and the relationships between them across various industrial
assets. Failure modes, rooted in the theoretical framework of reliability engineering, represent potential points of failure within an asset or system. In contrast, sensors are physical manifestations that collect real-time data from operational systems. By combining these two concepts, our proposed dataset offers an opportunity to assess an LLM’s ability to reason across both theoretical and real-world domains, providing insights into their capacity to understand complex industrial processes. 


## 2. Dataset
We introduce FailureSensorIQ, a multiple-choice QA dataset that explores the relationships between sensors and failure modes for 10 industrial assets. By only leveraging the information found in ISO documents, we developed a data generation pipeline that creates questions in two formats: (i) row-centric (FM2Sensor) and (ii) column-centric (Sensor2FM). Additionally, we designed questions
in a selection vs. elimination format, taking advantage of the fact that the absence of an ✓ in a cell (as shown in Table 1) indicates irrelevant information. The FailureSensorIQ dataset consists of 8,296
questions across 10 assets, with 2,667 single-correct-answer questions and 5,629 multi-correct-answer questions.

### Asset Distribution: 
Electric Motor (234), Steam Turbine (171), Aero Gas Turbine (336), Industrial Gas Turbine (240), Pump (152), Compressor (220), Reciprocating IC Engine (336), Electric Generator (234), Fan (200), Power Transformer (544)

### Option Distribution: 
Option A: 752, Option B: 729, Option C: 491, Option D: 408, Option E: 208

### Distribution of Questions
2-options: 487, 3-options: 266, 4-options: 389, 5-options: 1525

## 3. PertEval
Recent literature highlights concerns about LLMs’ ability to reliably select the correct answer in multiple-choice questions, raising the question of whether models select an answer first and then generate reasoning or reason before choosing. The tendency to favor specific options introduces biases that vary across models and are hard to quantify. To address these challenges, we evaluate model performance on both the original (ST-MCQA) and a perturbed dataset, which underwent rigorous modifications. We adopted the [PertEval toolkit](https://github.com/aigc-apps/PertEval) enabled us to create a copy of the perturbed dataset. We developed two versions of the perturbed dataset: (i) SimplePert, which modifies the formatting of the questions by reordering the options, adding a right parenthesis to each option, and changing the option labels from A, B, C, etc., to P, Q, R, and so on. (ii) ComplexPert, apply all the question permutation as well as use LLM (llama-3-70b in this case) to change the questions also.

## 4. Uncertainty Quantification
We adopt the [LLM Uncertainty Bench framework](https://github.com/smartyfh/LLM-Uncertainty-Bench) to assess model uncertainty in multi-choice question answering. Each LLM is prompted with their Base Prompting method to output prediction probabilities for all answer options. To calibrate uncertainty estimates, we partition the dataset by asset type into a calibration set and a test set. Using the calibration set, we compute conformal scores that define a confidence threshold ˆq. For the test set, any answer option with a probability exceeding ˆq is selected as a prediction. This approach allows a variable number of predicted options per question, ranging from zero to all available choices.
