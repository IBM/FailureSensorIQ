import ast
import json
from typing import Any, Dict, List, Optional, Callable, Tuple

import pandas as pd
from langchain.globals import set_verbose
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
)
from industrialqa_fmsr.baseline.function.self_guess.self_guess_prompting import \
    generate_final_answer as self_guess_generate_final_answer
from langchain_core.runnables.retry import RunnableRetry
import random
import re
from collections import Counter
import uuid

set_verbose(True)


MAX_RETRIES = 5

rits_models = [
    "rits/mistralai/mistral-large-instruct-2407",
    # "rits/meta-llama/Llama-3.2-90B-Vision-Instruct",
    "rits/meta-llama/llama-3-3-70b-instruct",
    "rits/meta-llama/llama-3-1-70b-instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/llama-3-1-405b-instruct-fp8",
]

litellm_models = [
    "watsonx/mistralai/mistral-large",
    "watsonx/meta-llama/llama-3-405b-instruct",
    "watsonx/meta-llama/llama-3-1-70b-instruct",
    "watsonx/meta-llama/llama-3-70b-instruct",
    "watsonx/mistralai/mixtral-8x7b-instruct-v01",
    "watsonx/ibm/granite-3-8b-instruct",
    "watsonx/ibm/granite-13b-chat-v2",
]

llm_wrapper = 'rits'

if llm_wrapper == 'lite_llm':
    voters = litellm_models
elif llm_wrapper == 'rits':
    voters = rits_models


class ProblemGenerator:
    def _create_question(
            self,
            inputs: Dict[str, Any],
            num_batches: int,
            question_paraphrase: bool,
            random_seed: int,
            elimination: bool
    ):
        print('_create_question')
        random.seed(random_seed)
        if elimination:
            question_candidates = self.elimination_question_candidates
            if not question_paraphrase:
                question_candidates = [self.elimination_question_candidates[0]]
        else:
            question_candidates = self.selection_question_candidates
            if not question_paraphrase:
                question_candidates = [self.selection_question_candidates[0]]

        questions = [random.choice(question_candidates).format(**inputs) for _ in
                     range(num_batches)]

        return questions

    def _create_choices(
            self,
            inputs: Dict[str, Any],
            num_batches: int,
            choice_candidates: List[str],
            num_choices: int,
            random_seed: int,
            llm: str = None,
            elimination: bool = False
    ):
        # PICK UP TOP N RELEVANT SENSORS FOR EACH ASSET CLASS
        print('_create_choices')
        print(inputs)
        random.seed(random_seed)
        relevant_criteria = self.criteria[0].format(**inputs)
        irrelevant_criteria = self.criteria[1].format(**inputs)
        relevant_sensor_candidates, irrelevant_sensor_candidates = self._group_choices(
            criteria=(relevant_criteria, irrelevant_criteria), choices=choice_candidates, llm=llm)
        choices = []
        for i in range(num_batches):
            if elimination:
                correct_choice = irrelevant_sensor_candidates[i % len(irrelevant_sensor_candidates)]
                incorrect_choices = random.sample(relevant_sensor_candidates, num_choices - 1)
            else:
                correct_choice = relevant_sensor_candidates[i % len(relevant_sensor_candidates)]
                incorrect_choices = random.sample(irrelevant_sensor_candidates, num_choices - 1)
            all_choices = [correct_choice] + incorrect_choices
            random.shuffle(all_choices)
            choices.append(all_choices)
        return choices

    def _create_answer(
            self,
            prompt: str,
            num_voters: int = 3,
            agreement_threshold: float = 0.5,
            confidence_threshold: float = 80,
            unique_id: str = None,
    ):
        selected_voters = voters[:num_voters]
        votes = []
        for selected_voter in selected_voters:
            retry = 0
            while retry < MAX_RETRIES:
                try:
                    print('---')
                    print('unique_id', unique_id)
                    print(selected_voter)
                    print(prompt)
                    response = self_guess_generate_final_answer(question_prompt=prompt, model_name=selected_voter,
                                                                params={"temperature": 0}, few_shot=True)
                    print(json.dumps(response, indent=4))
                    assert 'answers' in response and len(response['answers']) > 1 \
                        and 'confidence_score' in response['answers'][-1] and 'generated_answer' in response['answers'][-1] \
                        and response['answers'][-1]['generated_answer'] in [chr(i) for i in range(65, 91)]
                    votes.append(response['answers'][-1])
                    print('---')
                    break
                except Exception as e:
                    
                    print(str(e) + ' ...Retrying...')

                    retry += 1

        with open(f"./tmp/{unique_id}.json", "w") as f:
            json.dump(votes, f, indent=4)

        generations = [(vote['generated_answer'], vote['confidence_score']) for vote in votes]
        generations = sorted(generations, key=lambda x: x[0])
        letter2confidence_score = {}
        for key, value in generations:
            letter2confidence_score.setdefault(key, []).append(value)

        most_agreed_letter = max(letter2confidence_score, key=lambda k: len(letter2confidence_score[k]))
        agreement_reached = len(
            letter2confidence_score[most_agreed_letter]) / num_voters >= agreement_threshold and all(
            x >= confidence_threshold for x in letter2confidence_score[most_agreed_letter])
        print("agreement_reached", agreement_reached)
        print("most_agreed_letter", most_agreed_letter)
        print("letter2confidence_score", letter2confidence_score)
        return agreement_reached, most_agreed_letter, letter2confidence_score

    def _create_prompt(
            self,
            inputs: Dict[str, Any],
            choice_candidates: List[str],
            num_batches: int,
            num_choices: int,
            question_paraphrase: bool,
            random_seed: int,
            llm: str = None,
            elimination: bool = False

    ):
        self._validate_question_keys(inputs)
        questions = self._create_question(
            inputs=inputs,
            num_batches=num_batches,
            question_paraphrase=question_paraphrase,
            random_seed=random_seed,
            elimination=elimination
        )
        choices = self._create_choices(
            inputs=inputs,
            num_batches=num_batches,
            choice_candidates=choice_candidates,
            num_choices=num_choices,
            random_seed=random_seed,
            llm=llm,
            elimination=elimination
        )
        prompts = self._generate_prompts_from_questions_and_choices(
            questions=questions,
            choices=choices,
        )

        return prompts

    def _create_reasoning(
            self,
            problem: Dict[str, Any],
            reasoning_llm: str = None,
            reasoning_prompting: Callable = self_guess_generate_final_answer,
    ):
        prompt = problem['prompt']
        expected_letter = problem['answer']
        unique_id = problem['unique_id']
        retry = 0
        temperature = 0

        final_resp = {"prompt": prompt, "answer": "", 'unique_id': unique_id,
                                  'letter': expected_letter}
        return final_resp
    
        while retry < MAX_RETRIES:
            try:
                response = reasoning_prompting(question_prompt=prompt, model_name=reasoning_llm,
                                               params={"temperature": temperature}, few_shot=True)
                resp = response['answers'][-1]
                if resp['generated_answer'] == expected_letter:
                    json_obj = json.loads(resp['answer'])
                    explanation = json_obj['explanation']
                    letter_with_choice = json_obj['answer']
                    rationale = json_obj['rationale']

                    final_answer = f"""{explanation}\n{rationale}\nSo the final answer is {letter_with_choice}."""
                    final_resp = {"prompt": prompt, "answer": final_answer, 'unique_id': unique_id,
                                  'letter': expected_letter, 'letter_with_choice': letter_with_choice}
                    return final_resp
                else:
                    temperature += 0.1
                    retry += 1
            except Exception as ex:
                print("!! Error catched here\n" + str(ex))
                retry += 1

        return None

    def generate(self, inputs: Dict[str, Any]):
        pass

    def _validate_inputs(self, inputs: Dict[str, Any]):
        for key in self.input_keys:
            if key not in inputs.keys():
                raise ValueError(f"Validating inputs: Input key {key} is missing.")

    def _validate_question_keys(self, inputs: Dict[str, Any]):
        for key in self.question_static_keys:
            if key not in inputs.keys():
                raise ValueError(f"Validating question keys: Input key {key} is missing.")

    @staticmethod
    def _group_choices(criteria: Tuple[str, str], choices: List[str], llm: str = None):
        retry = 0
        choices = str(choices)
        question = f"""Divide the following choices into two groups. First group is {criteria[0]}. Second group is {criteria[1]}. Output the first group in the first line.Output the second group in the second line.
Here are a list of choices: {choices}
Output the first group in the first line. Output the second group in the second line. Format of the output should be:
First group: ["choice1", "choice2", "choice3", ...]
Second group: ["choice4", "choice5", "choice6", ...]
"""
        print("group_choices question:", question)
        while retry < MAX_RETRIES:
            try:
                if llm_wrapper == 'rits':
                    from industrialqa_fmsr.wrapper.rits_llm import get_chat_response as get_response
                    print('group choice llm:', llm)
                    response = get_response(messages=question, model_id=llm)
                elif llm_wrapper == 'lite_llm':
                    from industrialqa_fmsr.wrapper.lite_llm import get_response
                    
                    response = get_response(messages=question, model_id=llm)
                print("group_choices response:\n", response)
                lines = [line for line in response.splitlines() if line.strip()]
                assert len(lines) == 2
                relevant_list = lines[0]
                irrelevant_list = lines[1]
                match = re.search(r"\[(.*?)\]", relevant_list)
                relevant_list = [item.strip().strip("'").strip('"') for item in match.group(1).split(',')]
                match = re.search(r"\[(.*?)\]", irrelevant_list)
                irrelevant_list = [item.strip().strip("'").strip('"') for item in match.group(1).split(',')]
                print("relevant_list", relevant_list)
                print("irrelevant_list", irrelevant_list)
                return relevant_list, irrelevant_list
            except Exception as ex:
                print("Error caught here\n" + str(ex))
                retry += 1
        raise Exception("Failed to group choices!")

    @staticmethod
    def _generate_prompts_from_questions_and_choices(questions: List[str], choices: List[str]):
        prompts = []

        for question, choice_list in zip(questions, choices):
            choice_list_str = '\n'.join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choice_list)])

            new_prompt = f"Question: {question}\n{choice_list_str}\nAnswer:"
            prompts.append(new_prompt)

        return prompts


class SensorPG(ProblemGenerator):

    @property
    def input_keys(self) -> List[str]:
        return ["asset_class", "sensor_candidates"]

    @property
    def question_static_keys(self) -> List[str]:
        return ["asset_class"]

    @property
    def elimination_question_candidates(self) -> List[str]:
        return ["Which sensor could not be installed on this asset {asset_class}?",
                "Is there a sensor that cannot be mounted on this asset {asset_class}?",
                "Can you identify a sensor that could not work with this asset {asset_class}?",
                "Which sensor is not proper to track performance and identify anomalies for this asset {asset_class}?"]

    
    @property
    def selection_question_candidates(self) -> List[str]:
        return ["Which sensor could be installed on this asset {asset_class}?",
                "Is there a sensor that can be mounted on this asset {asset_class}?",
                "Can you identify a sensor that could work with this asset {asset_class}?",
                "Which sensor is recommended to track performance and identify anomalies for this asset {asset_class}?"]

    @property
    def criteria(self) -> List[str]:
        return ["the sensors that can be installed on asset {asset_class}",
                "the sensors that cannot be installed on asset {asset_class}"]

    def generate(
            self,
            inputs: Dict[str, Any],
            num_problems_per_batch: int = 10,
            question_paraphrase: bool = True,
            num_choices: int = 5,
            random_seed: int = 32,
            num_voters: int = 3,
            agreement_threshold: float = 0.5,
            llm: str = voters[0],
            reasoning_prompting: Callable = self_guess_generate_final_answer,
            elimination=True
    ):
        self._validate_inputs(inputs)
        asset_class = inputs[self.input_keys[0]]
        sensor_candidates = inputs[self.input_keys[1]]
        if isinstance(asset_class, str):
            asset_class = [asset_class]
        if not isinstance(asset_class, List):
            raise ValueError(f"asset_class should be a string or a list of strings. Input asset_class: {asset_class}")
        if not isinstance(sensor_candidates, List):
            raise ValueError(
                f"sensor_candidates should be a list of strings. Input sensor_candidates: {sensor_candidates}")
        if len(sensor_candidates) < num_choices:
            raise ValueError(
                f"sensor_candidates should have at least {num_choices} candidates. Input sensor_candidates: {sensor_candidates}")

        num_problems_per_asset_class = num_problems_per_batch
        total_problems_with_reasoning = []

        dir = "./elimination/sensor" if elimination else "./selection/sensor"

        for an_asset_class in asset_class:
            print('asset_class', an_asset_class)
            prompts: List = self._create_prompt(
                inputs={"asset_class": an_asset_class},
                choice_candidates=sensor_candidates,
                num_batches=num_problems_per_asset_class,
                num_choices=num_choices,
                question_paraphrase=question_paraphrase,
                random_seed=random_seed,
                llm=llm,
                elimination=elimination
            )
            problems = []
            for prompt in prompts:
                unique_id = str(uuid.uuid4()).replace('-', '')[:8]
                agreement_reached, most_agreed_letter, letter2confidence_score = self._create_answer(  # self guess
                    prompt=prompt,
                    num_voters=num_voters,
                    agreement_threshold=agreement_threshold,
                    unique_id=unique_id,
                )
                if agreement_reached:
                    problems.append({
                        'prompt': prompt,
                        'answer': most_agreed_letter,
                        'unique_id': unique_id
                    })

            problems_with_reasoning = []
            for problem in problems:
                problem_with_reasoning = self._create_reasoning(
                    problem=problem,
                    reasoning_llm=llm,
                    reasoning_prompting=reasoning_prompting
                )
                if problem_with_reasoning:
                    problems_with_reasoning.append(problem_with_reasoning)

                    with open(f"{dir}/{an_asset_class}.jsonl", "a") as f:
                        f.write(json.dumps(problem_with_reasoning) + "\n")

            total_problems_with_reasoning.extend(problems_with_reasoning)

            with open(f"{dir}/{an_asset_class}.json", "w") as f_out:
                json.dump(problems_with_reasoning, f_out, indent=4)

        return total_problems_with_reasoning


class FailureModePG(ProblemGenerator):
    @property
    def input_keys(self) -> List[str]:
        return ["asset_class", "failure_mode_candidates"]

    @property
    def question_static_keys(self) -> List[str]:
        return ["asset_class"]


    @property
    def elimination_question_candidates(self) -> List[str]:
        return ["Which is the least common failure mode associated with the asset {asset_class}?",
                "Which failure mode should not be monitored for the asset {asset_class}?",
                "Which failure mode cannot occur in the asset {asset_class} during operation?",
                "Which is the failure scenario that the asset {asset_class} never encounter?",
                "Which failure mode is most likely not to occur with the asset {asset_class}?"]

    
    @property
    def selection_question_candidates(self) -> List[str]:
        return ["Which is the most common failure mode associated with the asset {asset_class}?",
                "Which failure mode should be monitored for the asset {asset_class}?",
                "Which failure mode can occur in the asset {asset_class} during operation?",
                "Which is the failure scenario that the asset {asset_class} might encounter?",
                "Which failure mode is most likely to occur with the asset {asset_class}?"]

    @property
    def criteria(self) -> List[str]:
        return ["the failure modes that are the common failure modes associated with the asset {asset_class}",
                "the failure modes that are unlikely to occur with the asset {asset_class}"]

    def generate(
            self,
            inputs: Dict[str, Any],
            num_problems_per_batch: int = 10,
            question_paraphrase: bool = True,
            num_choices: int = 5,
            random_seed: int = 32,
            num_voters: int = 3,
            agreement_threshold: float = 0.5,
            llm: str = voters[0],
            reasoning_prompting: Callable = self_guess_generate_final_answer,
            elimination=True
    ):
        self._validate_inputs(inputs)
        asset_class = inputs[self.input_keys[0]]
        failure_mode_candidates = inputs[self.input_keys[1]]
        if isinstance(asset_class, str):
            asset_class = [asset_class]
        if not isinstance(asset_class, List):
            raise ValueError(f"asset_class should be a string or a list of strings. Input asset_class: {asset_class}")
        if not isinstance(failure_mode_candidates, List):
            raise ValueError(
                f"failure_mode_candidates should be a list of strings. Input failure_mode_candidates: {failure_mode_candidates}")
        if len(failure_mode_candidates) < num_choices:
            raise ValueError(
                f"failure_mode_candidates should have at least {num_choices} candidates. Input failure_mode_candidates: {failure_mode_candidates}")

        num_problems_per_asset_class = num_problems_per_batch
        total_problems_with_reasoning = []
        dir = "./elimination/failure_mode" if elimination else "./selection/failure_mode"

        for an_asset_class in asset_class:
            print('asset_class', an_asset_class)
            prompts: List = self._create_prompt(
                inputs={"asset_class": an_asset_class},
                choice_candidates=failure_mode_candidates,
                num_batches=num_problems_per_asset_class,
                num_choices=num_choices,
                question_paraphrase=question_paraphrase,
                random_seed=random_seed,
                llm=llm,
                elimination=elimination

            )
            problems = []
            for prompt in prompts:
                unique_id = str(uuid.uuid4()).replace('-', '')[:8]
                agreement_reached, most_agreed_letter, letter2confidence_score = self._create_answer(  # self guess
                    prompt=prompt,
                    num_voters=num_voters,
                    agreement_threshold=agreement_threshold,
                    unique_id=unique_id,
                )
                if agreement_reached:
                    problems.append({
                        'prompt': prompt,
                        'answer': most_agreed_letter,
                        'unique_id': unique_id
                    })

            problems_with_reasoning = []
            for problem in problems:
                problem_with_reasoning = self._create_reasoning(
                    problem=problem,
                    reasoning_llm=llm,
                    reasoning_prompting=reasoning_prompting
                )
                if problem_with_reasoning:
                    problems_with_reasoning.append(problem_with_reasoning)

                    with open(f"{dir}/{an_asset_class}.jsonl", "a") as f:
                        f.write(json.dumps(problem_with_reasoning) + "\n")

            total_problems_with_reasoning.extend(problems_with_reasoning)

            with open(f"{dir}/{an_asset_class}.json", "w") as f_out:
                json.dump(problems_with_reasoning, f_out, indent=4)

        return total_problems_with_reasoning


class Sensor2FMPG(ProblemGenerator):
    @property
    def input_keys(self) -> List[str]:
        return ["asset_class", "relevant_sensors_for_each_asset", "failure_mode_candidates"]

    @property
    def question_static_keys(self) -> List[str]:
        return ["asset_class", "relevant_sensor"]

    
    
    @property
    def elimination_question_candidates(self) -> List[str]:
        return [
            "In the context of {asset_class}, which failure mode is least relevant when {relevant_sensor} shows abnormal readings?",
            "Which is the irrelevant failure mode for {asset_class} if {relevant_sensor} exhibits abnormal readings?",
            "Which failure mode should not be considered for {asset_class} when abnormal readings are detected by {relevant_sensor}?",
            "When {relevant_sensor} in {asset_class} displays abnormal readings, which failure mode is the least applicable?",
            "What is the most irrelevant failure mode for {asset_class} when {relevant_sensor} indicates abnormal behavior?",
            "In {asset_class}, which failure mode should not be considered when {relevant_sensor} reports unusual readings?",
            "Which failure mode is typically irrelevant to abnormal {relevant_sensor} readings in the context of {asset_class}?",
            "For {asset_class}, which failure mode never correlates with abnormal values reported by {relevant_sensor}?",
            "In the scenario where {relevant_sensor} shows unusual readings, which failure mode is least critical for {asset_class}?"]
    
    @property
    def selection_question_candidates(self) -> List[str]:
        return [
            "In the context of {asset_class}, which failure mode is most relevant when {relevant_sensor} shows abnormal readings?",
            "Which is the most relevant failure mode for {asset_class} if {relevant_sensor} exhibits abnormal readings?",
            "Which failure mode should be considered for {asset_class} when abnormal readings are detected by {relevant_sensor}?",
            "When {relevant_sensor} in {asset_class} displays abnormal readings, which failure mode is the most applicable?",
            "For {asset_class}, what is the key failure mode when {relevant_sensor} has abnormal readings?",
            "What is the most likely failure mode for {asset_class} when {relevant_sensor} indicates abnormal behavior?",
            "In {asset_class}, which failure mode should be considered when {relevant_sensor} reports unusual readings?",
            "Which failure mode is typically linked to abnormal {relevant_sensor} readings in the context of {asset_class}?",
            "For {asset_class}, which failure mode correlates with abnormal values reported by {relevant_sensor}?",
            "In the scenario where {relevant_sensor} shows unusual readings, which failure mode is most critical for {asset_class}?"]

    @property
    def criteria(self) -> List[str]:
        return [
            "the failure modes that are likely to happen when asset {asset_class} has abnormal readings of sensor {relevant_sensor}",
            "the failure modes that are unlikely to happen when asset {asset_class} has abnormal readings of sensor {relevant_sensor}"]

    def generate(
            self,
            inputs: Dict[str, Any],
            num_problems_per_batch: int = 10,
            question_paraphrase: bool = True,
            num_choices: int = 5,
            random_seed: int = 32,
            num_voters: int = 3,
            agreement_threshold: float = 0.5,
            llm: str = voters[0],
            reasoning_prompting: Callable = self_guess_generate_final_answer,
            elimination=True
    ):
        self._validate_inputs(inputs)
        asset_class = inputs[self.input_keys[0]]
        relevant_sensors_for_each_asset: Dict[str, List[str]] = inputs[self.input_keys[1]]
        failure_mode_candidates = inputs[self.input_keys[2]]

        if isinstance(asset_class, str):
            asset_class = [asset_class]
        if not isinstance(asset_class, List):
            raise ValueError(f"asset_class should be a string or a list of strings. Input asset_class: {asset_class}")
        if not isinstance(failure_mode_candidates, List):
            raise ValueError(
                f"failure_mode_candidates should be a list of strings. Input failure_mode_candidates: {failure_mode_candidates}")
        if len(failure_mode_candidates) < num_choices:
            raise ValueError(
                f"failure_mode_candidates should have at least {num_choices} candidates. Input failure_mode_candidates: {failure_mode_candidates}")

        dir = "./elimination/sensor2fm" if elimination else "./selection/sensor2fm"

        total_problems_with_reasoning = []
        for an_asset_class in asset_class:
            print('asset_class', an_asset_class)
            if an_asset_class not in relevant_sensors_for_each_asset:
                print(f"asset_class {an_asset_class} is not in relevant_sensors_for_each_asset.")
                continue

            problems = []

            for relevant_sensor in relevant_sensors_for_each_asset[an_asset_class]:
                prompts: List = self._create_prompt(
                    inputs={"asset_class": an_asset_class, "relevant_sensor": relevant_sensor},
                    choice_candidates=failure_mode_candidates,
                    num_batches=num_problems_per_batch,
                    num_choices=num_choices,
                    question_paraphrase=question_paraphrase,
                    random_seed=random_seed,
                    llm=llm,
                    elimination=elimination

                )
                for prompt in prompts:
                    unique_id = str(uuid.uuid4()).replace('-', '')[:8]
                    agreement_reached, most_agreed_letter, letter2confidence_score = self._create_answer(  # self guess
                        prompt=prompt,
                        num_voters=num_voters,
                        agreement_threshold=agreement_threshold,
                        unique_id=unique_id,
                    )
                    if agreement_reached:
                        problems.append({
                            'prompt': prompt,
                            'answer': most_agreed_letter,
                            'unique_id': unique_id
                        })

            problems_with_reasoning = []
            for problem in problems:
                problem_with_reasoning = self._create_reasoning(
                    problem=problem,
                    reasoning_llm=llm,
                    reasoning_prompting=reasoning_prompting
                )
                if problem_with_reasoning:
                    problems_with_reasoning.append(problem_with_reasoning)

                    with open(f"{dir}/{an_asset_class}.jsonl", "a") as f:
                        f.write(json.dumps(problem_with_reasoning) + "\n")

            total_problems_with_reasoning.extend(problems_with_reasoning)

            with open(f"{dir}/{an_asset_class}.json", "w") as f_out:
                json.dump(problems_with_reasoning, f_out, indent=4)

        return total_problems_with_reasoning


class FM2SensorPG(ProblemGenerator):
    @property
    def input_keys(self) -> List[str]:
        return ["asset_class", "relevant_failure_modes_for_each_asset", "sensor_candidates"]

    @property
    def question_static_keys(self) -> List[str]:
        return ["asset_class", "relevant_failure_mode"]


    @property
    def elimination_question_candidates(self) -> List[str]:
        return [
            'Which sensor cannot be used to monitor asset {asset_class} for failure mode {relevant_failure_mode}?',
            "What sensor is least suitable for monitoring {asset_class} to detect {relevant_failure_mode}?",
            "What sensor cannot be utilized to monitor {asset_class} for signs of {relevant_failure_mode}?",
            "Which sensor is irrelevant to monitor {asset_class} for the occurrence of {relevant_failure_mode}?",
            "In an {asset_class}, which sensor is not designed to track {relevant_failure_mode}?",
            "In the context of {asset_class}, which sensor cannot help in identifying {relevant_failure_mode}?",
            "Which sensor has nothing to do with monitoring {asset_class} to detect {relevant_failure_mode}?",
            "Which sensor is not able to effectively monitor {asset_class} for potential {relevant_failure_mode}?"
        ]
    
    @property
    def selection_question_candidates(self) -> List[str]:
        return [
            'Which sensor can be used to monitor asset {asset_class} for failure mode {relevant_failure_mode}?',
            "What sensor is suitable for monitoring {asset_class} to detect {relevant_failure_mode}?",
            "What sensor can be utilized to monitor {asset_class} for signs of {relevant_failure_mode}?",
            "Which sensor is best suited to monitor {asset_class} for the occurrence of {relevant_failure_mode}?",
            "In an {asset_class}, which sensor is designed to track {relevant_failure_mode}?",
            "In the context of {asset_class}, which sensor can help in identifying {relevant_failure_mode}?",
            "Which sensor would you recommend for monitoring {asset_class} to detect {relevant_failure_mode}?",
            "Which sensor can effectively monitor {asset_class} for potential {relevant_failure_mode}?"
        ]

    @property
    def criteria(self) -> List[str]:
        return [
            "the sensors that can be used to monitor asset {asset_class} for failure mode {relevant_failure_mode}",
            "the sensors that cannot be used to monitor asset {asset_class} for failure mode {relevant_failure_mode}"]

    def generate(
            self,
            inputs: Dict[str, Any],
            num_problems_per_batch: int = 10,
            question_paraphrase: bool = True,
            num_choices: int = 5,
            random_seed: int = 32,
            num_voters: int = 3,
            agreement_threshold: float = 0.5,
            llm: str = voters[0],
            reasoning_prompting: Callable = self_guess_generate_final_answer,
            elimination=True
    ):
        self._validate_inputs(inputs)
        asset_class = inputs[self.input_keys[0]]
        relevant_failure_modes_for_each_asset: Dict[str, List[str]] = inputs[self.input_keys[1]]
        sensor_candidates = inputs[self.input_keys[2]]

        if isinstance(asset_class, str):
            asset_class = [asset_class]
        if not isinstance(asset_class, List):
            raise ValueError(f"asset_class should be a string or a list of strings. Input asset_class: {asset_class}")
        if not isinstance(sensor_candidates, List):
            raise ValueError(
                f"sensor_candidates should be a list of strings. Input sensor_candidates: {sensor_candidates}")
        if len(sensor_candidates) < num_choices:
            raise ValueError(
                f"sensor_candidates should have at least {num_choices} candidates. Input sensor_candidates: {sensor_candidates}")

        dir = "./elimination/fm2sensor" if elimination else "./selection/fm2sensor"

        total_problems_with_reasoning = []
        for an_asset_class in asset_class:
            print('asset_class', an_asset_class)
            if an_asset_class not in relevant_failure_modes_for_each_asset:
                raise ValueError(f"asset_class {an_asset_class} is not in relevant_failure_modes_for_each_asset.")

            problems = []

            for relevant_failure_mode in relevant_failure_modes_for_each_asset[an_asset_class]:
                prompts: List = self._create_prompt(
                    inputs={"asset_class": an_asset_class, "relevant_failure_mode": relevant_failure_mode},
                    choice_candidates=sensor_candidates,
                    num_batches=num_problems_per_batch,
                    num_choices=num_choices,
                    question_paraphrase=question_paraphrase,
                    random_seed=random_seed,
                    llm=llm,
                    elimination=elimination
                )
                for prompt in prompts:
                    unique_id = str(uuid.uuid4()).replace('-', '')[:8]
                    agreement_reached, most_agreed_letter, letter2confidence_score = self._create_answer(  # self guess
                        prompt=prompt,
                        num_voters=num_voters,
                        agreement_threshold=agreement_threshold,
                        unique_id=unique_id,
                    )
                    if agreement_reached:
                        problems.append({
                            'prompt': prompt,
                            'answer': most_agreed_letter,
                            'unique_id': unique_id
                        })

            problems_with_reasoning = []
            for problem in problems:
                problem_with_reasoning = self._create_reasoning(
                    problem=problem,
                    reasoning_llm=llm,
                    reasoning_prompting=reasoning_prompting
                )
                if problem_with_reasoning:
                    problems_with_reasoning.append(problem_with_reasoning)

                    with open(f"{dir}/{an_asset_class}.jsonl", "a") as f:
                        f.write(json.dumps(problem_with_reasoning) + "\n")

            total_problems_with_reasoning.extend(problems_with_reasoning)

            with open(f"{dir}/{an_asset_class}.json", "w") as f_out:
                json.dump(problems_with_reasoning, f_out, indent=4)

        return total_problems_with_reasoning
