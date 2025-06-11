# coding:utf-8

from abc import ABC, abstractmethod
from functools import partial
from sentence_transformers import SentenceTransformer, util
import copy
import json
import numpy as np
import re
import spacy
from typing import List
import logging
import traceback
import sys
import random

from GeneralLLM import LargeLanguageModel, Qwen, ChatGPT


def _add_left_parenthesis(s: str) -> str:
    return f"({s}"


def _add_left_bracket(s: str) -> str:
    return f"[{s}"


def _add_left_brace(s: str) -> str:
    return f"{{{s}"


def _add_left_wave(s: str) -> str:
    return f"~{s}"


def _add_right_parenthesis(s: str) -> str:
    return f"{s})"


def _add_right_bracket(s: str) -> str:
    return f"{s}]"


def _add_right_brace(s: str) -> str:
    return f"{s}}}"


def _add_right_wave(s: str) -> str:
    return f"{s}~"


def _add_right_eq(s: str) -> str:
    return f"{s}="


def _add_parentheses(s: str) -> str:
    return f"({s})"


def _add_brackets(s: str) -> str:
    return f"[{s}]"


def _add_braces(s: str) -> str:
    return f"{{{s}}}"


def _add_waves(s: str) -> str:
    return f"~{s}~"


def _caesar(s: str, delta: int = 10) -> str:
    return chr(ord(s) + delta)


class MultipleChoiceQuestion:
    """
    The class of multiple choice questions.
    """

    def __init__(
        self,
        question: str = "",
        option_ids: List[str] = [],
        options: List[str] = [],
        correct: List[bool] = None,
        question_first: bool = True,
        text_type="choice",
        trigger_statement=None,
    ):
        """
        Args:
            question: the question text
            option_ids: the list of option indeces (with formats), e.g., ['(A)', '(B)', '(C)', '(D)']
            options: the option contents, each of which appears after the corresponding option index.
            correct: the list indicating the correctness of each option. E.g., [True, False, False, True]
                indicates that only the first and the last options are correct answers.
            question_first: the switch controlling whether the question appears before the option.
            text_type: 'choice' or 'judgement', indicating whether the question text appears as a multiple
                choice question or a multiple judgement question
        """
        self.question = question
        self.option_ids = option_ids
        self.options = options
        self.question_first = question_first
        self.correct = correct
        self.text_type = text_type
        self.trigger_statement = trigger_statement
        assert len(option_ids) == len(options)
        if self.correct is not None:
            assert len(option_ids) == len(self.correct)
        assert text_type == "choice" or text_type == "judgement"

    def add_trigger_statement(self, trigger_statement):
        self.trigger_statement = trigger_statement

    def to_dict(self) -> dict:
        """Return the dictionary form of the current question."""
        result = {
            "question": self.question,
            "option_ids": self.option_ids,
            "options": self.options,
            "question_first": self.question_first,
            "correct": self.correct,
            "text_type": self.text_type,
            "trigger_statement": self.trigger_statement,
        }
        return result

    def load_dict(self, data: dict):
        """Load information of the question from a dictionary."""
        trigger_statement = data.get("trigger_statement", None)
        self.__init__(
            question=data["question"],
            option_ids=data["option_ids"],
            options=data["options"],
            question_first=data["question_first"],
            correct=data["correct"],
            text_type=data["text_type"],
            trigger_statement=trigger_statement,
        )

    def __str__(self):
        assert len(self.option_ids) == len(self.options)
        prompt = "Options:\n"
        for key, value in zip(self.option_ids, self.options):
            prompt += f"{key} {value}\n"
        prompt = f"Question: {self.question}\n" + prompt
        prompt += f"Answer:{self.correct}\n"
        prompt += f"question_first:{self.question_first}\n"
        prompt += f"text_type:{self.text_type}\n"
        prompt += f"trigger_statement:{self.trigger_statement}"
        return prompt

    def get_prompt(self):
        """Get the prompt of the current question for LLMs."""
        assert len(self.option_ids) == len(self.options)
        assert self.text_type == "choice" or self.text_type == "judgement"
        if self.text_type == "choice" and self.trigger_statement:
            prefix = f"Please select the correct option(s) from the following options given the question. To solve the problem, follow the {self.trigger_statement} reasoning strategy.\n"
        elif self.text_type == "choice":
            prefix = "Please select the correct option(s) from the following options given the question:\n"
        elif self.text_type == "judgement":
            prefix = "Please judge whether each of the options is correct or incorrect given the question:\n"
        prompt = "Options:\n"
        for key, value in zip(self.option_ids, self.options):
            prompt += f"{key} {value}\n"
        if self.question_first:
            prompt = f"Question: {self.question}\n" + prompt
        else:
            prompt = prompt + f"Question: {self.question}\n"
        if self.text_type == "choice":
            option_or = ", ".join([f'"{option}"' for option in self.option_ids])
            if self.trigger_statement:
                # If trigger_statement is provided, reasoning should come first, followed by the selected options
                # prompt += (
                #     "Your output must strictly follow this format:\n"
                #     '{"reasoning": <"Your reasoning step-by-step">, "answer": <the list of selected options, e.g., [%s]>}\n'
                #     % option_or
                # )
                # prompt += (
                #     '{"step_1": "<Step 1 of your reasoning>", "step_2": "<Step 2 of your reasoning>", "step_n": "<Step n of your reasoning>", "answer": <the list of selected option, e.g., ["A", "B", "C", "D", "E"]>}\n'
                # )
                prompt += '{"option_a": "<Reasoning for option a>", "option_b": "<Reasoning for option b>", ..., "answer": <the list of selected option, e.g., ["A", "B", "C", "D", "E"]>}'

            else:
                prompt += (
                    'Your output must strictly follow this format:\n{"answer": <the list of selected options, e.g., [%s]>}\n'
                    % option_or
                )
        elif self.text_type == "judgement":
            output_fmt = ", ".join(
                [f'"{option}": <"True" or "False">' for option in self.option_ids]
            )
            output_fmt = "{" + output_fmt + "}"
            prompt += f"Your output must strictly follow this format:\n{output_fmt}\n"
        prompt = prefix + prompt
        prompt += "\nYour output in a single line:"
        return prompt

    def get_formatted_answer(self):
        result = None
        if self.text_type == "choice":
            answers = []
            for option, correct in zip(self.option_ids, self.correct):
                if correct:
                    answers.append(option)

            content = ", ".join(['"' + option + '"' for option in answers])
            result = f'"answer": [{content}]'
            result = "{" + result + "}"
        elif self.text_type == "judgement":
            result = ", ".join(
                [
                    f'"{option}":"{correct}"'
                    for option, correct in zip(self.option_ids, self.correct)
                ]
            )
            result = "{" + result + "}"
        return result


def fix_json_string_1(json_string):
    # Ensure keys are properly quoted
    json_string = re.sub(r"(\breasoning\b|\banswer\b):", r'"\1":', json_string)

    # Wrap unquoted reasoning values with double quotes
    json_string = re.sub(
        r'(?<="reasoning":\s)([^{\[\"].*?)(?=,?\s*"?answer")',
        lambda match: f'"{match.group(0).strip()}"',
        json_string,
        flags=re.DOTALL,
    )

    # Add a comma if missing between reasoning and answer
    json_string = re.sub(
        r'("reasoning":.*?[^}])\s*("answer":)', r"\1, \2", json_string, flags=re.DOTALL
    )

    return json_string


def fix_json_string(json_string):
    # Ensure keys are properly quoted
    json_string = re.sub(r"(\breasoning\b|\banswer\b):", r'"\1":', json_string)

    # Wrap unquoted reasoning values with double quotes
    json_string = re.sub(
        r'(?<="reasoning":\s)([^"\[{].*?)(?=,?\s*"?answer")',
        lambda match: f'"{match.group(0).strip()}"',
        json_string,
        flags=re.DOTALL,
    )

    # Add a comma if missing between reasoning and answer
    json_string = re.sub(
        r'("reasoning":.*?[^}],?)\s*("answer":)',
        r"\1, \2",
        json_string,
        flags=re.DOTALL,
    )

    return json_string


def get_mcq_llm_answer(mcq: MultipleChoiceQuestion, llm: LargeLanguageModel) -> tuple:
    """Get the answer of an LLM to a multiple-choice question.
    Args:
        mcq:MultipleChoiceQuestion, the question to answer
        llm:LargeLauguageModel, the model that answer the question
    Return:
        List[bool]: the list that indicates whether each of the
            options is selected by the model.
        str: the original response of the model.
    """
    prompt = mcq.get_prompt()
    response_ok = False
    max_retry = 3
    n_retry = 1
    result = [False] * len(mcq.correct)
    llm.refresh()
    while response_ok is False and n_retry <= max_retry:
        try:
            original_response = ""
            if n_retry > 1:
                original_response = llm.listen_and_response(
                    "Failed to extract proper json block as requested"
                )
            else:
                original_response = llm.listen_and_response(prompt)
            if random.random() < 0.05:
                logging.info("original response: " + original_response)
            response = re.sub(r"\n", " ", original_response)
            response = re.findall(r'[{]\s*"[^{]*[}]', response)[0]
            # response = re.sub(r'("[^"]*"\s*:\s*[^,}]+)(?=\s*"[^{]*[}])', r'\1,', response)
            if mcq.text_type == "choice":
                try:
                    response = json.loads(response)
                except:
                    response = fix_json_string(response)
                    response = json.loads(response)

                for i in range(len(mcq.option_ids)):
                    if mcq.option_ids[i] in response["answer"]:
                        result[i] = True
            elif mcq.text_type == "judgement":
                response = re.sub(r"\s+True", ' "True"', response)
                response = re.sub(r"\s+False", ' "False"', response)
                response = json.loads(response)
                oid2pos = {}
                for i in range(len(mcq.option_ids)):
                    oid2pos[mcq.option_ids[i]] = i
                for key in response.keys():
                    result[oid2pos[key]] = eval(response[key])
            else:
                logging.error(
                    f"get_mcq_llm_answer: Invalid text type '{mcq.text_type}'"
                )
            response_ok = True
        except Exception as ex:
            
            logging.error(f"original_llm_answer = {original_response}")
            logging.error(traceback.format_exc())
            logging.error(
                f"get_mcq_llm_answer: Format error, try again. n_retry = {n_retry}"
            )
            n_retry += 1
    return result, original_response


class KPPerturbation(ABC):
    def __init__(self):
        self.method = "default"
        pass

    @abstractmethod
    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        pass


class OptionFormatPerturbation(KPPerturbation):
    def __init__(
        self,
        method: str = "add_parentheses",
    ):
        """
        Args:
            method:str, the perturbation method
        """
        super().__init__()
        self.method = method
        self.formatter = None
        if method == "add_left_parenthesis":
            self.formatter = _add_left_parenthesis
        elif method == "add_left_bracket":
            self.formatter = _add_left_bracket
        elif method == "add_left_brace":
            self.formatter = _add_left_brace
        elif method == "add_left_wave":
            self.formatter = _add_left_wave
        elif method == "add_right_parenthesis":
            self.formatter = _add_right_parenthesis
        elif method == "add_right_bracket":
            self.formatter = _add_right_bracket
        elif method == "add_right_brace":
            self.formatter = _add_right_brace
        elif method == "add_right_wave":
            self.formatter = _add_right_wave
        elif method == "add_right_eq":
            self.formatter = _add_right_eq
        elif method == "add_parentheses":
            self.formatter = _add_parentheses
        elif method == "add_brackets":
            self.formatter = _add_brackets
        elif method == "add_braces":
            self.formatter = _add_braces
        elif method == "add_waves":
            self.formatter = _add_waves
        else:
            raise Exception("Invalid option format perturbation method.")

    def __str__(self):
        return f"OptionFormatPerturbation.{self.method}"

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        assert len(mcq.option_ids) == len(mcq.options)
        try:
            new_option_ids = [self.formatter(elem) for elem in mcq.option_ids]
            result = copy.deepcopy(mcq)
            result.option_ids = new_option_ids
        except:
            logging.error("OptionFormatPerturbation error. Keep the original result.")
            result = copy.deepcopy(mcq)
        return result


class CaesarPerturbation(KPPerturbation):
    def __init__(self, delta: int = 20):
        """
        Args:
            delta:int, the offset value in ASCII of option ids.
        """
        super().__init__()
        self.formatter = partial(_caesar, delta=delta)

    def __str__(self):
        return f"CaesarPerturbation.{self.method}"

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        assert len(mcq.option_ids) == len(mcq.options)

        try:
            new_option_ids = [self.formatter(elem) for elem in mcq.option_ids]
            result = copy.deepcopy(mcq)
            result.option_ids = new_option_ids
        except:
            logging.error("CaesarPerturbation error. Keep the original result.")
            result = copy.deepcopy(mcq)
        return result


class OptionPermutationPerturbation(KPPerturbation):
    def __init__(
        self,
        permutation_map={
            2: {0: 1, 1: 0},  # For 2 options
            3: {0: 2, 1: 0, 2: 1},  # For 3 options
            4: {0: 3, 1: 2, 2: 1, 3: 0},  # For 4 options
            5: {0: 4, 1: 3, 2: 2, 3: 1, 4: 0},  # For 5 options
        },
    ):
        """
        Args:
            permutation_map:dict. The key denotes the original position of option contents,
                while the value denotes the target position to place option contents.
                If it is None, then the permutation map is randomly generated.
        """
        super().__init__()
        self.map = permutation_map

    def _generate_r_map(self, mcq_len):
        self.rmap = {}
        for k in self.map[mcq_len].keys():
            self.rmap[self.map[mcq_len][k]] = k

    def __str__(self):
        return f"OptionPermutationPerturbation.{self.method}"

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        assert len(mcq.option_ids) == len(mcq.options)
        result = copy.deepcopy(mcq)
        self._generate_r_map(len(mcq.options))
        new_options = [
            mcq.options[self.rmap.get(i, i)] for i in range(len(mcq.options))
        ]
        new_correct = [
            mcq.correct[self.rmap.get(i, i)] for i in range(len(mcq.correct))
        ]
        result.options = new_options
        result.correct = new_correct
        # else: raise Exception("Invalid permutation perturbation method.")
        return result


class ChangeQuestionPosPerturbation(KPPerturbation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"ChangeQuestionPosPerturbation.{self.method}"

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        result = copy.deepcopy(mcq)
        result.question_first = not result.question_first
        return result


class ChangeTypePerturbation(KPPerturbation):
    def __init__(self):
        super().__init__()
        self.type_dict = {0: "choice", 1: "judgement"}
        self.rev_type_dict = {}
        for key in self.type_dict:
            self.rev_type_dict[self.type_dict[key]] = key

    def __str__(self):
        return f"ChangeTypePerturbation.{self.method}"

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        type_id = self.rev_type_dict[mcq.text_type]
        new_type_id = (type_id + 1) % len(self.type_dict)
        result = copy.deepcopy(mcq)
        result.text_type = self.type_dict[new_type_id]
        return result


class QuestionRewriter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.n_requests = 0

    def rewrite(
        self,
        mcq: MultipleChoiceQuestion,
        rewriter: ChatGPT,
        n_candidates=2,
        similarity_score=0.6,
    ) -> List[List[str]]:
        assert n_candidates > 1
        self.n_requests = 0
        doc = self.nlp(mcq.question)
        sentences = [item.text for item in doc.sents]
        result = []
        for i in range(0, len(sentences)):
            context = " ".join(sentences[:i])
            sentence = sentences[i]
            if len(sentences) > 1:
                prompt = f"""Here is a sentence in a multiple choice question. Please rewrite the sentence given its context and the expected similarity score. Here are necessary requirements:
[Requirements Start]
1. Be consistent with its context.
2. The rewrited sentence should keep the semantic of the original sentence.
3. If the sentence contains blanks/underlines to be filled, these blanks/underlines should be kept after paraphrasing.
4. You can utilize various rewriting skills (e.g., add/replace/delete words, paraphrase) to make it looks different from the original.
[Requirements End]
[Meaning of Expected Similarity Score Start]
For the expected similarity score (0.0~1.0)，1.0 denotes that the rewrited is exactly the same as the original; 0.8 denotes that the the there exist word-level differences between the rewrited and the original; 0.6 denotes that there exist not only word-level, but lots of sentence structure-level differences between the rewrited and the original; 0.4 denotes that you are allowed to entirely paraphrase the sentence by your own; 0.2 denotes that you are allowed to add misleading statements to the current sentence.
[Meaning of Expected Similarity Score End]
You should only output the rewrited sentence without any extra content.
Expected similarity score: {similarity_score}
Context: {context}
Sentence: {sentence}
Your output:"""
            else:
                prompt = f"""Here is a sentence in a multiple choice question. Please rewrite the sentence given its context and the expected similarity score. Here are necessary requirements:
[Requirements Start]
1. Be consistent with its context.
2. The rewrited sentence should keep the semantic of the original sentence.
3. If the sentence contains blanks/underlines to be filled, these blanks/underlines should be kept after paraphrasing.
4. You can utilize various rewriting skills (e.g., add/replace/delete words, paraphrase) to make it looks different from the original.
[Requirements End]
[Meaning of Expected Similarity Score Start]
For the expected similarity score (0.0~1.0)，1.0 denotes that the rewrited is exactly the same as the original; 0.8 denotes that the the there exist word-level differences between the rewrited and the original; 0.6 denotes that there exist not only word-level, but lots of sentence structure-level differences between the rewrited and the original; 0.4 denotes that you are allowed to entirely paraphrase the sentence by your own; 0.2 denotes that you are allowed to add misleading statements to the current sentence.
[Meaning of Expected Similarity Score End]
You should only output the rewrited sentence without any extra content.
Expected similarity score: {similarity_score}
Sentence: {sentence}
Your output:"""
            rewriter.refresh()
            response = rewriter.listen_and_response(prompt, n_outputs=n_candidates)
            self.n_requests += 1
            response = [sentence] + response
            result.append(response)
        return result


class QuestionGenerator:
    def __init__(self):
        self.transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def generate(self, candidates: List[List[str]]) -> tuple:
        embeddings = []
        similarities = []
        priorities = []
        question_texts = []
        question_similarities = []

        # for j in range(len(candidates[0])):
        #     candidates[0][j] = candidates[0][0]

        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                if "Sentence:" in candidates[i][j]:
                    candidates[i][j] = re.findall(
                        ".*Sentence:\s*(.*)", candidates[i][j]
                    )[0]

        for elem in candidates:
            emb = self.transformer.encode(elem, convert_to_tensor=True)
            embeddings.append(emb)
            similarity = []
            for j in range(emb.shape[0]):
                j_sim = (
                    util.pytorch_cos_sim(emb[0], emb[j]).detach().cpu().numpy()[0, 0]
                )
                # j_sim = util.pytorch_cos_sim(emb[0], emb[j]).detach().numpy()[0,0]
                similarity.append(j_sim)
            priority = np.argsort(similarity)[::-1]
            similarities.append(similarity)
            priorities.append(priority.tolist())
        priorities = np.array(priorities)
        original_text = " ".join(candidates[i][0] for i in range(priorities.shape[0]))
        embedding_1 = self.transformer.encode(original_text, convert_to_tensor=True)

        for top_k in range(priorities.shape[1]):
            question_text = " ".join(
                candidates[i][priorities[i][top_k]] for i in range(priorities.shape[0])
            )
            embedding_2 = self.transformer.encode(question_text, convert_to_tensor=True)
            q_sim = (
                util.pytorch_cos_sim(embedding_1, embedding_2)
                .detach()
                .cpu()
                .numpy()[0, 0]
            )
            question_texts.append(question_text)
            question_similarities.append(q_sim)
        return question_texts, question_similarities


class ParaphrasingPerturbation(KPPerturbation):
    def __init__(self, paraphrase_config: dict, rewriter: ChatGPT):
        """
        Args:
            paraphrase_config:dict, the configuration of paraphrasing.
                <key:value> = {"n_candidates":int,
                               "similarity_score":float}
                "n_candidates" denotes the number of generated candidates for each
                question sentence. "similarity_score" denotes the expected similarity
                score of the paraphrasing result, between 0 and 1. The larger, the similar.
            rewriter:ChatGPT (gpt-4-turbo is recommended), the rewriter that paraphrases questions.
        """
        super().__init__()
        self.questionRewriter = QuestionRewriter()
        self.questionGenerator = QuestionGenerator()
        self.rewriterLM = rewriter
        self.n_candidates = paraphrase_config["n_candidates"]
        self.similarity_score = paraphrase_config["similarity_score"]

    def __str__(self):
        return f"ParaphrasingPerturbation_n_candidates_{self.n_candidates}_similarity_score_{self.similarity_score}"

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        candidate_texts = self.questionRewriter.rewrite(
            mcq=mcq,
            rewriter=self.rewriterLM,
            n_candidates=self.n_candidates,
            similarity_score=self.similarity_score,
        )
        candidate_questions, similarities = self.questionGenerator.generate(
            candidate_texts
        )
        mid = int(len(candidate_questions) / 2)
        result = copy.deepcopy(mcq)
        result.question = candidate_questions[mid]
        return result


class MixedPerturbation(KPPerturbation):
    """The MixedPerturbation is the composite of atomic knowledge-invariant perturbations."""

    def __init__(self, perturbations: List[KPPerturbation] = None):
        super().__init__()
        self.perturbations = perturbations if isinstance(perturbations, list) else []

    def __str__(self):
        result = (
            "MixedPerturbation = [\n"
            + ",\n".join(" " * 4 + elem.__str__() for elem in self.perturbations)
            + "\n]\n"
        )
        return result

    def refresh(self):
        self.perturbations = []

    def push(self, elem: KPPerturbation):
        self.perturbations.append(elem)
        return

    def pop(self):
        if len(self.perturbations > 0):
            del self.perturbations[-1]
        return

    def perturb(self, mcq: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
        assert len(mcq.option_ids) == len(mcq.options)
        result = copy.deepcopy(mcq)
        for pt_elem in self.perturbations:
            result = pt_elem.perturb(result)
        return result


if __name__ == "__main__":
    mcq = MultipleChoiceQuestion(
        question="__________ memory is the aspect of memory that is involved in the recall of information acquired within the past few hours to days.",
        option_ids=["A", "B", "C", "D"],
        options=["Working", "Sensory", "Long-term", "Prospective"],
        question_first=True,
        correct=[False, False, False, True],
    )

    llm = ChatGPT(
        name="chatgpt",
        description="The chatgpt assistant.",
        model="gpt-4-turbo",
        temperature=1.0,
    )
    paraphrase_config = {"n_candidates": 3, "similarity_score": 0.6}
    ptb = ParaphrasingPerturbation(rewriter=llm, paraphrase_config=paraphrase_config)
    result = ptb.perturb(mcq)
    print(f"Original question: {mcq.get_prompt()}")
    print(f"Paraphrased question: {result.get_prompt()}")
