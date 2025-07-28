import json
from ssee_metrics.recall import (
    structured_semantic_entity_evaluation_recall as ssee_recall,
)
from ssee_metrics.precision import (
    structured_semantic_entity_evaluation_precision as ssee_precision,
)
from ssee_metrics.semantic_utils import SentenceTransformerSemanticUtils
import ast
import math


def evaluate_file(file_path, recall_thresh=1.0, precision_thresh=0.9):
    stsu = SentenceTransformerSemanticUtils()
    recall_scores = []
    precision_scores = []
    c_len = []

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                E_gold_str = data.get("answer", [])
                E_cand = data.get("canswer", [])

                # Add quotes around each item if missing
                if not '"' in E_gold_str and not "'" in E_gold_str:
                    # Split by comma and strip whitespace
                    items = [item.strip() for item in E_gold_str.strip("[]").split(",")]
                    # Re-wrap as proper JSON-style string list
                    E_gold = items
                else:
                    E_gold = ast.literal_eval(E_gold_str)

                c_len.append(len(E_gold))
                recall = ssee_recall(E_cand, E_gold, recall_thresh, stsu)
                precision = ssee_precision(E_cand, E_gold, precision_thresh, stsu)
                recall_scores.append(recall)
                precision_scores.append(precision)

            except Exception as e:
                print (line)
                print(f"⚠️ Skipping line {line_num} due to error: {e}")

    # Filter out NaN values
    precision_scores_clean = [p for p in precision_scores if not math.isnan(p)]
    recall_scores_clean = [r for r in recall_scores if not math.isnan(r)]

    # Compute averages
    avg_precision = (
        sum(precision_scores_clean) / len(precision_scores_clean)
        if precision_scores_clean
        else 0.0
    )
    avg_recall = (
        sum(recall_scores_clean) / len(recall_scores_clean)
        if recall_scores_clean
        else 0.0
    )
    print (sum(c_len)/len(c_len))
    return avg_precision, avg_recall

# Example usage
fileset = [
    "senarios_10_asset_class_with_answers_model_1_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_1_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_2_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_2_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_0_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_0_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_20_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_20_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_19_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_19_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_7_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_7_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_6_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_6_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_12_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_12_control_10.jsonl",
    "senarios_10_asset_class_with_answers_model_16_control_5.jsonl",
    "senarios_10_asset_class_with_answers_model_16_control_10.jsonl",
]

for file in fileset[:4]:
    try:
        precision, recall = evaluate_file(file)
        print(f"📄 File: {file}")
        print(f"✅ Average Precision: {precision:.4f}")
        print(f"✅ Average Recall: {recall:.4f}")
    except Exception as ex:
        print ('eror ' + str(ex))
        pass
