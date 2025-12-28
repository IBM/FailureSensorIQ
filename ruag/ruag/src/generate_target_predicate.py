import pandas as pd
import numpy as np
import json
from openai import OpenAI


def generate_atomic_predicates(
    df: pd.DataFrame, max_unique_categorical=20, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)
):
    """
    Automatically generate ALL atomic logical predicates:
      • For categorical columns → equality predicates
      • For numeric columns → threshold predicates
    Returns: enriched DataFrame, list of predicate column names
    """

    df = df.copy()
    predicates = []

    for col in df.columns:
        # Skip already binary predicate columns
        if set(df[col].dropna().unique()).issubset({0, 1}):
            continue

        # CATEGORICAL → equality predicates
        if df[col].dtype == "object" or df[col].nunique() <= max_unique_categorical:
            for v in df[col].dropna().unique():
                name = f"{col}__eq__{str(v).replace(' ','_')}"
                df[name] = (df[col] == v).astype(int)
                predicates.append(name)

        # NUMERIC → threshold predicates
        else:
            qs = df[col].quantile(quantiles).unique()
            for q in qs:
                name = f"{col}__le__{round(float(q), 4)}"
                df[name] = (df[col] <= q).astype(int)
                predicates.append(name)

    return df, predicates


client = OpenAI()

SYSTEM_PROMPT = """
You are a Target-Derived Predicate Oracle.

You are given a dataset description, feature descriptions, a target variable, and a list of existing predicates.

Your task is to generate new symbolic predicates derived from the target variable, including:
- Alternative thresholds (high, medium, low)
- Combinations with other features
- Trend or temporal conditions
- Aggregated or discretized versions
- Multivariate interactions with the target

Constraints:
- Do NOT duplicate existing predicates
- Must be computable from the dataset
- Must remain discrete (boolean)
- Must meaningfully relate to the target variable
- Must be interpretable and actionable

Return STRICT JSON:

{
  "new_predicates": [
    {
      "predicate_name": "",
      "meaning": "",
      "logic": "",
      "python_code": "",
      "justification": ""
    }
  ]
}
"""

def generate_target_derived_predicates(
    df: pd.DataFrame,
    dataset_description: str,
    feature_descriptions: dict,
    target_column: str,
    existing_predicates: list,
    model="gpt-4.1"
):
    """
    Uses LLM to generate new target-derived predicates and materialize them in the DataFrame.

    Returns: updated DataFrame, list of new predicate metadata
    """
    user_prompt = f"""
Dataset description:
{dataset_description}

Feature descriptions:
{json.dumps(feature_descriptions, indent=2)}

Target variable: {target_column}

Existing predicates: {json.dumps(existing_predicates)}
Return JSON only.
"""

    # Call the LLM
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_output_tokens=800
    )

    # Parse LLM JSON output
    try:
        pred_info = json.loads(response.output_text.strip())
        new_preds = pred_info.get("new_predicates", [])
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON")

    # Safely materialize new predicates in the DataFrame
    df_copy = df.copy()
    for pred in new_preds:
        try:
            exec(pred["python_code"], {}, {"df": df_copy})
        except Exception as e:
            print(f"Failed to execute predicate {pred['predicate_name']}: {e}")

    return df_copy, new_preds
