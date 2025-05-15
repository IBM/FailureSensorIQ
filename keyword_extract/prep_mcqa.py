import json

def format_mcqa_from_jsonl(jsonl_path, limit=None):
    """
    Reads a .jsonl file and formats MCQA questions.

    Args:
        jsonl_path (str): Path to the .jsonl file.
        limit (int or None): Number of questions to format (useful for preview/testing).

    Returns:
        list[str]: Formatted MCQ strings.
    """
    formatted_questions = []

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            data = json.loads(line)

            question_id = data.get("id", "UnknownID")
            question = data.get("question", "No question provided")
            options = data.get("options", [])
            option_ids = data.get("option_ids", [])
            correct_flags = data.get("correct", [])

            # Find the correct option ID
            correct_option_id = option_ids[correct_flags.index(True)] if True in correct_flags else "?"

            # Format question
            output = [f"Q{question_id}: {question}\n"]
            for opt_id, opt in zip(option_ids, options):
                output.append(f"{opt_id}. {opt}")
            output.append(f"\nAnswer: {correct_option_id}\n")

            formatted_questions.append("\n".join(output))

    return formatted_questions
