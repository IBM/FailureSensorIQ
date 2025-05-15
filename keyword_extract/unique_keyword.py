import json

# Function to read a JSONL file and convert it to a list of dictionaries
def read_jsonl(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line.strip()) for line in f]  # Strip whitespace/newline from each line

# Function to extract all unique keywords from the "search_keywords" field
def extract_unique_keywords(responses):
    unique_keywords = set()  # Using a set to store unique keywords
    for entry in responses:
        # Debugging: Print the type and structure of entry to identify any hidden issues
        print(f"Processing entry (Type: {type(entry)}): {entry}")
        
        # Ensure the entry is a dictionary and contains 'search_keywords'
        if isinstance(entry, dict):
            if "search_keywords" in entry:
                keywords = entry["search_keywords"]
                if isinstance(keywords, list):  # Ensure that search_keywords is a list
                    unique_keywords.update(keywords)  # Add keywords to the set
                else:
                    print(f"Invalid 'search_keywords' format in entry: {entry}")
            else:
                print(f"'search_keywords' missing in entry: {entry}")
        else:
            print(f"Skipping non-dict entry: {entry}")  # Handle entries that aren't dictionaries
    return unique_keywords

# Function to save the unique keywords to a text file
def save_keywords_to_file(keywords, filepath="unique_keywords.txt"):
    with open(filepath, 'w') as f:
        for keyword in keywords:
            f.write(f"{keyword}\n")  # Write each keyword on a new line

# Main execution
if __name__ == "__main__":
    # Read the JSONL file containing the responses
    filepath = "fmsr_llm_responses.jsonl"
    
    # Read the file and parse each line as a JSON object
    responses = []
    with open(filepath, 'r') as f:
        for line in f:
            # Parse the line into a Python dictionary if it's in string format
            try:
                entry = json.loads(line.strip())  # Convert string to JSON
                if isinstance(entry, dict):
                    responses.append(entry)
                else:
                    responses.append(json.loads(entry))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}, skipping line.")
    
    # Extract unique keywords from the responses
    unique_keywords = extract_unique_keywords(responses)
    
    # Print the unique keywords
    print("Unique Keywords:")
    print(unique_keywords)
    
    # Save the unique keywords to a file
    save_keywords_to_file(unique_keywords, "unique_keywords.txt")
    print(f"Unique keywords saved to 'unique_keywords.txt'")
