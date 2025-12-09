# clean_json.py

def clean_json_response(raw_text):
    """
    Removes markdown code block formatting from LLM response
    and returns a clean JSON string.
    """
    # Remove leading/trailing whitespace
    raw_text = raw_text.strip()
    
    # Remove markdown code block delimiters if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]  # Remove '```json' from start
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]  # Remove '```' from end
    
    return raw_text.strip()
