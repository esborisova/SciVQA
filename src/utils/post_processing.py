import json
import ast

def str_to_json(text):
    try:
        cleaned_str = text.replace("```json\n", "").replace("\n```", "")
        json_data = json.loads(cleaned_str)
        return json_data

    except Exception as e:
        return text
    
def convert_to_list(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val
