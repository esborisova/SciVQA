import json


def str_to_json(text):
    try:
        cleaned_str = text.replace("```json\n", "").replace("\n```", "")
        json_data = json.loads(cleaned_str)
        return json_data

    except Exception as e:
        return text
