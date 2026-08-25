import os
import json

def redact_secrets(data, key=None):
    if key is None:
        key = os.environ.get('MINIMAX_API_KEY')
    if not key:
        return data
    
    if isinstance(data, str):
        return data.replace(key, "***REDACTED***")
    elif isinstance(data, dict):
        return {k: redact_secrets(v, key) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_secrets(v, key) for v in data]
    return data

def safe_json_dumps(data):
    return json.dumps(redact_secrets(data))
