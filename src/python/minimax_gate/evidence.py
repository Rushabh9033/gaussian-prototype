import os
import json
from .redaction import safe_json_dumps

def append_evidence(out_dir, record):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'evidence.jsonl'), 'a', encoding='utf-8') as f:
        f.write(safe_json_dumps(record) + '\n')
