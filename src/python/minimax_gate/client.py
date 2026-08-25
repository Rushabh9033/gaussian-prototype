import os
import time
import requests
import base64
from .configuration import PROMPT, WIDTH, HEIGHT, MODEL_ID, API_ENDPOINT
from .call_budget import check_budget
from .redaction import redact_secrets

class MissingAPIKeyError(Exception): pass
class APIError(Exception): pass
class VehicleReferenceRejectedError(Exception): pass
class AuthorizationError(Exception): pass
class QuotaError(Exception): pass

def generate_image(reference_url, seed):
    key = os.environ.get('MINIMAX_API_KEY')
    if not key:
        raise MissingAPIKeyError("MINIMAX_API_KEY is not set.")
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "subject_reference": {
            "type": "character", 
            "image_file": reference_url
        },
        "width": WIDTH,
        "height": HEIGHT,
        "response_format": "base64",
        "prompt_optimizer": False,
        "n": 1,
        "prompt": PROMPT,
        "seed": seed
    }
    
    def _make_call():
        check_budget()
        try:
            return requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=45)
        except requests.exceptions.RequestException as e:
            raise APIError(redact_secrets(str(e)))
            
    response = _make_call()
    
    # Retry on 5xx once
    if 500 <= response.status_code < 600:
        time.sleep(2)
        response = _make_call()
        
    try:
        # Check for rejection or auth errors
        if response.status_code == 400 and "character" in response.text.lower():
            raise VehicleReferenceRejectedError("Vehicle reference rejected.")
        if response.status_code == 401 or response.status_code == 403:
            raise AuthorizationError(f"Authentication failed: {response.status_code}")
        if response.status_code == 429:
            raise QuotaError("Quota exceeded or rate limited.")
            
        response.raise_for_status()
        data = response.json()
        
        # Schema validation
        if 'base64_image' not in data:
            if 'choices' in data and len(data['choices']) > 0 and 'base64_image' in data['choices'][0]:
                base64_str = data['choices'][0]['base64_image']
            elif 'data' in data and len(data['data']) > 0 and 'base64_image' in data['data'][0]:
                base64_str = data['data'][0]['base64_image']
            else:
                base64_str = data.get('base64_image')
                if not base64_str:
                    raise APIError("Response schema missing base64_image.")
        else:
            base64_str = data['base64_image']
            
        image_data = base64.b64decode(base64_str)
        return image_data, redact_secrets(data)
    except Exception as e:
        if isinstance(e, (VehicleReferenceRejectedError, AuthorizationError, QuotaError)):
            raise
        raise APIError(redact_secrets(str(e)))
