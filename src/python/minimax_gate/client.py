import os
import time
import requests
import base64
import numpy as np
import cv2
from .configuration import PROMPT, WIDTH, HEIGHT, MODEL_ID, API_ENDPOINT
from .call_budget import check_budget
from .redaction import redact_secrets

class MissingAPIKeyError(Exception): pass
class APIError(Exception): pass
class VehicleReferenceRejectedError(Exception): pass
class AuthorizationError(Exception): pass
class QuotaError(Exception): pass
class ImageValidationError(Exception): pass

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
        "subject_reference": [{
            "type": "character", 
            "image_file": reference_url
        }],
        "width": WIDTH,
        "height": HEIGHT,
        "response_format": "base64",
        "prompt_optimizer": False,
        "n": 1,
        "prompt": PROMPT,
        "seed": seed
    }
    
    check_budget()
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=45)
    except requests.exceptions.RequestException as e:
        raise APIError(redact_secrets(str(e)))
        
    if response.status_code == 400 and "character" in response.text.lower():
        raise VehicleReferenceRejectedError("Vehicle reference rejected.")
    if response.status_code == 401 or response.status_code == 403:
        raise AuthorizationError(f"Authentication failed: {response.status_code}")
    if response.status_code == 429:
        raise QuotaError("Quota exceeded or rate limited.")
        
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise APIError(redact_secrets(str(e)))
        
    data = response.json()
    
    base_resp = data.get('base_resp', {})
    if base_resp.get('status_code') == 2056:
        raise QuotaError(f"Quota exceeded: {base_resp.get('status_msg')}")
    if base_resp.get('status_code') != 0:
        raise APIError(f"API Error: {base_resp.get('status_msg')}")
        
    metadata = data.get('metadata', {})
    if str(metadata.get('success_count')) != "1":
        raise APIError(f"Expected exactly 1 success, got metadata: {metadata}")
        
    resp_data = data.get('data', {})
    if not isinstance(resp_data, dict) or 'image_base64' not in resp_data:
        raise APIError("Response data missing 'image_base64' list.")
        
    image_list = resp_data['image_base64']
    if not isinstance(image_list, list) or len(image_list) != 1:
        raise APIError("Expected exactly one image in 'image_base64'.")
        
    base64_str = image_list[0]
    try:
        image_bytes = base64.b64decode(base64_str)
        img_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        raise ImageValidationError("Failed to decode base64 into a valid image.")
        
    if img_np is None:
        raise ImageValidationError("Decoded bytes are not a valid image.")
        
    if img_np.shape[:2] != (HEIGHT, WIDTH):
        raise ImageValidationError(f"Invalid image dimensions: {img_np.shape[:2]}, expected {(HEIGHT, WIDTH)}")
        
    return image_bytes, redact_secrets(data)
