import hashlib

def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_seed(source_sha256: str, box_normalized: tuple, zoom_factor: float, version: str) -> int:
    """
    Derive each level seed deterministically from:
    source SHA-256 + normalized bounding box + zoom factor + algorithm version.
    """
    cx, cy, w, h = box_normalized
    seed_str = f"{source_sha256}_{cx:.6f}_{cy:.6f}_{w:.6f}_{h:.6f}_{zoom_factor:.6f}_{version}"
    hash_obj = hashlib.sha256(seed_str.encode('utf-8'))
    # Convert first 8 bytes of hash to an integer seed
    seed = int(hash_obj.hexdigest()[:16], 16) % (2**32 - 1)
    return seed
