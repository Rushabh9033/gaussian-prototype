import os
import pytest
import numpy as np
import torch
import zipfile
import json
from PIL import Image

from src.python.encoder import encode_image
from src.python.package import create_package, read_package
from src.python.renderer import render_image
from src.python.cli import hash_file

@pytest.fixture
def dummy_image(tmp_path):
    path = tmp_path / "dummy.png"
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]
    Image.fromarray(img).save(path)
    return str(path)

def test_hash_generation(dummy_image):
    h = hash_file(dummy_image)
    assert len(h) == 64
    assert isinstance(h, str)

def test_deterministic_encoding():
    img_t = torch.zeros(3, 32, 32)
    model1, _ = encode_image(img_t, num_gaussians=10, steps=2, seed=42)
    model2, _ = encode_image(img_t, num_gaussians=10, steps=2, seed=42)
    
    pos1 = model1.pos.detach().cpu().numpy()
    pos2 = model2.pos.detach().cpu().numpy()
    
    np.testing.assert_allclose(pos1, pos2)

def test_package_round_trip(tmp_path, dummy_image):
    out_zip = tmp_path / "test.zip"
    
    pos = np.random.rand(10, 2).astype(np.float32)
    scale = np.random.rand(10, 2).astype(np.float32)
    rot = np.random.rand(10).astype(np.float32)
    color = np.random.rand(10, 3).astype(np.float32)
    opacity = np.random.rand(10).astype(np.float32)
    
    metrics = {
        "original_width": 32,
        "original_height": 32,
        "timestamp": "now",
        "source_sha256": "abcdef"
    }
    
    create_package(str(out_zip), pos, scale, rot, color, opacity, 32, 32, metrics, seed=123)
    
    assert out_zip.exists()
    
    manifest, data, read_metrics = read_package(str(out_zip))
    
    assert manifest["gaussian_count"] == 10
    assert manifest["deterministic_seed"] == 123
    assert read_metrics["source_sha256"] == "abcdef"
    
    # Check data shape
    assert data.shape == (10, 9)
    np.testing.assert_allclose(data[:, 0:2], pos)
    np.testing.assert_allclose(data[:, 4], rot)

def test_corrupt_package(tmp_path):
    out_zip = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(out_zip, 'w') as zf:
        zf.writestr("manifest.json", "{}")
        # missing splats.bin
    
    with pytest.raises(KeyError):
        read_package(str(out_zip))

def test_reference_rendering():
    pos = torch.tensor([[16.0, 16.0]])
    scale = torch.tensor([[5.0, 5.0]])
    rot = torch.tensor([0.0])
    color = torch.tensor([[1.0, 0.0, 0.0]])
    opacity = torch.tensor([1.0])
    
    img = render_image(32, 32, pos, scale, rot, color, opacity)
    assert img.shape == (3, 32, 32)
    
    # center should be red
    assert img[0, 16, 16] > 0.9
    assert img[1, 16, 16] < 0.1
