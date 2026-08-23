import pytest
import torch
import os
import zipfile
import numpy as np
from src.python.encoder import encode_image
from src.python.package import create_package, read_package
from src.python.quantize import quantize_gaussians, dequantize_gaussians

def test_v2_quantize_roundtrip():
    target = torch.rand(3, 32, 32)
    model, _ = encode_image(target, 20, 2, strategy='adaptive', seed=42)
    pos, scale, rot, color, opacity = model.get_params()
    pos = pos.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rot = rot.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    
    bin_data, rules = quantize_gaussians(pos, scale, rot, color, opacity, 32, 32)
    assert len(bin_data) == 20 * 14
    
    dequant = dequantize_gaussians(bin_data, rules)
    assert dequant.shape == (20, 9)
    # Check bounds or basic finite
    assert np.isfinite(dequant).all()

def test_v1_backward_compat(tmp_path):
    target = torch.rand(3, 32, 32)
    model, _ = encode_image(target, 20, 2, strategy='adaptive', seed=42)
    pos, scale, rot, color, opacity = model.get_params()
    pos = pos.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rot = rot.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    
    zip_path = str(tmp_path / "v1.zip")
    create_package(zip_path, pos, scale, rot, color, opacity, 32, 32, {}, format_version=1.0)
    
    manifest, data, metrics = read_package(zip_path)
    assert manifest["format_version"] == "1.0"
    assert data.shape == (20, 9)

def test_v2_manifest_and_layers(tmp_path):
    target = torch.rand(3, 32, 32)
    model, _ = encode_image(target, 20, 2, strategy='adaptive', seed=42)
    pos, scale, rot, color, opacity = model.get_params()
    pos = pos.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rot = rot.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    
    zip_path = str(tmp_path / "v2.zip")
    create_package(zip_path, pos, scale, rot, color, opacity, 32, 32, {}, format_version=2.0)
    
    manifest, data_all, metrics = read_package(zip_path, layers="all")
    assert manifest["format_version"] == "2.0"
    assert manifest["total_gaussian_count"] == 20
    assert manifest["record_stride"] == 14
    assert manifest["byte_order"] == "little-endian"
    assert "quantization_rules" in manifest
    assert len(manifest["layers"]) == 2
    assert manifest["layers"][0]["gaussian_count"] == 12
    assert manifest["layers"][1]["gaussian_count"] == 8
    assert data_all.shape == (20, 9)
    
    # Check base only
    manifest_base, data_base, _ = read_package(zip_path, layers="base")
    assert data_base.shape == (12, 9)

def test_v2_exceptions(tmp_path):
    target = torch.rand(3, 32, 32)
    model, _ = encode_image(target, 20, 2, strategy='adaptive', seed=42)
    pos, scale, rot, color, opacity = model.get_params()
    pos = pos.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rot = rot.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    
    zip_path = str(tmp_path / "v2_bad.zip")
    
    # Test NaN pos
    pos_bad = pos.copy()
    pos_bad[0, 0] = float('nan')
    with pytest.raises(ValueError, match="NaN or Infinity in pos, rot, color, or opacity"):
        create_package(zip_path, pos_bad, scale, rot, color, opacity, 32, 32, {}, format_version=2.0)
        
    # Test zero scale
    scale_bad = scale.copy()
    scale_bad[0, 0] = 0.0
    with pytest.raises(ValueError, match="Scale must be strictly positive"):
        create_package(zip_path, pos, scale_bad, rot, color, opacity, 32, 32, {}, format_version=2.0)
        
    # Test negative scale
    scale_bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="Scale must be strictly positive"):
        create_package(zip_path, pos, scale_bad, rot, color, opacity, 32, 32, {}, format_version=2.0)

    # Test Inf scale
    scale_bad[0, 0] = float('inf')
    with pytest.raises(ValueError, match="NaN or Infinity in scale"):
        create_package(zip_path, pos, scale_bad, rot, color, opacity, 32, 32, {}, format_version=2.0)

def test_v2_deterministic(tmp_path):
    target = torch.rand(3, 32, 32)
    model1, _ = encode_image(target, 20, 2, strategy='adaptive', seed=42)
    pos1, scale1, rot1, color1, opacity1 = model1.get_params()
    pos1 = pos1.detach().cpu().numpy()
    scale1 = scale1.detach().cpu().numpy()
    rot1 = rot1.detach().cpu().numpy()
    color1 = color1.detach().cpu().numpy()
    opacity1 = opacity1.detach().cpu().numpy()
    z1 = str(tmp_path / "1.zip")
    create_package(z1, pos1, scale1, rot1, color1, opacity1, 32, 32, {}, seed=42, format_version=2.0)
    
    model2, _ = encode_image(target, 20, 2, strategy='adaptive', seed=42)
    pos2, scale2, rot2, color2, opacity2 = model2.get_params()
    pos2 = pos2.detach().cpu().numpy()
    scale2 = scale2.detach().cpu().numpy()
    rot2 = rot2.detach().cpu().numpy()
    color2 = color2.detach().cpu().numpy()
    opacity2 = opacity2.detach().cpu().numpy()
    z2 = str(tmp_path / "2.zip")
    create_package(z2, pos2, scale2, rot2, color2, opacity2, 32, 32, {}, seed=42, format_version=2.0)
    
    with open(z1, "rb") as f1, open(z2, "rb") as f2:
        assert f1.read() == f2.read()
