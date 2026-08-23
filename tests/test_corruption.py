import pytest
import torch
import os
import zipfile
import json
from src.python.encoder import encode_image
from src.python.package import create_package, read_package

def test_v2_corruption(tmp_path):
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
    
    # Missing detail layer
    missing_path = str(tmp_path / "v2_missing.zip")
    with zipfile.ZipFile(zip_path, 'r') as zin, zipfile.ZipFile(missing_path, 'w') as zout:
        for item in zin.infolist():
            if "detail.bin" not in item.filename:
                zout.writestr(item, zin.read(item.filename))
                
    with pytest.raises(ValueError, match="Detail layer not found"):
        read_package(missing_path, layers="all")
        
    # Corrupt checksum
    corrupt_path = str(tmp_path / "v2_corrupt.zip")
    with zipfile.ZipFile(zip_path, 'r') as zin, zipfile.ZipFile(corrupt_path, 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if "base.bin" in item.filename:
                data = data[:-1] + b'\x00' # corrupt last byte
            zout.writestr(item, data)
            
    with pytest.raises(ValueError, match="Checksum mismatch"):
        read_package(corrupt_path, layers="base")

def test_v1_corruption(tmp_path):
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
    
    corrupt_path = str(tmp_path / "v1_corrupt.zip")
    with zipfile.ZipFile(zip_path, 'r') as zin, zipfile.ZipFile(corrupt_path, 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if "splats.bin" in item.filename:
                data = data[:-4] # truncate 4 bytes
            zout.writestr(item, data)
            
    with pytest.raises(ValueError, match="length mismatch"):
        read_package(corrupt_path)
