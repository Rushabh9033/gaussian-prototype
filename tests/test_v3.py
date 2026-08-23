import pytest
import os
import json
import zipfile
import torch
import numpy as np
from PIL import Image

from src.python.generator import generate_branch
from src.python.package import read_package, add_generated_branch

def test_v3_branch_generation(tmp_path):
    # create dummy source image
    img_path = str(tmp_path / "src.png")
    Image.new("RGB", (360, 220), color=(100, 100, 100)).save(img_path)
    
    # run test-mode generation
    manifest_add, data_add = generate_branch(
        source_img_path=img_path,
        cx=180,
        cy=110,
        region_size=50,
        levels=3,
        seed_val=123,
        prompt="test prompt",
        test_mode=True
    )
    
    # Verify manifest fields
    assert manifest_add["branch_id"] == "branch-001"
    assert manifest_add["level_count"] == 3
    assert len(manifest_add["levels"]) == 3
    assert manifest_add["root_bbox_pixels"] == [130, 60, 100, 100]  # center - size, size * 2
    
    # Check level 1
    l1 = manifest_add["levels"][0]
    assert l1["level"] == 1
    assert l1["cumulative_zoom"] == 4
    
    # Check level 2
    l2 = manifest_add["levels"][1]
    assert l2["level"] == 2
    assert l2["cumulative_zoom"] == 16
    assert l2["parent_hash"] == l1["sha256"]
    
    # Verify exact 4x zoom math is respected in root_bbox
    assert np.isclose(l2["root_bbox"][2] * 4, l1["root_bbox"][2])
    
    assert len(data_add) == 3

def test_v3_package_integration(tmp_path):
    img_path = str(tmp_path / "src.png")
    Image.new("RGB", (360, 220), color=(100, 100, 100)).save(img_path)
    
    # create dummy v2 package
    from src.python.package import create_package
    pos = np.zeros((10, 2), dtype=np.float32)
    scale = np.ones((10, 2), dtype=np.float32)
    rot = np.zeros((10,), dtype=np.float32)
    color = np.zeros((10, 3), dtype=np.float32)
    opacity = np.ones((10,), dtype=np.float32)
    
    v2_zip = str(tmp_path / "v2.zip")
    create_package(v2_zip, pos, scale, rot, color, opacity, 360, 220, {"psnr": 20.0, "ssim": 0.5}, format_version=2.0)
    
    manifest_add, data_add = generate_branch(
        source_img_path=img_path,
        cx=180,
        cy=110,
        region_size=50,
        levels=2,
        seed_val=123,
        prompt="test",
        test_mode=True
    )
    
    v3_zip = str(tmp_path / "v3.zip")
    add_generated_branch(v2_zip, v3_zip, manifest_add, data_add, source_img_path=img_path)
    
    # Read V3
    manifest, data, metrics = read_package(v3_zip)
    assert manifest["format_version"] == "3.0"
    assert "generated_branches" in manifest
    assert len(manifest["generated_branches"]) == 1
    assert "base_pre_quantization" in metrics
    assert "generated_branch" in metrics
    assert "v2_decoded_base" in metrics
    assert "v2_decoded_full" in metrics
    
    # check zip contents
    with zipfile.ZipFile(v3_zip, 'r') as z:
        names = z.namelist()
        assert "generated_branches/branch-001/level-01.webp" in names
        assert "generated_branches/branch-001/level-02.webp" in names

