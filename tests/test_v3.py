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
        region_size=128,
        levels=5,
        seed_val=123,
        prompt="test prompt",
        test_mode=True
    )
    
    # Verify manifest fields
    assert manifest_add["branch_id"] == "branch-001"
    assert manifest_add["level_count"] == 5
    assert len(manifest_add["levels"]) == 5
    assert manifest_add["root_bbox_pixels"] == [116, 46, 128, 128]  # center - 64
    
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
    
    assert len(data_add) == 5

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
        region_size=128,
        levels=5,
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


def test_v3_validation_rules(tmp_path):
    img_path = str(tmp_path / "src.png")
    Image.new("RGB", (360, 220), color=(100, 100, 100)).save(img_path)
    
    from src.python.package import create_package
    pos = np.zeros((1, 2), dtype=np.float32)
    scale = np.ones((1, 2), dtype=np.float32)
    rot = np.zeros((1,), dtype=np.float32)
    color = np.zeros((1, 3), dtype=np.float32)
    opacity = np.ones((1,), dtype=np.float32)
    v2_zip = str(tmp_path / "v2.zip")
    import hashlib
    with open(img_path, "rb") as f:
        real_hash = hashlib.sha256(f.read()).hexdigest()
    create_package(v2_zip, pos, scale, rot, color, opacity, 360, 220, {"source_sha256": real_hash}, format_version=2.0)
    
    from src.python.generator import generate_branch
    from src.python.package import add_generated_branch, read_package
    
    manifest_add, data_add = generate_branch(
        source_img_path=img_path,
        cx=180,
        cy=110,
        region_size=128,
        levels=5,
        seed_val=123,
        prompt="test",
        test_mode=True
    )
    
    # 1. Test hash mismatch
    bad_img = str(tmp_path / "bad.png")
    Image.new("RGB", (360, 220), color=(50, 50, 50)).save(bad_img)
    with pytest.raises(ValueError, match="Source image hash mismatch"):
        add_generated_branch(v2_zip, str(tmp_path / "out.zip"), manifest_add, data_add, source_img_path=bad_img)
        
    # 2. Test boundary rejection
    with pytest.raises(ValueError, match="Invalid coordinates: Bounding box falls outside source image."):
        generate_branch(
            source_img_path=img_path,
            cx=10, cy=10, region_size=128, levels=5, seed_val=123, prompt="t", test_mode=True
        )

def test_v3_missing_or_corrupted(tmp_path):
    img_path = str(tmp_path / "src.png")
    Image.new("RGB", (360, 220), color=(100, 100, 100)).save(img_path)
    from src.python.package import create_package
    pos = np.zeros((1, 2), dtype=np.float32)
    scale = np.ones((1, 2), dtype=np.float32)
    rot = np.zeros((1,), dtype=np.float32)
    color = np.zeros((1, 3), dtype=np.float32)
    opacity = np.ones((1,), dtype=np.float32)
    v2_zip = str(tmp_path / "v2.zip")
    import hashlib
    with open(img_path, "rb") as f:
        real_hash = hashlib.sha256(f.read()).hexdigest()
    create_package(v2_zip, pos, scale, rot, color, opacity, 360, 220, {"source_sha256": real_hash}, format_version=2.0)
    
    from src.python.generator import generate_branch
    from src.python.package import add_generated_branch, read_package
    manifest_add, data_add = generate_branch(
        source_img_path=img_path,
        cx=180, cy=110, region_size=128, levels=5, seed_val=123, prompt="t", test_mode=True
    )
    v3_zip = str(tmp_path / "v3.zip")
    add_generated_branch(v2_zip, v3_zip, manifest_add, data_add, source_img_path=img_path)
    
    # manually corrupt a file
    bad_zip = str(tmp_path / "v3_bad.zip")
    with zipfile.ZipFile(v3_zip, 'r') as z_in, zipfile.ZipFile(bad_zip, 'w') as z_out:
        for item in z_in.infolist():
            if "level-01.webp" in item.filename:
                z_out.writestr(item, b"corrupted bytes")
            else:
                z_out.writestr(item, z_in.read(item.filename))
                
    with pytest.raises(ValueError, match="Hash mismatch"):
        read_package(bad_zip)
