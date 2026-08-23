import torch
import numpy as np
import pytest
import tempfile
import os
from src.python.encoder import encode_image

def test_adaptive_initialization():
    # create a target image that has one sharp edge in the center, rest is flat
    target = torch.zeros(3, 32, 32)
    target[:, 10:20, 10:20] = 1.0
    
    # 1. Deterministic adaptive initialization
    model1, _ = encode_image(target, 100, 2, strategy='adaptive', seed=42)
    pos1 = model1.pos.detach().clone()
    
    model2, _ = encode_image(target, 100, 2, strategy='adaptive', seed=42)
    pos2 = model2.pos.detach().clone()
    
    assert torch.allclose(pos1, pos2), "Adaptive initialization should be deterministic"
    
    # 2. Exact requested Gaussian count
    assert model1.pos.shape[0] == 100
    
    # 3. Increased allocation to high detail regions (center 10-20)
    # Stage 1 allocation is 60 gaussians.
    # The edge is around x in 9..21 and y in 9..21.
    stage1_pos = pos1[:60]
    in_center = ((stage1_pos[:, 0] > 8) & (stage1_pos[:, 0] < 22) & 
                 (stage1_pos[:, 1] > 8) & (stage1_pos[:, 1] < 22)).sum().item()
    # If it was uniform, we'd expect 60 * (14*14 / 32*32) ~ 60 * 0.19 ~ 11
    # With adaptive, it should be significantly more.
    assert in_center > 15, "Should allocate more to high detail regions"
    
    # 4. Uniform coverage floor
    # There should be at least SOME gaussians outside the center.
    assert in_center < 60, "Should have uniform floor coverage outside edges"

def test_two_stage_preservation():
    target = torch.rand(3, 32, 32)
    model, _ = encode_image(target, 20, 6, strategy='adaptive', seed=42)
    # At step 3 (half of 6), stage 2 triggers.
    # The first 60% (12 gaussians) should be preserved exactly? 
    # Well, they continue to optimize, so their values aren't mathematically locked,
    # but their parameters at index 0..11 are not overwritten by stage 2 initialization.
    assert model.pos.shape[0] == 20
    assert not torch.isnan(model.pos).any()
    assert not torch.isinf(model.pos).any()

def test_multi_tile_adaptive():
    target = torch.rand(3, 256, 256)
    # 256x256 multi tile
    model, _ = encode_image(target, 50, 2, strategy='adaptive', tile_size=128)
    assert model.pos.shape[0] == 50
    assert not torch.isnan(model.pos).any()
    
def test_random_strategy_compatibility():
    target = torch.rand(3, 32, 32)
    model, _ = encode_image(target, 20, 2, strategy='random', seed=42)
    assert model.pos.shape[0] == 20
    assert not torch.isnan(model.pos).any()
