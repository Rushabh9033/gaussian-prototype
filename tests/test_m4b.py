import json
import os
from PIL import Image

def test_m4b_status_is_valid():
    """Verify that M4B results reflect a valid state and no mocks are present."""
    results_path = "m4b_results.json"
    assert os.path.exists(results_path), "m4b_results.json must exist"
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    valid_statuses = ["NEEDS_USER_VISUAL_CHECK", "NO_CANDIDATE_PASSED", "BLOCKED"]
    assert data["overall_status"] in valid_statuses, f"Invalid overall status: {data['overall_status']}"
    
    assert "A" in data["candidates"], "Candidate A must be in results"
    assert "B" in data["candidates"], "Candidate B must be in results"
    assert "C" in data["candidates"], "Candidate C must be in results"
    
    if data["overall_status"] == "BLOCKED":
        assert not os.path.exists("m4b_results/candidate_c/level_1.png"), "No mock images should be generated when blocked"
        assert not os.path.exists("porche_10k_generated_v3.zip"), "Stale V3 zip should not be restored"
        
    if data["overall_status"] == "NEEDS_USER_VISUAL_CHECK":
        out_dir = "m4b_results/candidate_c"
        for i in range(1, 6):
            img_path = os.path.join(out_dir, f"level_{i}.png")
            assert os.path.exists(img_path), f"Missing level_{i}.png"
            with Image.open(img_path) as img:
                assert img.size == (512, 512), f"Image level_{i}.png must be 512x512, got {img.size}"
        assert os.path.exists(os.path.join(out_dir, "contact_sheet.png")), "Missing contact_sheet.png"
        assert os.path.exists(os.path.join(out_dir, "metrics.json")), "Missing metrics.json"
        assert os.path.exists(os.path.join(out_dir, "prompts.json")), "Missing prompts.json"
        assert os.path.exists(os.path.join(out_dir, "provenance.json")), "Missing provenance.json"
        
        with open(os.path.join(out_dir, "metrics.json"), "r") as f:
            metrics = json.load(f)
            assert "level_2" in metrics, "Missing metrics for level 2"
            assert "edge_iou" in metrics["level_2"] or "edge_f1" in metrics["level_2"], "Metrics must contain edge_iou or edge_f1"
