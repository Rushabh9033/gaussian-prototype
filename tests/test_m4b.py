import json
import os

def test_m4b_status_is_valid():
    """Verify that M4B results reflect a valid state and no mocks are present."""
    results_path = "m4b_results.json"
    assert os.path.exists(results_path), "m4b_results.json must exist"
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    valid_statuses = ["NEEDS_USER_VISUAL_CHECK", "NO_CANDIDATE_PASSED", "BLOCKED"]
    assert data["overall_status"] in valid_statuses, f"Invalid overall status: {data['overall_status']}"
    
    # If blocked, we shouldn't have mock contact sheets
    if data["overall_status"] == "BLOCKED":
        assert not os.path.exists("m4b_results"), "No mock images should be generated when blocked"
        assert not os.path.exists("porche_10k_generated_v3.zip"), "Stale V3 zip should not be restored"
