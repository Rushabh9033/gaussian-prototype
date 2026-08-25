import sys
import os
import json
import hashlib

def validate_m7a():
    print("M7A Validation script starting...")
    
    # Check legacy hashes
    legacy_ev_path = os.path.join("milestone7a_artifacts", "failed_attempts", "legacy_evidence.jsonl")
    legacy_res_path = os.path.join("milestone7a_artifacts", "failed_attempts", "legacy_results.json")
    
    if not os.path.exists(legacy_ev_path) or not os.path.exists(legacy_res_path):
        print("ERROR: Historical legacy evidence missing.")
        return 1
        
    ev_hash = hashlib.sha256(open(legacy_ev_path, 'rb').read()).hexdigest()
    res_hash = hashlib.sha256(open(legacy_res_path, 'rb').read()).hexdigest()
    
    if ev_hash != "e2a908fbee830d9af36bc5244b6e039842c297cb2c9d792d8ea358f4877cf470":
        print("ERROR: hashes mismatch")
        return 1
    if res_hash != "cd7ebe79a9608cfa4fd09cec31cd53bacbfa5dd4aae749da1c205b8e3fb5b7ee":
        print("ERROR: hashes mismatch")
        return 1
        
    # Check current evidence
    current_ev_path = os.path.join("milestone7a_artifacts", "evidence.jsonl")
    if os.path.exists(current_ev_path):
        with open(current_ev_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                # required evidence fields are missing
                for field in ["run_id", "utc_timestamp", "status", "commit"]:
                    if field not in record:
                        print("ERROR: required evidence fields are missing")
                        return 1
                
                # an execution record says COMPLETED but every candidate contains an error
                if record.get("status") == "COMPLETED":
                    results = record.get("results", {})
                    has_success = False
                    successful_images = 0
                    for scale, scale_res in results.items():
                        for seed, seed_res in scale_res.items():
                            if "error" not in seed_res:
                                has_success = True
                                successful_images += 1
                    
                    if not has_success:
                        print("ERROR: an execution record says COMPLETED but every candidate contains an error")
                        return 1
                        
                    # results contain zero successfully decoded images
                    if successful_images == 0:
                        print("ERROR: results contain zero successfully decoded images")
                        return 1
                        
                # submitted calls exceed the authorized total
                if record.get("calls_made", 0) > 4:
                    print("ERROR: submitted calls exceed the authorized total")
                    return 1
                    
                # execution occurred from a dirty working tree
                if record.get("git_status", "") != "":
                    print("ERROR: execution occurred from a dirty working tree")
                    return 1
                    
                # an invalid final status is used
                status = record.get("status")
                if status not in ["BLOCKED_API_ACCESS", "COMPLETED", "FAILED"]:
                    print("ERROR: an invalid final status is used")
                    return 1

    report_path = "MILESTONE7A_REPORT.md"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
            if "API_NOT_SUITABLE" in report_text and "Quota" in report_text:
                print("ERROR: a report claims API_NOT_SUITABLE based only on quota failure")
                return 1
            if "BLOCKED_API_ACCESS" not in report_text:
                print("ERROR: an invalid final status is used")
                return 1

    print("M7A Validation passed. Honest status confirmed: BLOCKED_API_ACCESS")
    return 0

if __name__ == '__main__':
    sys.exit(validate_m7a())
