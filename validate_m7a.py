import sys
import os
import json
import hashlib
import re

def validate_m7a():
    print("M7A Validation script starting...")
    
    legacy_ev_path = os.path.join("milestone7a_artifacts", "failed_attempts", "legacy_evidence.jsonl")
    legacy_res_path = os.path.join("milestone7a_artifacts", "failed_attempts", "legacy_results.json")
    
    if not os.path.exists(legacy_ev_path) or not os.path.exists(legacy_res_path):
        print("ERROR: Historical legacy evidence missing.")
        return 1
        
    ev_hash = hashlib.sha256(open(legacy_ev_path, 'rb').read()).hexdigest()
    res_hash = hashlib.sha256(open(legacy_res_path, 'rb').read()).hexdigest()
    
    if ev_hash != "e2a908fbee830d9af36bc5244b6e039842c297cb2c9d792d8ea358f4877cf470":
        print("ERROR: hashes mismatch for evidence")
        return 1
    if res_hash != "cd7ebe79a9608cfa4fd09cec31cd53bacbfa5dd4aae749da1c205b8e3fb5b7ee":
        print("ERROR: hashes mismatch for results")
        return 1
        
    with open(legacy_ev_path, 'r') as f:
        legacy_lines = f.readlines()
        
    if len(legacy_lines) != 5:
        print("ERROR: five historical records do not exist")
        return 1
        
    blocked_count = 0
    exec_records = 0
    total_calls = 0
    total_images = 0
    
    for line in legacy_lines:
        record = json.loads(line)
        if record.get("status") == "BLOCKED_MISSING_API_KEY":
            blocked_count += 1
        else:
            calls = record.get("calls_made", 0)
            if calls != 4:
                print("ERROR: execution record did not report exactly four attempted calls")
                return 1
            exec_records += 1
            total_calls += calls
            
            results = record.get("results", {})
            for scale, scale_res in results.items():
                for seed, seed_res in scale_res.items():
                    if "error" not in seed_res:
                        total_images += 1
                    else:
                        pass # error is present, which is expected

    if blocked_count != 1:
        print("ERROR: exactly one BLOCKED_MISSING_API_KEY record not found")
        return 1
    if exec_records != 4:
        print("ERROR: exactly four execution records not found")
        return 1
    if total_calls != 16:
        print("ERROR: total historical attempted calls do not equal 16")
        return 1
    if total_images != 0:
        print("ERROR: total successful generated images not equal 0")
        return 1
        
    if os.path.exists(os.path.join("milestone7a_artifacts", "evidence.jsonl")):
        print("ERROR: active evidence.jsonl exists")
        return 1
    if os.path.exists(os.path.join("milestone7a_artifacts", "results.json")):
        print("ERROR: active results.json exists")
        return 1

    report_path = "MILESTONE7A_REPORT.md"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
            
            # Require exactly: **Final Status:** `BLOCKED_API_ACCESS`
            match = re.search(r'\*\*Final Status:\*\*\s*`BLOCKED_API_ACCESS`', report_text)
            if not match:
                print("ERROR: Final status exact text not found in report")
                return 1
                
            if "16 historical attempts" not in report_text:
                print("ERROR: report does not state 16 historical attempts")
                return 1
            if "0 successful images" not in report_text and "Zero images were successfully generated" not in report_text and "Successful Image Count:** 0" not in report_text:
                print("ERROR: report does not state 0 successful images")
                return 1
            if "cost is unknown" not in report_text and "Cost:** Unknown" not in report_text:
                print("ERROR: report does not state confirmed cost is unknown")
                return 1

    print("M7A Validation passed. Honest status confirmed: BLOCKED_API_ACCESS")
    return 0

if __name__ == '__main__':
    sys.exit(validate_m7a())
