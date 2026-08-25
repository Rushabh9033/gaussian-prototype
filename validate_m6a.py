#!/usr/bin/env python3
"""Validate M6A: run tests, build viewer, check artifacts."""
import subprocess
import sys
import os
import json
import hashlib

def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def main():
    results = {}
    validation_passed = True

    # 1. Run tests
    print("Running pytest...")
    cmd_pytest = [sys.executable, "-m", "pytest", "tests", "-v"]
    r = subprocess.run(cmd_pytest, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr)
    results['pytest'] = {
        'command': " ".join(cmd_pytest),
        'exit_code': r.returncode,
        'stdout': r.stdout,
        'stderr': r.stderr
    }
    if r.returncode != 0:
        print("TESTS FAILED")
        validation_passed = False

    # 2. Build viewer
    print("\nBuilding viewer...")
    cmd_npm = ["npm", "run", "build"]
    r2 = subprocess.run(cmd_npm, cwd="src/viewer", capture_output=True, text=True, shell=True)
    print(r2.stdout)
    if r2.stderr:
        print(r2.stderr)
    results['viewer_build'] = {
        'command': "cd src/viewer && npm run build",
        'exit_code': r2.returncode,
        'stdout': r2.stdout,
        'stderr': r2.stderr
    }
    if r2.returncode != 0:
        print("VIEWER BUILD FAILED")
        validation_passed = False

    # 3. Read evidence
    out_dir = 'm6a_results'
    evidence_path = os.path.join(out_dir, 'evidence.jsonl')
    recorded_commits = set()
    expected_artifacts = {}
    evidence_malformed = False

    if not os.path.isfile(evidence_path):
        print("No evidence.jsonl found!")
        validation_passed = False
    else:
        try:
            with open(evidence_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    if 'git_commit' in record:
                        recorded_commits.add(record['git_commit'])
                    if 'artifacts' in record:
                        for art in record['artifacts']:
                            path = art['path']
                            if path in expected_artifacts:
                                print(f"Duplicate artifact record: {path}")
                                validation_passed = False
                            expected_artifacts[path] = art
        except Exception as e:
            print(f"Malformed evidence: {e}")
            evidence_malformed = True
            validation_passed = False

    # 4. Hash all artifacts
    results['artifact_verification'] = {}
    actual_artifacts = set()
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            if fname in ['evidence.jsonl', 'validation.json', 'results.json', 'provenance.json']:
                continue
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                actual_artifacts.add(fname)
                actual_hash = sha256_file(fpath)
                actual_size = os.path.getsize(fpath)
                
                res = {'actual_hash': actual_hash, 'actual_size': actual_size, 'status': 'passed'}
                if fname not in expected_artifacts:
                    res['status'] = 'unexpected'
                    validation_passed = False
                else:
                    exp = expected_artifacts[fname]
                    res['expected_hash'] = exp['sha256']
                    res['expected_size'] = exp['size']
                    if actual_hash != exp['sha256'] or actual_size != exp['size']:
                        res['status'] = 'mismatch'
                        validation_passed = False
                results['artifact_verification'][fname] = res

    for fname, exp in expected_artifacts.items():
        if fname not in actual_artifacts:
            results['artifact_verification'][fname] = {
                'expected_hash': exp['sha256'],
                'expected_size': exp['size'],
                'status': 'missing'
            }
            validation_passed = False

    if evidence_malformed:
        results['artifact_verification']['evidence'] = 'malformed'

    # 5. Git Commit Verification
    results['git_verification'] = {}
    for commit in recorded_commits:
        r_git = subprocess.run(["git", "cat-file", "-e", commit], capture_output=True)
        if r_git.returncode == 0:
            results['git_verification'][commit] = 'reachable'
        else:
            results['git_verification'][commit] = 'unreachable'
            validation_passed = False
            print(f"Commit {commit} is unreachable!")

    results['final_status'] = 'PASSED' if validation_passed else 'FAILED'

    with open(os.path.join(out_dir, 'validation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if validation_passed:
        print("\nValidation complete. PASSED.")
    else:
        print("\nValidation FAILED.")
        sys.exit(1)

if __name__ == '__main__':
    main()
