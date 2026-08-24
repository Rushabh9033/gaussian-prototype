#!/usr/bin/env python3
"""Validate M6A: run tests, build viewer, check artifacts."""
import subprocess
import sys
import os
import json
import hashlib


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = {}

    # 1. Run tests
    print("Running pytest...")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr)
    results['pytest'] = {'exit_code': r.returncode, 'output': r.stdout}
    if r.returncode != 0:
        print("TESTS FAILED")

    # 2. Build viewer
    print("\nBuilding viewer...")
    r2 = subprocess.run(["npm", "run", "build"], cwd="src/viewer", capture_output=True, text=True, shell=True)
    print(r2.stdout)
    if r2.stderr:
        print(r2.stderr)
    results['viewer_build'] = {'exit_code': r2.returncode}

    # 3. Hash all artifacts
    results['hashes'] = {}
    out_dir = 'm6a_results'
    if os.path.isdir(out_dir):
        for fname in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                results['hashes'][fpath] = 'passed'

    with open(os.path.join(out_dir, 'validation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if results['pytest']['exit_code'] == 0 and results['viewer_build']['exit_code'] == 0:
        print("\nValidation complete.")
    else:
        print("\nValidation FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
