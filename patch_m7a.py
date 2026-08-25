import sys
import re

with open("run_m7a_benchmark.py", "r") as f:
    content = f.read()

import_argparse = "import argparse\n"
if "import argparse" not in content:
    content = content.replace("import sys\n", "import sys\n" + import_argparse)

new_main_start = """def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-api", action="store_true", help="Attempt to execute live API generation")
    parser.add_argument("--max-calls", type=int, default=4, help="Maximum authorized API calls")
    args = parser.parse_args()

    if args.execute_api:
        print("Milestone 7A API budget exhausted; additional requests are not authorized.")
        sys.exit(1)

    print("=== Milestone 7A: MiniMax Suitability Gate ===")"""

content = re.sub(r'def main\(\):\n    print\("=== Milestone 7A: MiniMax Suitability Gate ==="\)', new_main_start, content)

with open("run_m7a_benchmark.py", "w") as f:
    f.write(content)
