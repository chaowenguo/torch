import sys
import json

def parse_output(output_str: str) -> int:
    # Basic parser checking pytest summary output
    if "PASSED" in output_str and "FAILED" not in output_str:
        return 1
    return 0

if __name__ == "__main__":
    output = sys.stdin.read()
    print(parse_output(output))
