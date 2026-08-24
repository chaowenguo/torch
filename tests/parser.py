import sys
import re

def parse_output(output_str: str) -> float:
    # Look for pytest summary line like "3 passed, 1 failed in 0.05s"
    passed_match = re.search(r'(\d+)\s+passed', output_str)
    failed_match = re.search(r'(\d+)\s+failed', output_str)
    
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    
    total = passed + failed
    if total == 0:
        return 0.0
        
    # Graded bounded continuous reward (fraction passed)
    return float(passed) / float(total)

if __name__ == "__main__":
    output = sys.stdin.read()
    score = parse_output(output)
    
    # Write the reward to the standard verifier path
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(f"{score:.2f}")
    
    print(f"Calculated reward: {score:.2f}")
