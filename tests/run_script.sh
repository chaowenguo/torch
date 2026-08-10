set -uo pipefail

mkdir -p /logs/verifier/rewards
cd /workspace
if pytest -v /tests/test_behavior.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
