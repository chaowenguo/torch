#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
mkdir -p /verifier

cd /workspace

# 运行 pytest 并将结果写回所有可能的奖励路径
if pytest -v /tests/test_behavior.py; then
  echo "1.0" > /logs/verifier/reward.txt
  echo "1.0" > /verifier/reward
else
  echo "0.0" > /logs/verifier/reward.txt
  echo "0.0" > /verifier/reward
fi
