#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

# Run pytest capturing stdout/stderr and pipe it through parser.py
pytest -v /tests/test_behavior.py 2>&1 | python3 /tests/parser.py
