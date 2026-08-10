#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
cd /workspace

# Run the test script and pipe through parser.py
bash /tests/run_script.sh
