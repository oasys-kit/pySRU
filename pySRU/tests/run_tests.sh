#!/bin/bash
#
# Runs all the pySRU tests (the suite defined in AllTests.py).
#
# Usage (from any directory):
#   ./run_tests.sh [python]
#   e.g., ./run_tests.sh ~/venv/oasys2/bin/python
#
# The python interpreter can also be given with the environment variable PYTHON (default: python3).
#
PYTHON=${1:-${PYTHON:-python3}}

# repository root (two levels above this script), so that the local pySRU is imported
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

cd "$ROOT" || exit 1
echo "Running pySRU tests with $($PYTHON --version 2>&1) in $ROOT"
MPLBACKEND=Agg PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m unittest pySRU.tests.AllTests.suite -v "${@:2}"
