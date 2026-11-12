#!/bin/bash
# Verifier for the evolve-harness task. Runs in the same container as the
# agent. Writes reward + ctrf to /logs/verifier/. The edited workspace itself
# travels back to the host via harbor's `artifacts` mechanism (declared in
# task.toml), regardless of reward — this script only grades.

mkdir -p /logs/verifier

python -m pytest /tests/test_evolution.py --ctrf /logs/verifier/ctrf.json -rA
code=$?

if [ $code -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
