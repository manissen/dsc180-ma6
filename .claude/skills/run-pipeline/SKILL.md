---
name: run-pipeline
description: Run one pipeline stage (ingest, features, train, or all) and verify its output against the contract.
argument-hint: "[ingest|features|train|all]"
---

Run pipeline stage: $ARGUMENTS (default: all)

Current committed metrics:
!`cat reports/metrics.json 2>/dev/null`

Steps:
1. Run `make $ARGUMENTS` (use `make all` if no argument). Show the real output.
2. Verify against the contract and STOP on any mismatch:
   - ingest: 728 rows, 2011-01-03 -> 2012-12-31
   - features: 714 rows; no column from src/config.FORBIDDEN in the feature list
   - train/all: n_train 348, n_valid 182, n_test 184; both baselines present
3. Run `make test` and paste the real output.
4. If train/all ran, run `git diff --stat HEAD -- reports/metrics.json` and report
   whether metrics changed vs the last commit. Do not edit reports/metrics.json by hand.
5. Report only numbers you printed. Never compare models on the test split.
