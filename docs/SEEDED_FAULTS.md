# Seeded Faults

## Fault 1: forbidden column added to feature_columns()

**Change**
- return [c for c in numeric if c not in FORBIDDEN]
+ return [c for c in numeric if c not in FORBIDDEN] + ["casual"]

**Red Test Output**
def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_7', 'dayofweek', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
================================================================================== short test summary info ==================================================================================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: forbidden columns in feature set: {'casual'}
1 failed, 6 passed in 0.69s

**Hook Stderr**
The hook fired immediately after the Edit tool call; Claude Code showed
"PostToolUse:Edit hook reported a problem".

PostToolUse:Edit hook reported a problem
[${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh]: Leakage tests failed after editing /Users/margotnissen/Desktop/Capstone/DSC_180A_MA_6/bikeshare-open/src/features.py:
________________________________________________________________________________ test_no_forbidden_features _________________________________________________________________________________

feats =         dteday  instant  season      yr mnth holiday weekday      temp  ...  is_holiday  is_workday     month_sin  mon...    0           1 -2.449294e-16   1.000000               1.0     1796.0     920.0  1531.428571

[714 rows x 28 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_7', 'dayofweek', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
================================================================================== short test summary info ==================================================================================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: forbidden columns in feature set: {'casual'}
1 failed, 6 passed in 0.69s

**Turns to Fix**
The agent fixed it by restoring the file from the last commit (`git checkout src/features.py`),
not by editing the line. It then verified: `git status --porcelain src/` empty,
`git diff --stat` empty, 7/7 leakage tests passed. It did not touch FORBIDDEN
or the test.
