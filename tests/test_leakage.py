"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.

It is name-based, and that is its limit -- it only knows a column is leaky if
someone wrote the name down in src/config.py. test_no_feature_restates_target
and test_known_leaks_never_features close that from two directions: one refits
the arithmetic, one refuses to read config.py.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.config import FORBIDDEN, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw

# A candidate feature set is rejected if cnt is reproducible to within this many
# rides. The known leak reproduces to 3.6e-12 (docs/DECISIONS.md 2026-11-20,
# Step 3) and the best non-leaky single or pair on the current 14 features
# leaves max |residual| = 3936 rides (temp + windspeed), so the threshold sits
# ~6 orders of magnitude above one and ~10 below the other. A precision floor
# on float64 lstsq, not a tuning knob.
RESTATES_TOL = 1e-6

# Deliberately a literal, not `from src.config import LEAKAGE_COLS`. See
# test_known_leaks_never_features for why this must be allowed to duplicate.
KNOWN_LEAKS = {"casual", "registered", "cnt"}


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_known_leaks_never_features(feats):
    """The guard is checked against a hardcoded list, not against FORBIDDEN.

    test_no_forbidden_features reads config.FORBIDDEN, so deleting "casual" from
    LEAKAGE_COLS removes it from the guard in the same edit that lets
    feature_columns() -- which is numeric *minus* FORBIDDEN -- hand it to the
    model. Both sides of the assertion move together and the test stays green.
    That is fault 1 in docs/SEEDED_FAULTS.md, and this is its permanent
    counterpart.

    KNOWN_LEAKS is a literal on purpose, so `casual + registered == cnt` is
    stated in two places that cannot be edited in lockstep: a config change and
    a test change are two commits, not one. This repo's no-hardcoding-twice rule
    is about constants the pipeline imports, not about an expected value inside
    the test that checks the constant.
    """
    cols = set(feature_columns(feats))
    assert not (cols & KNOWN_LEAKS), (
        f"known leaky columns in feature set: {sorted(cols & KNOWN_LEAKS)}; "
        f"they are components of {TARGET}, so a model scored on them has learned "
        f"addition"
    )


def test_no_feature_restates_target(feats):
    """A column, or a pair of them, may not reproduce cnt exactly.

    The arithmetic check, independent of the column-name check above: least
    squares cnt ~ a*x + b over every feature and cnt ~ a1*x1 + a2*x2 + b over
    every pair, then require a real residual. casual/registered are the known
    answer (max |residual| 3.6e-12 over 714 rows), and this is the test that
    notices one nobody wrote down in config.py -- a future "cnt_smoothed", a
    "rides_scaled" a transform step introduces, or a lag someone forgets to
    shift.

    Scope is singles and pairs, which is where the identity this dataset is
    famous for actually lives: casual + registered is a pair. A three-way or
    larger identity would slip through, and no cheaper test than this one would
    catch it either, so the gap is written down rather than papered over.
    """
    cols = feature_columns(feats)
    y = feats[TARGET].to_numpy(dtype=float)
    for group in itertools.chain(
        ((c,) for c in cols), itertools.combinations(cols, 2)
    ):
        design = np.column_stack(
            [*(feats[c].to_numpy(dtype=float) for c in group), np.ones(len(y))]
        )
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = float(np.max(np.abs(design @ coef - y)))
        assert resid > RESTATES_TOL, (
            f"{list(group)} restate {TARGET} exactly: max |residual| {resid:.3g} "
            f"rides over {len(y)} rows (tolerance {RESTATES_TOL:g}). At prediction "
            f"time you would not know any of these for the day being predicted -- "
            f"they are the target. Drop the column (REJECTED_COLS in "
            f"src/config.py) rather than raising RESTATES_TOL."
        )


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_roll_3_excludes_today_hand_computed(feats):
    """cnt_roll_3 on 2012-06-15 == mean(cnt) over 06-12, 06-13, 06-14.

    Hand-computed from the immutable raw CSV (data/raw/bike_daily.csv lines
    527-529), not from src.features: cnt is 4972, 7421, 7363 on those three
    days, so (4972 + 7421 + 7363) / 3 == 19756 / 3 == 6585.333...

    2012-06-15 is a Friday and all three window days are weekdays, so a
    misaligned window cannot pass by coincidence -- one day late would pick
    up Friday's own 7665 and give 7483.0, one day early 6917.666..., against
    6585.333... here.
    """
    assert "cnt_roll_3" in feats.columns, "build_features does not create cnt_roll_3 yet"
    row = feats.loc[feats.dteday == "2012-06-15"]
    assert len(row) == 1, "2012-06-15 must survive the warmup dropna"
    assert row.cnt_roll_3.iloc[0] == pytest.approx(19756 / 3)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique
