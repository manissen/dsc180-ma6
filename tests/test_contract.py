"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import (
    FORBIDDEN,
    LAG_WARMUP,
    RAW,
    RAW_SHA256,
    SOURCE_COLS,
    TARGET,
    UNUSED_COLS,
    sha256_file,
)
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw
from src.train import MODELS, temporal_split

# Raw columns that are features as-is, with no derivation. Pinned here rather
# than in src/config.py because it is an expected value, which is what a test
# asserts and not what the pipeline imports. See test_only_audited_raw_columns_
# reach_the_model for why the classification check alone cannot supply it.
RAW_PASSTHROUGH_FEATURES = {"temp", "atemp", "hum", "windspeed"}


def unclassified(columns, feature_cols):
    """Raw columns that no bucket of the audit accounts for.

    `columns` is the raw CSV's header, `feature_cols` the pipeline's output. A
    column is accounted for if it is forbidden, is used as a feature directly,
    is raw material build_features derives from, is present but read by nothing
    (UNUSED_COLS), or is the date index. Anything else is a column nobody has
    looked at -- and because feature_columns() subtracts FORBIDDEN from every
    numeric column, an unlooked-at *numeric* column is already in the model.

    Returns a list, in input order, so a failure names the columns in the order
    they appear in the CSV.
    """
    known = (
        set(FORBIDDEN)
        | set(feature_cols)
        | set(SOURCE_COLS)
        | set(UNUSED_COLS)
        | {"dteday"}
    )
    return [c for c in columns if c not in known]



def test_raw_data_unchanged():
    """data/raw is immutable; permission rules bind only Edit/Write, and a Bash
    cp or `python -m src.ingest --force` bypasses them, so this catches any
    modification.

    Byte-level, so it also catches a re-download that is *almost* the same CSV:
    a drifted row is invisible to load_raw()'s shape checks and would silently
    change every metric. The hash is the only place the identity of the raw data
    is pinned, so it is the thing to fix if the dataset is deliberately
    replaced -- see docs/DECISIONS.md for the convention.
    """
    assert sha256_file(RAW) == RAW_SHA256, (
        f"{RAW} has changed; expected sha256 {RAW_SHA256}. Restore it with "
        f"`git checkout -- {RAW.relative_to(RAW.parent.parent)}`."
    )


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_every_raw_column_is_classified(raw, feats):
    """A column must be accounted for before it can reach the model.

    feature_columns() subtracts FORBIDDEN from the numeric columns, so anything
    the mirror adds is a feature by default -- a new numeric column enters the
    model on the same commit that downloads it, and nothing downstream says
    so. This makes the audit total: every column in data/raw/bike_daily.csv
    must be in exactly one of the five buckets, so a new one fails here with
    its name in the message.

    Companion to test_raw_data_unchanged: that pins the file's bytes, this
    pins what the pipeline is allowed to conclude from them. A deliberate
    dataset change is a RAW_SHA256 edit plus a new bucket entry with a reason.
    """
    assert unclassified(raw.columns, feature_columns(feats)) == []


def test_unclassified_flags_a_new_column(raw, feats):
    """Proves the check above can fail, without touching data/raw.

    A classification test whose helper is wrong -- a typo'd set name, a `|`
    that should have been an `&` -- passes for the current columns and stays
    green forever. This feeds the helper a column that is in no bucket and
    requires it to come back. Test 3 alone cannot distinguish "everything is
    classified" from "nothing is ever unclassified".
    """
    assert unclassified([*raw.columns, "new_col"], feature_columns(feats)) == [
        "new_col"
    ]


def test_only_audited_raw_columns_reach_the_model(raw, feats):
    """Which raw columns are model features must be an audited list, not an
    accident of dtype.

    test_every_raw_column_is_classified cannot catch a new *numeric* column on
    its own: feature_columns() classifies it as a feature the moment it
    appears, so it lands in a bucket and the audit passes. The hole is closed
    by checking the feature set against an expected list instead.

    Deliberately minimal -- raw passthroughs only, not the full feature set.
    The derived features (lags, is_weekend, month_sin) are each pinned by
    name in tests/test_leakage.py, and re-listing them here would mean two
    places to update for every feature added or dropped. What this pins is the
    set feature_columns() derives from the raw frame by dtype alone.
    """
    passthrough = set(feature_columns(feats)) & set(raw.columns)
    assert passthrough == RAW_PASSTHROUGH_FEATURES, (
        f"raw columns used as features unchanged: {sorted(passthrough)}; "
        f"a new one needs a reason before it needs a bucket"
    )


def test_beats_the_mean_baseline(feats):
    """A model that cannot beat predicting the training mean is not a result."""
    cols = feature_columns(feats)
    train, _, test = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(test[TARGET], ridge.predict(test[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS
