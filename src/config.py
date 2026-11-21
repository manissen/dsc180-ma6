"""Single source of truth for paths, columns, and split dates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bike_daily.csv"
PROCESSED = ROOT / "data" / "processed" / "features.parquet"
METRICS = ROOT / "reports" / "metrics.json"

# Daily Capital Bikeshare counts (Fanaee-T & Gama). Canonical home:
#   https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
# We use the processed copy from Molnar's "Interpretable Machine Learning"
# because the UCI host blocks some campus networks. It differs from UCI:
#   - 728 rows, 2011-01-03 to 2012-12-31 (UCI has 731 from 2011-01-01)
#   - 2011-03-10 is absent (the UCI row records humidity 0, a sensor
#     artefact, and the mirror's cleaning dropped the whole day)
#   - `weather` is the labels GOOD / MISTY / BAD, not UCI's codes 1-4
#   - two extra columns: `days_since_2011` and `cnt_2d_bfr`
DATA_URL = (
    "https://raw.githubusercontent.com/christophM/"
    "interpretable-ml-book/master/data/bike.csv"
)

TARGET = "cnt"

# --- Four reasons a column is kept out of the feature set --------------

# 1. Leakage: the target or an arithmetic restatement of it.
#    casual + registered == cnt exactly, on every row. A model handed these
#    scores R^2 ~ 1.0 and has learned addition. At prediction time you would
#    not know either input: they are components of the thing being predicted.
LEAKAGE_COLS = {"casual", "registered", "cnt"}

# 2. Time indexes: not leakage, but a bad bet. A tree splits on "after date
#    X" and predicts a flat constant for every future date; a linear model
#    extrapolates a two-year trend we have no reason to trust. Excluding
#    them keeps the model comparison about what each does with lagged demand.
TIME_INDEX_COLS = {"instant", "yr", "days_since_2011"}

# 3. Unverified: `cnt_2d_bfr` is a two-day lag of cnt that shipped with the
#    mirror. It is NOT leakage -- a two-day-old count is known at prediction
#    time -- but we did not build it and cannot test how it was aligned, so we
#    build our own lags from the target and test those instead.
UNVERIFIED_COLS = {"cnt_2d_bfr"}

# 4. Rejected on evidence, not on principle. The column is safe -- it passes
#    the same availability argument as the lags that stayed -- but it does not
#    earn its place, so it is excluded rather than deleted: the feature code
#    and the hand-computed leakage test in tests/test_leakage.py both remain,
#    so the negative result is reproducible and cannot be re-litigated later
#    by someone who assumes nobody tried it.
#    cnt_roll_3: built and tested, excluded after validation R^2 moved
#    ridge -0.004, hgb -0.020 (inside noise; 182 validation days).
REJECTED_COLS = {"cnt_roll_3"}

# What feature_columns() and the tests actually enforce.
FORBIDDEN = LEAKAGE_COLS | TIME_INDEX_COLS | UNVERIFIED_COLS | REJECTED_COLS

# Temporal split: train on 2011, validate on H1 2012, test on H2 2012.
TRAIN_END = "2011-12-31"
VALID_END = "2012-06-30"

# Longest lag window in src/features.py. The first LAG_WARMUP calendar days
# have no complete history and are dropped; tests/test_contract.py uses this
# to bound how many rows may legitimately disappear.
LAG_WARMUP = 7
