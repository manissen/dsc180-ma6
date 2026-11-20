# CLAUDE.md

Forecast daily Capital Bikeshare `cnt` (2011-2012) from calendar, weather, and lagged demand.

## Never
- Put `casual` or `registered` in a feature set. `casual + registered == cnt` exactly, so a model handed them scores R^2 1.0 having learned addition.
- Split randomly. Use `src.train.temporal_split`: train 2011, valid H1 2012, test H2 2012.
- Tune on test. Iterate on validation; test gets touched once, after all feature work.
- Hand-type a feature list. `feature_columns()` subtracts `FORBIDDEN` from the numeric columns, so every new column is included by default. To exclude one, add it to `FORBIDDEN` in `src/config.py` with the reason.
- Write to `data/raw/`. The CSV is checked in and immutable.
- Add a dependency. Standard library, pandas, scikit-learn only; ask first.

## Lags and rolling features
- Run `reindex_to_calendar` first. The mirror is missing 2011-03-10 and `.shift()` moves by rows, so an unpatched gap silently ships a 2-day lag labelled 1-day.
- `.shift(1)` before any aggregation, otherwise the window contains the day's own `cnt`.
- Window longer than 7 days: raise `LAG_WARMUP` in `src/config.py`, or `tests/test_contract.py` stops bounding row loss.
- Add a hand-computed test in `tests/test_leakage.py` for every new lag. Use the `/new-feature` skill, which enforces this order.
- Build the feature after the `reindex_to_calendar` call in `build_features`, never in `src/train.py`.

## Reporting
- Report the mean baseline **and** the persistence baseline (`cnt_lag_1`) next to every model score. Persistence is the one a lag-driven model can lose to; say so out loud when it does.
- A negative R^2 is a finding, not a bug to hide.
- Do not "fix" hgb. Trees fit on 2011 cannot extrapolate a year that grew ~65%. Test R^2 to expect: ridge 0.583, persistence 0.510, hgb -0.50, mean -2.14.
- Do not "fix" `src/leakage_demo.py`. It trains on the leaky columns on purpose to print R^2 1.000.
- Do not drop 2012-10-29 (Hurricane Sandy, 22 rides) to improve a score. It is real demand and it is in the test set.
- Append one line to `docs/DECISIONS.md` for every methodological decision: what, why, what changed.

## Commands
- `make all` = data -> features -> train. `make test` before claiming anything works.
- Always `python -m pytest ...`, never bare `pytest`. Only `-m` puts the repo root on `sys.path`; bare `pytest` fails collection with `ModuleNotFoundError: No module named 'src'`.
- One test: `python -m pytest tests/test_leakage.py::test_no_forbidden_features`
- `make data` will not re-fetch an existing CSV. Use `python -m src.ingest --force`.
- Run `make features` before `/new-feature`; the skill reads the feature list from the parquet.

## Style
- Paths and constants live in `src/config.py`. Nothing hardcoded twice.
- Every module runs standalone: `python -m src.<module>` prints one `OK ...` line.
- `MODELS` in `src/train.py` holds fitted instances, and sklearn refits in place. Copy a model before reaching into it from a test or script.
