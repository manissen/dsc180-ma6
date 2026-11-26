## Scope

Requested as a diff audit. **There is no diff to audit** — working tree is
clean, `HEAD == origin/master`, nothing unpushed, and the pipeline files have
not changed since `fd2284a`. Scope was therefore widened to the whole pipeline
at HEAD: `src/config.py`, `src/features.py`, `src/train.py`, `src/ingest.py`,
`tests/`, and the numbers claimed in `REPORT.md`.

## Repo state (verified)

- Path: `../partner-repo`, branch `master`, HEAD `5248262 "REPORT: work in progress"`
- `git status`: clean. `git log origin/master..HEAD`: empty.
- Remote: `https://github.com/m1chun/bikeshare-open.git`
- `src/`: `__init__.py`, `config.py`, `features.py`, `ingest.py`,
  `leakage_demo.py`, `train.py`
- Also present: `agent/` (contains its own `audit.py`, with an audit run
  committed at `d5fe40e`), `scripts/`, `reports/`, `model/`, `docs/`

## History of the files under audit

    fd2284a config: add SAFE_INPUT_COLS (fix: missing from previous commit)
    d2b28c2 Fix fault 3: restore raise on unmapped weather labels
    3cd7280 Fault 3 (seeded): raise removed from encode_weather
    8f8525c Fix fault 2: restore shift(1) on cnt_roll_7
    ad1675a Fault 2 (seeded): shift(1) removed from cnt_roll_7
    800630f Fix fault 1: remove casual from feature_columns
    a198939 Fault 1 (seeded): casual added to feature_columns
    3cc46ea Ex1: add temp_sq (ridge valid R² 0.6475 -> 0.5811, negative result)
    0c5aca0 Starter (unmodified)

## Findings

## Checks to perform

1. **Target leakage** — read `src/config.py` `FORBIDDEN` / `SAFE_INPUT_COLS`;
   check arithmetic combinations in `feature_columns()`, not column names.
   Note `800630f` removed `casual`; `fd2284a`'s own message admits
   `SAFE_INPUT_COLS` was missing from the commit before it.
2. **Temporal ordering** — every aggregation `.shift(1)` before aggregating;
   split by date; no fitted transformer sees valid/test rows.
3. **Calendar gaps** — lag features vs missing days in the index.
4. **Silent row loss** — `.map()` / `.merge()` / `.dropna()` without a
   bounding assertion.
5. **Unreported baseline** — mean *and* persistence next to every score.
