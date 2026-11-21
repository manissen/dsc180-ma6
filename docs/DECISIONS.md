# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- 2026-11-20: Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- 2026-11-20: Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- 2026-11-20: Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- 2026-11-20: Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- 2026-11-20: Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-20: Switched agent from local bikeshare-coder to NRP-hosted qwen3 (8 GB RAM can't run local model).
- 2026-11-20:Removed CLAUDE_CODE_EFFORT_LEVEL/MAX_OUTPUT_TOKENS from settings; caused "malformed stream" errors.
- 2026-11-20: Trimmed CLAUDE.md to hard rules + commands.
- 2026-11-20: Agent profile verified: 728 rows, 2011-01-03→2012-12-31, 2011-03-10 missing, weather 463/245/20.
- 2026-11-20: Agent error: claimed median nearest-train distance "one second" under random split; daily data → one day.
- 2026-11-20: Agent error: said 2012-10-29 (cnt=22) is ~700× below mean; actual ~205× (4520/22).
- 2026-11-20: Agent error (self-corrected): reported mnth/dteday mismatch due to case bug in its own comparison.
- 2026-11-20: Agent argued cnt_2d_bfr is verifiably calendar-aligned (725/725); kept it in UNVERIFIED_COLS anyway — contract uses our own lags, no exception rows needed.
- 2026-11-20: Verified instant 3–731, min cnt 22 on 2012-10-29, cnt_2d_bfr calendar-aligned 725/725.
- 2026-11-20: Agent error: misreported SEP as SPRING/SUMMER; actual 16 FALL / 44 SUMMER.
- 2026-11-20: Agent error: called season "two conventions, contradicts mnth on ~20% of rows". Counts (MAR 11/yr SPRING, JUN 10/yr SUMMER, DEC 11/yr WINTER) match a single astronomical (solstice/equinox) rule — season is consistent. Using cyclical month encoding; season redundant with it.
- 2026-11-20: cnt_2d_bfr verified calendar-aligned (725/725) but kept in UNVERIFIED_COLS; contract builds its own lags.
- 2026-11-20: Step 3 naive prompt: CLAUDE.md rules held, agent avoided leaky model by default; literal all-columns run gave R² 1.000 (leakage).
- 2026-11-20: Ridge beats persistence on test R² (0.583 vs 0.510) but loses on MAE (947 vs 840); input for Exercise 5.
- 2026-11-20: Agent reported test-split metrics during Step 3 (no tuning); added CLAUDE.md rule against test reporting until final results.
- 2026-11-20: Step 3 residual check: agent computed (not explained) lstsq over subsets ≤3; only {casual, registered} exact, coef [1,1], intercept 0, max|resid| 3.6e-12 on 714 feature rows. Confirmed independently on 728 raw rows.
- 2026-11-20: Agent claimed no size-4+ identity is "plausible"; unverified — search limited to subsets ≤3.
- 2026-11-20: Verified agent's data description by hand (728 rows, 2011-01-03→2012-12-31, GOOD/MISTY/BAD 463/245/20, missing 2011-03-10, 714 feature rows). All correct.
- 2026-11-20: Step 6 rewrite of train.py from spec. Ridge = StandardScaler + Ridge(alpha=1.0) sklearn default, all four fitted on 2011 only, scored on identical rows. Mean/persistence/ridge reproduce the reference to 4 decimals (test -2.1441 / 0.5099 / 0.5833).
- 2026-11-20: hgb kept at max_iter=300. sklearn's default 100 gives test R² -0.6179; 300 gives the documented -0.4975 (valid 0.2185/1296.6) — the whole CLAUDE.md gap was one parameter, derived independently by sweeping before confirming it in git history (b8e90c0). Restored for reproducibility: requirements.txt pins versions precisely so `make all` reproduces reports/metrics.json to the printed digit. Not tuning — 300 was never chosen by looking at validation or test, and both values carry the same conclusion (trees fit on 2011 cannot extrapolate a year that grew ~65%; hgb loses to persistence on MAE 1972.6 vs 840.1).
- 2026-11-20: Persistence baseline implemented as a sklearn estimator (PersistenceRegressor) that predicts the cnt_lag_1 column, so it is fit/predict-scored on the same 182/184 rows as the learned models. Subsetting to rows where a lag exists would score it on an easier sample and void the comparison.
- 2026-11-20: Step 6 diagnostic: agent, unprompted, refit hgb on train+valid and scored it on test (R² 0.609), while claiming it had not touched test. Discarded; not used for any decision. Test remains scored once, by make all.

