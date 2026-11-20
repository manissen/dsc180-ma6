# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- Switched agent from local bikeshare-coder to NRP-hosted qwen3 (8 GB RAM can't run local model).
- Removed CLAUDE_CODE_EFFORT_LEVEL/MAX_OUTPUT_TOKENS from settings; caused "malformed stream" errors.
- Trimmed CLAUDE.md to hard rules + commands.
- Agent profile verified: 728 rows, 2011-01-03→2012-12-31, 2011-03-10 missing, weather 463/245/20.
- Agent error: claimed median nearest-train distance "one second" under random split; daily data → one day.
- Agent error: said 2012-10-29 (cnt=22) is ~700× below mean; actual ~205× (4520/22).
- Agent error (self-corrected): reported mnth/dteday mismatch due to case bug in its own comparison.
- Agent argued cnt_2d_bfr is verifiably calendar-aligned (725/725); kept it in UNVERIFIED_COLS anyway — contract uses our own lags, no exception rows needed.
- Verified instant 3–731, min cnt 22 on 2012-10-29, cnt_2d_bfr calendar-aligned 725/725.
- Agent error: misreported SEP as SPRING/SUMMER; actual 16 FALL / 44 SUMMER.
- Agent error: called season "two conventions, contradicts mnth on ~20% of rows". Counts (MAR 11/yr SPRING, JUN 10/yr SUMMER, DEC 11/yr WINTER) match a single astronomical (solstice/equinox) rule — season is consistent. Using cyclical month encoding; season redundant with it.
- cnt_2d_bfr verified calendar-aligned (725/725) but kept in UNVERIFIED_COLS; contract builds its own lags.
- Step 3 naive prompt: CLAUDE.md rules held, agent avoided leaky model by default; literal all-columns run gave R² 1.000 (leakage).
- Ridge beats persistence on test R² (0.583 vs 0.510) but loses on MAE (947 vs 840); input for Exercise 5.
- Agent reported test-split metrics during Step 3 (no tuning); added CLAUDE.md rule against test reporting until final results.
