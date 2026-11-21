1. Results

Temporal split: train = 2011, validation = H1 2012, test = H2 2012 (src.train.temporal_split). Features exclude everything in FORBIDDEN (casual, registered, cnt, instant, yr, days_since_2011, cnt_2d_bfr).

Model	Valid R^2	Test R^2	Test MAE
Mean baseline (2011 mean)	-0.932	-2.144	2848
Persistence (yesterday)	0.451	0.510	840
Ridge (standardized, L2, 14 features)	0.647	0.583	947
Gradient boosting (HGB)	0.219	-0.498	1973

These match the handout's reference values (-2.14 / 0.51 / 0.58 / -0.50).

What the numbers say

Mean baseline (-2.14): predicting the 2011 average does much worse than predicting a constant in 2012. This measures the distribution shift: mean cnt was 3427 in 2011 and about 5600 in 2012 (+63%).
Gradient boosting (valid 0.219 -> test -0.498): trees fit on 2011 can't predict above the 2011 range, so they fail as demand grows through 2012. I kept this result in on purpose. It's a real finding, not a bug.
Ridge vs persistence: ridge wins on R^2 (0.583 vs 0.510) and loses on MAE (947 vs 840). See Exercise 5.

Data facts verified by hand (not taken from the agent): 728 rows, 2011-01-03 -> 2012-12-31, one internal gap (2011-03-10), weather GOOD/MISTY/BAD = 463/245/20, instant 3-731, min cnt = 22 on 2012-10-29, raw CSV SHA-256 begins a6bed1577f6e4de5.

2. Headline Finding

A perfect score was a bug report. Asked naively to "predict cnt from the other columns," the pipeline reaches R^2 = 1.000 (python -m src.leakage_demo). I made the agent prove the cause by computing it, not explaining it: numpy.linalg.lstsq of cnt ~ casual + registered gives coefficients [1.0000, 1.0000], intercept 0, and max |residual| = 3.6e-12. That's zero up to float64 precision, on 714 feature rows, and I confirmed it separately on the 728 raw rows. No other subset of up to 3 columns comes close: the best non-leaky subset has max |residual| ~ 1.8e3, so the gap is 15 orders of magnitude.

The standing rules changed what the agent did. With CLAUDE.md loaded, the agent did not build the leaky model by default. It flagged the trap, used the audited feature set, and ran the all-columns version only as a demonstration. TODO (optional): the same prompt with CLAUDE.md removed, for a before/after comparison.

I verified the agent's claims and didn't trust its fluency. Its data profile got every core fact right, but it also made confident mistakes I only caught by checking (all logged in docs/DECISIONS.md):

It said nearest-training-day distance under a random split was "one second" (the data is daily, so it's one day).
It said cnt = 22 was "~700x below the mean" (actually 4520/22 ~ 205x).
It invented a story that season mixes "two conventions" and contradicts mnth on about 20% of rows. The boundary-month counts (MAR: 11 SPRING days per year, i.e. Mar 21-31; JUN: 10 SUMMER; SEP: 8 FALL; DEC: 11 WINTER) fit one astronomical rule, so the column is consistent.
It misreported September as SPRING/SUMMER (actually 16 FALL / 44 SUMMER).

The tests had a blind spot. test_no_forbidden_features checks against config.FORBIDDEN, so removing casual from LEAKAGE_COLS takes it out of the guard at the same moment it enters the features, and the test stays green. I added a hardcoded test_known_leaks_never_features, which turns red on that edit. TODO: paste the red/green pytest output.

3. Exercise 4 - Target Transformation≈

Gradient boosting fails on the raw target because a tree cannot predict above
the largest value it saw in training, and 39% of validation days exceed the
2011 maximum. I tried three targets that express demand relative to recent
demand, so that 2012 falls inside the 2011 range: today minus yesterday
(diff_lag1), today divided by the trailing 7-day mean (ratio_roll7), and log
growth from yesterday (logdiff_lag1). Each model fits the transformed target;
predictions are inverted back to rides before scoring. The two baselines stay
on rides: routing persistence through the difference transform turns it into
2 × yesterday (R² −9.99), which is not a baseline.

On validation, every transform fixed the coverage problem (days out of range:
39% → 2.2%, 0.0%, 0.5%) and lifted hgb from R² 0.219 to 0.608, 0.745 and 0.573.
They did not help ridge. diff_lag1 left it unchanged (0.6475), as expected:
cnt_lag_1 is already a feature, so predicting y − cnt_lag_1 is the same linear
model with one coefficient shifted by 1. ratio_roll7 and logdiff_lag1 made it
worse (0.388, 0.265). I chose ratio_roll7 on validation, committed that choice,
and then scored test once: hgb R² 0.642 and MAE 810.1, against persistence at
0.510 / 840.1 and the untransformed ridge at 0.583 / 947.1. It is the only
model in the project to beat persistence on MAE.

I think the transformation is defensible, but the conclusion it supports is
narrower than "gradient boosting is better." It is defensible because
ridership grew by about 65%, a multiplicative change, and modelling today
relative to last week is a standard way to handle that; it removes the cause
of the failure (out-of-range days 39% → 0%) rather than masking it. But I
picked the best of three on a 182-day validation set, and the same target that
lifts the tree breaks the linear model (ridge falls to 0.023 on test). So the
result is that gradient boosting can compete once the target is put on a
stationary scale, not that it is the better model; the comparison depends on
which target you choose. The code is behind --target in src/train.py (with
src/transforms.py); the committed reports/metrics.json is the untransformed
pipeline.

4. Exercise 5

Ridge is worth deploying, narrowly, if large misses are what cost you. It
beats persistence on R² (test 0.583 vs 0.510; valid 0.648 vs 0.451) and on
RMSE (valid 1,020 vs 1,273), and its worst validation day is off by 3,536
rides against persistence's 5,597. It is not just "yesterday again": lagged
demand is 63% of its prediction on validation, but weather adds +0.04 valid
R² over a lags-only ridge (0.606 → 0.648), and a BAD-weather day after a GOOD
one moves the forecast by about −984 rides, which persistence cannot do. The
cost is bias: ridge shrinks toward the 2011 mean (slope 0.72 on yesterday),
so in a growing year it under-predicts every day (mean error −555 on test vs
+16 for persistence). That is why it loses on MAE (947 vs 840). If the
operational loss is linear in error (MAE), persistence is the better choice;
if big misses dominate (staffing before storms), ridge is.

5. Attribution

Agent model: qwen3, hosted on NRP Nautilus (https://ellm.nrp-nautilus.io/anthropic). TODO: add the parameter count from curl .../v1/models.
Why not the handout model: my Mac has 8 GB RAM, and the local qwen3.5:9b (64k context) returned HTTP 500. The hosted fallback qwen3.5:27b-cloud was retired by Ollama on 2026-09-25. I used the professor's NRP fallback instead.
Config changes: removed CLAUDE_CODE_EFFORT_LEVEL=max and CLAUDE_CODE_MAX_OUTPUT_TOKENS from the NRP config, because every response failed with "malformed stream."
Harness: Claude Code 2.1.329.
Other AI help: I used Claude (claude.ai) for environment troubleshooting, for reviewing the agent's claims, and to draft this report's structure. All numbers were produced by the repo's code and checked by me.
Me: chose the rules, verified every data fact and agent claim listed above, found the blind spot in the forbidden-features test, and made every decision in docs/DECISIONS.md.

