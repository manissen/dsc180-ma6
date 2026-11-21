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

The naive model, which uses every column, scores test R² = 1.000, and that result is invalid: casual + registered reproduces cnt exactly (max |residual| 3.6e-12), so the model learns addition instead of forecasting. Under a temporal split, the mean baseline scores −2.14 on test because ridership grew (the test mean is 1.73× the 2011 mean). A standardised ridge reaches 0.58 against a persistence ("yesterday's count") baseline of 0.51, and loses to persistence on MAE (947 vs 840 rides). Gradient boosting scores −0.50: a tree cannot predict above the largest value it saw in training (6,043 rides), and 56.5% of test days exceed it. Raising cnt_lag_1 from 6,043 to 8,714 leaves its prediction fixed at 4,863 while ridge's rises by 716, and 72% of its test MSE is bias rather than noise. It also loses to persistence on validation (0.22 vs 0.45). Most of that gap comes from out-of-range days (39% of H1 2012 exceeds the 2011 maximum), but on in-range days it still reaches only R² 0.33, consistent with 348 training rows and default settings. Re-expressing the target relative to recent demand removes the extrapolation problem and lets gradient boosting compete (Exercise 4), but that changes which target the models fit, not which model is better.

The feature I added, cnt_roll_3 (mean count over the three preceding days), moved validation R² by −0.004 for ridge and −0.020 for gradient boosting. Both are below the 0.01 bar and inside the noise of a 182-day validation set, and the feature mostly duplicates cnt_lag_1 and cnt_roll_7. I rejected it: it remains built and tested, but is excluded from the models through REJECTED_COLS in src/config.py.

Responding to the batch audit (reports/audit.jsonl, run with qwen3; 7 findings, and it passed all four of its checks: leakage none, temporal split valid, baselines reported, reproducible). I fixed five findings. First, a derived restatement of cnt (e.g. cnt * 1.0 added in build_features) passed all 19 tests and scored ridge test R² 0.9999. It is now caught by test_no_feature_restates_target, a least-squares check over every feature and pair. Second, deleting casual from LEAKAGE_COLS was caught only incidentally; test_known_leaks_never_features now checks a hardcoded list (both shown red in docs/SEEDED_FAULTS.md). Third, LAG_WARMUP silently set two window sizes, so raising it as CLAUDE.md instructed would have turned cnt_lag_7 into a 14-day lag; the windows are now literals. Fourth, a test fitted the shared ridge in place; it now fits a copy. Fifth, my claim that transformed gradient boosting was the only model to beat persistence on MAE was false as written, because ridge also beats it on validation MAE; I reworded it to "on both splits", which is true. I did not fix two findings. The expected metrics in CLAUDE.md and the deliberate leakage demo are part of the starter's design and are excluded from feature_columns() by construction. The claimed ImportError for agent/audit.py is wrong: requirements.txt installs claude-agent-sdk, and the audit itself ran on it. None of the fixes moved a reported number; make all reproduces reports/metrics.json byte for byte, and make test runs 21 tests.

3. Exercise 4 - Target Transformation≈

Gradient boosting fails on the raw target because a tree cannot predict above the largest value it saw in training, and 39% of validation days exceed the 2011 maximum. I tried three targets that express demand relative to recent demand, so that 2012 falls inside the 2011 range: today minus yesterday (diff_lag1), today divided by the trailing 7-day mean (ratio_roll7), and log growth from yesterday (logdiff_lag1). Each model fits the transformed target, and predictions are converted back to rides before scoring. The two baselines stay on rides: routing persistence through the difference transform turns it into 2 × yesterday (R² −9.99), which is no longer a baseline.

Every transform fixed the coverage problem and lifted gradient boosting. None helped ridge. diff_lag1 left it unchanged, as expected: cnt_lag_1 is already a feature, so predicting y − cnt_lag_1 is the same linear model with one coefficient shifted by 1. The other two made it worse. I chose ratio_roll7 on validation, committed that choice, and then scored test once: gradient boosting reached R² 0.642 and MAE 810.1, against persistence at 0.510 / 840.1 and the untransformed ridge at 0.583 / 947.1; ridge under the same transform fell to 0.023. Transformed gradient boosting is the only model in the project to beat persistence on MAE on both splits (validation 687.2 vs 900.6, test 810.1 vs 840.1). Untransformed ridge beats persistence on validation (832.9 vs 900.6) but loses on test (947.1 vs 840.1).

I think the transformation is defensible, but the conclusion it supports is narrower than "gradient boosting is better." It is defensible because ridership grew multiplicatively (the test mean is 1.73× the 2011 mean), and modelling today relative to last week is a standard way to handle that. It also removes the cause of the failure (out-of-range days 39% → 0%) rather than masking it. But I picked the best of three on a 182-day validation set, and the same target that lifts the tree breaks the linear model. So the result is that gradient boosting can compete once the target is put on a stationary scale, not that it is the better model; the comparison depends on which target you choose. The code is behind --target and --show-test in src/train.py, with the transforms in src/transforms.py; the committed reports/metrics.json is the untransformed pipeline.

4. Exercise 5

Ridge is worth deploying, narrowly, if large misses are what cost you. It beats persistence on R² (test 0.583 vs 0.510; validation 0.648 vs 0.451) and on RMSE (validation 1,020 vs 1,273), and its worst validation day is off by 3,536 rides against persistence's 5,597. It is not just "yesterday again". Lagged demand accounts for 63% of the variance of its validation predictions, and its largest weight is on the 7-day mean (+567 per SD), not on yesterday (+362). Weather adds +0.04 validation R² over a lags-only ridge (0.606 → 0.648), and moving from GOOD to BAD weather lowers the forecast by about 984 rides, which persistence cannot do. The cost is bias. Ridge shrinks toward its 2011 intercept (slope 0.72 when regressed on yesterday), so in a growing year it under-predicts almost every day (mean error −555 on test, vs +16 for persistence). That is why it loses on MAE (947 vs 840). If the operational loss is roughly linear in error, persistence is the better and simpler choice. If occasional large misses dominate the cost, such as positioning bikes before a storm, ridge is. The gain, 0.07 test R² over doing nothing clever, is the honest size of the result.

5. Attribution

Agent model: qwen3, hosted on NRP Nautilus (https://ellm.nrp-nautilus.io/anthropic). TODO: add the parameter count from curl .../v1/models.
Why not the handout model: my Mac has 8 GB RAM, and the local qwen3.5:9b (64k context) returned HTTP 500. The hosted fallback qwen3.5:27b-cloud was retired by Ollama on 2026-09-25. I used the professor's NRP fallback instead.
Config changes: removed CLAUDE_CODE_EFFORT_LEVEL=max and CLAUDE_CODE_MAX_OUTPUT_TOKENS from the NRP config, because every response failed with "malformed stream."
Harness: Claude Code 2.1.329.
Other AI help: I used Claude (claude.ai) for environment troubleshooting, for reviewing the agent's claims, and to draft this report's structure. All numbers were produced by the repo's code and checked by me.
Me: chose the rules, verified every data fact and agent claim listed above, found the blind spot in the forbidden-features test, and made every decision in docs/DECISIONS.md.

Where the agent was wrong and I caught it: it reported a "checked-in" metrics conflict in a gitignored file; it said "running it now" on several turns without running anything; it attributed a DECISIONS.md line I wrote to a hook; it described the LAG_WARMUP coupling as intentional; and it scored test on a configuration I had not asked for. Where it corrected me: my specification for the unclassified-column test named mnth as a source column, which nothing reads, and my test design could not catch a new numeric column. The agent found both and added the passthrough test that closes the second. The batch audit then found that a derived restatement of the target passed every test, which I fixed.

My decisions (each has a line in docs/DECISIONS.md)
- What to predict (cnt), what counts as leakage (casual, registered), and that time indexes and the unverified cnt_2d_bfr are excluded for separate reasons
- The temporal split and the two baselines, accepted from the starter after checking them
- Verifying the agent's description of the data by hand (728 rows, the missing 2011-03-10, weather counts 463/245/20, 714 feature rows)
- Seeding the four faults and judging each fix. For fault 4 the agent refused the cp into data/raw/; I ran it myself to show that no structural layer blocks it, then had the hash test added
- Discarding the agent's unrequested refit of gradient boosting on train+valid scored on test, which it ran while reporting that it had not touched the test set
- Rejecting cnt_roll_3 (validation R² below the 0.01 bar), but keeping it built and tested
- Choosing ratio_roll7 on validation before scoring test once, and judging it defensible but not evidence that gradient boosting is the better model
- Which audit findings to fix and which to reject
- Which result is believable: ridge's 0.58 is real but only 0.07 better than persistence, and the gradient-boosting results say more about extrapolation and target choice than about the model

