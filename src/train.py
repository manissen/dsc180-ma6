"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

    python -m src.train                              # the committed run
    python -m src.train --target diff_lag1           # fit a transformed target
    python -m src.train --target diff_lag1 --show-test

`--target` (src/transforms.py) fits the model on a transformed target and
inverts the prediction back to rides before scoring, so R^2 and MAE stay in
rides and stay comparable to the `none` row. It prints VALIDATION only and
writes reports/metrics_target_<name>.json; reports/metrics.json belongs to
`--target none` alone, because `make all` regenerates it. `--show-test` is a
separate flag and prints the test numbers -- do not reach for it to tune.

Run: python -m src.train
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, ROOT, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns
from src.transforms import (
    RIDES,
    TARGET_TRANSFORMS,
    TargetTransform,
    positivity_ok,
    roundtrip_max_error,
)


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}

# The two baselines stay on rides under --target. They are reference points,
# not models of the target: "yesterday again" and "the 2011 average" mean the
# same thing on any scale, and only the learned models are being re-expressed.
# See fit_and_score for what goes wrong if they are routed through the
# transform -- persistence inverts to 2 * cnt_lag_1, the mean inverts to
# persistence -- and the rows stop being comparable to reports/metrics.json.
RIDES_BASELINES = ("baseline_mean", "baseline_persistence")


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def score_in_rides(
    tf: TargetTransform,
    model,
    part: pd.DataFrame,
    cols: list[str],
    diagnostics: bool = False,
) -> dict:
    """Fit-target -> predict -> invert -> score.

    With `--target none` the transform is the identity and this is exactly the
    arithmetic the committed baseline runs -- including the returned dict, so
    reports/metrics.json keeps its shape. With any other transform the model
    sees z = f(cnt) and its output comes back to rides before the metric is
    computed, so the number below is comparable to a rides-per-day truth.

    The score is left raw -- no clipping at zero. A negative inverted
    prediction is the model saying "fewer rides than the anchor allowed", and
    hiding it would understate a failure the report should show. It is counted
    instead, in `n_pred_rides_negative`.
    """
    y = part[TARGET].to_numpy(dtype=float)
    z = tf.transform(y, part)
    pred = tf.invert(np.asarray(model.predict(part[cols]), dtype=float), part)
    scored = {
        "r2": round(float(r2_score(y, pred)), 4),
        "mae": round(float(mean_absolute_error(y, pred)), 1),
    }
    if diagnostics:
        scored["n_pred_rides_negative"] = int((pred < 0).sum())
        scored["pred_rides_min"] = round(float(pred.min()), 1)
    return scored


def coverage_outside_train_range(tf: TargetTransform, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """Share of validation days whose transformed target falls outside the
    range the 2011 training days span.

    This is the extrapolation audit for --target. A tree cannot predict a z it
    never saw, and a linear model can only reach it by extrapolating, so the
    transform that looks hardest to fit is also the one whose predictions are
    least trustworthy. It is only meaningful against a non-empty 2011 range,
    which is why an empty training frame raises instead of returning 1.0.
    """
    if train.empty:
        raise ValueError("coverage needs a non-empty 2011 training range")
    z_train = tf.transform(train[TARGET].to_numpy(dtype=float), train)
    lo, hi = np.nanmin(z_train), np.nanmax(z_train)
    z = tf.transform(valid[TARGET].to_numpy(dtype=float), valid)
    outside = (z < lo) | (z > hi)
    return {
        "train_z_lo": round(float(lo), 6),
        "train_z_hi": round(float(hi), 6),
        "valid_z_lo": round(float(np.nanmin(z)), 6),
        "valid_z_hi": round(float(np.nanmax(z)), 6),
        "n_valid": int(len(valid)),
        "n_outside_train_range": int(outside.sum()),
        "n_below_train_range": int((z < lo).sum()),
        "n_above_train_range": int((z > hi).sum()),
        "share_outside_train_range": round(float(outside.mean()), 4),
    }


def _models() -> dict:
    """Copies of MODELS to fit, so a --target run cannot leave the shared
    instances fitted at a transformed target for the next import in this
    process. `clone` is not enough: it rebuilds an empty estimator and drops
    the lag column PersistenceBaseline was constructed with."""
    return {name: copy.deepcopy(model) for name, model in MODELS.items()}


# The feature each transform is inverted against. These are features, so they
# exist at prediction time; the check below is a guard, not a discovery.
INVERSE_COLUMN = {
    "diff_lag1": "cnt_lag_1",
    "logdiff_lag1": "cnt_lag_1",
    "ratio_roll7": "cnt_roll_7",
}


def fit_and_score(target: str = "none", show_test: bool = False) -> dict:
    """Score every model under one target transform.

    `target="none"` is the committed run and must stay byte-identical, so the
    transform threads through with a default rather than changing this path:
    same dict keys, same numbers, `reports/metrics.json` untouched.

    Under any other transform, test is not scored unless `show_test` is
    passed -- not merely left unprinted. "Test gets touched once" is about not
    looking at it, and a metrics file full of test numbers is lookable.
    """
    tf = TARGET_TRANSFORMS[target]
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    diagnostics = target != "none"
    if diagnostics:
        results["target"] = target
        results["target_description"] = tf.description
        results["fit_target"] = f"models fit {tf.description}; every prediction is " \
            "inverted back to rides before scoring"
        if target in INVERSE_COLUMN:
            needed = INVERSE_COLUMN[target]
            assert needed in cols, f"{target} cannot be inverted without {needed} in the feature set"
        for name, part in [("train", train), ("valid", valid), ("test", test)]:
            ok, why = positivity_ok(target, part)
            if not ok:
                raise ValueError(f"--target {target} is not defined on the {name} split: {why}")
            results[f"domain_{name}"] = why
        # 3 significant figures, not decimal places: rounding 9.1e-13 to 12 dp
        # reports 0.0 and throws away the only interesting thing about it.
        results["invert_roundtrip_max_rides"] = float(
            f"{max(roundtrip_max_error(tf, part) for part in (train, valid, test)):.3e}"
        )
        results["coverage"] = {"valid": coverage_outside_train_range(tf, train, valid)}

    scored = {"valid": valid}
    if target == "none" or show_test:
        scored["test"] = test

    z_train = tf.transform(train[TARGET].to_numpy(dtype=float), train)
    y_train = train[TARGET].to_numpy(dtype=float)
    for name, model in _models().items():
        # The baselines are reference points on rides, not models of the target.
        # "Yesterday again" and "the 2011 average" say the same thing whichever
        # scale you choose to fit on, so they stay fitted on rides and their
        # predictions are never inverted. Route them through z and
        # PersistenceBaseline's output is read as a change and inverted to
        # 2 * cnt_lag_1 -- which on diff_lag1 scores R^2 -9.99 while looking
        # like a baseline that collapsed, and DummyRegressor fit on the change
        # inverts to lag1 + 15, i.e. persistence under another name. Either way
        # the row stops being comparable to the committed run, which is the
        # whole reason predictions are inverted to rides in the first place.
        # Under `--target none` both branches are the identity, so this changes
        # nothing about the committed numbers.
        model_tf, model_y = (RIDES, y_train) if name in RIDES_BASELINES else (tf, z_train)
        model.fit(train[cols], model_y)
        results[name] = {
            split: score_in_rides(model_tf, model, part, cols, diagnostics=diagnostics)
            for split, part in scored.items()
        }
    # Only under a transform: an extra key in reports/metrics.json would make
    # the no-flag run not the committed run.
    if diagnostics:
        results["splits_scored"] = list(scored)
    return results


def metrics_path(target: str) -> Path:
    """reports/metrics.json for the committed run,
    reports/metrics_target_<name>.json for every transform, so no --target run
    can overwrite the baseline `make all` depends on."""
    if target == "none":
        return METRICS
    return ROOT / "reports" / f"metrics_target_{target}.json"


def parse_args(argv=None) -> argparse.Namespace:
    # prog from sys.argv[0] so it reads "python -m src.train" in the usage
    # line, including when launched bare as src/train.py.
    parser = argparse.ArgumentParser(
        prog=f"python -m {__name__}",
        description=__doc__.splitlines()[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=list(TARGET_TRANSFORMS),
        default="none",
        help="fit this transform of cnt, invert predictions back to rides before "
        "scoring (default: none, the committed run). Validation only; writes "
        "reports/metrics_target_<name>.json.",
    )
    parser.add_argument(
        "--show-test",
        action="store_true",
        help="print test-set numbers too. Test gets touched once, after all "
        "feature work -- not to compare transforms.",
    )
    return parser.parse_args(argv)


SPLIT_TITLES = {"valid": "VALIDATION", "test": "TEST"}


def format_report(res: dict) -> str:
    """What a --target run prints: what was fit, the scores in rides, and how
    much of the validation target the 2011 model never saw.

    The splits printed are exactly the splits that were scored, so this
    cannot show a number that fit_and_score was not asked to compute.
    """
    target = res["target"]
    if target == "none":
        note = "target: none -- the committed run, so test is scored and printed too."
    elif res["splits_scored"] == ["valid", "test"]:
        note = f"target: {target} -- TEST NUMBERS SHOWN, deliberately (--show-test)."
    else:
        note = f"target: {target} -- VALIDATION ONLY. --show-test prints test; tuning on it is off limits."
    lines = [
        note,
        f"  fit target: {target} = {res['target_description']}",
        f"  scored:     rides (predictions inverted back to cnt before R^2 and MAE)",
        f"  rows:       train {res['n_train']} (2011) / valid {res['n_valid']} (H1 2012)"
        + (f" / test {res['n_test']} (H2 2012)" if len(res["splits_scored"]) > 1 else " / test not scored"),
        f"  round-trip: max |inverse(f(cnt)) - cnt| = {res['invert_roundtrip_max_rides']} rides",
        f"  features:   {res['n_features']} (unchanged; the lags the inverse uses are already in them)",
        "",
    ]
    header = ["split", "model", "R^2", "MAE (rides)", "min pred", "n<0"]
    rows = []
    for key in res["splits_scored"]:
        for name in MODELS:
            got = res[name][key]
            rows.append([
                SPLIT_TITLES[key],
                name,
                f"{got['r2']:.4f}",
                f"{got['mae']:.1f}",
                f"{got['pred_rides_min']:.1f}",
                str(got["n_pred_rides_negative"]),
            ])
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    cov = res["coverage"]["valid"]
    lines += [
        "",
        f"  validation days whose transformed target is outside the 2011 range:",
        f"    {cov['n_outside_train_range']}/{cov['n_valid']} = {cov['share_outside_train_range']:.1%}"
        f"   (2011 z in [{cov['train_z_lo']:.4f}, {cov['train_z_hi']:.4f},"
        f" validation z in [{cov['valid_z_lo']:.4f}, {cov['valid_z_hi']:.4f}])",
        "    a model fit on 2011 z cannot reach outside that interval except by",
        "    extrapolating, and trees never do: their prediction is a leaf average.",
        "",
        f"  -> {metrics_path(target).relative_to(ROOT)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    res = fit_and_score(args.target, show_test=args.show_test)
    path = metrics_path(args.target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    if args.target == "none":
        # The committed output shape. Byte-identical to before --target.
        print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
    else:
        print(format_report(res))
