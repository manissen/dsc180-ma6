"""Target transforms for `python -m src.train --target NAME`.

Why transform the target at all: 2012 grew ~65% over 2011, so a model fit on
`cnt` in 2011 is asked to extrapolate a level it never saw. Each transform
here moves the fitting problem somewhere more stationary -- closer to
yesterday (difference), closer to a ratio (log) -- without changing the
arithmetic of the report: every prediction is inverted back to rides before
scoring, so R^2 and MAE stay in rides and line up with the `--target none`
run row for row.

Two invariants, enforced by tests/test_target.py:

  1. `log` only ever sees values verified strictly positive. A difference
     target can be negative -- that is the point -- so it is log-transformed
     as a ratio, which is only defined on positive rides. Both inputs of
     every `log` here are counts of rides, and the minimum over the whole
     modelling table and over the 2011 training rows is checked, not assumed.
     `log(0)` is -inf, which a later `dropna()` deletes without comment --
     the exact silent-NaN failure `encode_weather` guards against.
  2. Each transform is invertible on the rows it is used, and the inverse
     maps f(y) back to y for every row of the modelling table. That is what
     keeps --target runs comparable to the baseline instead of quietly
     scoring a different quantity.

The base columns (`cnt_lag_1`, `cnt_roll_7`) are features, so they are known
at prediction time: inverting a prediction uses information the forecaster
actually has, never the day's own `cnt`.

Run: python -m src.transforms
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import PROCESSED, TARGET, TRAIN_END


@dataclass(frozen=True)
class TargetTransform:
    """z = transform(cnt, known-at-t columns); rides = invert(z_hat, same)."""

    name: str
    description: str
    transform: Callable[[np.ndarray, pd.DataFrame], np.ndarray]
    invert: Callable[[np.ndarray, pd.DataFrame], np.ndarray]
    domain: str  # where the transform is defined; the honesty column


def _rides(part: pd.DataFrame) -> np.ndarray:
    return np.asarray(part[TARGET], dtype=float)


def _y(y) -> np.ndarray:
    """The target the caller actually passed, as float.

    transform() honours its `y` argument rather than re-reading part[TARGET]:
    a transform that quietly returns f(part.cnt) whatever you hand it is a
    wrong-numbers machine the first time a caller passes a target from another
    frame. Length mismatch raises inside numpy/pandas, which is the loud
    failure we want.
    """
    return np.asarray(y, dtype=float)


# --- none ------------------------------------------------------------------
def _none_y(y, part):
    return _y(y)


def _none_inv(pred, part):
    return np.asarray(pred, dtype=float)


# --- diff_lag1: fit the change since yesterday -----------------------------
def _diff_y(y, part):
    return _y(y) - part.cnt_lag_1.to_numpy(dtype=float)


def _diff_inv(pred, part):
    return np.asarray(pred, dtype=float) + part.cnt_lag_1.to_numpy(dtype=float)


# --- ratio_roll7: fit today as a multiple of the week before ---------------
def _ratio_y(y, part):
    return _y(y) / part.cnt_roll_7.to_numpy(dtype=float)


def _ratio_inv(pred, part):
    return np.asarray(pred, dtype=float) * part.cnt_roll_7.to_numpy(dtype=float)


# --- logdiff_lag1: fit the log ratio, i.e. a multiplicative change ---------
# log(y) - log(lag1) == log(y / lag1). The difference form is only defined
# because both counts are strictly positive, which is checked before use.
def _logdiff_y(y, part):
    return np.log(_y(y)) - np.log(part.cnt_lag_1.to_numpy(dtype=float))


def _logdiff_inv(pred, part):
    return np.exp(np.asarray(pred, dtype=float) + np.log(part.cnt_lag_1.to_numpy(dtype=float)))


TARGET_TRANSFORMS: dict[str, TargetTransform] = {
    "none": TargetTransform(
        name="none",
        description="cnt (the committed baseline: nothing is transformed)",
        transform=_none_y,
        invert=_none_inv,
        domain="all real y",
    ),
    "diff_lag1": TargetTransform(
        name="diff_lag1",
        description="y - cnt_lag_1",
        transform=_diff_y,
        invert=_diff_inv,
        domain="all real y (a difference can be negative); inverse needs cnt_lag_1",
    ),
    "ratio_roll7": TargetTransform(
        name="ratio_roll7",
        description="y / cnt_roll_7",
        transform=_ratio_y,
        invert=_ratio_inv,
        domain="cnt_roll_7 > 0 (mean of 7 positive counts, min 1022.7 on the table)",
    ),
    "logdiff_lag1": TargetTransform(
        name="logdiff_lag1",
        description="log y - log cnt_lag_1",
        transform=_logdiff_y,
        invert=_logdiff_inv,
        domain="y > 0 and cnt_lag_1 > 0 (min cnt 22 on 2012-10-29, min cnt_lag_1 22)",
    ),
}


# The identity, exported so callers that need "a transform that does nothing"
# can say so instead of hand-writing the identity.
RIDES: TargetTransform = TARGET_TRANSFORMS["none"]


def positivity_ok(name: str, part: pd.DataFrame) -> tuple[bool, str]:
    """Do the inputs this transform needs exist and, where log is involved,
    are they strictly positive? `True` is the safe answer, so the caller only
    applies the transform when this says so."""
    y = _rides(part)
    if len(part) == 0:
        return False, "empty part: the 2011 range would be vacuous"
    if name == "logdiff_lag1":
        lag = part.cnt_lag_1.to_numpy(dtype=float)
        if not (y.min() > 0 and lag.min() > 0):
            return False, f"log needs positive inputs: cnt min {y.min()}, cnt_lag_1 min {lag.min()}"
    if name == "ratio_roll7":
        roll = part.cnt_roll_7.to_numpy(dtype=float)
        if not roll.min() > 0:
            return False, f"ratio needs cnt_roll_7 > 0: min {roll.min()}"
    return True, "ok"


def roundtrip_max_error(tf: TargetTransform, part: pd.DataFrame) -> float:
    """max |invert(transform(y)) - y| over a frame. 0.0 means the inversion is
    exact where it is used; anything else means scoring is measuring something
    other than rides."""
    y = _rides(part)
    return float(np.max(np.abs(tf.invert(tf.transform(y, part), part) - y)))


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED)
    train = df[df.dteday <= TRAIN_END]
    for name, tf in TARGET_TRANSFORMS.items():
        ok, why = positivity_ok(name, df)
        train_ok, train_why = positivity_ok(name, train)
        err = roundtrip_max_error(tf, df) if ok else float("nan")
        z = tf.transform(_rides(df), df) if ok else None
        span = f"z [{z.min():.4f}, {z.max():.4f}]" if ok else "not applied"
        print(
            f"OK transform: {name:13s} roundtrip max|inv(f(y))-y| = {err:.3e}  "
            f"domain ok on all {len(df)} rows: {ok} ({why}) / 2011: {train_ok} ({train_why})  {span}"
        )
