"""
Survival tools: time from injury to return, with censoring.

Some players never come back within follow-up. Dropping them makes returns
look faster; counting them as "never" makes returns look slower. The
Kaplan-Meier estimator keeps each player in the data exactly as long as he
was observed. Most of this module comes from nba-return-after-injury.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff


def km_curve(time, event) -> pd.DataFrame:
    """Kaplan-Meier with a 95% interval built on the log(-log) scale."""
    sf = SurvfuncRight(np.asarray(time, float), np.asarray(event, int))
    s = np.asarray(sf.surv_prob, float)
    se = np.asarray(sf.surv_prob_se, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(-np.log(s))
        se_ll = se / (s * np.abs(np.log(s)))
        lo = np.exp(-np.exp(ll + 1.96 * se_ll))
        hi = np.exp(-np.exp(ll - 1.96 * se_ll))
    return pd.DataFrame({
        "time": np.asarray(sf.surv_times, float),
        "surv": s,
        "lo": np.clip(np.nan_to_num(lo, nan=0.0), 0, 1),
        "hi": np.clip(np.nan_to_num(hi, nan=1.0), 0, 1),
        "at_risk": np.asarray(sf.n_risk, float),
    })


def median_time(time, event) -> float:
    """Day on which the share still out drops to one half."""
    km = km_curve(time, event)
    reached = km[km.surv <= 0.5]
    return float(reached.time.iloc[0]) if len(reached) else np.nan


def returned_by(time, event, day) -> tuple[float, float, float]:
    """Share back by a given day, with its 95% interval."""
    km = km_curve(time, event)
    upto = km[km.time <= day]
    if upto.empty:
        return 0.0, 0.0, 0.0
    row = upto.iloc[-1]
    return float(1 - row.surv), float(1 - row.hi), float(1 - row.lo)


def bootstrap_median(time, event, n=2000, seed=7) -> tuple[float, float]:
    """Percentile interval for the median. Resamples where the median is not
    reached count as the largest observed time, which keeps the upper end honest."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(time), len(time))
        m = median_time(time[idx], event[idx])
        out[i] = time.max() if np.isnan(m) else m
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def logrank(time, event, group) -> tuple[float, float]:
    chi, p = survdiff(np.asarray(time, float), np.asarray(event, int), np.asarray(group))
    return float(chi), float(p)


def cox_table(data: pd.DataFrame, formula: str, strata: str | None = None) -> pd.DataFrame:
    """Cox regression as a readable table. HR above 1 = earlier return."""
    kw = {"strata": data[strata].values} if strata else {}
    m = PHReg.from_formula(formula, data, status=data.event.values, **kw).fit()
    ci = np.asarray(m.conf_int(), float)
    return pd.DataFrame({
        "HR": np.exp(np.asarray(m.params, float)),
        "low": np.exp(ci[:, 0]),
        "high": np.exp(ci[:, 1]),
        "p": np.asarray(m.pvalues, float),
    }, index=list(m.model.exog_names))


def loglog_slope(time, event) -> tuple[float, float]:
    """Slope of log(-log S) against log(time). Similar slopes between groups
    mean the proportional-hazards assumption holds."""
    from scipy import stats
    km = km_curve(time, event)
    km = km[(km.surv > 0.01) & (km.surv < 0.99) & (km.time > 0)]
    if len(km) < 5:
        return np.nan, np.nan
    fit = stats.linregress(np.log(km.time), np.log(-np.log(km.surv)))
    return float(fit.slope), float(fit.stderr)
