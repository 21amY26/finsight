"""Walk-forward score against a trailing-average baseline."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from finsight.forecast import daily_balance, forecast


def _books_as_of(
    transactions: pd.DataFrame,
    invoices: pd.DataFrame,
    origin: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx[tx["date"].dt.date <= origin]
    inv = invoices.copy()
    inv["issue_date"] = pd.to_datetime(inv["issue_date"])
    inv["paid_date"] = pd.to_datetime(inv["paid_date"], errors="coerce")
    inv = inv[inv["issue_date"].dt.date <= origin].copy()
    paid_after = inv["paid_date"].notna() & (inv["paid_date"].dt.date > origin)
    inv.loc[paid_after, "paid_date"] = pd.NaT
    return tx, inv


def _first_negative(series: pd.Series, start: date, end: date) -> date | None:
    for day, value in series.items():
        if start < day <= end and float(value) < 0:
            return day
    return None


def _lead_days(
    transactions: pd.DataFrame,
    invoices: pd.DataFrame,
    *,
    opening: float,
    history_start: date,
    origins: list[date],
    event: date,
    horizon: int,
    use_baseline: bool,
) -> int | None:
    for origin in origins:
        if origin >= event:
            break
        tx, inv = _books_as_of(transactions, invoices, origin)
        result = forecast(
            tx,
            inv,
            opening=opening,
            history_start=history_start,
            as_of=origin,
            horizon=horizon,
            n_sims=0,
        )
        path = result.baseline if use_baseline else result.balance
        flagged = any(
            day <= event and value < 0
            for day, value in zip(result.dates, path)
            if day > origin
        )
        if flagged:
            return (event - origin).days
    return None


def evaluate(
    transactions: pd.DataFrame,
    invoices: pd.DataFrame,
    *,
    opening: float,
    history_start: date,
    as_of: date,
    holdout_days: int = 60,
    horizon: int = 21,
) -> dict:
    """Score daily cash-balance error on weekly origins inside the holdout.

    Lead time is the earliest warning, in days, before the first below-zero
    close in the holdout. It is null when that event does not occur or the
    method never flags it.
    """
    holdout_start = as_of - timedelta(days=holdout_days)
    actual = daily_balance(transactions, opening, history_start, as_of)
    origins: list[date] = []
    origin = holdout_start
    while origin + timedelta(days=horizon) <= as_of:
        origins.append(origin)
        origin += timedelta(days=7)
    if not origins:
        raise ValueError("holdout is shorter than the forecast horizon")

    model_errors: list[float] = []
    baseline_errors: list[float] = []
    for origin in origins:
        tx, inv = _books_as_of(transactions, invoices, origin)
        result = forecast(
            tx,
            inv,
            opening=opening,
            history_start=history_start,
            as_of=origin,
            horizon=horizon,
            n_sims=0,
        )
        for day, model_balance, baseline_balance in zip(result.dates, result.balance, result.baseline):
            if day <= origin or day > as_of or day not in actual.index:
                continue
            actual_balance = float(actual.loc[day])
            model_errors.append(abs(model_balance - actual_balance))
            baseline_errors.append(abs(baseline_balance - actual_balance))

    event = _first_negative(actual, holdout_start, as_of)
    model_lead = None
    baseline_lead = None
    if event is not None:
        model_lead = _lead_days(
            transactions,
            invoices,
            opening=opening,
            history_start=history_start,
            origins=origins,
            event=event,
            horizon=horizon,
            use_baseline=False,
        )
        baseline_lead = _lead_days(
            transactions,
            invoices,
            opening=opening,
            history_start=history_start,
            origins=origins,
            event=event,
            horizon=horizon,
            use_baseline=True,
        )

    model_mae = float(np.mean(model_errors))
    baseline_mae = float(np.mean(baseline_errors))
    return {
        "holdout_start": holdout_start,
        "holdout_end": as_of,
        "origins": len(origins),
        "horizon_days": horizon,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "model_improves_mae": model_mae < baseline_mae,
        "shortfall_event": event,
        "model_lead_days": model_lead,
        "baseline_lead_days": baseline_lead,
    }
