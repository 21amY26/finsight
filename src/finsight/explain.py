"""Explanations taken from the forecast path, not from a language model."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from finsight.forecast import ForecastResult, as_date


def _scheduled(result: ForecastResult) -> pd.DataFrame:
    frame = result.scheduled.copy()
    if frame.empty:
        return frame
    frame["date"] = frame["date"].map(as_date)
    return frame.sort_values("date")


def crossing_item(result: ForecastResult) -> dict | None:
    """Scheduled cash item that first takes the base path below zero."""
    if result.shortfall_date is None:
        return None
    frame = _scheduled(result)
    if frame.empty:
        return None
    variable = result.variable_daily
    future = result.dates[1:]
    var_by_day = dict(zip(future, variable))
    running = result.cash_today
    previous = result.as_of
    culprit = None
    for _, row in frame.iterrows():
        day = as_date(row["date"])
        if day <= result.as_of:
            continue
        cursor = previous + timedelta(days=1)
        while cursor <= day and cursor <= result.dates[-1]:
            if cursor in var_by_day and not (
                cursor == day and row["category"] != "client_payment"
            ):
                running -= var_by_day[cursor]
            if cursor == day:
                break
            cursor += timedelta(days=1)
        before = running
        running += float(row["amount"])
        if before >= 0 > running and culprit is None:
            culprit = {
                "date": day,
                "amount": float(row["amount"]),
                "category": row["category"],
                "counterparty": row["counterparty"],
                "balance_after": running,
            }
        previous = day
    return culprit


def unusual_expenses(transactions: pd.DataFrame, as_of: date) -> list[str]:
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.date
    tx = tx[(tx["amount"] < 0) & (~tx["category"].isin(["opening", "rent", "payroll", "loan"]))]
    notes: list[str] = []
    for category, rows in tx.groupby("category"):
        def window_total(end: date) -> float:
            start = end - timedelta(days=29)
            picked = rows[(rows["date"] >= start) & (rows["date"] <= end)]
            return abs(float(picked["amount"].sum())) if not picked.empty else 0.0

        recent_total = window_total(as_of)
        prior = [window_total(as_of - timedelta(days=shift)) for shift in range(30, 180, 30)]
        prior = [value for value in prior if value > 0]
        if len(prior) < 3 or recent_total == 0:
            continue
        spread = float(np.std(prior))
        mean = float(np.mean(prior))
        if spread == 0:
            if recent_total <= mean * 1.5:
                continue
            notes.append(
                f"{category} spending over the last 30 days is ${recent_total:,.0f}, "
                f"against a prior 30-day average of ${mean:,.0f} that did not vary."
            )
            continue
        score = (recent_total - mean) / spread
        if score > 2:
            notes.append(
                f"{category} spending over the last 30 days is ${recent_total:,.0f}, "
                f"which is {score:.1f} standard deviations above the prior 30-day average "
                f"of ${mean:,.0f}."
            )
    return notes


def explain(result: ForecastResult, transactions: pd.DataFrame) -> dict:
    frame = _scheduled(result)
    horizon_end = result.as_of + timedelta(days=30)
    window = frame[frame["date"].map(lambda day: result.as_of < as_date(day) <= horizon_end)] if not frame.empty else frame
    if window.empty:
        inflows = pd.Series(dtype=float)
        outflows = pd.Series(dtype=float)
    else:
        inflows = (
            window[window["amount"] > 0]
            .groupby("counterparty")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        outflows = (
            window[window["amount"] < 0]
            .groupby("category")["amount"]
            .sum()
            .sort_values()
        )
    cross = crossing_item(result)
    bullets: list[str] = []
    if result.shortfall_date is None:
        minimum = min(result.balance)
        bullets.append(
            f"The base path stays non-negative over {len(result.dates) - 1} days. "
            f"The lowest projected balance is ${minimum:,.0f}."
        )
    else:
        days_out = (result.shortfall_date - result.as_of).days
        bullets.append(
            f"The base path drops below zero on {result.shortfall_date.isoformat()} "
            f"({days_out} days out)."
        )
    if cross is not None:
        bullets.append(
            f"{cross['counterparty']} ({cross['category']}) of ${cross['amount']:,.0f} "
            f"on {cross['date'].isoformat()} is the scheduled item that takes cash through zero, "
            f"to ${cross['balance_after']:,.0f}."
        )
    bullets.extend(result.delay_notes)
    bullets.extend(unusual_expenses(transactions, result.as_of))
    low_30 = min(result.balance[:31])
    low_90 = min(result.balance)
    return {
        "inflows": inflows,
        "outflows": outflows,
        "crossing": cross,
        "bullets": bullets,
        "assumptions": result.assumptions,
        "min_30": low_30,
        "min_90": low_90,
        "prob_30": result.shortfall_prob_30,
        "prob_90": result.shortfall_prob_90,
    }
