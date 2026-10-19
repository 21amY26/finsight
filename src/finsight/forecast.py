"""Schedule-aware cash forecast and a trailing-average baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCHEDULED_CATEGORIES = ("rent", "payroll", "loan", "marketing")
EXCLUDED_FROM_VARIABLE = SCHEDULED_CATEGORIES + ("client_payment", "opening")


def as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _dated(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column]).dt.date


@dataclass
class ForecastResult:
    dates: list[date]
    balance: list[float]
    low: list[float]
    high: list[float]
    baseline: list[float]
    shortfall_date: date | None
    shortfall_prob_30: float
    shortfall_prob_90: float
    scheduled: pd.DataFrame
    assumptions: list[str]
    cash_today: float
    as_of: date
    delay_notes: list[str] = field(default_factory=list)
    variable_daily: list[float] = field(default_factory=list)


def daily_balance(
    transactions: pd.DataFrame,
    opening: float,
    start: date,
    end: date,
) -> pd.Series:
    """End-of-day cash from `opening` (cash before `start`) through `end`."""
    if end < start:
        raise ValueError("end is before start")
    idx = pd.date_range(start, end, freq="D")
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    net = tx.groupby("date")["amount"].sum().reindex(idx, fill_value=0.0)
    balance = float(opening) + net.cumsum()
    balance.index = [ts.date() for ts in idx]
    return balance


def cash_on(transactions: pd.DataFrame, opening: float, start: date, day: date) -> float:
    return float(daily_balance(transactions, opening, start, day).iloc[-1])


def _history(transactions: pd.DataFrame, as_of: date) -> pd.DataFrame:
    tx = transactions.copy()
    tx["date"] = _dated(tx, "date")
    return tx[tx["date"] <= as_of]


def recurring_events(transactions: pd.DataFrame, as_of: date, horizon: int) -> list[dict]:
    history = _history(transactions, as_of)
    end = as_of + timedelta(days=horizon)
    events: list[dict] = []
    for category in SCHEDULED_CATEGORIES:
        rows = history[history["category"] == category].sort_values("date")
        if rows.empty:
            continue
        day_of_month = int(np.median([d.day for d in rows["date"]]))
        day_of_month = min(max(day_of_month, 1), 28)
        amount = float(np.median(rows["amount"].tail(6)))
        counterparty = str(rows.iloc[-1]["counterparty"])
        year, month = as_of.year, as_of.month
        for _ in range(horizon // 28 + 3):
            event_day = date(year, month, day_of_month)
            if as_of < event_day <= end:
                events.append(
                    {
                        "date": event_day,
                        "amount": amount,
                        "category": category,
                        "counterparty": counterparty,
                        "source": "recurring",
                    }
                )
            if month == 12:
                year, month = year + 1, 1
            else:
                month += 1
    events.sort(key=lambda item: (item["date"], item["category"]))
    return events


def _delay_frame(invoices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    inv = invoices.copy()
    inv["due_date"] = _dated(inv, "due_date")
    inv["paid_date"] = pd.to_datetime(inv["paid_date"], errors="coerce")
    paid = inv[inv["paid_date"].notna()].copy()
    paid["paid_date"] = paid["paid_date"].dt.date
    paid = paid[paid["paid_date"] <= as_of]
    paid["delay"] = [
        (paid_on - due).days for paid_on, due in zip(paid["paid_date"], paid["due_date"])
    ]
    return paid


def delay_stats(invoices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    paid = _delay_frame(invoices, as_of)
    if paid.empty:
        return pd.DataFrame(columns=["median", "std", "n"])
    grouped = paid.groupby("customer")["delay"]
    stats = grouped.agg(median="median", std="std", n="count")
    return stats.fillna({"std": 0.0})


def open_invoice_frame(invoices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    inv = invoices.copy()
    inv["issue_date"] = _dated(inv, "issue_date")
    inv["due_date"] = _dated(inv, "due_date")
    paid = pd.to_datetime(inv["paid_date"], errors="coerce")
    is_open = paid.isna() | (paid.dt.date > as_of)
    visible = inv["issue_date"] <= as_of
    return inv[is_open & visible].copy()


def expected_collection(
    due: date,
    customer: str,
    stats: pd.DataFrame,
    as_of: date,
) -> tuple[date, float, float]:
    if not stats.empty and customer in stats.index and stats.loc[customer, "n"] >= 2:
        delay = float(stats.loc[customer, "median"])
        spread = float(stats.loc[customer, "std"])
    else:
        delay, spread = 10.0, 5.0
    expected = due + timedelta(days=max(int(round(delay)), 0))
    if expected <= as_of:
        expected = as_of + timedelta(days=1)
    return expected, delay, spread


def _variable_spend(transactions: pd.DataFrame, start: date, as_of: date) -> pd.Series:
    history = _history(transactions, as_of)
    mask = ~history["category"].isin(EXCLUDED_FROM_VARIABLE)
    daily = history.loc[mask].groupby("date")["amount"].sum()
    idx = pd.date_range(start, as_of, freq="D")
    spend = pd.Series(0.0, index=idx)
    for ts in idx:
        if ts.date() in daily.index:
            spend.loc[ts] = max(0.0, -float(daily.loc[ts.date()]))
    return spend


def _fit_variable(spend: pd.Series) -> tuple[Ridge, float]:
    rows = []
    targets = []
    values = spend.to_numpy(dtype=float)
    for i in range(7, len(values)):
        features = np.zeros(9, dtype=float)
        features[spend.index[i].dayofweek] = 1.0
        features[7] = values[i - 1]
        features[8] = values[i - 7]
        rows.append(features)
        targets.append(values[i])
    model = Ridge(alpha=1.0)
    model.fit(np.vstack(rows), np.array(targets))
    fitted = model.predict(np.vstack(rows))
    residual_std = float(np.std(np.array(targets) - fitted))
    return model, residual_std


def _predict_variable(model: Ridge, spend: pd.Series, future: list[date]) -> list[float]:
    values = list(spend.to_numpy(dtype=float))
    preds: list[float] = []
    for day in future:
        features = np.zeros(9, dtype=float)
        features[pd.Timestamp(day).dayofweek] = 1.0
        features[7] = values[-1]
        features[8] = values[-7]
        pred = max(0.0, float(model.predict(features.reshape(1, -1))[0]))
        preds.append(pred)
        values.append(pred)
    return preds


def _sample_delays(paid: pd.DataFrame, customer: str) -> np.ndarray:
    delays = paid.loc[paid["customer"] == customer, "delay"].to_numpy(dtype=float)
    if len(delays) < 2:
        return np.array([10.0, 14.0, 20.0])
    return delays


def forecast(
    transactions: pd.DataFrame,
    invoices: pd.DataFrame,
    *,
    opening: float,
    history_start: date,
    as_of: date,
    horizon: int = 90,
    n_sims: int = 200,
    collection_overrides: dict[str, date] | None = None,
    expense_cuts: dict[str, float] | None = None,
    seed: int = 7,
) -> ForecastResult:
    """Project cash from the close of `as_of` for `horizon` days."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    cuts = expense_cuts or {}
    overrides = collection_overrides or {}
    history = _history(transactions, as_of)
    cash_today = cash_on(history, opening, history_start, as_of)
    future = [as_of + timedelta(days=i) for i in range(1, horizon + 1)]
    future_index = {day: i for i, day in enumerate(future)}

    stats = delay_stats(invoices, as_of)
    paid = _delay_frame(invoices, as_of)
    open_inv = open_invoice_frame(invoices, as_of)
    recurring = recurring_events(history, as_of, horizon)

    nets = np.zeros(horizon, dtype=float)
    scheduled_rows: list[dict] = []
    delay_notes: list[str] = []

    for event in recurring:
        cut = float(cuts.get(event["category"], 0.0))
        cut = min(max(cut, 0.0), 1.0)
        amount = float(event["amount"]) * (1.0 - cut)
        offset = future_index.get(event["date"])
        if offset is None:
            continue
        nets[offset] += amount
        scheduled_rows.append({**event, "amount": amount, "cut": cut})

    for _, inv in open_inv.iterrows():
        invoice_id = str(inv["invoice_id"])
        customer = str(inv["customer"])
        due = as_date(inv["due_date"])
        expected, delay, spread = expected_collection(due, customer, stats, as_of)
        if invoice_id in overrides:
            expected = as_date(overrides[invoice_id])
            spread = 0.0
        amount = float(inv["amount"])
        if expected in future_index:
            nets[future_index[expected]] += amount
        scheduled_rows.append(
            {
                "date": expected,
                "amount": amount,
                "category": "client_payment",
                "counterparty": customer,
                "source": "invoice",
                "invoice_id": invoice_id,
                "due_date": due,
                "delay_days": delay,
                "delay_std": spread,
                "cut": 0.0,
            }
        )
        if spread >= 7 and amount >= 5_000:
            delay_notes.append(
                f"{customer} invoices have historically cleared about {delay:.0f} days "
                f"after the due date, with a spread of {spread:.0f} days. "
                f"{invoice_id} for ${amount:,.0f} is large enough that this timing "
                "moves the cash forecast."
            )

    spend = _variable_spend(history, history_start, as_of)
    model, residual_std = _fit_variable(spend)
    variable = _predict_variable(model, spend, future)
    for i, amount in enumerate(variable):
        nets[i] -= amount

    balance = [cash_today]
    for value in nets:
        balance.append(balance[-1] + float(value))

    window = history[history["date"] >= as_of - timedelta(days=29)]
    window = window[window["date"] <= as_of]
    daily_net = (
        window.groupby("date")["amount"].sum().reindex(
            pd.date_range(as_of - timedelta(days=29), as_of, freq="D").date,
            fill_value=0.0,
        )
    )
    trailing = float(np.mean(daily_net.to_numpy(dtype=float)))
    baseline = [cash_today + trailing * step for step in range(horizon + 1)]

    dates = [as_of, *future]
    low = list(balance)
    high = list(balance)
    prob_30 = 1.0 if any(value < 0 for value in balance[1:31]) else 0.0
    prob_90 = 1.0 if any(value < 0 for value in balance[1:]) else 0.0

    if n_sims > 0:
        rng = np.random.default_rng(seed)
        paths = np.zeros((n_sims, horizon + 1), dtype=float)
        paths[:, 0] = cash_today
        for sim in range(n_sims):
            sim_nets = np.zeros(horizon, dtype=float)
            for event in recurring:
                cut = min(max(float(cuts.get(event["category"], 0.0)), 0.0), 1.0)
                offset = future_index.get(event["date"])
                if offset is not None:
                    sim_nets[offset] += float(event["amount"]) * (1.0 - cut)
            for _, inv in open_inv.iterrows():
                invoice_id = str(inv["invoice_id"])
                customer = str(inv["customer"])
                due = as_date(inv["due_date"])
                if invoice_id in overrides:
                    expected = as_date(overrides[invoice_id])
                else:
                    delay = float(rng.choice(_sample_delays(paid, customer)))
                    expected = due + timedelta(days=max(int(round(delay)), 0))
                    if expected <= as_of:
                        expected = as_of + timedelta(days=1)
                offset = future_index.get(expected)
                if offset is not None:
                    sim_nets[offset] += float(inv["amount"])
            noise = rng.normal(0.0, residual_std, size=horizon)
            sim_nets -= np.maximum(0.0, np.array(variable) + noise)
            paths[sim, 1:] = cash_today + np.cumsum(sim_nets)
        low = np.quantile(paths, 0.10, axis=0).tolist()
        high = np.quantile(paths, 0.90, axis=0).tolist()
        prob_30 = float(np.mean(np.min(paths[:, 1:31], axis=1) < 0))
        prob_90 = float(np.mean(np.min(paths[:, 1:], axis=1) < 0))

    shortfall = next((day for day, value in zip(dates, balance) if day > as_of and value < 0), None)
    assumptions = [
        "Rent, payroll, loan, and marketing continue on their recent calendar days at the median amount.",
        "Open invoices are collected on the due date plus that customer's median historical delay, unless a scenario sets the date.",
        "Other daily spending is projected with a ridge regression on day-of-week and recent variable outflow.",
        "The band resamples historical payment delays and variable-spend residuals. It is a range of simulated paths, not a guarantee.",
        "Scenario results are calculations under these assumptions.",
    ]
    scheduled = pd.DataFrame(scheduled_rows)
    if scheduled.empty:
        scheduled = pd.DataFrame(
            columns=["date", "amount", "category", "counterparty", "source", "cut"]
        )
    return ForecastResult(
        dates=dates,
        balance=balance,
        low=low,
        high=high,
        baseline=baseline,
        shortfall_date=shortfall,
        shortfall_prob_30=prob_30,
        shortfall_prob_90=prob_90,
        scheduled=scheduled,
        assumptions=assumptions,
        cash_today=cash_today,
        as_of=as_of,
        delay_notes=delay_notes,
        variable_daily=variable,
    )
