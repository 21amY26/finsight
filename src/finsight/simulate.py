"""What-if changes on top of the same cash engine."""

from __future__ import annotations

from datetime import date

import pandas as pd

from finsight.forecast import ForecastResult, forecast


def apply_scenario(
    transactions: pd.DataFrame,
    invoices: pd.DataFrame,
    *,
    opening: float,
    history_start: date,
    as_of: date,
    invoice_id: str | None = None,
    expected_date: date | None = None,
    category: str | None = None,
    reduction_pct: float = 0.0,
    horizon: int = 90,
    n_sims: int = 200,
) -> ForecastResult:
    overrides = None
    if invoice_id and expected_date is not None:
        overrides = {invoice_id: expected_date}
    cuts = None
    if category and reduction_pct:
        cuts = {category: reduction_pct}
    return forecast(
        transactions,
        invoices,
        opening=opening,
        history_start=history_start,
        as_of=as_of,
        horizon=horizon,
        n_sims=n_sims,
        collection_overrides=overrides,
        expense_cuts=cuts,
    )


def balance_delta(base: ForecastResult, scenario: ForecastResult) -> list[float]:
    return [s - b for s, b in zip(scenario.balance, base.balance)]
