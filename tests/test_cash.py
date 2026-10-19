"""Cash engine tests that do not depend on the demo narrative."""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.evaluate import evaluate
from finsight.forecast import cash_on, forecast
from finsight.generate import AS_OF, START, build_books
from finsight.simulate import apply_scenario


def _tx(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _inv(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("issue_date", "due_date", "paid_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _payroll_history(as_of: date, amount: float = 1_500.0) -> list[dict]:
    rows = []
    cursor = date(as_of.year, as_of.month, 1)
    for _ in range(5):
        cursor = date(cursor.year - 1, 12, 1) if cursor.month == 1 else date(cursor.year, cursor.month - 1, 1)
        rows.append(
            {
                "date": date(cursor.year, cursor.month, 23),
                "amount": -amount,
                "category": "payroll",
                "counterparty": "Payroll",
                "description": "payroll",
            }
        )
    return rows


class ForecastTests(unittest.TestCase):
    def test_shortfall_matches_hand_ledger(self) -> None:
        as_of = date(2026, 3, 2)
        start = date(2025, 10, 1)
        rows = _payroll_history(as_of, 1_500.0)
        transactions = _tx(rows)
        level = cash_on(transactions, 0.0, start, as_of)
        opening = 1_000.0 - level
        invoices = _inv(
            [
                {
                    "invoice_id": "NONE",
                    "customer": "Nobody",
                    "amount": 1.0,
                    "issue_date": start,
                    "due_date": start,
                    "paid_date": start,
                }
            ]
        )
        result = forecast(
            transactions,
            invoices,
            opening=opening,
            history_start=start,
            as_of=as_of,
            horizon=30,
            n_sims=0,
        )
        self.assertEqual(result.shortfall_date, date(2026, 3, 23))
        self.assertAlmostEqual(result.cash_today, 1_000.0, places=2)
        self.assertLess(result.balance[(result.shortfall_date - as_of).days], 0)

    def test_shifting_payment_changes_balances_by_that_amount(self) -> None:
        as_of = date(2026, 3, 2)
        start = as_of - timedelta(days=40)
        amount = 1_000.0
        transactions = _tx(
            [
                {
                    "date": start,
                    "amount": 5_000.0,
                    "category": "opening",
                    "counterparty": "Co",
                    "description": "opening",
                }
            ]
        )
        paid_rows = []
        for index, issued in enumerate(
            [start + timedelta(days=1), start + timedelta(days=8), start + timedelta(days=15)]
        ):
            due = issued + timedelta(days=5)
            paid = due + timedelta(days=5)
            paid_rows.append(
                {
                    "invoice_id": f"P-{index}",
                    "customer": "Northwind",
                    "amount": 400.0,
                    "issue_date": issued,
                    "due_date": due,
                    "paid_date": paid,
                }
            )
            transactions = pd.concat(
                [
                    transactions,
                    _tx(
                        [
                            {
                                "date": paid,
                                "amount": 400.0,
                                "category": "client_payment",
                                "counterparty": "Northwind",
                                "description": f"P-{index}",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        open_due = as_of + timedelta(days=5)
        invoices = _inv(
            paid_rows
            + [
                {
                    "invoice_id": "OPEN",
                    "customer": "Northwind",
                    "amount": amount,
                    "issue_date": as_of - timedelta(days=2),
                    "due_date": open_due,
                    "paid_date": pd.NaT,
                }
            ]
        )
        base = forecast(
            transactions,
            invoices,
            opening=0.0,
            history_start=start,
            as_of=as_of,
            horizon=30,
            n_sims=0,
        )
        shifted = apply_scenario(
            transactions,
            invoices,
            opening=0.0,
            history_start=start,
            as_of=as_of,
            invoice_id="OPEN",
            expected_date=as_of + timedelta(days=16),
            horizon=30,
            n_sims=0,
        )
        for day_index in range(10, 16):
            self.assertAlmostEqual(
                shifted.balance[day_index],
                base.balance[day_index] - amount,
                places=2,
            )
        for day_index in range(16, 31):
            self.assertAlmostEqual(shifted.balance[day_index], base.balance[day_index], places=2)

    def test_expense_cut_matches_cash_engine(self) -> None:
        as_of = date(2026, 3, 2)
        start = date(2025, 9, 1)
        rows = []
        cursor = date(2025, 10, 18)
        while cursor <= as_of:
            rows.append(
                {
                    "date": cursor,
                    "amount": -2_000.0,
                    "category": "marketing",
                    "counterparty": "Ads",
                    "description": "marketing",
                }
            )
            month = 1 if cursor.month == 12 else cursor.month + 1
            year = cursor.year + (1 if cursor.month == 12 else 0)
            cursor = date(year, month, 18)
        transactions = _tx(rows)
        invoices = _inv(
            [
                {
                    "invoice_id": "NONE",
                    "customer": "Nobody",
                    "amount": 1.0,
                    "issue_date": start,
                    "due_date": start,
                    "paid_date": start,
                }
            ]
        )
        opening = 10_000.0 - cash_on(transactions, 0.0, start, as_of)
        base = forecast(
            transactions,
            invoices,
            opening=opening,
            history_start=start,
            as_of=as_of,
            horizon=40,
            n_sims=0,
        )
        cut = apply_scenario(
            transactions,
            invoices,
            opening=opening,
            history_start=start,
            as_of=as_of,
            category="marketing",
            reduction_pct=0.5,
            horizon=40,
            n_sims=0,
        )
        event_index = (date(2026, 3, 18) - as_of).days
        self.assertAlmostEqual(cut.balance[event_index - 1], base.balance[event_index - 1], places=2)
        self.assertAlmostEqual(cut.balance[event_index], base.balance[event_index] + 1_000.0, places=2)
        self.assertAlmostEqual(cut.balance[-1], base.balance[-1] + 1_000.0, places=2)

    def test_evaluate_returns_both_models(self) -> None:
        transactions, invoices, meta = build_books()
        metrics = evaluate(
            transactions,
            invoices,
            opening=meta["opening_cash"],
            history_start=START,
            as_of=AS_OF,
            holdout_days=60,
            horizon=21,
        )
        self.assertIn("model_mae", metrics)
        self.assertIn("baseline_mae", metrics)
        self.assertGreater(metrics["model_mae"], 0)
        self.assertGreater(metrics["baseline_mae"], 0)
        self.assertEqual(metrics["origins"] > 0, True)

    def test_demo_flags_shortfall_baseline_misses(self) -> None:
        transactions, invoices, meta = build_books()
        result = forecast(
            transactions,
            invoices,
            opening=meta["opening_cash"],
            history_start=START,
            as_of=AS_OF,
            horizon=90,
            n_sims=40,
        )
        self.assertIsNotNone(result.shortfall_date)
        days_out = (result.shortfall_date - AS_OF).days
        self.assertGreaterEqual(days_out, 18)
        self.assertLessEqual(days_out, 25)
        self.assertGreaterEqual(min(result.baseline[:31]), 0)
        recent = transactions[
            (transactions["date"] >= pd.Timestamp(AS_OF - timedelta(days=90)))
            & (transactions["date"] <= pd.Timestamp(AS_OF))
        ]
        invoiced = invoices[
            (invoices["issue_date"] >= pd.Timestamp(AS_OF - timedelta(days=90)))
            & (invoices["issue_date"] <= pd.Timestamp(AS_OF))
        ]["amount"].sum()
        expenses = -recent.loc[recent["amount"] < 0, "amount"].sum()
        self.assertGreater(float(invoiced), float(expenses) * 0.5)


if __name__ == "__main__":
    unittest.main()
