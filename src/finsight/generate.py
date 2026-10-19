"""Build a reproducible set of books for Harbor & Co."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.forecast import cash_on

AS_OF = date(2026, 3, 2)
START = date(2024, 9, 1)
COMPANY = "Harbor & Co."
TARGET_CASH = 16_000.0
HERO_INVOICE_ID = "BR-109"
HERO_CUSTOMER = "Brightline Studio"
DISCRETIONARY = "marketing"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

SCHEDULED = (
    ("rent", 1, 4_800.0, "Oak Street Landlord"),
    ("payroll", 23, 22_000.0, "Payroll"),
    ("loan", 25, 1_800.0, "North Bank"),
    ("marketing", 18, 2_400.0, "Ad Network"),
)


def build_books(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return transactions, invoices, and metadata.

    Cash on the as-of date is scaled to TARGET_CASH. The only open invoice is
    Brightline's, and its usual collection lag puts the cash after payroll.
    """
    rng = np.random.default_rng(seed)
    transactions: list[dict] = []
    invoices: list[dict] = []

    def spend(day: date, amount: float, category: str, counterparty: str, description: str) -> None:
        if day < START or day > AS_OF:
            return
        transactions.append(
            {
                "date": day,
                "amount": -abs(float(amount)),
                "category": category,
                "counterparty": counterparty,
                "description": description,
            }
        )

    def collect(day: date, amount: float, customer: str, invoice_id: str) -> None:
        transactions.append(
            {
                "date": day,
                "amount": abs(float(amount)),
                "category": "client_payment",
                "counterparty": customer,
                "description": invoice_id,
            }
        )

    cursor = date(START.year, START.month, 1)
    while cursor <= AS_OF:
        for category, day_of_month, amount, counterparty in SCHEDULED:
            event_day = date(cursor.year, cursor.month, day_of_month)
            spend(event_day, amount, category, counterparty, category)
            if category == "marketing" and event_day == date(2026, 2, 18):
                spend(event_day, 4_500.0, category, counterparty, "campaign overrun")
        utilities = 620.0 + float(rng.normal(0, 40))
        spend(date(cursor.year, cursor.month, 6), utilities, "utilities", "City Power", "utilities")
        spend(date(cursor.year, cursor.month, 3), 420.0, "software", "Toolstack", "software")
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    day = START
    while day <= AS_OF:
        if day.weekday() < 5:
            spend(day, float(rng.uniform(35, 80)), "supplies", "Office", "supplies")
        if day.weekday() == 4:
            spend(day, 900.0, "contractor", "Lane Design", "contractor")
        day += timedelta(days=1)

    monthly_customers = (
        ("Maple Goods", 7_200.0, 5, 10, (4, 5, 6)),
        ("Harbor Civic", 9_400.0, 12, 14, (6, 7, 8)),
        ("Brightline Studio", 19_000.0, 8, 14, (8, 14, 20, 26, 32)),
    )
    for name, amount, issue_day, terms, delays in monthly_customers:
        cursor = date(START.year, START.month, 1)
        sequence = 1
        while cursor <= AS_OF:
            issued = date(cursor.year, cursor.month, issue_day)
            due = issued + timedelta(days=terms)
            delay = delays[(sequence - 1) % len(delays)]
            paid = due + timedelta(days=delay)
            # February Brightline is the open hero invoice, added separately.
            is_hero_slot = name == HERO_CUSTOMER and issued.year == 2026 and issued.month == 2
            if not is_hero_slot and issued <= AS_OF and paid <= AS_OF:
                invoice_id = f"{name[:2].upper()}-{sequence:03d}"
                invoices.append(
                    {
                        "invoice_id": invoice_id,
                        "customer": name,
                        "amount": amount,
                        "issue_date": issued,
                        "due_date": due,
                        "paid_date": paid,
                    }
                )
                collect(paid, amount, name, invoice_id)
            sequence += 1
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)

    # A cleared retainer in late February keeps the trailing 30-day average
    # positive. It is not an open invoice, so it does not fund March payroll.
    retainer_paid = date(2026, 2, 28)
    invoices.append(
        {
            "invoice_id": "SR-014",
            "customer": "Seasonal Retainer",
            "amount": 32_000.0,
            "issue_date": date(2026, 2, 10),
            "due_date": date(2026, 2, 20),
            "paid_date": retainer_paid,
        }
    )
    collect(retainer_paid, 32_000.0, "Seasonal Retainer", "SR-014")

    invoices.append(
        {
            "invoice_id": HERO_INVOICE_ID,
            "customer": HERO_CUSTOMER,
            "amount": 20_000.0,
            "issue_date": date(2026, 2, 18),
            "due_date": date(2026, 3, 10),
            "paid_date": pd.NA,
        }
    )

    tx = pd.DataFrame(transactions)
    tx["date"] = pd.to_datetime(tx["date"])
    inv = pd.DataFrame(invoices)
    for column in ("issue_date", "due_date", "paid_date"):
        inv[column] = pd.to_datetime(inv[column])

    level = cash_on(tx, 0.0, START, AS_OF)
    opening = TARGET_CASH - level
    tx = tx.sort_values("date").reset_index(drop=True)
    meta = {
        "company": COMPANY,
        "as_of": AS_OF.isoformat(),
        "history_start": START.isoformat(),
        "opening_cash": opening,
        "target_cash": TARGET_CASH,
        "hero_invoice_id": HERO_INVOICE_ID,
        "hero_customer": HERO_CUSTOMER,
        "discretionary_category": DISCRETIONARY,
    }
    return tx, inv, meta


def write_books(directory: Path | None = None) -> Path:
    directory = directory or DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    transactions, invoices, meta = build_books()
    transactions.to_csv(directory / "transactions.csv", index=False)
    invoices.to_csv(directory / "invoices.csv", index=False)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    return directory


def load_books(directory: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = directory or DATA_DIR
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        write_books(directory)
    meta = json.loads(meta_path.read_text())
    transactions = pd.read_csv(directory / "transactions.csv", parse_dates=["date"])
    invoices = pd.read_csv(directory / "invoices.csv", parse_dates=["issue_date", "due_date", "paid_date"])
    return transactions, invoices, meta


if __name__ == "__main__":
    print(write_books())


__all__ = [
    "AS_OF",
    "START",
    "COMPANY",
    "TARGET_CASH",
    "HERO_INVOICE_ID",
    "build_books",
    "write_books",
    "load_books",
]
