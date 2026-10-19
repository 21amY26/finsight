# FinSight AI

Predictive cash-flow intelligence for a small business. The prototype forecasts Harbor & Co., a synthetic studio that looks busy on recent invoices and still runs short of cash when payroll lands before a large customer pays.

The schedule-aware forecast places recurring bills on their calendar days and open invoices on each customer's usual collection lag. A trailing-average baseline only extends the last 30 days of net cash. On the held-out 60 days in the sample books, the schedule-aware forecast has lower daily cash-balance error, and it flags the January shortfall before it happens. The baseline does not. Those figures are shown in the app and apply only to this synthetic protocol.

Scenario sliders recalculate the same engine. They are calculations under the stated assumptions, not guarantees. There is no chatbot.

## Setup

Use a project virtual environment. Do not install these packages into the system interpreter.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Generate the sample books (also written automatically the first time the app loads) and start the dashboard:

```bash
PYTHONPATH=src .venv/bin/python -m finsight.generate
PYTHONPATH=src .venv/bin/streamlit run app.py
```

Tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_cash
```

## What the screen shows

- **Position.** Cash today, 90-day invoicing versus expenses, the projected low, and the share of simulated paths that hit zero within 30 days.
- **Forecast.** History, the schedule-aware path, a 10th–90th percentile band, the baseline, and the shortfall date.
- **Why.** The collections and bills behind the next 30 days, including the item that takes cash through zero.
- **What-if.** Move an open invoice's collection date, or cut a discretionary expense, and compare the path.
- **Evaluation.** Holdout mean absolute error and shortfall lead time for both methods.
