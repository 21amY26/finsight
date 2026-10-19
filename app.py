"""FinSight AI dashboard."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finsight.evaluate import evaluate
from finsight.explain import explain
from finsight.forecast import as_date, daily_balance, forecast
from finsight.generate import load_books
from finsight.simulate import apply_scenario

st.set_page_config(page_title="FinSight AI", layout="wide")


def money(value: float) -> str:
    return f"${value:,.0f}"


def prose_money(value: float) -> str:
    """Dollar amounts for Streamlit markdown, which treats $ as math."""
    return f"USD {value:,.0f}"


@st.cache_data(show_spinner=False)
def books():
    return load_books()


transactions, invoices, meta = books()
as_of = date.fromisoformat(meta["as_of"])
history_start = date.fromisoformat(meta["history_start"])
opening = float(meta["opening_cash"])

base = forecast(
    transactions,
    invoices,
    opening=opening,
    history_start=history_start,
    as_of=as_of,
    horizon=90,
    n_sims=200,
)
history = daily_balance(transactions, opening, history_start, as_of)
recent_start = as_of - timedelta(days=90)
recent_tx = transactions[
    (transactions["date"] >= pd.Timestamp(recent_start))
    & (transactions["date"] <= pd.Timestamp(as_of))
]
invoiced = float(
    invoices.loc[
        (invoices["issue_date"] >= pd.Timestamp(recent_start))
        & (invoices["issue_date"] <= pd.Timestamp(as_of)),
        "amount",
    ].sum()
)
collected = float(recent_tx.loc[recent_tx["category"] == "client_payment", "amount"].sum())
expenses = float(-recent_tx.loc[recent_tx["amount"] < 0, "amount"].sum())
story = explain(base, transactions)

st.title("FinSight AI")
st.caption(
    f"{meta['company']} · books through {as_of.isoformat()} · "
    "synthetic data. Scenario results are calculations under the stated assumptions, not guarantees."
)

st.header("Position")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cash today", money(base.cash_today))
c2.metric("Invoiced, 90 days", money(invoiced))
c3.metric("Collected, 90 days", money(collected))
c4.metric("Expenses, 90 days", money(expenses))
c5, c6, c7, c8 = st.columns(4)
c5.metric("Lowest cash, 30 days", money(story["min_30"]))
c6.metric("Lowest cash, 90 days", money(story["min_90"]))
c7.metric(
    "Shortfall date",
    base.shortfall_date.isoformat() if base.shortfall_date else "None in 90 days",
)
c8.metric("Paths below zero within 30 days", f"{base.shortfall_prob_30:.0%}")
if invoiced > expenses and base.shortfall_date is not None:
    st.info(
        f"Invoices over the last 90 days ({prose_money(invoiced)}) exceed expenses ({prose_money(expenses)}), "
        f"but the base cash path still crosses zero on {base.shortfall_date.isoformat()}."
    )
for note in base.delay_notes:
    st.warning(note.replace("$", "USD "))

st.header("Forecast")
chart_start = as_of - timedelta(days=90)
hist = history[history.index >= chart_start]
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(hist.index), y=list(hist.values), name="History", line={"color": "#334155"}))
fig.add_trace(
    go.Scatter(
        x=base.dates,
        y=base.high,
        name="p90",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=base.dates,
        y=base.low,
        name="10th–90th percentile",
        fill="tonexty",
        line={"width": 0},
        fillcolor="rgba(37, 99, 235, 0.18)",
    )
)
fig.add_trace(go.Scatter(x=base.dates, y=base.balance, name="Schedule-aware", line={"color": "#2563eb", "width": 3}))
fig.add_trace(
    go.Scatter(
        x=base.dates,
        y=base.baseline,
        name="Trailing-average baseline",
        line={"color": "#b45309", "dash": "dash"},
    )
)
if base.shortfall_date is not None:
    fig.add_vline(
        x=base.shortfall_date.isoformat(),
        line_dash="dot",
        line_color="#b91c1c",
        annotation_text="Shortfall",
        annotation_position="top left",
    )
fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
fig.update_layout(
    height=420,
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
    legend={"orientation": "h", "y": 1.12},
    yaxis_title="Cash",
    xaxis_title=None,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The baseline extends the last 30 days of average daily net cash and ignores payroll, rent, and open invoices. "
    "The band resamples historical payment delays and variable-spend residuals."
)

st.header("Why")
left, right = st.columns(2)
with left:
    st.subheader("Inflows in the next 30 days")
    if story["inflows"].empty:
        st.write("No collections land in the next 30 days under the base assumptions.")
    else:
        st.dataframe(
            story["inflows"].rename("amount").map(lambda value: money(float(value))).reset_index(),
            hide_index=True,
            use_container_width=True,
        )
with right:
    st.subheader("Scheduled outflows in the next 30 days")
    st.dataframe(
        story["outflows"].rename("amount").map(lambda value: money(float(value))).reset_index(),
        hide_index=True,
        use_container_width=True,
    )
for bullet in story["bullets"]:
    st.write("• " + bullet.replace("$", "USD "))
with st.expander("Assumptions"):
    for item in story["assumptions"]:
        st.write("• " + item)

st.header("What-if")
st.write(
    "Change one collection date or reduce one expense. The chart uses the same cash engine as the base forecast."
)
open_inv = invoices[invoices["paid_date"].isna()].copy()
open_ids = open_inv["invoice_id"].astype(str).tolist()
default_id = meta["hero_invoice_id"] if meta["hero_invoice_id"] in open_ids else open_ids[0]
pick, date_col, cut_col = st.columns(3)
with pick:
    invoice_id = st.selectbox("Open invoice", open_ids, index=open_ids.index(default_id))
selected = open_inv[open_inv["invoice_id"].astype(str) == invoice_id].iloc[0]
base_expected = base.scheduled.loc[
    base.scheduled.get("invoice_id", pd.Series(dtype=str)).astype(str) == invoice_id, "date"
]
if len(base_expected):
    starting_day = as_date(base_expected.iloc[0])
else:
    starting_day = as_date(selected["due_date"])
with date_col:
    expected_date = st.date_input(
        "Expected collection date",
        value=starting_day,
        min_value=as_of + timedelta(days=1),
        max_value=as_of + timedelta(days=90),
    )
with cut_col:
    category = st.selectbox("Expense to reduce", ["marketing", "contractor", "software"], index=0)
reduction = st.number_input("Reduction (%)", min_value=0, max_value=100, value=0, step=10)

scenario = apply_scenario(
    transactions,
    invoices,
    opening=opening,
    history_start=history_start,
    as_of=as_of,
    invoice_id=invoice_id,
    expected_date=expected_date,
    category=category,
    reduction_pct=reduction / 100,
    horizon=90,
    n_sims=80,
)
s1, s2, s3 = st.columns(3)
s1.metric(
    "Scenario shortfall",
    scenario.shortfall_date.isoformat() if scenario.shortfall_date else "None in 90 days",
)
s2.metric("Scenario lowest cash, 30 days", money(min(scenario.balance[:31])))
s3.metric(
    "Change in 30-day low vs base",
    money(min(scenario.balance[:31]) - story["min_30"]),
)
compare = go.Figure()
compare.add_trace(go.Scatter(x=base.dates, y=base.balance, name="Base", line={"color": "#2563eb"}))
compare.add_trace(go.Scatter(x=scenario.dates, y=scenario.balance, name="Scenario", line={"color": "#059669"}))
compare.add_hline(y=0, line_color="#94a3b8", line_width=1)
compare.update_layout(height=360, margin={"l": 20, "r": 20, "t": 20, "b": 20}, legend={"orientation": "h"})
st.plotly_chart(compare, use_container_width=True)
scenario_story = explain(scenario, transactions)
for bullet in scenario_story["bullets"]:
    st.write("• " + bullet.replace("$", "USD "))

st.header("Evaluation")
st.write(
    "Weekly origins in the 60 days before the as-of date each forecast the next 21 days. "
    "Error is the mean absolute difference from the actual daily cash balance. "
    "Lead time is how far ahead a method first projected a below-zero balance on or before the holdout shortfall."
)
metrics = evaluate(
    transactions,
    invoices,
    opening=opening,
    history_start=history_start,
    as_of=as_of,
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Schedule-aware MAE", money(metrics["model_mae"]))
m2.metric("Baseline MAE", money(metrics["baseline_mae"]))
m3.metric(
    "Model lead time",
    "—" if metrics["model_lead_days"] is None else f"{metrics['model_lead_days']} days",
)
m4.metric(
    "Baseline lead time",
    "—" if metrics["baseline_lead_days"] is None else f"{metrics['baseline_lead_days']} days",
)
if metrics["model_improves_mae"]:
    st.success(
        f"On this holdout the schedule-aware forecast MAE is {prose_money(metrics['model_mae'])}, "
        f"compared with {prose_money(metrics['baseline_mae'])} for the trailing average."
    )
else:
    st.warning("On this holdout the trailing-average baseline has the lower cash-balance error.")
event = metrics["shortfall_event"]
if event is None:
    st.write("Cash did not close below zero during the holdout, so lead time is not defined.")
else:
    model_lead = metrics["model_lead_days"]
    baseline_lead = metrics["baseline_lead_days"]
    model_text = "did not flag it" if model_lead is None else f"flagged it {model_lead} days ahead"
    baseline_text = "did not flag it" if baseline_lead is None else f"flagged it {baseline_lead} days ahead"
    st.write(
        f"The holdout cash balance first closed below zero on {event.isoformat()}. "
        f"The schedule-aware forecast {model_text}; the baseline {baseline_text}."
    )
