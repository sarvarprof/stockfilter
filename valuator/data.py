"""
Data loading layer.

Wraps yfinance with:
  - retries + exponential backoff for the flaky info dict
  - explicit logging when fields come back None
  - direct extraction from financial statements (preferred over info)
  - period-appropriate share counts for historical EPS

The class is intentionally narrow: it only fetches and exposes raw data.
All cleaning, normalisation, and assumption-building lives in assumptions.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Common alternate row names used by yfinance across different ticker types.
# Yahoo standardises imperfectly, so we have to tolerate variants.
ROW_ALIASES = {
    "operating_cash_flow": [
        "Operating Cash Flow", "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ],
    "capex": [
        "Capital Expenditure", "Capital Expenditures",
        "Net PPE Purchase And Sale", "Purchase Of PPE",
    ],
    "interest_expense": [
        "Interest Expense", "Interest Expense Non Operating",
        "Net Interest Income",  # banks
    ],
    "tax_provision": ["Tax Provision", "Income Tax Expense", "Provision For Income Taxes"],
    "pretax_income": ["Pretax Income", "Income Before Tax", "Earnings Before Tax"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "diluted_eps": ["Diluted EPS", "Diluted EPS Including Extraordinary Items"],
    "diluted_shares": ["Diluted Average Shares", "Weighted Average Shs Out (Dil)"],
    "basic_shares": ["Basic Average Shares", "Weighted Average Shs Out"],
    "total_assets": ["Total Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "stockholders_equity": ["Stockholders Equity", "Total Stockholder Equity"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "revenue": ["Total Revenue", "Operating Revenue"],
}


def _row(df: pd.DataFrame, key: str) -> pd.Series | None:
    """Look up a row by any of the known alias names."""
    if df is None or df.empty:
        return None
    for name in ROW_ALIASES.get(key, [key]):
        if name in df.index:
            row = df.loc[name].dropna()
            if len(row):
                return row.sort_index()  # ascending by date
    return None


@dataclass
class TickerData:
    """Raw data bundle for a single ticker. All fields nullable; cleaning later."""
    ticker: str

    # Price / market
    current_price: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None
    beta_raw: float | None = None  # yahoo's 5y monthly beta

    # Per-share fundamentals (point-in-time from `info`)
    trailing_eps: float | None = None
    forward_eps: float | None = None
    book_value_per_share: float | None = None
    current_dividend: float | None = None
    payout_ratio: float | None = None
    return_on_equity: float | None = None
    earnings_growth: float | None = None  # analyst forward
    trailing_pe: float | None = None
    forward_pe: float | None = None

    # Sector / industry
    sector: str = ""
    industry: str = ""

    # Whole financial statements (for proper analysis)
    income_stmt: pd.DataFrame | None = None
    balance_sheet: pd.DataFrame | None = None
    cash_flow: pd.DataFrame | None = None
    dividends: pd.Series | None = None

    # Derived series we extract once (with period-appropriate share counts)
    fcf_series: pd.Series | None = None         # annual FCF, oldest to newest
    eps_series: pd.Series | None = None         # diluted EPS, period-appropriate shares
    interest_expense_series: pd.Series | None = None
    tax_rate_series: pd.Series | None = None
    revenue_series: pd.Series | None = None
    ebitda_point: float | None = None           # most recent annual EBITDA

    # Diagnostics
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        log.warning("[%s] %s", self.ticker, msg)
        self.warnings.append(msg)


def _retry(fn, attempts: int = 3, base_delay: float = 0.5):
    """Simple exponential-backoff retry for the frequently-flaky info dict."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(base_delay * (2 ** i))
    if last_exc:
        log.debug("retry exhausted: %s", last_exc)
    return None


def load(ticker_symbol: str) -> TickerData:
    """Load a TickerData bundle from yfinance, with logging on every gap."""
    import yfinance as yf

    log.info("Loading %s...", ticker_symbol)
    t = yf.Ticker(ticker_symbol)
    info = _retry(lambda: t.info) or {}

    td = TickerData(ticker=ticker_symbol)

    # ── Top-line price/market data ──
    td.current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    td.shares_outstanding = info.get("sharesOutstanding")
    td.market_cap = info.get("marketCap")
    td.beta_raw = info.get("beta")

    # ── Per-share + ratio fundamentals (informational; we'll usually re-derive) ──
    td.trailing_eps = info.get("trailingEps")
    td.forward_eps = info.get("forwardEps")
    td.book_value_per_share = info.get("bookValue")
    td.current_dividend = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    td.payout_ratio = info.get("payoutRatio")
    td.return_on_equity = info.get("returnOnEquity")
    td.earnings_growth = info.get("earningsGrowth")
    td.trailing_pe = info.get("trailingPE")
    td.forward_pe = info.get("forwardPE")
    td.sector = info.get("sector") or ""
    td.industry = info.get("industry") or ""

    # Required: we can't value a company without shares
    if not td.shares_outstanding or td.shares_outstanding <= 0:
        td.warn("sharesOutstanding missing — cannot proceed")
        td.missing_fields.append("sharesOutstanding")
    if not td.current_price or td.current_price <= 0:
        td.warn("currentPrice missing")
        td.missing_fields.append("currentPrice")

    # ── Statements ──
    td.income_stmt = _retry(lambda: t.financials)
    td.balance_sheet = _retry(lambda: t.balance_sheet)
    td.cash_flow = _retry(lambda: t.cash_flow)
    div = _retry(lambda: t.dividends)
    td.dividends = div if (div is not None and len(div) > 0) else None

    if td.income_stmt is None or td.income_stmt.empty:
        td.warn("income statement unavailable")
    if td.balance_sheet is None or td.balance_sheet.empty:
        td.warn("balance sheet unavailable")
    if td.cash_flow is None or td.cash_flow.empty:
        td.warn("cash flow statement unavailable")

    _derive_series(td)
    return td


def _derive_series(td: TickerData) -> None:
    """Extract the time series we'll actually use, with period-aware share counts."""

    # ── FCF series: OCF + Capex (capex is reported negative on Yahoo) ──
    if td.cash_flow is not None:
        ocf = _row(td.cash_flow, "operating_cash_flow")
        capex = _row(td.cash_flow, "capex")
        if ocf is not None and capex is not None:
            # Reindex on intersection; capex is signed negative, so adding them
            # is the standard "OCF - |capex|" calculation.
            common = ocf.index.intersection(capex.index)
            if len(common) >= 2:
                td.fcf_series = (ocf.loc[common] + capex.loc[common]).sort_index()
            else:
                td.warn("FCF series too short")
        else:
            td.warn("OCF or capex missing from cash flow stmt")

    # ── EPS series: prefer Diluted EPS, else NI / period-appropriate shares ──
    if td.income_stmt is not None:
        diluted_eps = _row(td.income_stmt, "diluted_eps")
        if diluted_eps is not None and len(diluted_eps) >= 3:
            # Yahoo reports EPS already; trust it.
            td.eps_series = diluted_eps[diluted_eps > 0].sort_index()
        else:
            # Fallback: NI divided by the appropriate period's diluted shares.
            # The previous bug was using *current* shares for *historical* NI,
            # which gives wrong CAGRs for buyback-heavy names.
            ni = _row(td.income_stmt, "net_income")
            shares = _row(td.income_stmt, "diluted_shares")
            if shares is None:
                shares = _row(td.income_stmt, "basic_shares")
            if ni is not None and shares is not None:
                common = ni.index.intersection(shares.index)
                if len(common) >= 3:
                    eps = (ni.loc[common] / shares.loc[common]).sort_index()
                    td.eps_series = eps[eps > 0]
                else:
                    td.warn("NI/share series too short for EPS reconstruction")
            else:
                td.warn("Could not derive EPS series (no diluted EPS, NI, or shares)")

        # ── Tax rate series: Tax Provision / Pretax Income ──
        tax = _row(td.income_stmt, "tax_provision")
        pretax = _row(td.income_stmt, "pretax_income")
        if tax is not None and pretax is not None:
            common = tax.index.intersection(pretax.index)
            if len(common) >= 1:
                # Avoid division blow-ups when pretax is near zero
                ratio = (tax.loc[common] / pretax.loc[common]).sort_index()
                ratio = ratio[(ratio > -0.5) & (ratio < 0.7)]  # sanity bands
                if len(ratio) >= 1:
                    td.tax_rate_series = ratio

        # ── Interest expense series (for cost of debt) ──
        ie = _row(td.income_stmt, "interest_expense")
        if ie is not None:
            # Some sources sign it negative; we want positive magnitude
            td.interest_expense_series = ie.abs()

        # ── Revenue (for sanity-checking FCF margins) ──
        rev = _row(td.income_stmt, "revenue")
        if rev is not None:
            td.revenue_series = rev

        # ── EBITDA (point estimate) ──
        ebitda = _row(td.income_stmt, "ebitda")
        if ebitda is not None and len(ebitda) > 0:
            td.ebitda_point = float(ebitda.iloc[-1])  # most recent
        else:
            # Fall back to info dict
            pass  # caller can still use info["ebitda"] via assumptions layer
