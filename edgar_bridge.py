"""
edgar_bridge.py — thin adapter that exposes the valuator/edgar.py data layer
to penny_filter, growth_filter, and value_filter.

Design principles:
  • Never imported at module level inside filters — each filter calls
    get_ticker_data(ticker) lazily so the EDGAR hit only happens when
    that specific sub-check runs.
  • Falls back silently to None when EDGAR is unavailable (non-SEC filer,
    network error, no user-agent).  Callers always check for None.
  • Reuses the valuator/edgar.py disk cache (3-day TTL) so a ticker that
    was already fetched for valuation costs zero network round-trips.

Public API (all return None on failure):
  get_ticker_data(ticker)          → TickerData | None
  get_revenue_series(ticker)       → pd.Series | None   (annual, $, ascending)
  get_fcf_series(ticker)           → pd.Series | None   (annual, ascending)
  get_eps_series(ticker)           → pd.Series | None   (annual, ascending)
  get_shares_series(ticker)        → pd.Series | None   (annual, ascending)
  get_cash_debt_ocf(ticker)        → dict | None
      keys: cash, total_debt, equity, ocf, d_to_e
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── bootstrap: add valuator/ to path once ─────────────────────────────────
_HERE = Path(__file__).parent
_VALUATOR_DIR = _HERE / "valuator"
if str(_VALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VALUATOR_DIR))

_DEFAULT_UA = os.environ.get(
    "EDGAR_USER_AGENT", "stockscreener saidrakhmonov94@gmail.com"
)

# ── lazy imports (only resolve on first use) ───────────────────────────────
def _edgar_load(ticker: str):
    """Import edgar module and load TickerData. Returns None on any failure."""
    try:
        import edgar as _edgar
        ua = os.environ.get("EDGAR_USER_AGENT") or _DEFAULT_UA
        td = _edgar.load(ticker, user_agent=ua)
        return td
    except Exception as exc:
        log.debug("[edgar_bridge] %s load failed: %s", ticker, exc)
        return None


def _row(df, key: str):
    """Re-export data._row so callers don't need to import it."""
    try:
        from data import _row as _dr
        return _dr(df, key)
    except Exception:
        return None


# ── in-process cache (complements the on-disk 3-day EDGAR cache) ──────────
_td_cache: dict[str, Any] = {}


def get_ticker_data(ticker: str):
    """
    Load TickerData for *ticker* (EDGAR primary + yfinance market data).
    Returns None when EDGAR is unavailable or the ticker isn't a SEC filer.
    Results are held in a process-level dict so repeated calls within one
    request are free.
    """
    ticker = ticker.upper()
    if ticker in _td_cache:
        return _td_cache[ticker]
    td = _edgar_load(ticker)
    _td_cache[ticker] = td          # cache even None to avoid retries
    return td


def get_revenue_series(ticker: str):
    """Annual revenue, oldest-to-newest (pd.Series[float])."""
    td = get_ticker_data(ticker)
    if td is None:
        return None
    return getattr(td, "revenue_series", None)


def get_fcf_series(ticker: str):
    """Annual FCF (OCF + CapEx), oldest-to-newest."""
    td = get_ticker_data(ticker)
    if td is None:
        return None
    return getattr(td, "fcf_series", None)


def get_eps_series(ticker: str):
    """Annual diluted EPS, oldest-to-newest."""
    td = get_ticker_data(ticker)
    if td is None:
        return None
    return getattr(td, "eps_series", None)


def get_shares_series(ticker: str):
    """
    Annual diluted shares outstanding, oldest-to-newest.
    Derived from the income statement 'Diluted Average Shares' row.
    """
    td = get_ticker_data(ticker)
    if td is None:
        return None
    if td.income_stmt is None or td.income_stmt.empty:
        return None
    row = _row(td.income_stmt, "diluted_shares")
    if row is None:
        row = _row(td.income_stmt, "basic_shares")
    return row   # already sorted ascending by _row


def get_cash_debt_ocf(ticker: str) -> dict | None:
    """
    Returns a dict with audited balance-sheet and cash-flow figures:
      cash        float  – most recent cash & equivalents ($)
      total_debt  float  – LTD + STD properly summed ($)
      equity      float  – stockholders' equity ($), may be None
      ocf         float  – most recent annual operating cash flow ($)
      d_to_e      float  – debt / equity as a fraction (not %; None if equity ≤ 0)

    All values come from the annual statements.  Returns None if the balance
    sheet is unavailable.
    """
    td = get_ticker_data(ticker)
    if td is None:
        return None
    if td.balance_sheet is None or td.balance_sheet.empty:
        return None

    def _latest(key: str):
        r = _row(td.balance_sheet, key)
        if r is not None and len(r):
            return float(r.iloc[-1])
        return None

    cash       = _latest("cash")
    total_debt = _latest("total_debt")
    equity     = _latest("stockholders_equity")

    # OCF from cash flow statement
    ocf = None
    if td.cash_flow is not None and not td.cash_flow.empty:
        r = _row(td.cash_flow, "operating_cash_flow")
        if r is not None and len(r):
            ocf = float(r.iloc[-1])

    if cash is None and total_debt is None and ocf is None:
        return None

    d_to_e = None
    if total_debt is not None and equity is not None and equity > 0:
        d_to_e = total_debt / equity     # fraction (e.g. 0.45 = 45%)

    return {
        "cash":       cash       or 0.0,
        "total_debt": total_debt or 0.0,
        "equity":     equity,
        "ocf":        ocf        or 0.0,
        "d_to_e":     d_to_e,
    }


def clear_cache() -> None:
    """Flush the process-level cache (useful between test runs)."""
    _td_cache.clear()
