"""Tests for the FRED client and cost-of-debt fallback path."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Use a fresh tmp cache for tests so we never touch the real one
_TMP_CACHE = Path(tempfile.mkdtemp(prefix="valuation_test_cache_"))
import os
os.environ["VALUATION_CACHE"] = str(_TMP_CACHE)

# IMPORTANT: import macro AFTER setting env var so its CACHE_DIR uses the tmp path
from macro import (
    FredClient,
    MacroEnvironment,
    credit_spread_for_rating,
    DAMODARAN_ERP,
)
import macro as macro_mod


def _clear_cache():
    """Wipe the test cache directory between tests."""
    for f in macro_mod.CACHE_DIR.glob("*.json"):
        f.unlink()


def assert_close(a, b, tol=1e-4, label=""):
    if a is None or b is None:
        ok = a == b
    elif isinstance(a, str) or isinstance(b, str):
        ok = a == b
    else:
        ok = abs(a - b) <= tol
    status = "✓" if ok else "✗"
    print(f"  {status} {label}: got {a}, expected ≈ {b}")
    if not ok:
        sys.exit(1)


def assert_true(cond, label):
    status = "✓" if cond else "✗"
    print(f"  {status} {label}")
    if not cond:
        sys.exit(1)


# ─── FredClient: API path ────────────────────────────────────────────────

def test_fred_latest_via_api():
    print("\n[FredClient.latest_observation via API]")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "observations": [
            {"date": "2026-04-21", "value": "4.32"},
            {"date": "2026-04-18", "value": "4.30"},
            {"date": "2026-04-17", "value": "."},  # missing
        ]
    }

    client = FredClient(api_key="dummy_key_for_test")
    with patch.object(client._session, "get", return_value=fake_response):
        result = client.latest_observation("DGS10")

    assert_true(result is not None, "Got a non-None result")
    val, date = result
    assert_close(val, 4.32, label="Latest value extracted (4.32)")
    assert_close(date, "2026-04-21", label="Latest date extracted")


def test_fred_skips_missing_dot():
    """FRED uses '.' for missing observations — must be filtered."""
    print("\n[FredClient skips '.' missing values]")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "observations": [
            {"date": "2026-04-21", "value": "."},
            {"date": "2026-04-18", "value": "."},
            {"date": "2026-04-17", "value": "4.25"},
        ]
    }
    client = FredClient(api_key="dummy")
    with patch.object(client._session, "get", return_value=fake_response):
        result = client.latest_observation("DGS10")
    assert_true(result is not None, "Skipped past missing observations")
    assert_close(result[0], 4.25, label="Took first non-missing value")


def test_fred_no_key_uses_csv():
    """When no API key is provided, must fall back to CSV path."""
    print("\n[FredClient with no key uses CSV fallback]")
    csv_content = "DATE,DGS10\n2026-04-18,4.30\n2026-04-21,4.32\n"
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()
    fake_response.text = csv_content

    client = FredClient(api_key=None)  # explicit None
    assert_true(not client.has_key(), "has_key() returns False")
    with patch.object(client._session, "get", return_value=fake_response):
        result = client.latest_observation("DGS10")

    assert_true(result is not None, "CSV fallback returned a result")
    assert_close(result[0], 4.32, label="Latest value from CSV (4.32)")


def test_fred_series_full():
    """Full series fetch returns a Series with proper datetime index."""
    print("\n[FredClient.series via API]")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "observations": [
            {"date": "2024-01-01", "value": "3.95"},
            {"date": "2024-04-01", "value": "4.10"},
            {"date": "2024-07-01", "value": "4.20"},
            {"date": "2024-10-01", "value": "."},
            {"date": "2025-01-01", "value": "4.30"},
        ]
    }
    client = FredClient(api_key="dummy")
    with patch.object(client._session, "get", return_value=fake_response):
        s = client.series("DGS10", start="2024-01-01")

    assert_true(s is not None, "Series returned")
    assert_true(len(s) == 4, f"Skipped one missing value (got {len(s)} of 5)")
    assert_true(isinstance(s.index, pd.DatetimeIndex), "Index is datetime")
    assert_close(s.iloc[0], 3.95, label="First value correct")
    assert_close(s.iloc[-1], 4.30, label="Last value correct")


# ─── credit_spread_for_rating: Damodaran ICR table ───────────────────────

def test_credit_spread_table():
    print("\n[credit_spread_for_rating mapping]")
    # AAA quality (high coverage)
    s = credit_spread_for_rating(10.0, baa_spread=0.020, aaa_spread=0.010)
    assert_close(s, 0.010, label="ICR=10 → AAA spread (1.0%)")

    # BBB (mid IG)
    s = credit_spread_for_rating(3.0, baa_spread=0.020, aaa_spread=0.010)
    assert_close(s, 0.020, label="ICR=3 → BBB spread (2.0%)")

    # Distressed (capped at 10%)
    s = credit_spread_for_rating(0.5, baa_spread=0.030, aaa_spread=0.020)
    assert_close(s, 0.030 * 3.0, label="ICR<1 with low BAA → 3×BAA = 9%")
    s = credit_spread_for_rating(0.5, baa_spread=0.050, aaa_spread=0.020)
    assert_close(s, 0.10, label="ICR<1 with high BAA → cap at 10%")

    # No ICR: returns midpoint
    s = credit_spread_for_rating(None, baa_spread=0.020, aaa_spread=0.010)
    assert_close(s, 0.015, label="No ICR → midpoint of AAA, BAA")

    # No FRED data either: hardcoded fallback
    s = credit_spread_for_rating(None, baa_spread=None, aaa_spread=None)
    assert_close(s, 0.015, label="No data → 1.5% default")


# ─── MacroEnvironment integration with mocked FRED ───────────────────────

def test_macro_environment_with_mock_fred():
    """Full MacroEnvironment.fetch flow using mocked FRED responses."""
    print("\n[MacroEnvironment.fetch end-to-end with mocks]")

    # Mock the FRED client to return canned values for each series
    series_values = {
        "DGS10":  ("4.30", "2026-04-21"),
        "DFII10": ("1.95", "2026-04-21"),
        "T10YIE": ("2.35", "2026-04-21"),
        "BAA10Y": ("1.85", "2026-04-21"),
        "AAA10Y": ("0.95", "2026-04-21"),
    }

    def fake_latest(series_id):
        if series_id in series_values:
            v, d = series_values[series_id]
            return (float(v), d)
        return None

    def fake_series(series_id, start=None, limit=None):
        if series_id == "GDP":
            # Synthesise 80 quarterly points growing at ~4% nominal
            dates = pd.date_range("2004-01-01", periods=80, freq="QS")
            vals = [12000 * (1.04 ** (i / 4)) for i in range(80)]
            return pd.Series(vals, index=dates, name="GDP")
        return None

    with patch.object(FredClient, "latest_observation", side_effect=fake_latest), \
         patch.object(FredClient, "series", side_effect=fake_series), \
         patch("macro._spy_implied_erp", return_value=None):  # force Damodaran default

        env = MacroEnvironment.fetch(fred_key="dummy")

    assert_close(env.risk_free_rate, 0.0430, label="rf from mocked DGS10")
    assert_close(env.real_risk_free_rate, 0.0195, label="real rf from DFII10")
    assert_close(env.expected_inflation, 0.0235, label="breakeven inflation from T10YIE")
    assert_close(env.baa_spread, 0.0185, label="BAA spread from BAA10Y")
    assert_close(env.aaa_spread, 0.0095, label="AAA spread from AAA10Y")
    assert_close(env.equity_risk_premium, DAMODARAN_ERP,
                 label="ERP fell back to Damodaran when SPY unavailable")
    assert_true(0.038 < env.long_run_gdp < 0.042,
                f"GDP CAGR ≈ 4% (got {env.long_run_gdp:.4f})")
    assert_true("FRED GDP CAGR" in env.source_gdp, "GDP source attributed correctly")
    assert_true("FRED DGS10" in env.source_rf, "rf source attributed correctly")


def test_macro_falls_back_to_yfinance_then_default():
    """When FRED returns nothing, must try yfinance, then default."""
    print("\n[MacroEnvironment fallback chain]")
    with patch.object(FredClient, "latest_observation", return_value=None), \
         patch.object(FredClient, "series", return_value=None), \
         patch("macro._yfinance_rf", return_value=0.0445), \
         patch("macro._spy_implied_erp", return_value=None):
        env = MacroEnvironment.fetch(fred_key="dummy")
    assert_close(env.risk_free_rate, 0.0445, label="Used yfinance RF fallback")
    assert_true("yfinance" in env.source_rf, "Provenance string mentions yfinance")
    assert_close(env.long_run_gdp, 0.040, label="GDP fell back to default")

    # Now turn off yfinance too
    with patch.object(FredClient, "latest_observation", return_value=None), \
         patch.object(FredClient, "series", return_value=None), \
         patch("macro._yfinance_rf", return_value=None), \
         patch("macro._spy_implied_erp", return_value=None):
        env = MacroEnvironment.fetch(fred_key="dummy")
    assert_close(env.risk_free_rate, 0.0430, label="Hardcoded RF default kicked in")
    assert_true("fallback" in env.source_rf.lower(), "Marked as fallback")


# ─── Integration: cost of debt now uses FRED spreads ─────────────────────

def test_cost_of_debt_uses_fred_when_no_statements():
    """When there's no interest_expense data, kd should use FRED BAA/AAA."""
    print("\n[Cost of debt uses FRED credit spreads as fallback]")
    from data import TickerData
    from assumptions import _cost_of_debt
    from macro import MacroEnvironment

    # Synthetic ticker with no statements at all
    td = TickerData(ticker="GHOST", shares_outstanding=100)
    macro = MacroEnvironment(
        risk_free_rate=0.043,
        equity_risk_premium=0.04,
        long_run_gdp=0.04,
        baa_spread=0.018,
        aaa_spread=0.009,
    )
    kd, source = _cost_of_debt(td, rf=0.043, macro=macro)

    # No ICR, no spread — uses midpoint of BAA & AAA
    expected_spread = (0.018 + 0.009) / 2
    assert_close(kd, 0.043 + expected_spread,
                 label=f"kd = rf + (BAA+AAA)/2 = {(0.043+expected_spread)*100:.2f}%")
    assert_true("FRED" in source, "Source attributed to FRED")


def test_cost_of_debt_prefers_real_data():
    """When interest_expense IS available, it wins over FRED spreads."""
    print("\n[Cost of debt prefers actual financials when available]")
    from data import TickerData
    from data import _derive_series
    from assumptions import _cost_of_debt
    from macro import MacroEnvironment

    # Build a synthetic ticker with interest expense and debt
    dates = pd.date_range("2022-12-31", periods=3, freq="YE")
    td = TickerData(ticker="REAL", shares_outstanding=100)
    td.income_stmt = pd.DataFrame(
        {d: {"Operating Income": 1000, "Net Income": 800,
             "Tax Provision": 200, "Pretax Income": 1000,
             "Interest Expense": 60, "Total Revenue": 5000} for d in dates}
    )
    td.balance_sheet = pd.DataFrame(
        {d: {"Total Assets": 5000, "Current Liabilities": 1000,
             "Total Debt": 1000, "Cash And Cash Equivalents": 200,
             "Stockholders Equity": 3000} for d in dates}
    )
    _derive_series(td)

    macro = MacroEnvironment(risk_free_rate=0.043, equity_risk_premium=0.04,
                             long_run_gdp=0.04, baa_spread=0.018, aaa_spread=0.009)
    kd, source = _cost_of_debt(td, rf=0.043, macro=macro)

    # IE/Debt = 60/1000 = 6%
    assert_close(kd, 0.06, label="kd = IE/Debt = 6.0%")
    assert_true("interest_expense / total_debt" in source,
                "Source attributed to actual financials, not FRED")


def run():
    tests = [
        ("FredClient.latest_observation via API", test_fred_latest_via_api),
        ("FredClient skips '.' missing values",   test_fred_skips_missing_dot),
        ("FredClient no-key falls back to CSV",   test_fred_no_key_uses_csv),
        ("FredClient.series via API",             test_fred_series_full),
        ("Damodaran ICR → spread mapping",        test_credit_spread_table),
        ("MacroEnvironment with mocked FRED",     test_macro_environment_with_mock_fred),
        ("Macro fallback chain (FRED→yf→def)",    test_macro_falls_back_to_yfinance_then_default),
        ("Cost of debt uses FRED spreads",        test_cost_of_debt_uses_fred_when_no_statements),
        ("Cost of debt prefers real financials",  test_cost_of_debt_prefers_real_data),
    ]
    for name, fn in tests:
        _clear_cache()  # ensure each test starts from a fresh cache
        fn()
    # Clean up tmp cache
    shutil.rmtree(_TMP_CACHE, ignore_errors=True)
    print(f"\n{'='*60}\nAll {len(tests)} FRED tests passed ✓\n{'='*60}")


if __name__ == "__main__":
    run()
