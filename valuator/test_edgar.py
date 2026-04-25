"""
Tests for SEC EDGAR loader.

Uses mocked HTTP responses shaped like real EDGAR companyfacts payloads.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Fresh test cache, set BEFORE importing edgar module
import os
_TMP = Path(tempfile.mkdtemp(prefix="valuation_edgar_test_"))
os.environ["VALUATION_CACHE"] = str(_TMP)

import edgar
from edgar import (
    EdgarClient,
    _RateLimiter,
    _extract_annual_series,
    _construct_total_debt,
    _sic_to_sector,
    CONCEPT_ALIASES,
    load_from_edgar,
)


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


def _clear_cache():
    for f in edgar.CACHE_DIR.glob("*.json"):
        f.unlink()


# ─── Synthetic EDGAR payload, AAPL-shaped ──────────────────────────────────

def make_synthetic_companyfacts():
    """A miniature companyfacts JSON in EDGAR's exact format."""
    def annual_obs(years_values, unit_dim="USD"):
        # years_values = [(2021, val), (2022, val), ...]
        return [
            {
                "end": f"{y}-09-30",
                "val": v,
                "fy": y,
                "fp": "FY",
                "form": "10-K",
                "filed": f"{y}-11-01",
                "accn": f"00000-{y}",
            }
            for y, v in years_values
        ]

    return {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues", "description": "Total revenue",
                    "units": {"USD": annual_obs([
                        (2021, 365_817_000_000),
                        (2022, 394_328_000_000),
                        (2023, 383_285_000_000),
                        (2024, 391_035_000_000),
                    ])},
                },
                "NetIncomeLoss": {
                    "units": {"USD": annual_obs([
                        (2021,  94_680_000_000),
                        (2022,  99_803_000_000),
                        (2023,  96_995_000_000),
                        (2024,  93_736_000_000),
                    ])},
                },
                "OperatingIncomeLoss": {
                    "units": {"USD": annual_obs([
                        (2021, 108_949_000_000),
                        (2022, 119_437_000_000),
                        (2023, 114_301_000_000),
                        (2024, 123_216_000_000),
                    ])},
                },
                "IncomeLossBeforeIncomeTaxes": {
                    "units": {"USD": annual_obs([
                        (2021, 109_207_000_000),
                        (2022, 119_103_000_000),
                        (2023, 113_736_000_000),
                        (2024, 122_269_000_000),
                    ])},
                },
                "IncomeTaxExpenseBenefit": {
                    "units": {"USD": annual_obs([
                        (2021, 14_527_000_000),
                        (2022, 19_300_000_000),
                        (2023, 16_741_000_000),
                        (2024, 28_533_000_000),
                    ])},
                },
                "InterestExpense": {
                    "units": {"USD": annual_obs([
                        (2021, 2_645_000_000),
                        (2022, 2_931_000_000),
                        (2023, 3_933_000_000),
                        (2024, 4_034_000_000),
                    ])},
                },
                "EarningsPerShareDiluted": {
                    "units": {"USD/shares": annual_obs([
                        (2021, 5.61),
                        (2022, 6.11),
                        (2023, 6.13),
                        (2024, 6.08),
                    ], unit_dim="USD/shares")},
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": annual_obs([
                        (2021, 16_864_919_000),
                        (2022, 16_325_819_000),
                        (2023, 15_812_547_000),
                        (2024, 15_408_095_000),
                    ], unit_dim="shares")},
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": annual_obs([
                        (2021, 104_038_000_000),
                        (2022, 122_151_000_000),
                        (2023, 110_543_000_000),
                        (2024, 118_254_000_000),
                    ])},
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": annual_obs([
                        (2021, 11_085_000_000),
                        (2022, 10_708_000_000),
                        (2023, 10_959_000_000),
                        (2024,  9_447_000_000),
                    ])},
                },
                "Assets": {
                    "units": {"USD": annual_obs([
                        (2021, 351_002_000_000),
                        (2022, 352_755_000_000),
                        (2023, 352_583_000_000),
                        (2024, 364_980_000_000),
                    ])},
                },
                "LiabilitiesCurrent": {
                    "units": {"USD": annual_obs([
                        (2021, 125_481_000_000),
                        (2022, 153_982_000_000),
                        (2023, 145_308_000_000),
                        (2024, 176_392_000_000),
                    ])},
                },
                "LongTermDebt": {
                    "units": {"USD": annual_obs([
                        (2021, 109_106_000_000),
                        (2022,  98_959_000_000),
                        (2023,  95_281_000_000),
                        (2024,  85_750_000_000),
                    ])},
                },
                "ShortTermBorrowings": {
                    "units": {"USD": annual_obs([
                        (2021,  9_613_000_000),
                        (2022, 11_128_000_000),
                        (2023,  5_985_000_000),
                        (2024, 10_912_000_000),
                    ])},
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {"USD": annual_obs([
                        (2021, 34_940_000_000),
                        (2022, 23_646_000_000),
                        (2023, 29_965_000_000),
                        (2024, 29_943_000_000),
                    ])},
                },
                "StockholdersEquity": {
                    "units": {"USD": annual_obs([
                        (2021, 63_090_000_000),
                        (2022, 50_672_000_000),
                        (2023, 62_146_000_000),
                        (2024, 56_950_000_000),
                    ])},
                },
                "CommonStockSharesOutstanding": {
                    "units": {"shares": annual_obs([
                        (2021, 16_426_786_000),
                        (2022, 15_943_425_000),
                        (2023, 15_550_061_000),
                        (2024, 15_115_823_000),
                    ], unit_dim="shares")},
                },
            }
        },
    }


def make_synthetic_ticker_map():
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


def make_synthetic_submissions():
    return {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "sic": "3571",
        "sicDescription": "Electronic Computers",
    }


# ─── Tests ─────────────────────────────────────────────────────────────────

def test_user_agent_required():
    print("\n[EdgarClient requires User-Agent with email]")
    try:
        EdgarClient(user_agent="")
        print("  ✗ Should have raised on empty user agent")
        sys.exit(1)
    except ValueError:
        print("  ✓ Empty user agent rejected")
    try:
        EdgarClient(user_agent="just-a-name-no-email")
        print("  ✗ Should have raised on missing email")
        sys.exit(1)
    except ValueError:
        print("  ✓ User agent without '@' rejected")
    EdgarClient(user_agent="test test@example.com")
    print("  ✓ Valid user agent accepted")


def test_rate_limiter():
    print("\n[Rate limiter enforces minimum interval]")
    import time
    rl = _RateLimiter(rate_per_sec=20.0)  # 50ms minimum
    rl.wait()
    start = time.time()
    rl.wait()
    elapsed = time.time() - start
    assert_true(0.040 < elapsed < 0.080,
                f"Second call delayed ~50ms (got {elapsed*1000:.0f}ms)")


def test_extract_annual_series_picks_first_alias():
    print("\n[_extract_annual_series picks first matching concept]")
    facts = make_synthetic_companyfacts()
    s = _extract_annual_series(facts, ["NonexistentTag", "Revenues"])
    assert_true(s is not None and len(s) == 4, "Got 4 annual revenue obs")
    assert_close(s.iloc[-1], 391_035_000_000, label="2024 revenue")
    assert_true(s.is_monotonic_increasing or
                s.index.is_monotonic_increasing,
                "Series sorted by date ascending")


def test_extract_annual_series_handles_restatements():
    """If two 10-Ks cover the same period end, take the one with later 'filed'."""
    print("\n[_extract_annual_series prefers later-filed restatements]")
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [
                        # Original filing
                        {"end": "2023-09-30", "val": 380_000_000_000,
                         "fp": "FY", "form": "10-K", "filed": "2023-11-01"},
                        # Restated filing
                        {"end": "2023-09-30", "val": 383_285_000_000,
                         "fp": "FY", "form": "10-K/A", "filed": "2024-03-15"},
                    ]}
                }
            }
        }
    }
    s = _extract_annual_series(facts, ["Revenues"])
    assert_true(s is not None and len(s) == 1, "One observation per period")
    assert_close(s.iloc[0], 383_285_000_000, label="Took the restated value")


def test_extract_filters_to_full_year():
    """fp != FY should be excluded (don't take Q-period 10-K/A by accident)."""
    print("\n[_extract_annual_series filters to FY observations]")
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [
                        {"end": "2023-09-30", "val": 100, "fp": "FY",
                         "form": "10-K", "filed": "2023-11-01"},
                        {"end": "2023-06-30", "val": 75, "fp": "Q3",
                         "form": "10-K", "filed": "2023-08-01"},  # quarterly oddball
                    ]}
                }
            }
        }
    }
    s = _extract_annual_series(facts, ["Revenues"])
    assert_true(s is not None and len(s) == 1, "Only FY observation kept")
    assert_close(s.iloc[0], 100, label="Correct value")


def test_construct_total_debt():
    print("\n[_construct_total_debt sums LTD + STD period-aligned]")
    facts = make_synthetic_companyfacts()
    debt = _construct_total_debt(facts)
    assert_true(debt is not None and len(debt) == 4,
                f"4 periods of total debt (got {len(debt) if debt is not None else 0})")
    # 2024: LTD 85.75B + STD 10.91B = 96.66B
    assert_close(debt.iloc[-1], 85_750_000_000 + 10_912_000_000,
                 label="2024 total debt = LTD + STD")


def test_sic_to_sector():
    print("\n[SIC code → sector mapping]")
    assert_close(_sic_to_sector("3571"), "Technology", label="3571 (computers) → Technology")
    assert_close(_sic_to_sector("6020"), "Financial Services", label="6020 (banks) → Financial")
    assert_close(_sic_to_sector("6798"), "Financial Services", label="6798 (REIT trust) → Financial Services")
    assert_close(_sic_to_sector("6500"), "Real Estate", label="6500 → Real Estate")
    assert_close(_sic_to_sector("4911"), "Utilities", label="4911 (electric utility) → Utilities")
    assert_close(_sic_to_sector(""), "", label="Empty SIC → empty string")
    assert_close(_sic_to_sector("garbage"), "", label="Bad SIC → empty string")


def test_ticker_map_caching():
    print("\n[Ticker map cached after first fetch]")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = make_synthetic_ticker_map()

    client = EdgarClient(user_agent="test test@example.com")
    with patch.object(client._session, "get", return_value=fake_response) as mock_get:
        m1 = client._ticker_map()
        m2 = client._ticker_map()  # should hit cache, not network
    assert_true(mock_get.call_count == 1, f"Only 1 HTTP call (got {mock_get.call_count})")
    assert_close(m1.get("AAPL"), "0000320193", label="AAPL CIK zero-padded")
    assert_close(m2.get("MSFT"), "0000789019", label="MSFT CIK from same map")


def test_cik_lookup_case_insensitive():
    print("\n[CIK lookup is case-insensitive]")
    fake = MagicMock()
    fake.raise_for_status = MagicMock()
    fake.json.return_value = make_synthetic_ticker_map()
    client = EdgarClient(user_agent="test test@example.com")
    with patch.object(client._session, "get", return_value=fake):
        c1 = client.cik_for_ticker("aapl")
        c2 = client.cik_for_ticker("AAPL")
    assert_close(c1, c2, label="lowercase and uppercase resolve to same CIK")
    assert_close(c1, "0000320193", label="CIK is 10-digit zero-padded")


def test_unknown_ticker_returns_none():
    print("\n[Unknown ticker (not in EDGAR) returns None]")
    fake = MagicMock()
    fake.raise_for_status = MagicMock()
    fake.json.return_value = make_synthetic_ticker_map()
    client = EdgarClient(user_agent="test test@example.com")
    with patch.object(client._session, "get", return_value=fake):
        result = client.cik_for_ticker("NOTINTHEMAP")
    assert_close(result, None, label="Returns None, not crash")


def test_full_load_from_edgar():
    print("\n[load_from_edgar end-to-end with synthetic AAPL]")

    # Mock EDGAR's three endpoints with appropriate responses
    facts = make_synthetic_companyfacts()
    ticker_map = make_synthetic_ticker_map()
    submissions = make_synthetic_submissions()

    def fake_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.status_code = 200
        if "company_tickers" in url:
            resp.json.return_value = ticker_map
        elif "companyfacts" in url:
            resp.json.return_value = facts
        elif "submissions" in url:
            resp.json.return_value = submissions
        else:
            resp.status_code = 404
            resp.json.return_value = {}
        return resp

    with patch.object(EdgarClient, "_get",
                      side_effect=lambda url, retries=3: fake_get(url).json()):
        td = load_from_edgar("AAPL", user_agent="test test@example.com")

    assert_true(td is not None, "TickerData built")
    assert_close(td.ticker, "AAPL", label="Ticker preserved")
    assert_close(td.sector, "Technology", label="Sector mapped from SIC 3571")
    assert_close(td.industry, "Electronic Computers", label="Industry from sicDescription")

    # Income statement: 4 periods, all required rows
    assert_true(td.income_stmt is not None, "Income statement built")
    assert_true(len(td.income_stmt.columns) == 4, "4 fiscal years")
    assert_true("Total Revenue" in td.income_stmt.index, "Revenue row present")
    assert_true("Net Income" in td.income_stmt.index, "Net Income row present")
    assert_true("Diluted EPS" in td.income_stmt.index, "Diluted EPS row present")
    assert_true("Tax Provision" in td.income_stmt.index, "Tax row present")

    # Series derivation
    assert_true(td.fcf_series is not None and len(td.fcf_series) == 4,
                f"FCF series has 4 periods (got {len(td.fcf_series) if td.fcf_series is not None else 0})")
    # 2024 FCF = OCF 118.254B - capex 9.447B = 108.807B
    assert_close(td.fcf_series.iloc[-1] / 1e9, 108.807, tol=0.01,
                 label="2024 FCF ≈ $108.8B")

    assert_true(td.eps_series is not None and len(td.eps_series) == 4,
                "EPS series has 4 periods")
    assert_close(td.eps_series.iloc[-1], 6.08, label="2024 diluted EPS = $6.08")

    assert_true(td.tax_rate_series is not None,
                "Tax rate series derived from tax/pretax")
    # 2024: 28.533 / 122.269 = 23.3%
    assert_close(float(td.tax_rate_series.iloc[-1]), 28.533 / 122.269, tol=1e-3,
                 label="2024 tax rate ≈ 23.3%")

    # Balance sheet
    assert_true(td.balance_sheet is not None, "Balance sheet built")
    debt_row = td.balance_sheet.loc["Total Debt"].dropna()
    # 2024: 85.75 + 10.912 = 96.662B
    assert_close(debt_row.iloc[-1] / 1e9, 96.662, tol=0.01,
                 label="2024 Total Debt = LTD + STD ≈ $96.66B")

    # Shares outstanding (point estimate from CommonStockSharesOutstanding)
    assert_close(td.shares_outstanding, 15_115_823_000,
                 label="Shares from CommonStockSharesOutstanding")
    assert_close(td.trailing_eps, 6.08, label="Trailing EPS from latest filing")


def test_load_returns_none_for_unknown_ticker():
    print("\n[load_from_edgar returns None for non-SEC filer]")
    fake = MagicMock()
    fake.raise_for_status = MagicMock()
    fake.json.return_value = make_synthetic_ticker_map()  # only AAPL, MSFT
    with patch.object(EdgarClient, "_get",
                      side_effect=lambda url, retries=3: fake.json.return_value if "company_tickers" in url else None):
        td = load_from_edgar("TSM", user_agent="test test@example.com")  # ADR, not in map
    assert_close(td, None, label="ADR not in EDGAR returns None (caller falls back)")


def run():
    tests = [
        ("EdgarClient User-Agent validation",     test_user_agent_required),
        ("Rate limiter enforces interval",        test_rate_limiter),
        ("Concept alias resolution",              test_extract_annual_series_picks_first_alias),
        ("Restatement → latest filing wins",      test_extract_annual_series_handles_restatements),
        ("Filters to full-year (FY) only",        test_extract_filters_to_full_year),
        ("Total Debt = LTD + STD",                test_construct_total_debt),
        ("SIC code → sector",                     test_sic_to_sector),
        ("Ticker map cached after first fetch",   test_ticker_map_caching),
        ("CIK lookup case-insensitive",           test_cik_lookup_case_insensitive),
        ("Unknown ticker → None",                 test_unknown_ticker_returns_none),
        ("Full load_from_edgar(AAPL) integration",test_full_load_from_edgar),
        ("Non-SEC filer returns None",            test_load_returns_none_for_unknown_ticker),
    ]
    for name, fn in tests:
        _clear_cache()
        fn()
    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{'='*60}\nAll {len(tests)} EDGAR tests passed ✓\n{'='*60}")


if __name__ == "__main__":
    run()
