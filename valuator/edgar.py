"""
SEC EDGAR data loader.

EDGAR is the authoritative source for SEC-registered company fundamentals:
auditable, point-in-time, with restatement history. This module:

  1. Resolves ticker → CIK via company_tickers.json (cached on disk)
  2. Fetches companyfacts/CIK{cik}.json (contains every concept ever filed)
  3. Resolves concept aliases — different filers tag the same economic
     fact differently (e.g. "Revenues" vs "RevenueFromContractWithCustomerExcludingAssessedTax")
  4. Filters to annual 10-K observations, ordered by fiscal period
  5. Builds the same TickerData shape as the yfinance loader so downstream
     code is unchanged

EDGAR coverage:
  ✓ All US SEC filers (NYSE, NASDAQ, OTC)
  ✗ ADRs reporting on Form 6-K instead of 10-K (TSM, NVO, ASML — partial)
  ✗ Foreign listings without SEC registration
  ✗ Private companies

Use yfinance as fallback for ADRs/non-SEC filers.

Rate limit: SEC enforces ≤10 req/s per IP. Token bucket built-in.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from data import TickerData, _row, _derive_series  # reuse the shape

log = logging.getLogger(__name__)


# ─── Cache config (separate dir from FRED to keep things tidy) ─────────────
CACHE_DIR = Path(os.environ.get("VALUATION_CACHE", "~/.valuation_cache")).expanduser() / "edgar"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CIK lookup file is small; companyfacts files are 1-5MB each.
TICKER_CACHE_TTL = 7 * 24 * 3600          # weekly — tickers change slowly
COMPANYFACTS_CACHE_TTL = 3 * 24 * 3600    # 3 days — fundamentals update on filing


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _cache_get(name: str, ttl: int) -> dict | None:
    p = _cache_path(name)
    if not p.exists() or time.time() - p.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_put(name: str, data: dict) -> None:
    try:
        _cache_path(name).write_text(json.dumps(data))
    except Exception as e:
        log.warning("EDGAR cache write failed: %s", e)


# ─── Rate limiter (10 req/s leaky bucket) ──────────────────────────────────

class _RateLimiter:
    """Simple thread-safe rate limiter. SEC enforces 10 req/s per IP."""

    def __init__(self, rate_per_sec: float = 8.0):  # leave headroom under 10
        self.min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()


# ─── EDGAR client ──────────────────────────────────────────────────────────

EDGAR_BASE = "https://data.sec.gov"
EDGAR_FILES_BASE = "https://www.sec.gov/files"


class EdgarClient:
    """SEC EDGAR HTTP client with rate limiting, retries, and caching."""

    def __init__(self, user_agent: str, rate_per_sec: float = 8.0, timeout: int = 15):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "EDGAR requires a User-Agent with contact info. "
                "Pass something like 'YourName your@email.com'."
            )
        self.user_agent = user_agent
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",  # required by SEC
        })
        self._rate_limiter = _RateLimiter(rate_per_sec)
        self.timeout = timeout

    # ── Network ──

    def _get(self, url: str, retries: int = 3) -> dict | None:
        for attempt in range(retries):
            self._rate_limiter.wait()
            try:
                # data.sec.gov vs www.sec.gov require different Host headers
                headers = dict(self._session.headers)
                if "www.sec.gov" in url:
                    headers["Host"] = "www.sec.gov"
                r = self._session.get(url, headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    log.warning("EDGAR rate-limited, backing off")
                    time.sleep(2 ** (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                log.debug("EDGAR get %s attempt %d failed: %s", url, attempt + 1, e)
                if attempt < retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
        return None

    # ── Ticker → CIK ──

    def _ticker_map(self) -> dict[str, str] | None:
        """{ticker_upper: zero-padded 10-char CIK}. Cached for a week."""
        cached = _cache_get("ticker_map", TICKER_CACHE_TTL)
        if cached:
            return cached

        # SEC publishes this file at sec.gov/files (not data.sec.gov)
        url = f"{EDGAR_FILES_BASE}/company_tickers.json"
        raw = self._get(url)
        if raw is None:
            return None

        # Format: {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."},...}
        out: dict[str, str] = {}
        for v in raw.values():
            ticker = v.get("ticker", "").upper()
            cik = v.get("cik_str")
            if ticker and cik is not None:
                out[ticker] = str(cik).zfill(10)

        if out:
            _cache_put("ticker_map", out)
        return out

    def cik_for_ticker(self, ticker: str) -> str | None:
        m = self._ticker_map()
        if m is None:
            return None
        return m.get(ticker.upper())

    # ── Companyfacts (the big payload) ──

    def companyfacts(self, cik_padded: str) -> dict | None:
        cache_key = f"facts_{cik_padded}"
        cached = _cache_get(cache_key, COMPANYFACTS_CACHE_TTL)
        if cached:
            return cached
        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        data = self._get(url)
        if data:
            _cache_put(cache_key, data)
        return data

    # ── Submissions (for sector/SIC, name, exchange) ──

    def submissions(self, cik_padded: str) -> dict | None:
        cache_key = f"sub_{cik_padded}"
        cached = _cache_get(cache_key, COMPANYFACTS_CACHE_TTL)
        if cached:
            return cached
        url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
        data = self._get(url)
        if data:
            _cache_put(cache_key, data)
        return data


# ─── XBRL concept aliases ──────────────────────────────────────────────────
#
# The same economic concept gets tagged with different us-gaap names across
# filers and fiscal years. We try each alias in order; the first one with
# usable data wins. Order is roughly "most-modern preferred name first".

CONCEPT_ALIASES = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperations",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossBeforeIncomeTaxes",
    ],
    "tax_provision": [
        "IncomeTaxExpenseBenefit",
        "CurrentIncomeTaxExpenseBenefit",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ],
    "diluted_eps": [
        "EarningsPerShareDiluted",
    ],
    "basic_eps": [
        "EarningsPerShareBasic",
    ],
    "diluted_shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDilutedAdjusted",
    ],
    "basic_shares": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",  # in dei namespace
    ],
    # Cash flow
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
    ],
    # Balance sheet
    "total_assets": ["Assets"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_debt": [
        # No single perfect tag; we sum LongTermDebt + ShortTermBorrowings later
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "DebtLongtermAndShorttermCombinedAmount",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent",  # current portion of LTD
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "dividends_per_share": [
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ],
}


def _extract_annual_series(
    facts: dict,
    concept_keys: list[str],
    namespaces: tuple[str, ...] = ("us-gaap", "ifrs-full", "dei"),
) -> pd.Series | None:
    """
    Pull annual (10-K) observations for the first matching concept.

    Returns a pd.Series indexed by fiscal period end date, sorted ascending,
    with one observation per fiscal year (taking the latest 10-K filing
    if a concept has restatements).
    """
    if "facts" not in facts:
        return None
    f = facts["facts"]

    for ns in namespaces:
        if ns not in f:
            continue
        for concept in concept_keys:
            if concept not in f[ns]:
                continue
            units = f[ns][concept].get("units", {})
            # Prefer USD; fall back to any currency or shares unit
            unit_priority = ["USD", "USD/shares", "shares"] + list(units.keys())
            for unit in unit_priority:
                if unit not in units:
                    continue
                obs = units[unit]
                # Annuals only: 10-K and 10-K/A
                annuals = [
                    o for o in obs
                    if o.get("form", "").startswith("10-K")
                    and o.get("fp") == "FY"
                ]
                if not annuals:
                    # Fall back to any 10-K-ish observation with full-year period
                    annuals = [o for o in obs if o.get("form", "").startswith("10-K")]
                if not annuals:
                    continue

                # Group by fiscal year end; take the latest 'filed' for each
                by_period: dict[str, dict] = {}
                for o in annuals:
                    end = o.get("end")
                    if not end:
                        continue
                    existing = by_period.get(end)
                    if existing is None or o.get("filed", "") > existing.get("filed", ""):
                        by_period[end] = o

                if not by_period:
                    continue

                rows = sorted(by_period.values(), key=lambda x: x["end"])
                idx = pd.to_datetime([o["end"] for o in rows])
                vals = [float(o["val"]) for o in rows]
                return pd.Series(vals, index=idx, name=concept)

    return None


def _construct_total_debt(facts: dict) -> pd.Series | None:
    """
    Total debt = Long-term debt + Short-term debt, period-aligned.
    Better than the unreliable single 'Debt' tag.
    """
    ltd = _extract_annual_series(facts, CONCEPT_ALIASES["long_term_debt"])
    std = _extract_annual_series(facts, CONCEPT_ALIASES["short_term_debt"])
    if ltd is None and std is None:
        return None
    if ltd is None:
        return std
    if std is None:
        return ltd
    common = ltd.index.intersection(std.index)
    if len(common) == 0:
        # Periods don't line up — return whichever is longer
        return ltd if len(ltd) >= len(std) else std
    return (ltd.loc[common] + std.loc[common]).sort_index()


# ─── Main entry: load TickerData from EDGAR ────────────────────────────────

def load_from_edgar(ticker: str, user_agent: str) -> TickerData | None:
    """
    Build a TickerData from EDGAR. Returns None if ticker isn't a SEC filer
    (caller should then try yfinance fallback).

    Note: market data (current price, market cap, beta, current shares) is
    NOT in EDGAR. We populate fundamentals here; the orchestrator merges
    with yfinance market data afterwards.
    """
    client = EdgarClient(user_agent=user_agent)
    cik = client.cik_for_ticker(ticker)
    if cik is None:
        log.info("[%s] not in EDGAR ticker map (likely non-SEC filer)", ticker)
        return None

    facts = client.companyfacts(cik)
    if facts is None:
        log.warning("[%s] EDGAR companyfacts fetch failed", ticker)
        return None

    sub = client.submissions(cik)

    td = TickerData(ticker=ticker)

    # Sector/industry from submissions (SIC code)
    if sub:
        sic_desc = sub.get("sicDescription", "")
        td.sector = _sic_to_sector(sub.get("sic", ""))
        td.industry = sic_desc

    # ── Income statement series ──
    revenue = _extract_annual_series(facts, CONCEPT_ALIASES["revenue"])
    net_income = _extract_annual_series(facts, CONCEPT_ALIASES["net_income"])
    operating_income = _extract_annual_series(facts, CONCEPT_ALIASES["operating_income"])
    pretax = _extract_annual_series(facts, CONCEPT_ALIASES["pretax_income"])
    tax = _extract_annual_series(facts, CONCEPT_ALIASES["tax_provision"])
    interest = _extract_annual_series(facts, CONCEPT_ALIASES["interest_expense"])
    diluted_eps = _extract_annual_series(facts, CONCEPT_ALIASES["diluted_eps"])
    diluted_shares = _extract_annual_series(facts, CONCEPT_ALIASES["diluted_shares"])
    basic_shares = _extract_annual_series(facts, CONCEPT_ALIASES["basic_shares"])

    # Build the income_stmt DataFrame with the same row labels yfinance uses,
    # so the rest of the pipeline (data.py:_derive_series, assumptions.py)
    # works without modification.
    income_rows = {}
    if revenue is not None:           income_rows["Total Revenue"] = revenue
    if net_income is not None:        income_rows["Net Income"] = net_income
    if operating_income is not None:  income_rows["Operating Income"] = operating_income
    if pretax is not None:            income_rows["Pretax Income"] = pretax
    if tax is not None:               income_rows["Tax Provision"] = tax
    if interest is not None:          income_rows["Interest Expense"] = interest.abs()
    if diluted_eps is not None:       income_rows["Diluted EPS"] = diluted_eps
    if diluted_shares is not None:    income_rows["Diluted Average Shares"] = diluted_shares
    if basic_shares is not None:      income_rows["Basic Average Shares"] = basic_shares

    if income_rows:
        # All series may have different period coverage; build a DataFrame
        # using the *union* of dates as columns.
        all_dates = sorted(set().union(*[s.index for s in income_rows.values()]))
        td.income_stmt = pd.DataFrame(
            {d: {label: s.loc[d] if d in s.index else np.nan
                 for label, s in income_rows.items()}
             for d in all_dates}
        )

    # ── Cash flow ──
    ocf = _extract_annual_series(facts, CONCEPT_ALIASES["operating_cash_flow"])
    capex = _extract_annual_series(facts, CONCEPT_ALIASES["capex"])
    cf_rows = {}
    if ocf is not None:    cf_rows["Operating Cash Flow"] = ocf
    if capex is not None:  cf_rows["Capital Expenditure"] = -capex.abs()  # signed negative

    if cf_rows:
        all_dates = sorted(set().union(*[s.index for s in cf_rows.values()]))
        td.cash_flow = pd.DataFrame(
            {d: {label: s.loc[d] if d in s.index else np.nan
                 for label, s in cf_rows.items()}
             for d in all_dates}
        )

    # ── Balance sheet ──
    total_assets = _extract_annual_series(facts, CONCEPT_ALIASES["total_assets"])
    current_liab = _extract_annual_series(facts, CONCEPT_ALIASES["current_liabilities"])
    total_debt = _construct_total_debt(facts)
    cash = _extract_annual_series(facts, CONCEPT_ALIASES["cash"])
    equity = _extract_annual_series(facts, CONCEPT_ALIASES["stockholders_equity"])
    shares_now = _extract_annual_series(facts, CONCEPT_ALIASES["shares_outstanding"])

    bs_rows = {}
    if total_assets is not None:  bs_rows["Total Assets"] = total_assets
    if current_liab is not None:  bs_rows["Current Liabilities"] = current_liab
    if total_debt is not None:    bs_rows["Total Debt"] = total_debt
    if cash is not None:          bs_rows["Cash And Cash Equivalents"] = cash
    if equity is not None:        bs_rows["Stockholders Equity"] = equity

    if bs_rows:
        all_dates = sorted(set().union(*[s.index for s in bs_rows.values()]))
        td.balance_sheet = pd.DataFrame(
            {d: {label: s.loc[d] if d in s.index else np.nan
                 for label, s in bs_rows.items()}
             for d in all_dates}
        )

    # Most-recent share count from EDGAR (point estimate, not market data)
    if shares_now is not None and len(shares_now) > 0:
        td.shares_outstanding = float(shares_now.iloc[-1])

    # Most recent EPS / book value
    if diluted_eps is not None and len(diluted_eps) > 0:
        td.trailing_eps = float(diluted_eps.iloc[-1])
    if equity is not None and shares_now is not None:
        common = equity.index.intersection(shares_now.index)
        if len(common):
            td.book_value_per_share = float(equity.loc[common[-1]]) / float(shares_now.loc[common[-1]])

    # Run the same series-derivation as the yfinance loader
    _derive_series(td)

    log.info("[%s] EDGAR loaded from CIK %s (%d income rows, %d BS rows, %d CF rows)",
             ticker, cik,
             len(td.income_stmt.columns) if td.income_stmt is not None else 0,
             len(td.balance_sheet.columns) if td.balance_sheet is not None else 0,
             len(td.cash_flow.columns) if td.cash_flow is not None else 0)
    return td


# ─── Combined loader: EDGAR primary + yfinance for market data ─────────────

def load(ticker: str, user_agent: str | None = None,
         prefer_edgar: bool = True) -> TickerData:
    """
    Load TickerData with EDGAR as primary fundamentals source and yfinance
    for market data (price, shares, beta, market cap).

    Falls back to pure yfinance when:
      - User-Agent isn't set (EDGAR will refuse)
      - Ticker isn't in EDGAR (ADRs, foreign listings, private)
      - EDGAR fetch fails (network, rate limit)

    The resulting TickerData has EDGAR-quality statements but yfinance
    market data. Both sources are logged in td.warnings for transparency.
    """
    user_agent = user_agent or os.environ.get("EDGAR_USER_AGENT")

    edgar_td: TickerData | None = None
    if prefer_edgar and user_agent:
        try:
            edgar_td = load_from_edgar(ticker, user_agent)
        except Exception as e:
            log.warning("[%s] EDGAR loader crashed (%s); falling back to yfinance", ticker, e)

    # Always pull yfinance for market data (and as full fallback if EDGAR failed)
    import data as data_layer
    yf_td = data_layer.load(ticker)

    if edgar_td is None:
        yf_td.warn("Using yfinance for fundamentals (EDGAR unavailable)")
        return yf_td

    # Merge: EDGAR fundamentals + yfinance market data
    merged = edgar_td
    merged.current_price = yf_td.current_price
    merged.market_cap = yf_td.market_cap
    merged.beta_raw = yf_td.beta_raw
    merged.payout_ratio = yf_td.payout_ratio
    merged.return_on_equity = yf_td.return_on_equity
    merged.earnings_growth = yf_td.earnings_growth
    merged.trailing_pe = yf_td.trailing_pe
    merged.forward_pe = yf_td.forward_pe
    merged.forward_eps = yf_td.forward_eps
    merged.current_dividend = yf_td.current_dividend
    merged.dividends = yf_td.dividends
    # Sector/industry: yfinance is more user-friendly than SIC code
    if yf_td.sector:
        merged.sector = yf_td.sector
    if yf_td.industry:
        merged.industry = yf_td.industry
    # Use yfinance's current shares (point-in-time) if EDGAR didn't have one
    if not merged.shares_outstanding and yf_td.shares_outstanding:
        merged.shares_outstanding = yf_td.shares_outstanding

    merged.warnings.append("Fundamentals: SEC EDGAR (audited XBRL)")
    merged.warnings.append("Market data: yfinance")
    return merged


# ─── SIC code → sector heuristic ──────────────────────────────────────────
# Crude mapping of SEC's SIC codes to the Yahoo-style sectors the rest of
# the pipeline uses. Yahoo's mapping is finer-grained for some industries.

def _sic_to_sector(sic: str) -> str:
    if not sic:
        return ""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return ""
    # SIC ranges from https://www.sec.gov/info/edgar/siccodes.htm
    if 100 <= s <= 999:    return "Basic Materials"   # Agriculture
    if 1000 <= s <= 1499:  return "Basic Materials"   # Mining
    if 1500 <= s <= 1799:  return "Industrials"       # Construction
    if 2000 <= s <= 2099:  return "Consumer Defensive"  # Food
    if 2100 <= s <= 2199:  return "Consumer Defensive"  # Tobacco
    if 2200 <= s <= 2399:  return "Consumer Cyclical"   # Apparel
    if 2400 <= s <= 2499:  return "Industrials"
    if 2600 <= s <= 2699:  return "Basic Materials"   # Paper
    if 2800 <= s <= 2899:  return "Basic Materials"   # Chemicals
    if 2900 <= s <= 2999:  return "Energy"            # Petroleum
    if 3000 <= s <= 3099:  return "Consumer Cyclical" # Rubber
    if 3300 <= s <= 3399:  return "Basic Materials"   # Primary metals
    if 3400 <= s <= 3499:  return "Industrials"       # Fabricated metal
    if 3500 <= s <= 3569:  return "Industrials"       # Industrial machinery
    if 3570 <= s <= 3579:  return "Technology"        # Computer & office equipment
    if 3580 <= s <= 3599:  return "Industrials"       # Other machinery
    if 3600 <= s <= 3669:  return "Technology"        # Electronic equipment (general)
    if 3670 <= s <= 3679:  return "Technology"        # Semiconductors
    if 3680 <= s <= 3699:  return "Technology"        # Computer storage / peripherals
    if 3700 <= s <= 3799:  return "Consumer Cyclical" # Transportation equipment (autos)
    if 3800 <= s <= 3899:  return "Healthcare"        # Instruments (med devices)
    if 4000 <= s <= 4799:  return "Industrials"       # Transportation services
    if 4800 <= s <= 4899:  return "Communication Services"
    if 4900 <= s <= 4999:  return "Utilities"
    if 5000 <= s <= 5199:  return "Industrials"       # Wholesale
    if 5200 <= s <= 5999:  return "Consumer Cyclical" # Retail
    if 6000 <= s <= 6199:  return "Financial Services"  # Banks
    if 6200 <= s <= 6299:  return "Financial Services"  # Brokers
    if 6300 <= s <= 6399:  return "Financial Services"  # Insurance
    if 6500 <= s <= 6599:  return "Real Estate"
    if 6700 <= s <= 6799:  return "Financial Services"  # Holding/investment
    if 7000 <= s <= 7299:  return "Consumer Cyclical"   # Hotels/personal services
    if 7300 <= s <= 7399:  return "Technology"          # Business services / software
    if 7800 <= s <= 7999:  return "Communication Services"  # Entertainment
    if 8000 <= s <= 8099:  return "Healthcare"          # Health services
    if 8200 <= s <= 8299:  return "Consumer Defensive"  # Education
    if 9000 <= s <= 9999:  return "Industrials"
    return ""


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    ua = os.environ.get("EDGAR_USER_AGENT", "valuation-tool saidrakhmonov94@gmail.com")
    td = load_from_edgar(ticker, user_agent=ua)
    if td is None:
        print(f"{ticker}: not in EDGAR")
    else:
        print(f"{ticker}: {td.industry}")
        print(f"  Income stmt periods: {td.income_stmt.columns.tolist() if td.income_stmt is not None else 'none'}")
        print(f"  EPS series: {td.eps_series.tolist() if td.eps_series is not None else 'none'}")
        print(f"  FCF series: {[round(x/1e6, 1) for x in td.fcf_series.values] if td.fcf_series is not None else 'none'} ($M)")
