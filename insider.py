"""
insider.py — Dataroma insider-trade summary scraper.

Fetches the buy/sell totals (transactions count + dollar amount) for a
ticker over weekly / monthly / quarterly windows.

Data source: https://www.dataroma.com/m/ins/ins.php
No API key needed. Returns JSON-serializable dicts.

Usage:
    from insider import get_insider_summary
    data = get_insider_summary("AAPL")
    # {
    #   "ticker": "AAPL",
    #   "weekly":    {"buys": {"count": 2, "amount": 1500000}, "sells": {...}},
    #   "monthly":   {...},
    #   "quarterly": {...},
    #   "errors":    []
    # }
"""

from __future__ import annotations

import re
import time
import threading
from typing import Dict, Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from bs4 import BeautifulSoup

# ── Cache ────────────────────────────────────────────────────────────────────
_CACHE_TTL = 600          # 10 min (insider filings don't change by the minute)
_cache: Dict[str, tuple] = {}
_lock = threading.Lock()

def _cache_get(key: str) -> Optional[Any]:
    with _lock:
        rec = _cache.get(key)
        if rec and rec[0] > time.time():
            return rec[1]
    return None

def _cache_put(key: str, val: Any) -> None:
    with _lock:
        _cache[key] = (time.time() + _CACHE_TTL, val)

# ── HTTP helper ───────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def _fetch_html(url: str, retries: int = 2) -> str:
    req = Request(url, headers=_HEADERS)
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (URLError, HTTPError, OSError) as exc:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return ""


# ── Dollar-string parser ──────────────────────────────────────────────────────
def _parse_amount(s: str) -> int:
    """'$24,173,073' → 24173073    '$0' → 0"""
    s = s.strip().lstrip("$").replace(",", "").replace(" ", "")
    if not s or s == "0":
        return 0
    try:
        return int(re.sub(r"[^\d]", "", s))
    except ValueError:
        return 0


# ── Single-period fetch ────────────────────────────────────────────────────────
_PERIOD_CODES = {
    "weekly":    "w",
    "monthly":   "m",
    "quarterly": "q",
}

def _fetch_period(ticker: str, period_code: str) -> Dict[str, Any]:
    """
    Returns:
        {
            "buys":  {"count": int, "amount": int},
            "sells": {"count": int, "amount": int},
        }
    or raises on network / parse failure.
    """
    url = (
        f"https://www.dataroma.com/m/ins/ins.php"
        f"?t={period_code}&am=0&sym={ticker.upper()}&o=fd&d=d"
    )
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {"id": "sum"})
    if not table:
        raise ValueError(f"Summary table not found for {ticker}/{period_code}")

    result: Dict[str, Dict[str, int]] = {
        "buys":  {"count": 0, "amount": 0},
        "sells": {"count": 0, "amount": 0},
    }

    for row in table.find_all("tr"):
        cls = " ".join(row.get("class", [])).lower()
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 3:
            continue
        if "buys" in cls or cells[0].lower() == "buys":
            result["buys"] = {
                "count":  int(cells[1]) if cells[1].isdigit() else 0,
                "amount": _parse_amount(cells[2]),
            }
        elif "sells" in cls or cells[0].lower() == "sells":
            result["sells"] = {
                "count":  int(cells[1]) if cells[1].isdigit() else 0,
                "amount": _parse_amount(cells[2]),
            }

    return result


# ── Public API ────────────────────────────────────────────────────────────────
def get_insider_summary(ticker: str,
                        periods: tuple = ("weekly", "monthly", "quarterly"),
                        use_cache: bool = True) -> Dict[str, Any]:
    """
    Fetch insider buy/sell summary for a ticker across periods.

    Args:
        ticker:     Stock symbol (case-insensitive).
        periods:    Subset of ("weekly", "monthly", "quarterly").
        use_cache:  Use in-process 10-min TTL cache.

    Returns JSON-serializable dict:
        {
            "ticker": "AAPL",
            "weekly":    {"buys": {"count": 0, "amount": 0},
                          "sells": {"count": 8, "amount": 24173073}},
            "monthly":   {...},
            "quarterly": {...},
            "errors":    []        # non-empty if some periods failed
        }
    """
    ticker = ticker.upper().strip()
    cache_key = f"insider:{ticker}"

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    out: Dict[str, Any] = {"ticker": ticker, "errors": []}

    for period in periods:
        code = _PERIOD_CODES.get(period)
        if not code:
            out["errors"].append(f"Unknown period '{period}'")
            continue
        try:
            out[period] = _fetch_period(ticker, code)
        except Exception as exc:
            out[period] = {"buys": {"count": 0, "amount": 0},
                           "sells": {"count": 0, "amount": 0}}
            out["errors"].append(f"{period}: {exc}")

    if use_cache:
        _cache_put(cache_key, out)

    return out


# ── Superinvestor ownership table ────────────────────────────────────────────

def get_superinvestors(ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Scrape the superinvestor ownership table from dataroma.com/m/stock.php.

    Returns:
        {
            "ticker": "AAPL",
            "holders": [
                {
                    "manager": "Warren Buffett - Berkshire Hathaway",
                    "pct_portfolio": 22.60,        # % of manager's portfolio
                    "activity": "Reduce 4.32%",    # recent activity text
                    "activity_type": "sell",       # "buy" | "sell" | "hold"
                    "value": 61961735000,           # USD value of holding
                },
                ...
            ],
            "summary": {
                "total_bought_value": int,   # sum of value for buy/add rows
                "total_sold_value":   int,   # sum of value for sell/reduce rows
                "buy_count":  int,
                "sell_count": int,
                "hold_count": int,
            },
            "error": None | str,
        }
    """
    ticker = ticker.upper().strip()
    cache_key = f"superinvestors:{ticker}"

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    url = f"https://www.dataroma.com/m/stock.php?sym={ticker}"
    try:
        html = _fetch_html(url)
    except Exception as exc:
        result = {"ticker": ticker, "holders": [], "summary": {}, "error": str(exc)}
        return result

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "grid"})
    if not table:
        result = {"ticker": ticker, "holders": [], "summary": {},
                  "error": "Ownership table not found"}
        return result

    holders = []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 6:
            continue

        # cells[0] = hist link (skip)
        # cells[1] = firm / manager name
        # cells[2] = % of portfolio
        # cells[3] = recent activity (class "buy"/"sell")
        # cells[4] = shares
        # cells[5] = value ($)
        manager  = cells[1].get_text(strip=True)
        pct_text = cells[2].get_text(strip=True)
        act_cell = cells[3]
        act_text = act_cell.get_text(strip=True)
        act_cls  = " ".join(act_cell.get("class", [])).lower()
        val_text = cells[5].get_text(strip=True)

        try:
            pct = float(pct_text)
        except ValueError:
            pct = 0.0

        try:
            value = int(val_text.replace(",", ""))
        except ValueError:
            value = 0

        if "buy" in act_cls:
            atype = "buy"
        elif "sell" in act_cls:
            atype = "sell"
        else:
            atype = "hold"

        # Blank activity text means no change this quarter
        if not act_text:
            act_text = "No change"
            atype = "hold"

        holders.append({
            "manager":       manager,
            "pct_portfolio": pct,
            "activity":      act_text,
            "activity_type": atype,
            "value":         value,
        })

    # Summary row
    bought_val = sum(h["value"] for h in holders if h["activity_type"] == "buy")
    sold_val   = sum(h["value"] for h in holders if h["activity_type"] == "sell")
    summary = {
        "total_bought_value": bought_val,
        "total_sold_value":   sold_val,
        "buy_count":  sum(1 for h in holders if h["activity_type"] == "buy"),
        "sell_count": sum(1 for h in holders if h["activity_type"] == "sell"),
        "hold_count": sum(1 for h in holders if h["activity_type"] == "hold"),
    }

    result = {"ticker": ticker, "holders": holders,
              "summary": summary, "error": None}

    if use_cache:
        _cache_put(cache_key, result)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    mode   = sys.argv[2] if len(sys.argv) > 2 else "insider"
    if mode == "super":
        print(json.dumps(get_superinvestors(ticker, use_cache=False), indent=2))
    else:
        print(json.dumps(get_insider_summary(ticker, use_cache=False), indent=2))
