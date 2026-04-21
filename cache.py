"""
cache.py — In-memory TTL cache for ticker data.

Wraps yfinance.Ticker so that repeated calls (e.g. comparing all profiles
for the same ticker) re-use a single fetch of .info / .history /
.quarterly_financials / .calendar / .earnings_history / .news.

Also wraps the SEC EDGAR + news helpers in penny_filter so they are
shared across penny / growth / value evaluators.

Install once at app startup:

    import cache
    cache.install(ttl_seconds=600)
    cache.clear()              # to flush manually
"""

from __future__ import annotations

import time
import threading
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import yfinance as yf


_LOCK = threading.Lock()
_CACHE: Dict[Tuple, Tuple[float, Any]] = {}
_DEFAULT_TTL = 600  # 10 minutes


def _get(key, ttl: int):
    now = time.time()
    with _LOCK:
        rec = _CACHE.get(key)
        if rec and rec[0] > now:
            return rec[1]
    return None


def _put(key, value, ttl: int):
    expiry = time.time() + ttl
    with _LOCK:
        _CACHE[key] = (expiry, value)


def ttl_cache(ttl: int = _DEFAULT_TTL):
    """Decorator: TTL-cache a free function by its args/kwargs."""
    def deco(fn: Callable):
        ns = f"{fn.__module__}.{fn.__name__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (ns, args, tuple(sorted(kwargs.items())))
            cached = _get(key, ttl)
            if cached is not None:
                return cached
            val = fn(*args, **kwargs)
            _put(key, val, ttl)
            return val
        wrapper.__wrapped__ = fn
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# yfinance.Ticker wrapper
# ---------------------------------------------------------------------------

_ORIG_TICKER = yf.Ticker


class CachedTicker:
    """Drop-in replacement for yfinance.Ticker with per-symbol caching."""

    def __init__(self, symbol: str, *args, **kwargs):
        self._symbol = symbol.upper().strip() if isinstance(symbol, str) else symbol
        self._real = _ORIG_TICKER(symbol, *args, **kwargs)
        self._ttl = _DEFAULT_TTL

    def _memo(self, attr_key: str, compute):
        key = ("yf.Ticker", self._symbol, attr_key)
        cached = _get(key, self._ttl)
        if cached is not None:
            return cached
        val = compute()
        _put(key, val, self._ttl)
        return val

    # ---- properties (cached) ----
    @property
    def info(self):
        return self._memo("info", lambda: self._real.info)

    @property
    def fast_info(self):
        return self._memo("fast_info", lambda: self._real.fast_info)

    @property
    def quarterly_financials(self):
        return self._memo("qf", lambda: self._real.quarterly_financials)

    @property
    def quarterly_cashflow(self):
        return self._memo("qcf", lambda: self._real.quarterly_cashflow)

    @property
    def quarterly_balance_sheet(self):
        return self._memo("qbs", lambda: self._real.quarterly_balance_sheet)

    @property
    def financials(self):
        return self._memo("financials", lambda: self._real.financials)

    @property
    def cashflow(self):
        return self._memo("cashflow", lambda: self._real.cashflow)

    @property
    def balance_sheet(self):
        return self._memo("bs", lambda: self._real.balance_sheet)

    @property
    def calendar(self):
        return self._memo("calendar", lambda: self._real.calendar)

    @property
    def earnings_history(self):
        return self._memo("eh", lambda: self._real.earnings_history)

    @property
    def news(self):
        return self._memo("news", lambda: self._real.news)

    @property
    def institutional_holders(self):
        return self._memo("ih", lambda: self._real.institutional_holders)

    @property
    def major_holders(self):
        return self._memo("mh", lambda: self._real.major_holders)

    # ---- methods (cached by args) ----
    def history(self, *args, **kwargs):
        key = ("history", args, tuple(sorted(kwargs.items())))
        return self._memo(key, lambda: self._real.history(*args, **kwargs))

    # ---- pass-through for everything else (uncached) ----
    def __getattr__(self, name):
        # __getattr__ only fires for missing attrs; properties above take
        # precedence. Pass through to the real Ticker for anything we
        # haven't explicitly cached.
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

_INSTALLED = False


def install(ttl_seconds: int = _DEFAULT_TTL):
    """Replace yfinance.Ticker globally and wrap penny_filter helpers."""
    global _INSTALLED, _DEFAULT_TTL
    _DEFAULT_TTL = ttl_seconds

    if not _INSTALLED:
        yf.Ticker = CachedTicker
        _INSTALLED = True

    # Wrap penny_filter helper functions in place. Re-bind in growth_filter
    # and value_filter so their import-time references see the wrapped versions.
    import penny_filter
    import growth_filter
    import value_filter

    targets = ("sec_filings", "get_news", "yf_news", "yahoo_rss",
               "fetch_core", "http_get")
    for name in targets:
        if hasattr(penny_filter, name):
            fn = getattr(penny_filter, name)
            wrapped = ttl_cache(ttl_seconds)(fn)
            setattr(penny_filter, name, wrapped)
            for mod in (growth_filter, value_filter):
                if hasattr(mod, name):
                    setattr(mod, name, wrapped)


def clear():
    """Flush the entire ticker cache."""
    with _LOCK:
        _CACHE.clear()


def stats():
    """Return cache size + age of oldest entry (for debug)."""
    with _LOCK:
        n = len(_CACHE)
        if not n:
            return {"entries": 0}
        oldest = min(rec[0] for rec in _CACHE.values()) - time.time()
    return {"entries": n, "oldest_remaining_seconds": round(oldest, 1)}
