"""
Macro data sources, v2.

Primary: FRED API (https://fred.stlouisfed.org/docs/api/fred/) — requires
free API key, set via FRED_API_KEY env var or constructor arg.

Series fetched:
  DGS10    — 10y Treasury nominal yield (the "risk-free rate")
  DFII10   — 10y TIPS yield (real risk-free rate)
  T10YIE   — 10y breakeven inflation (DGS10 - DFII10)
  BAA10Y   — Moody's Baa - 10y Treasury (investment-grade credit spread)
  AAA10Y   — Moody's Aaa - 10y Treasury (high-grade credit spread)
  GDP      — Nominal US GDP (quarterly), used to compute long-run growth

Falls back to:
  1. Public FRED CSV (no key)
  2. yfinance ^TNX for risk-free rate
  3. Hardcoded Damodaran defaults

NEVER commit a FRED API key. Reads from env var only.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)


# ─── On-disk cache ────────────────────────────────────────────────────────
CACHE_DIR = Path(os.environ.get("VALUATION_CACHE", "~/.valuation_cache")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 24 * 3600


def _cache_get(key: str):
    p = CACHE_DIR / f"{key}.json"
    if not p.exists() or time.time() - p.stat().st_mtime > CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_put(key: str, value):
    try:
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))
    except Exception as e:
        log.warning("cache write failed: %s", e)


# ─── FRED API client ──────────────────────────────────────────────────────

FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FredClient:
    """
    Thin wrapper around FRED's REST API. Falls back to public CSV when no key.

    The api_key is read from (in order): explicit constructor arg → FRED_API_KEY
    env var → None. Never embedded as a default. Pass `api_key=None` to force
    the public-CSV path even if you have a key in the environment.
    """

    def __init__(self, api_key: str | None = "USE_ENV", timeout: int = 10):
        if api_key == "USE_ENV":
            self.api_key = os.environ.get("FRED_API_KEY") or None
        else:
            self.api_key = api_key  # may be None to force CSV path
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "valuation-tool/2.0"})

    def has_key(self) -> bool:
        return bool(self.api_key)

    def latest_observation(self, series_id: str) -> tuple[float, str] | None:
        """Most recent non-missing value, with its date. Tries API first, CSV fallback."""
        cached = _cache_get(f"fred_{series_id}_latest")
        if cached:
            return cached["value"], cached["date"]

        result = self._latest_via_api(series_id) if self.has_key() else None
        if result is None:
            result = self._latest_via_csv(series_id)

        if result is not None:
            _cache_put(f"fred_{series_id}_latest", {"value": result[0], "date": result[1]})
        return result

    def series(self, series_id: str, start: str | None = None,
               limit: int | None = None) -> pd.Series | None:
        cached = _cache_get(f"fred_{series_id}_series_{start}_{limit}")
        if cached:
            return pd.Series(cached["values"],
                             index=pd.to_datetime(cached["dates"]),
                             name=series_id)

        s = self._series_via_api(series_id, start, limit) if self.has_key() else None
        if s is None:
            s = self._series_via_csv(series_id, start)

        if s is not None:
            _cache_put(f"fred_{series_id}_series_{start}_{limit}",
                       {"dates": s.index.strftime("%Y-%m-%d").tolist(),
                        "values": s.tolist()})
        return s

    # ── implementations ──

    def _latest_via_api(self, series_id: str) -> tuple[float, str] | None:
        try:
            url = f"{FRED_API_BASE}/series/observations"
            params = {
                "series_id": series_id, "api_key": self.api_key,
                "file_type": "json", "limit": 10, "sort_order": "desc",
            }
            r = self._session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            for o in obs:
                if o["value"] != ".":
                    return float(o["value"]), o["date"]
        except Exception as e:
            log.debug("FRED API latest %s failed: %s", series_id, e)
        return None

    def _series_via_api(self, series_id: str, start: str | None,
                        limit: int | None) -> pd.Series | None:
        try:
            url = f"{FRED_API_BASE}/series/observations"
            params = {
                "series_id": series_id, "api_key": self.api_key,
                "file_type": "json", "sort_order": "asc",
            }
            if start:
                params["observation_start"] = start
            if limit:
                params["limit"] = limit
            r = self._session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            rows = [(o["date"], o["value"]) for o in obs if o["value"] != "."]
            if not rows:
                return None
            dates, vals = zip(*rows)
            return pd.Series(
                [float(v) for v in vals],
                index=pd.to_datetime(dates), name=series_id,
            )
        except Exception as e:
            log.debug("FRED API series %s failed: %s", series_id, e)
            return None

    def _latest_via_csv(self, series_id: str) -> tuple[float, str] | None:
        s = self._series_via_csv(series_id, None)
        if s is None or len(s) == 0:
            return None
        return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d")

    def _series_via_csv(self, series_id: str, start: str | None) -> pd.Series | None:
        try:
            url = f"{FRED_CSV_BASE}?id={series_id}"
            r = self._session.get(url, timeout=self.timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            val_col = next(
                (c for c in df.columns if c.upper() in (series_id.upper(), "VALUE")),
                df.columns[-1],
            )
            date_col = df.columns[0]
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            df = df.dropna(subset=[val_col])
            if start:
                df = df[df[date_col] >= start]
            if df.empty:
                return None
            return pd.Series(
                df[val_col].values,
                index=pd.to_datetime(df[date_col]), name=series_id,
            )
        except Exception as e:
            log.debug("FRED CSV %s failed: %s", series_id, e)
            return None


# ─── Yfinance fallback (only used if FRED is unreachable) ─────────────────

def _yfinance_rf() -> float | None:
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX")
        rf_raw = None
        try:
            rf_raw = tnx.fast_info.get("last_price")
        except Exception:
            pass
        if rf_raw is None:
            rf_raw = tnx.info.get("regularMarketPrice")
        if rf_raw is None:
            return None
        rf = float(rf_raw) / 100.0
        return rf if 0.005 < rf < 0.15 else None
    except Exception:
        return None


# ─── Damodaran ERP defaults ───────────────────────────────────────────────
# Last published implied ERP (Jan 2026): 4.23%
DAMODARAN_ERP = 0.0423
LONG_RUN_NOMINAL_GDP_DEFAULT = 0.040


def _spy_implied_erp(rf: float) -> float | None:
    """Quick implied ERP via SPY's earnings yield: ey + 2% (real growth) - rf."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        info = spy.info
        pe = info.get("trailingPE")
        if not pe or pe <= 0 or pe > 60:
            return None
        ey = 1.0 / float(pe)
        erp = ey - rf + 0.02
        return erp if 0.02 < erp < 0.10 else None
    except Exception:
        return None


# ─── Credit spread → cost-of-debt helper ──────────────────────────────────

def credit_spread_for_rating(
    interest_coverage: float | None,
    baa_spread: float | None,
    aaa_spread: float | None,
) -> float:
    """
    Map an interest coverage ratio to a credit spread, anchored on actual
    market BAA/AAA spreads from FRED instead of static bands.

    Damodaran's synthetic-rating ICR table (large firm version) maps:
      ICR > 8.5    → AAA  (use AAA spread)
      ICR 6.5-8.5  → AA   (~1.2 × AAA)
      ICR 4-6.5    → A    (~0.4 × (AAA+BAA) blend)
      ICR 2-4      → BBB  (use BAA spread)
      ICR 1-2      → BB   (~1.5 × BAA)
      ICR < 1      → B/CCC (~3 × BAA, capped)
    """
    aaa = aaa_spread if aaa_spread is not None else 0.010
    baa = baa_spread if baa_spread is not None else 0.020

    if interest_coverage is None:
        # Default to mid-IG: average of AAA and BAA
        return (aaa + baa) / 2

    if interest_coverage > 8.5:
        return aaa
    if interest_coverage > 6.5:
        return aaa * 1.2
    if interest_coverage > 4.0:
        return 0.4 * aaa + 0.6 * baa  # A-ish
    if interest_coverage > 2.0:
        return baa
    if interest_coverage > 1.0:
        return baa * 1.5
    return min(baa * 3.0, 0.10)  # cap distressed at 10% spread


# ─── MacroEnvironment ─────────────────────────────────────────────────────

@dataclass
class MacroEnvironment:
    risk_free_rate: float
    equity_risk_premium: float
    long_run_gdp: float
    real_risk_free_rate: float | None = None
    expected_inflation: float | None = None
    baa_spread: float | None = None
    aaa_spread: float | None = None
    source_rf: str = ""
    source_erp: str = ""
    source_gdp: str = ""
    notes: list[str] = field(default_factory=list)

    def note(self, m: str) -> None:
        log.info("[macro] %s", m)
        self.notes.append(m)

    @classmethod
    def fetch(cls, fred_key: str | None = "USE_ENV") -> "MacroEnvironment":
        client = FredClient(api_key=fred_key)
        env = cls(risk_free_rate=0.043,
                  equity_risk_premium=DAMODARAN_ERP,
                  long_run_gdp=LONG_RUN_NOMINAL_GDP_DEFAULT)

        # Risk-free rate
        rf_obs = client.latest_observation("DGS10")
        if rf_obs is not None:
            env.risk_free_rate = rf_obs[0] / 100.0
            env.source_rf = f"FRED DGS10 ({rf_obs[1]})"
        else:
            yf_rf = _yfinance_rf()
            if yf_rf is not None:
                env.risk_free_rate = yf_rf
                env.source_rf = "yfinance ^TNX"
            else:
                env.source_rf = "fallback 4.30%"

        # Inflation, real rates, credit spreads
        for sid, attr, label in (
            ("DFII10", "real_risk_free_rate", "10y TIPS"),
            ("T10YIE", "expected_inflation",  "10y breakeven inflation"),
            ("BAA10Y", "baa_spread",          "BAA-10y spread"),
            ("AAA10Y", "aaa_spread",          "AAA-10y spread"),
        ):
            obs = client.latest_observation(sid)
            if obs is not None:
                setattr(env, attr, obs[0] / 100.0)
                env.note(f"{label} = {obs[0]:.2f}% ({obs[1]})")

        # ERP: SPY earnings yield method, else Damodaran Jan 2026
        implied = _spy_implied_erp(env.risk_free_rate)
        if implied is not None:
            env.equity_risk_premium = implied
            env.source_erp = "SPY E/P-implied"
        else:
            env.source_erp = "Damodaran Jan 2026 (4.23%)"

        # Long-run GDP — 20y CAGR
        gdp = client.series("GDP", start="2004-01-01")
        if gdp is not None and len(gdp) > 40:
            n_years = (gdp.index[-1] - gdp.index[0]).days / 365.25
            if n_years > 5:
                cagr = (gdp.iloc[-1] / gdp.iloc[0]) ** (1 / n_years) - 1
                if 0.02 < cagr < 0.08:
                    env.long_run_gdp = float(cagr)
                    env.source_gdp = f"FRED GDP CAGR ({n_years:.0f}y, {cagr*100:.2f}%)"
        if not env.source_gdp:
            env.source_gdp = "default 4.0%"

        return env


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    env = MacroEnvironment.fetch()
    print(f"\nrf      = {env.risk_free_rate:.4f}  ({env.source_rf})")
    print(f"erp     = {env.equity_risk_premium:.4f}  ({env.source_erp})")
    print(f"gdp     = {env.long_run_gdp:.4f}  ({env.source_gdp})")
    if env.real_risk_free_rate is not None:
        print(f"rf real = {env.real_risk_free_rate:.4f}")
    if env.expected_inflation is not None:
        print(f"infl    = {env.expected_inflation:.4f}")
    if env.baa_spread is not None:
        print(f"BAA spr = {env.baa_spread:.4f}")
    if env.aaa_spread is not None:
        print(f"AAA spr = {env.aaa_spread:.4f}")
