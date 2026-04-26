"""
screener.py — Unified stock-screening dispatcher

Routes a ticker to the correct 10-filter framework based on market cap:

    Market Cap               Framework      Rationale
    -----------------------  -------------  -----------------------------------
    < $300M                  penny_filter   Nano + micro-cap. Float / offering /
                                            dilution / RS risk dominate.
    $300M – $10B             growth_filter  Small-to-mid cap. Revenue momentum,
                                            institutional entry, narrative.
    >= $10B                  value_filter   Large + mega-cap. Quality, balance
                                            sheet, consistent earnings, analyst
                                            targets.

Exposes a single entry point, screen(), that returns a JSON-serializable dict
for web/API consumption. CLI mirrors the sub-screeners' surface.

Usage (Python):
    from screener import screen, screen_all, detect_framework
    result = screen("AAPL")                    # auto-routes
    result = screen("AAPL", framework="value") # force
    panel  = screen_all("NVDA")                # run all three (hybrid view)

Usage (CLI):
    python screener.py AAPL
    python screener.py AAPL --framework value --profile quality
    python screener.py NVDA --compare-all-frameworks --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

import yfinance as yf

import penny_filter
import growth_filter
import value_filter


# ---------------------------------------------------------------------------
# Market-cap bands
# ---------------------------------------------------------------------------

PENNY_MAX   = 300_000_000        # $300M
GROWTH_MAX  = 2_000_000_000     # $2B

MCAP_BANDS = [
    ("nano",   0,                  50_000_000),
    ("micro",  50_000_000,         300_000_000),
    ("small",  300_000_000,        2_000_000_000),
    ("mid",    2_000_000_000,      10_000_000_000),
    ("large",  10_000_000_000,     200_000_000_000),
    ("mega",   200_000_000_000,    float("inf")),
]


def mcap_band(mcap: Optional[float]) -> str:
    if mcap is None or mcap <= 0:
        return "unknown"
    for name, lo, hi in MCAP_BANDS:
        if lo <= mcap < hi:
            return name
    return "unknown"


def detect_framework(mcap: Optional[float]) -> str:
    """Map a market cap to the recommended framework key."""
    if mcap is None or mcap <= 0:
        # Unknown cap — default to penny (most conservative w.r.t. data holes)
        return "penny"
    if mcap < PENNY_MAX:
        return "penny"
    if mcap < GROWTH_MAX:
        return "growth"
    return "value"


# ---------------------------------------------------------------------------
# Framework registry
# ---------------------------------------------------------------------------

FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "penny": {
        "module":  penny_filter,
        "label":   "Penny / Micro-cap (< $300M)",
        "accepts": {"profile", "sector", "scoring_mode", "apply_bonuses",
                    "verbose", "news_days", "filing_days", "f9_engine"},
    },
    "growth": {
        "module":  growth_filter,
        "label":   "Growth / Small-Mid-cap ($300M – $10B)",
        "accepts": {"profile", "sector", "scoring_mode", "apply_bonuses",
                    "verbose", "engine", "lang"},
    },
    "value": {
        "module":  value_filter,
        "label":   "Value / Large-cap (>= $10B)",
        "accepts": {"profile", "sector", "scoring_mode", "apply_bonuses",
                    "verbose", "lang", "engine"},
    },
}


# ---------------------------------------------------------------------------
# Shared ticker lookup
# ---------------------------------------------------------------------------

def fetch_ticker_snapshot(ticker: str) -> Dict[str, Any]:
    """Lightweight pull used by the dispatcher for routing + metadata."""
    ticker = ticker.upper().strip()
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    mcap = info.get("marketCap")
    price = (info.get("currentPrice") or info.get("regularMarketPrice")
             or info.get("navPrice") or info.get("previousClose"))
    return {
        "ticker":          ticker,
        "name":            info.get("longName") or info.get("shortName"),
        "market_cap":      mcap,
        "market_cap_band": mcap_band(mcap),
        "sector":          info.get("sector"),
        "industry":        info.get("industry"),
        "currency":        info.get("currency"),
        "exchange":        info.get("exchange"),
        "price":           price,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _filter_kwargs(framework_key: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    accepts = FRAMEWORKS[framework_key]["accepts"]
    return {k: v for k, v in kwargs.items() if k in accepts and v is not None}


def screen(ticker: str,
           framework: str = "auto",
           profile: str = "balanced",
           sector: str = "auto",
           scoring_mode: str = "normal",
           apply_bonuses: bool = True,
           verbose: bool = False,
           lang: str = "en",
           engine: str = "auto",
           news_days: Optional[int] = None,
           filing_days: Optional[int] = None) -> Dict[str, Any]:
    """
    Screen a ticker against the appropriate 10-filter framework.

    Args:
        ticker:         Symbol.
        framework:      'auto' | 'penny' | 'growth' | 'value'.
        profile:        Weighting profile (forwarded).
        sector:         'auto' | 'none' | overlay key (forwarded).
        scoring_mode:   'strict' | 'normal' | 'generous'.
        apply_bonuses:  Enable BONUS_RULES in the sub-screener.
        verbose:        Print progress. Default False (web/API use).
        lang:           'en' | 'uz' (growth + value only).
        engine:         Sentiment engine (penny: 'f9_engine', growth: 'engine').
        news_days:      Penny F9 lookback (days).
        filing_days:    Penny F5 lookback (days).

    Returns:
        JSON-serializable dict:
          {
            "ticker": ...,
            "framework_used": "penny" | "growth" | "value",
            "framework_auto_detected": bool,
            "framework_label": "...",
            "market_cap": float | None,
            "market_cap_band": "...",
            "snapshot": { ticker info },
            "result": { full sub-screener evaluation },
          }
    """
    ticker = ticker.upper().strip()
    snapshot = fetch_ticker_snapshot(ticker)
    mcap = snapshot["market_cap"]

    if framework == "auto":
        fw_key = detect_framework(mcap)
        auto_detected = True
    else:
        fw_key = framework.lower().strip()
        auto_detected = False

    if fw_key not in FRAMEWORKS:
        raise ValueError(
            f"Unknown framework '{framework}'. "
            f"Choose from: auto, {', '.join(FRAMEWORKS.keys())}"
        )

    mod = FRAMEWORKS[fw_key]["module"]

    # If the supplied profile is not valid for the resolved framework,
    # gracefully fall back to "balanced" (every framework defines it).
    valid_profiles = mod.WEIGHT_PROFILES.keys()
    if profile not in valid_profiles:
        profile = "balanced"

    # Same for sector overlays: auto/none always OK, otherwise verify.
    valid_sectors = mod.SECTOR_OVERLAYS.keys()
    if sector not in ("auto", "none", None, "") and sector not in valid_sectors:
        sector = "auto"

    # Map dispatcher kwargs to each sub-screener's signature
    raw_kwargs = {
        "profile":       profile,
        "sector":        sector,
        "scoring_mode":  scoring_mode,
        "apply_bonuses": apply_bonuses,
        "verbose":       verbose,
        "lang":          lang,
        "engine":        engine,       # growth
        "f9_engine":     engine,       # penny uses this name
        "news_days":     news_days,    # penny only
        "filing_days":   filing_days,  # penny only
    }
    call_kwargs = _filter_kwargs(fw_key, raw_kwargs)

    result = mod.evaluate(ticker, **call_kwargs)

    return {
        "ticker":                  ticker,
        "framework_used":          fw_key,
        "framework_auto_detected": auto_detected,
        "framework_label":         FRAMEWORKS[fw_key]["label"],
        "market_cap":              mcap,
        "market_cap_band":         snapshot["market_cap_band"],
        "snapshot":                snapshot,
        "result":                  result,
    }


def compare_all_profiles(ticker: str,
                         framework: str = "auto",
                         **kwargs) -> Dict[str, Any]:
    """
    Run every profile of the (resolved) framework and return a side-by-side
    panel. Designed to be backed by the ticker-data cache so the underlying
    yfinance / SEC fetches happen once and are reused across N profile runs.
    """
    ticker = ticker.upper().strip()
    snapshot = fetch_ticker_snapshot(ticker)
    mcap = snapshot["market_cap"]

    if framework == "auto":
        fw_key = detect_framework(mcap)
        auto_detected = True
    else:
        fw_key = framework.lower().strip()
        auto_detected = False

    if fw_key not in FRAMEWORKS:
        raise ValueError(f"Unknown framework '{framework}'.")

    mod = FRAMEWORKS[fw_key]["module"]
    profile_keys = list(mod.WEIGHT_PROFILES.keys())

    # Drop any profile from kwargs — we iterate them
    kwargs.pop("profile", None)

    runs: Dict[str, Any] = {}
    for p in profile_keys:
        try:
            runs[p] = screen(ticker, framework=fw_key, profile=p, **kwargs)
        except Exception as e:
            runs[p] = {"error": str(e), "profile": p,
                       "framework_used": fw_key}

    return {
        "ticker":                  ticker,
        "framework_used":          fw_key,
        "framework_auto_detected": auto_detected,
        "framework_label":         FRAMEWORKS[fw_key]["label"],
        "snapshot":                snapshot,
        "profiles":                runs,
    }


def screen_all(ticker: str, **kwargs) -> Dict[str, Any]:
    """
    Run the ticker against ALL three frameworks. Useful for hybrid names
    (e.g., a mid-cap that behaves like both growth and value) or for the
    web UI's compare-view tab.
    """
    ticker = ticker.upper().strip()
    snapshot = fetch_ticker_snapshot(ticker)
    suggested = detect_framework(snapshot["market_cap"])

    runs: Dict[str, Dict[str, Any]] = {}
    for fw_key in FRAMEWORKS.keys():
        try:
            runs[fw_key] = screen(ticker, framework=fw_key, **kwargs)
        except Exception as e:
            runs[fw_key] = {"error": str(e), "framework_used": fw_key}

    return {
        "ticker":               ticker,
        "snapshot":             snapshot,
        "suggested_framework":  suggested,
        "frameworks":           runs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_human(res: Dict[str, Any]) -> None:
    snap = res["snapshot"]
    mcap = res["market_cap"]
    mcap_s = f"${mcap:,.0f}" if mcap else "n/a"
    print()
    print("=" * 72)
    print(f"  {res['ticker']}   {snap.get('name') or ''}")
    print(f"  Market Cap:  {mcap_s}   [{res['market_cap_band']}-cap]")
    print(f"  Sector:      {snap.get('sector') or 'n/a'}  "
          f"/  {snap.get('industry') or 'n/a'}")
    print(f"  Framework:   {res['framework_used'].upper()}  "
          f"({'auto' if res['framework_auto_detected'] else 'forced'})")
    print(f"               {res['framework_label']}")
    print("=" * 72)

    inner = res["result"]
    # Sub-screeners return a verdict / score / score_max triplet.
    verdict = inner.get("verdict") or inner.get("final_verdict") or "n/a"
    score = inner.get("score") or inner.get("weighted_score") or inner.get("total_score")
    score_max = inner.get("score_max") or 100
    print(f"  Verdict:  {verdict}")
    if score is not None:
        print(f"  Score:    {score} / {score_max}")
    print()


def _print_compare(panel: Dict[str, Any]) -> None:
    snap = panel["snapshot"]
    mcap = snap["market_cap"]
    mcap_s = f"${mcap:,.0f}" if mcap else "n/a"
    print()
    print("=" * 72)
    print(f"  {panel['ticker']}   {snap.get('name') or ''}")
    print(f"  Market Cap:  {mcap_s}   [{snap['market_cap_band']}-cap]")
    print(f"  Suggested framework (auto): {panel['suggested_framework'].upper()}")
    print("=" * 72)

    for fw_key, run in panel["frameworks"].items():
        label = FRAMEWORKS[fw_key]["label"]
        print(f"\n--- {fw_key.upper()}  ({label}) ---")
        if "error" in run:
            print(f"  [error] {run['error']}")
            continue
        inner = run["result"]
        verdict = inner.get("verdict") or inner.get("final_verdict") or "n/a"
        score = inner.get("score") or inner.get("weighted_score") or inner.get("total_score")
        score_max = inner.get("score_max") or 100
        print(f"  Verdict:  {verdict}")
        if score is not None:
            print(f"  Score:    {score} / {score_max}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Unified stock screener — routes to penny / growth / "
                    "value framework based on market cap."
    )
    ap.add_argument("ticker", help="Ticker symbol (e.g. AAPL).")
    ap.add_argument("--framework", "-f",
                    default="auto",
                    choices=["auto", "penny", "growth", "value"],
                    help="Force a specific framework (default: auto by mcap).")
    ap.add_argument("--profile", "-p", default="balanced",
                    help="Weighting profile (see sub-screener profiles).")
    ap.add_argument("--sector", "-s", default="auto",
                    help="'auto' (detect) | 'none' (disable) | overlay key.")
    ap.add_argument("--scoring", default="normal",
                    choices=["strict", "normal", "generous"],
                    help="Scoring mode.")
    ap.add_argument("--no-bonuses", action="store_true",
                    help="Disable BONUS_RULES.")
    ap.add_argument("--lang", default="en", choices=["en", "uz"],
                    help="Output language (growth + value only).")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "keyword", "finbert", "claude"],
                    help="Sentiment engine for F9/F4 (penny/growth).")
    ap.add_argument("--news-days", type=int, default=None,
                    help="Penny F9 news lookback window.")
    ap.add_argument("--filing-days", type=int, default=None,
                    help="Penny F5 filings lookback window.")
    ap.add_argument("--compare-all-frameworks", action="store_true",
                    help="Run the ticker through ALL three frameworks.")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON to stdout (for web consumption).")
    ap.add_argument("--verbose", action="store_true",
                    help="Show sub-screener progress output.")
    args = ap.parse_args(argv)

    common = dict(
        profile=args.profile,
        sector=args.sector,
        scoring_mode=args.scoring,
        apply_bonuses=not args.no_bonuses,
        verbose=args.verbose,
        lang=args.lang,
        engine=args.engine,
        news_days=args.news_days,
        filing_days=args.filing_days,
    )

    if args.compare_all_frameworks:
        panel = screen_all(args.ticker, **common)
        if args.json:
            print(json.dumps(panel, indent=2, default=str))
        else:
            _print_compare(panel)
        return 0

    res = screen(args.ticker, framework=args.framework, **common)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        _print_human(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
