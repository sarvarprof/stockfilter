"""
Value / Quality Stock 10-Filter Framework Evaluator — FULLY AUTOMATED.

Hybrid quality-momentum screen (RS, earnings beats, analyst coverage
+ balance-sheet health + margin stability). Web-query optimized.

Usage:
    python value_filter.py TICKER
    python value_filter.py TICKER --profile quality --sector tech --scoring strict
    python value_filter.py TICKER --compare-all
    python value_filter.py TICKER --json           # for API/web use
"""

import os
import sys
import re
import json
import argparse
import statistics
from datetime import datetime, timedelta, timezone

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

# Reuse helpers from penny_filter to avoid duplication
from penny_filter import (
    http_get, sec_filings, get_news, match_any,
    _score_keyword, _score_finbert, _load_finbert,
)


# ============================================================
# i18n
# ============================================================

MSGS = {
    "uz": {
        "pass": "🟢 PASS", "fail": "🔴 FAIL", "neu": "🟡 NEUTRAL",
        "evaluating": "Baholanmoqda", "summary": "YAKUNIY NATIJA",
        "total": "JAMI BALL", "verdict": "XULOSA",
        "strong_buy": "🟢 STRONG BUY",
        "buy":        "🟡 BUY (Ehtiyot bilan)",
        "watch":      "🟡 WATCH — Kuzatish",
        "do_not_buy": "🔴 DO NOT BUY",
        "F1": "F1  Daromad o'sishi (Revenue Momentum)",
        "F2": "F2  Gross Margin barqarorligi",
        "F3": "F3  Institutsional qo'llab-quvvatlash",
        "F4": "F4  Earnings surprise tarixi",
        "F5": "F5  Earnings sanasiga masofa",
        "F6": "F6  Analitik konsensus",
        "F7": "F7  Relative Strength (RS)",
        "F8": "F8  Balans varag'i sog'lomligi",
        "F9": "F9  So'nggi katalizator (30d)",
        "F10":"F10 Sektor makro muhiti",
    },
    "en": {
        "pass": "🟢 PASS", "fail": "🔴 FAIL", "neu": "🟡 NEUTRAL",
        "evaluating": "Evaluating", "summary": "SUMMARY",
        "total": "TOTAL SCORE", "verdict": "VERDICT",
        "strong_buy": "🟢 STRONG BUY",
        "buy":        "🟡 BUY (cautious)",
        "watch":      "🟡 WATCH",
        "do_not_buy": "🔴 DO NOT BUY",
        "F1": "F1  Revenue momentum",
        "F2": "F2  Gross margin stability",
        "F3": "F3  Institutional support",
        "F4": "F4  Earnings surprise history",
        "F5": "F5  Distance to earnings",
        "F6": "F6  Analyst consensus",
        "F7": "F7  Relative Strength (RS)",
        "F8": "F8  Balance sheet health",
        "F9": "F9  Recent catalyst (30d)",
        "F10":"F10 Sector macro environment",
    },
}


# ============================================================
# WEIGHT PROFILES (analyst-tuned, sum = 100)
# ============================================================
# Tier Quality/Survival(26): F8=14, F2=12
# Tier Earnings Power(24):   F1=13, F4=11
# Tier Market Validation(21):F7=12, F3=9
# Tier Forward Look(17):     F6=10, F9=7
# Tier Timing(12):           F10=7, F5=5

WEIGHT_PROFILES = {
    # Default balanced analyst view
    "balanced": {
        "F1": 13, "F2": 12, "F3": 9,  "F4": 11, "F5": 5,
        "F6": 10, "F7": 12, "F8": 14, "F9": 7,  "F10": 7,
    },
    # Deep value — balance sheet + cheap moats above all
    "deep_value": {
        "F1": 10, "F2": 15, "F3": 10, "F4": 8,  "F5": 3,
        "F6": 8,  "F7": 6,  "F8": 22, "F9": 6,  "F10": 12,
    },
    # Quality compounder — margin + earnings power + institutional
    "quality": {
        "F1": 14, "F2": 18, "F3": 12, "F4": 14, "F5": 3,
        "F6": 10, "F7": 10, "F8": 13, "F9": 3,  "F10": 3,
    },
    # GARP (growth-at-reasonable-price) — growth + analyst upside
    "garp": {
        "F1": 18, "F2": 12, "F3": 9,  "F4": 13, "F5": 4,
        "F6": 14, "F7": 12, "F8": 10, "F9": 5,  "F10": 3,
    },
    # Momentum-lean quality — RS + sector macro dominate
    "momentum": {
        "F1": 10, "F2": 8,  "F3": 7,  "F4": 10, "F5": 6,
        "F6": 10, "F7": 22, "F8": 8,  "F9": 9,  "F10": 10,
    },
    # Earnings-week swing — avoid earnings-week risk, look for beat pattern
    "earnings_swing": {
        "F1": 10, "F2": 8,  "F3": 7,  "F4": 18, "F5": 15,
        "F6": 10, "F7": 12, "F8": 8,  "F9": 7,  "F10": 5,
    },
}

for _p, _w in WEIGHT_PROFILES.items():
    assert sum(_w.values()) == 100, f"Profile {_p} != 100"


# Veto rules — fail a critical filter, cap verdict
VETO_RULES = {
    "balanced":       {"F8": "WATCH"},            # bad balance sheet -> WATCH
    "deep_value":     {"F8": "DO_NOT_BUY"},       # no negotiation
    "quality":        {"F2": "WATCH", "F8": "WATCH"},
    "garp":           {"F1": "WATCH"},            # no growth -> WATCH
    "momentum":       {"F7": "WATCH", "F10": "WATCH"},
    "earnings_swing": {"F5": "DO_NOT_BUY"},       # earnings within 1w = no trade
}


# ============================================================
# SECTOR OVERLAYS
# ============================================================

SECTOR_OVERLAYS = {
    # Tech / SaaS — margin & RS matter most
    "tech": {
        "F1": +1, "F2": +5, "F3": +1, "F4": +1, "F5": 0,
        "F6": 0,  "F7": +3, "F8": -4, "F9": 0,  "F10": -7,
    },
    # Financials / Banks — balance sheet dominates
    "financial": {
        "F1": -2, "F2": -8, "F3": +3, "F4": +1, "F5": 0,
        "F6": +1, "F7": 0,  "F8": +8, "F9": -2, "F10": -1,
    },
    # Healthcare / Pharma — catalysts (trials, approvals) matter
    "healthcare": {
        "F1": -1, "F2": +2, "F3": +1, "F4": -2, "F5": -1,
        "F6": +2, "F7": 0,  "F8": +1, "F9": +5, "F10": -7,
    },
    # Energy / Commodity — cycle-driven; sector macro critical
    "energy": {
        "F1": -2, "F2": -3, "F3": +2, "F4": -1, "F5": 0,
        "F6": 0,  "F7": +1, "F8": +2, "F9": -1, "F10": +2,
    },
    # Consumer — brand moat (margin) + revenue growth
    "consumer": {
        "F1": +3, "F2": +3, "F3": 0,  "F4": +1, "F5": 0,
        "F6": 0,  "F7": 0,  "F8": -2, "F9": 0,  "F10": -5,
    },
    # Industrial / Defense — balance sheet + macro + catalysts
    "industrial": {
        "F1": +1, "F2": -1, "F3": +1, "F4": 0,  "F5": 0,
        "F6": +1, "F7": -1, "F8": +2, "F9": +2, "F10": -5,
    },
    # REIT — balance sheet (leverage) + macro rates
    "reit": {
        "F1": -2, "F2": -6, "F3": +2, "F4": -1, "F5": 0,
        "F6": +1, "F7": 0,  "F8": +6, "F9": -2, "F10": +2,
    },
    # Utility — low growth, stable, rate-sensitive
    "utility": {
        "F1": -5, "F2": -4, "F3": +2, "F4": -3, "F5": +1,
        "F6": +1, "F7": -2, "F8": +8, "F9": -1, "F10": +3,
    },
}

SECTOR_KEYWORDS = {
    "tech":       ["technology", "software", "semiconductor", "internet",
                   "saas", "cloud", "information technology"],
    "financial":  ["financial", "bank", "insurance", "capital markets",
                   "asset management"],
    "healthcare": ["healthcare", "biotech", "pharmaceutical",
                   "drug manufacturers", "medical device", "life sciences"],
    "energy":     ["energy", "oil", "gas", "pipeline", "uranium"],
    "consumer":   ["consumer", "retail", "apparel", "restaurant",
                   "beverage", "household products", "leisure"],
    "industrial": ["industrial", "aerospace", "defense", "machinery",
                   "construction", "transportation", "airlines"],
    "reit":       ["reit", "real estate"],
    "utility":    ["utilities", "utility", "electric"],
}


def _renormalize(w):
    w = {k: max(0, v) for k, v in w.items()}
    s = sum(w.values())
    if s == 0:
        return {k: 10 for k in w}
    scaled = {k: v * 100.0 / s for k, v in w.items()}
    rounded = {k: round(v) for k, v in scaled.items()}
    diff = 100 - sum(rounded.values())
    if diff != 0:
        fracs = sorted(scaled.items(),
                       key=lambda kv: (kv[1] - int(kv[1])),
                       reverse=(diff > 0))
        for k, _ in fracs[:abs(diff)]:
            rounded[k] += 1 if diff > 0 else -1
    return rounded


def resolve_weights(profile="balanced", sector_key=None):
    base = dict(WEIGHT_PROFILES[profile])
    if sector_key and sector_key in SECTOR_OVERLAYS:
        merged = {k: base[k] + SECTOR_OVERLAYS[sector_key].get(k, 0)
                  for k in base}
        return _renormalize(merged)
    return base


def detect_sector(info):
    haystack = " ".join([
        (info.get("sector") or ""),
        (info.get("industry") or ""),
        (info.get("industryKey") or ""),
    ]).lower()
    if not haystack.strip():
        return None
    for key, kws in SECTOR_KEYWORDS.items():
        if any(kw in haystack for kw in kws):
            return key
    return None


# ============================================================
# SCORING MODES + BONUSES
# ============================================================

SCORING_MODES = {
    "strict":   {"neutral_credit": 0.30,
                 "desc": "NEUTRAL = 30% credit"},
    "normal":   {"neutral_credit": 0.50,
                 "desc": "NEUTRAL = 50% credit (default)"},
    "generous": {"neutral_credit": 0.70,
                 "desc": "NEUTRAL = 70% credit"},
}

BONUS_RULES = [
    {"name": "Quality sweep",
     "requires": ["F1", "F2", "F8"], "points": 5},
    {"name": "Earnings credibility",
     "requires": ["F1", "F4"], "points": 3},
    {"name": "Market validation",
     "requires": ["F3", "F6", "F7"], "points": 3},
    {"name": "Tailwind confirmed",
     "requires": ["F9", "F10"], "points": 2},
]


# ============================================================
# FILTER IMPLEMENTATIONS
# ============================================================

def v1_revenue_momentum(ticker):
    """F1: Annual YoY revenue 10%+ = PASS (EDGAR primary); quarterly trend as supplement."""
    # EDGAR annual YoY (audited, most reliable for large-caps)
    annual_yoy = None
    annual_note = ""
    try:
        import edgar_bridge as _eb
        rev_s = _eb.get_revenue_series(ticker)  # ascending: oldest → newest
        if rev_s is not None and len(rev_s) >= 2:
            old = float(rev_s.iloc[-2])
            new = float(rev_s.iloc[-1])
            if old > 0:
                annual_yoy = (new - old) / old
                annual_note = f"Annual YoY {annual_yoy*100:+.1f}% (EDGAR)"
    except Exception:
        pass

    if annual_yoy is not None:
        # QoQ trend as supplemental signal
        recent_trend = "unknown"
        try:
            qf = yf.Ticker(ticker).quarterly_financials
            if qf is not None and not qf.empty:
                for idx in qf.index:
                    if "total revenue" in str(idx).lower():
                        vals = qf.loc[idx].dropna().tolist()
                        if len(vals) >= 3:
                            recent_trend = "up" if vals[0] > vals[1] > vals[2] else (
                                           "flat" if vals[0] >= vals[2] else "down")
                        break
        except Exception:
            pass
        note = annual_note
        if recent_trend != "unknown":
            note += f", QoQ trend: {recent_trend}"
        if annual_yoy >= 0.10:
            return True, note
        if annual_yoy >= 0:
            return None, f"{note} (0-10%)"
        return False, f"{note} declining"

    # Fallback: quarterly only
    try:
        qf = yf.Ticker(ticker).quarterly_financials
        if qf is None or qf.empty:
            return None, "Quarterly data unavailable"
        for idx in qf.index:
            if "total revenue" in str(idx).lower():
                vals = qf.loc[idx].dropna().tolist()
                if len(vals) < 4:
                    return None, "Insufficient quarters"
                yoy = None
                if len(vals) >= 5:
                    yoy = (vals[0] - vals[4]) / vals[4] if vals[4] else None
                recent_trend = "up" if vals[0] > vals[1] > vals[2] else (
                               "flat" if vals[0] >= vals[2] else "down")
                if yoy is not None and yoy >= 0.10:
                    return True, f"YoY {yoy*100:+.1f}%, trend: {recent_trend}"
                if yoy is not None and yoy >= 0:
                    return None, f"YoY {yoy*100:+.1f}% (0-10%), trend: {recent_trend}"
                if yoy is not None:
                    return False, f"YoY {yoy*100:+.1f}% declining"
                if recent_trend == "up":
                    return True, "2Q uptrend (no YoY data)"
                return None, "Flat/mixed trend"
        return None, "No revenue line"
    except Exception as e:
        return None, f"err: {e}"


def v2_gross_margin(ticker):
    """F2: Gross margin >=50% AND stable/expanding."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        gm = info.get("grossMargins")
        # Compute recent series from financials for stability check
        fin = t.financials
        series = []
        if fin is not None and not fin.empty:
            rev = cogs = None
            for idx in fin.index:
                low = str(idx).lower()
                if "total revenue" in low:
                    rev = fin.loc[idx].dropna().tolist()
                elif "cost of revenue" in low or "cost of goods" in low:
                    cogs = fin.loc[idx].dropna().tolist()
            if rev and cogs:
                n = min(len(rev), len(cogs), 4)
                series = [(rev[i] - cogs[i]) / rev[i]
                          for i in range(n) if rev[i]]

        if gm is None and not series:
            return None, "Gross margin unavailable"
        current = gm if gm is not None else (series[0] if series else None)
        current_pct = current * 100 if current and current < 1.5 else current

        if not series or len(series) < 2:
            # Only have point estimate
            if current_pct is not None and current_pct >= 50:
                return True, f"GM {current_pct:.1f}% (no trend data)"
            if current_pct is not None and current_pct >= 35:
                return None, f"GM {current_pct:.1f}% (<50%)"
            return False, f"GM {current_pct:.1f}% low"

        # Trend check: newest-first, so series[0] > series[-1] = expanding
        expanding = series[0] >= series[-1]
        avg = statistics.mean(series) * 100

        if avg >= 50 and expanding:
            return True, f"GM avg {avg:.1f}%, expanding"
        if avg >= 50:
            return None, f"GM avg {avg:.1f}% but contracting"
        if avg >= 35:
            return None, f"GM avg {avg:.1f}% (<50%)"
        return False, f"GM avg {avg:.1f}% low"
    except Exception as e:
        return None, f"err: {e}"


def v3_institutional_support(info):
    """F3: Top-tier fund accumulation. Proxy: heldPercentInstitutions + recent change."""
    inst = info.get("heldPercentInstitutions")
    if inst is None:
        return None, "Institutional data unavailable"
    pct = inst * 100 if inst < 1.5 else inst
    # We don't have QoQ delta easily from yfinance free tier.
    # Use level as proxy: >60% strong institutional base.
    if pct >= 60:
        return True, f"Institutional {pct:.1f}% (strong base)"
    if pct >= 30:
        return None, f"Institutional {pct:.1f}% (moderate)"
    return False, f"Institutional {pct:.1f}% (weak)"


def v4_earnings_surprises(ticker):
    """F4: Beats in 3+ of last 4 quarters."""
    try:
        t = yf.Ticker(ticker)
        # Try earnings_history first
        eh = None
        try:
            eh = t.earnings_history
        except Exception:
            pass
        if eh is None or (hasattr(eh, "empty") and eh.empty):
            try:
                eh = t.earnings_dates
            except Exception:
                eh = None

        if eh is None or (hasattr(eh, "empty") and eh.empty):
            return None, "Earnings history unavailable"

        beats = 0
        checked = 0
        for _, row in eh.iterrows():
            if checked >= 4:
                break
            # column names vary by yfinance version
            surprise = None
            for col in ("Surprise(%)", "surprise", "epsSurprisePct",
                        "Surprise Percent"):
                if col in row and row[col] is not None:
                    try:
                        surprise = float(row[col])
                        break
                    except Exception:
                        pass
            if surprise is None:
                est = reported = None
                for c in ("Estimate", "epsEstimate"):
                    if c in row:
                        est = row[c]
                for c in ("Reported EPS", "epsActual"):
                    if c in row:
                        reported = row[c]
                if est is not None and reported is not None:
                    try:
                        surprise = (float(reported) - float(est))
                    except Exception:
                        pass

            if surprise is None:
                continue
            checked += 1
            if surprise > 0:
                beats += 1

        if checked == 0:
            return None, "No parseable surprise data"
        if beats >= 3:
            return True, f"{beats}/{checked} beats"
        if beats == 2:
            return None, f"{beats}/{checked} beats (mixed)"
        return False, f"Only {beats}/{checked} beats"
    except Exception as e:
        return None, f"err: {e}"


def v5_earnings_distance(info, ticker):
    """F5: Next earnings > 3 weeks away = PASS."""
    ts = (info.get("earningsTimestamp")
          or info.get("earningsTimestampStart"))
    dt = None
    if ts:
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            dt = None
    if dt is None:
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is not None:
                if hasattr(cal, "loc"):
                    for key in ("Earnings Date", "earningsDate"):
                        try:
                            val = cal.loc[key].iloc[0]
                            dt = (val if isinstance(val, datetime)
                                  else datetime.fromisoformat(str(val)))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            break
                        except Exception:
                            continue
                elif isinstance(cal, dict):
                    ed = cal.get("Earnings Date") or cal.get("earningsDate")
                    if ed:
                        val = ed[0] if isinstance(ed, list) else ed
                        try:
                            dt = (val if isinstance(val, datetime)
                                  else datetime.fromisoformat(str(val)))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        except Exception:
                            pass
        except Exception:
            pass

    if dt is None:
        return None, "Earnings date unavailable"

    now = datetime.now(timezone.utc)
    days = (dt - now).days
    if days < 0:
        return None, f"Last earnings {abs(days)}d ago"
    if days >= 21:
        return True, f"Earnings in {days}d (>3 weeks)"
    if days >= 7:
        return None, f"Earnings in {days}d (1-3 weeks)"
    return False, f"Earnings in {days}d (<1 week)"


def v6_analyst_consensus(info):
    """F6: Consensus Buy/Strong Buy AND target > 10% above price."""
    rec = info.get("recommendationKey", "").lower()
    target = (info.get("targetMeanPrice")
              or info.get("targetMedianPrice"))
    price = (info.get("currentPrice")
             or info.get("regularMarketPrice"))
    n = info.get("numberOfAnalystOpinions") or 0

    if not rec and not target:
        return None, "Analyst data unavailable"

    upside = None
    if target and price:
        upside = (target - price) / price

    rec_pass = rec in ("buy", "strong_buy", "strong buy")
    rec_hold = rec in ("hold", "neutral")
    rec_sell = rec in ("sell", "strong_sell", "underperform")

    upside_pass = upside is not None and upside >= 0.10

    note = f"rec={rec or '?'}"
    if upside is not None:
        note += f", upside={upside*100:+.1f}%, n={n}"

    if rec_pass and upside_pass:
        return True, note
    if rec_sell or (upside is not None and upside < -0.05):
        return False, note
    if rec_hold or (upside is not None and -0.05 <= upside < 0.10):
        return None, note
    # Partial: one condition met
    if rec_pass or upside_pass:
        return None, note
    return None, note


def v7_relative_strength(ticker):
    """F7: 12-month return vs S&P 500. Maps to RS 0-100 proxy.
       RS 70+ = PASS, 50-70 NEUTRAL, <50 FAIL."""
    try:
        stock = yf.Ticker(ticker).history(period="1y")
        spy = yf.Ticker("SPY").history(period="1y")
        if stock is None or stock.empty or spy is None or spy.empty:
            return None, "Price history unavailable"
        s_ret = (stock["Close"].iloc[-1] / stock["Close"].iloc[0]) - 1
        m_ret = (spy["Close"].iloc[-1] / spy["Close"].iloc[0]) - 1
        # Relative outperformance
        delta = s_ret - m_ret
        # Map delta to RS 0-100 proxy: +30% outperf -> ~95, 0 -> 50, -30% -> ~5
        rs = max(0, min(100, 50 + delta * 150))
        note = (f"Stock {s_ret*100:+.1f}% vs SPY {m_ret*100:+.1f}%, "
                f"RS proxy={rs:.0f}")
        if rs >= 70:
            return True, note
        if rs >= 50:
            return None, note
        return False, note
    except Exception as e:
        return None, f"err: {e}"


def v8_balance_sheet(ticker, info):
    """F8: Cash > Debt OR D/E below sector, positive OCF. EDGAR primary."""
    score = 0
    total = 0
    notes = []
    src_tag = ""

    try:
        # EDGAR primary — properly-summed LTD+STD, audited OCF, derived D/E
        import edgar_bridge as _eb
        cdo = _eb.get_cash_debt_ocf(ticker)
        if cdo is not None:
            total_cash = cdo["cash"]
            total_debt = cdo["total_debt"]
            ocf        = cdo["ocf"]
            d_to_e     = cdo["d_to_e"]   # fraction (e.g. 0.45), or None
            src_tag = " (EDGAR)"

            if total_cash or total_debt:
                total += 1
                notes.append(f"cash ${total_cash/1e6:.0f}M vs debt ${total_debt/1e6:.0f}M{src_tag}")
                if total_cash > total_debt:
                    score += 1

            if d_to_e is not None:
                total += 1
                notes.append(f"D/E={d_to_e*100:.0f}%{src_tag}")
                if d_to_e < 1.0:   # fraction: 1.0 = 100% = threshold
                    score += 1

            if ocf is not None:
                total += 1
                notes.append(f"OCF=${ocf/1e6:.0f}M{src_tag}")
                if ocf > 0:
                    score += 1

            if total > 0:
                pct = score / total
                note = "; ".join(notes) + f"  ({score}/{total})"
                if pct >= 0.67:
                    return True, note
                if pct >= 0.34:
                    return None, note
                return False, note
    except Exception:
        pass

    # yfinance fallback
    try:
        total_cash = info.get("totalCash") or 0
        total_debt = info.get("totalDebt") or 0
        d2e = info.get("debtToEquity")
        ocf = info.get("operatingCashflow") or 0

        if total_cash or total_debt:
            total += 1
            notes.append(f"cash ${total_cash/1e6:.0f}M vs debt ${total_debt/1e6:.0f}M")
            if total_cash > total_debt:
                score += 1

        if d2e is not None:
            total += 1
            notes.append(f"D/E={d2e:.1f}")
            if d2e < 100:  # yfinance reports as percent
                score += 1

        if ocf:
            total += 1
            notes.append(f"OCF=${ocf/1e6:.0f}M")
            if ocf > 0:
                score += 1

        if total == 0:
            return None, "Balance sheet data unavailable"

        pct = score / total
        note = "; ".join(notes) + f"  ({score}/{total})"
        if pct >= 0.67:
            return True, note
        if pct >= 0.34:
            return None, note
        return False, note
    except Exception as e:
        return None, f"err: {e}"


VALUE_POSITIVE_CATALYSTS = [
    "new contract", "awarded", "partnership", "acquisition", "merger",
    "product launch", "launches", "expands", "expansion",
    "buyback", "share repurchase", "dividend increase", "raises dividend",
    "fda approval", "fda clearance", "breakthrough",
    "beats estimates", "raises guidance", "record revenue",
    "upgraded", "upgrade", "strategic alliance",
]
VALUE_NEGATIVE_CATALYSTS = [
    "lawsuit", "investigation", "downgrade", "cuts guidance",
    "lowered guidance", "misses", "recall", "fraud", "sec charges",
    "delisting", "going concern",
]


def v9_recent_catalyst(ticker, days=30, engine="auto"):
    """
    F9: Positive news in last 30d.

    engine: 'auto' | 'keyword' | 'finbert'
        - 'auto': use FinBERT if model is loadable, else keyword
        - 'keyword': value catalyst dictionary match
        - 'finbert': transformer sentiment classifier per-headline
    """
    news = get_news(ticker, days=days)
    if not news:
        return None, "No recent headlines"

    chosen = engine
    if engine == "auto":
        chosen = "finbert" if _load_finbert() else "keyword"

    pos, neg, neu = 0, 0, 0
    top_pos, top_neg = None, None

    if chosen == "finbert":
        for n in news:
            title = n.get("title") or ""
            label = _score_finbert(title)            # 'POS' | 'NEG' | 'NEU' | None
            if label is None:                        # FinBERT failed → keyword fallback
                label = _score_keyword(title)        # same shape from penny_filter
            label = (label or "NEU").upper()[:3]
            if label == "POS":
                pos += 1
                if not top_pos:
                    top_pos = title
            elif label == "NEG":
                neg += 1
                if not top_neg:
                    top_neg = title
            else:
                neu += 1
    else:  # keyword path uses value-specific catalyst dictionaries
        for n in news:
            t = (n.get("title") or "").lower()
            matched_pos = next((kw for kw in VALUE_POSITIVE_CATALYSTS
                                if kw in t), None)
            matched_neg = next((kw for kw in VALUE_NEGATIVE_CATALYSTS
                                if kw in t), None)
            if matched_pos and not matched_neg:
                pos += 1
                if not top_pos:
                    top_pos = matched_pos
            elif matched_neg and not matched_pos:
                neg += 1
                if not top_neg:
                    top_neg = matched_neg
            else:
                neu += 1

    note = (f"[{chosen}] {pos} pos, {neg} neg, {neu} neu in {days}d")
    if pos and not neg:
        return True, note + (f" (top: {top_pos})" if top_pos else "")
    if neg and neg > pos:
        return False, note + (f" (top: {top_neg})" if top_neg else "")
    if pos or neg:
        return None, note
    return None, note + " -> neutral"


SECTOR_ETF_MAP = {
    "tech":       "XLK",
    "financial":  "XLF",
    "healthcare": "XLV",
    "energy":     "XLE",
    "consumer":   "XLY",   # discretionary default; XLP for staples
    "industrial": "XLI",
    "reit":       "XLRE",
    "utility":    "XLU",
}


def v10_sector_macro(sector_key):
    """F10: Sector ETF above 50-day MA."""
    etf = SECTOR_ETF_MAP.get(sector_key, "SPY")
    try:
        h = yf.Ticker(etf).history(period="4mo")
        if h is None or h.empty or len(h) < 50:
            return None, f"{etf} history insufficient"
        close = h["Close"].iloc[-1]
        ma50 = h["Close"].iloc[-50:].mean()
        diff = (close - ma50) / ma50
        note = f"{etf} close {close:.2f} vs MA50 {ma50:.2f} ({diff*100:+.1f}%)"
        if close > ma50 and diff > 0.01:
            return True, note
        if abs(diff) <= 0.01:
            return None, note + " (at MA)"
        return False, note
    except Exception as e:
        return None, f"err: {e}"


# ============================================================
# RUNNER
# ============================================================

def verdict(score, lang="en"):
    """Weighted score (0-100) -> verdict."""
    m = MSGS[lang]
    if score >= 70:
        return m["strong_buy"]
    if score >= 55:
        return m["buy"]
    if score >= 40:
        return m["watch"]
    return m["do_not_buy"]


VERDICT_RANK = {
    "🟢 STRONG BUY": 3,
    "🟡 BUY (cautious)": 2, "🟡 BUY (Ehtiyot bilan)": 2,
    "🟡 WATCH": 1, "🟡 WATCH — Kuzatish": 1,
    "🔴 DO NOT BUY": 0,
}


def apply_vetoes(raw_verdict, fail_ids, profile, lang):
    m = MSGS[lang]
    vetoes = VETO_RULES.get(profile, {})
    cap_level = None
    triggered = []
    for fid, cap in vetoes.items():
        if fid in fail_ids:
            triggered.append(f"{fid}->{cap}")
            level = 0 if cap == "DO_NOT_BUY" else 1
            if cap_level is None or level < cap_level:
                cap_level = level
    if cap_level is None:
        return raw_verdict, triggered
    cap_verdict = m["watch"] if cap_level == 1 else m["do_not_buy"]
    if VERDICT_RANK.get(raw_verdict, 3) > cap_level:
        return cap_verdict, triggered
    return raw_verdict, triggered


PROFILE_DESCRIPTIONS = {
    "balanced":       "Analyst default — quality + momentum hybrid",
    "deep_value":     "Balance sheet + moats above all (F8=22)",
    "quality":        "Margin + beats + institutions (F2=18, F4=14)",
    "garp":           "Growth at reasonable price (F1=18, F6=14)",
    "momentum":       "RS + sector macro (F7=22, F10=10)",
    "earnings_swing": "Pre-earnings swing — beats + distance (F4=18, F5=15)",
}

SECTOR_DESCRIPTIONS = {
    "auto":       "Auto-detect from yfinance",
    "none":       "No sector overlay",
    "tech":       "Tech/SaaS — margin + RS emphasized",
    "financial":  "Banks/insurance — balance sheet dominates (F8 +8)",
    "healthcare": "Pharma/medical — catalysts (F9 +5)",
    "energy":     "Energy/commodity — sector macro + balance sheet",
    "consumer":   "Retail/brands — revenue + margin",
    "industrial": "Industrial/defense — balance sheet + catalysts",
    "reit":       "REITs — balance sheet + rate-sensitive macro",
    "utility":    "Utilities — balance sheet dominates, low growth OK",
}


def profile_menu():
    print("\nSelect weighting profile:")
    names = list(WEIGHT_PROFILES.keys())
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<16s} — {PROFILE_DESCRIPTIONS[n]}")
    while True:
        ans = input(f"Choice [1-{len(names)}, default 1]: ").strip()
        if not ans:
            return names[0]
        if ans.isdigit() and 1 <= int(ans) <= len(names):
            return names[int(ans) - 1]
        if ans in names:
            return ans


def sector_menu():
    print("\nSelect sector overlay:")
    names = list(SECTOR_DESCRIPTIONS.keys())
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<12s} — {SECTOR_DESCRIPTIONS[n]}")
    while True:
        ans = input(f"Choice [1-{len(names)}, default 1=auto]: ").strip()
        if not ans:
            return "auto"
        if ans.isdigit() and 1 <= int(ans) <= len(names):
            return names[int(ans) - 1]
        if ans in names:
            return ans


def scoring_menu():
    print("\nSelect scoring mode:")
    names = list(SCORING_MODES.keys())
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<10s} — {SCORING_MODES[n]['desc']}")
    while True:
        ans = input(f"Choice [1-{len(names)}, default 2=normal]: ").strip()
        if not ans:
            return "normal"
        if ans.isdigit() and 1 <= int(ans) <= len(names):
            return names[int(ans) - 1]
        if ans in names:
            return ans


def evaluate(ticker,
             profile="balanced",
             sector="auto",
             scoring_mode="normal",
             lang="en",
             apply_bonuses=True,
             verbose=True,
             engine="auto"):
    """
    Evaluate ticker against value/quality framework.

    Returns JSON-serializable dict for web/API use.
    """
    ticker = ticker.upper().strip()
    m = MSGS[lang]

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    if sector == "auto":
        sector_key = detect_sector(info)
    elif sector in ("none", None, ""):
        sector_key = None
    else:
        sector_key = sector

    weights = resolve_weights(profile, sector_key)
    neutral_credit = SCORING_MODES[scoring_mode]["neutral_credit"]

    if verbose:
        print(f"\n{'='*70}\n  {m['evaluating']}: ${ticker}   "
              f"[profile={profile} | sector={sector_key or 'none'} | "
              f"mode={scoring_mode}]\n{'='*70}")
        print(f"  Weights: " +
              ", ".join(f"{k}={v}" for k, v in weights.items()))

    filters = [
        ("F1",  m["F1"],  lambda: v1_revenue_momentum(ticker)),
        ("F2",  m["F2"],  lambda: v2_gross_margin(ticker)),
        ("F3",  m["F3"],  lambda: v3_institutional_support(info)),
        ("F4",  m["F4"],  lambda: v4_earnings_surprises(ticker)),
        ("F5",  m["F5"],  lambda: v5_earnings_distance(info, ticker)),
        ("F6",  m["F6"],  lambda: v6_analyst_consensus(info)),
        ("F7",  m["F7"],  lambda: v7_relative_strength(ticker)),
        ("F8",  m["F8"],  lambda: v8_balance_sheet(ticker, info)),
        ("F9",  m["F9"],  lambda: v9_recent_catalyst(ticker, 30, engine)),
        ("F10", m["F10"], lambda: v10_sector_macro(sector_key)),
    ]

    results = []
    total_score = 0.0
    pass_count = 0
    fail_ids = set()
    pass_ids = set()

    for fid, name, fn in filters:
        if verbose:
            print(f"\n--- {name} (weight {weights[fid]}) ---")
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = None, f"error: {e}"

        weight = weights[fid]
        if ok is True:
            status = m["pass"]
            earned = float(weight)
            pass_count += 1
            pass_ids.add(fid)
        elif ok is False:
            status = m["fail"]
            earned = 0.0
            fail_ids.add(fid)
        else:
            status = m["neu"]
            earned = weight * neutral_credit

        total_score += earned
        if verbose:
            print(f"  {status}  +{earned:.1f}/{weight}  {note}")
        results.append({
            "id": fid, "name": name, "status": status,
            "note": note, "earned": round(earned, 2),
            "weight": weight, "ok": ok,
        })

    # Bonuses
    bonus_applied = []
    if apply_bonuses:
        for rule in BONUS_RULES:
            if all(f in pass_ids for f in rule["requires"]):
                total_score += rule["points"]
                bonus_applied.append(rule)
        total_score = min(total_score, 100.0)

    raw_verdict = verdict(total_score, lang)
    final_verdict, triggered = apply_vetoes(
        raw_verdict, fail_ids, profile, lang)

    if verbose:
        print(f"\n{'='*70}\n  {m['summary']} — ${ticker}   "
              f"[profile={profile} | sector={sector_key or 'none'}]\n"
              f"{'='*70}")
        print(f"  {'Filter':<42s} {'Status':<14s} {'Score':>10s}")
        print("  " + "-" * 68)
        for r in results:
            print(f"  {r['name']:<42s} {r['status']:<14s} "
                  f"{r['earned']:>5.1f}/{r['weight']:<3d}")
        print("  " + "-" * 68)
        if bonus_applied:
            for b in bonus_applied:
                print(f"  + Bonus: {b['name']} (+{b['points']})")
        print(f"  {m['total']}: {total_score:.1f}/100   "
              f"(pass count: {pass_count}/10)")
        if triggered:
            print(f"  ⚠  Veto triggered: {', '.join(triggered)}  "
                  f"(raw verdict was {raw_verdict})")
        print(f"  {m['verdict']}:    {final_verdict}")
        print("=" * 70)

    return {
        "ticker": ticker,
        "profile": profile,
        "sector": sector_key,
        "scoring_mode": scoring_mode,
        "weights_used": weights,
        "score": round(total_score, 1),
        "max_score": 100,
        "pass_count": pass_count,
        "verdict": final_verdict,
        "raw_verdict": raw_verdict,
        "vetoes_triggered": triggered,
        "bonuses_applied": [{"name": b["name"], "points": b["points"]}
                            for b in bonus_applied],
        "results": results,
    }


def evaluate_all_scenarios(ticker, **kwargs):
    """Run all profiles (auto-sector) — for web comparison view."""
    out = []
    for p in WEIGHT_PROFILES.keys():
        out.append(evaluate(ticker, profile=p, sector="auto",
                            verbose=False, **kwargs))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--profile",
                    choices=list(WEIGHT_PROFILES.keys()) + ["ask"],
                    default="ask")
    ap.add_argument("--sector",
                    choices=list(SECTOR_DESCRIPTIONS.keys()) + ["ask"],
                    default="auto")
    ap.add_argument("--scoring",
                    choices=list(SCORING_MODES.keys()) + ["ask"],
                    default="normal")
    ap.add_argument("--lang", choices=["uz", "en"], default="en")
    ap.add_argument("--no-bonuses", action="store_true")
    ap.add_argument("--compare-all", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tk = args.ticker or input("Ticker: ").strip()

    profile = args.profile if args.profile != "ask" else profile_menu()
    sector  = args.sector  if args.sector  != "ask" else sector_menu()
    scoring = args.scoring if args.scoring != "ask" else scoring_menu()
    bonuses = not args.no_bonuses

    if args.compare_all:
        print(f"\n{'='*70}\n  All-profile comparison for ${tk.upper()}"
              f"  (sector={sector}, mode={scoring})\n{'='*70}")
        summary = []
        for pname in WEIGHT_PROFILES.keys():
            print(f"\n\n>>>>> Profile: {pname} <<<<<")
            r = evaluate(tk, profile=pname, sector=sector,
                         scoring_mode=scoring, lang=args.lang,
                         apply_bonuses=bonuses)
            summary.append(r)
        print(f"\n{'='*70}\n  CROSS-PROFILE COMPARISON — ${tk.upper()}\n"
              f"{'='*70}")
        print(f"  {'Profile':<16s} {'Sector':<12s} {'Score':>7s}   Verdict")
        print("  " + "-" * 66)
        for r in summary:
            print(f"  {r['profile']:<16s} {str(r['sector'] or '-'):<12s} "
                  f"{r['score']:>5.1f}/100   {r['verdict']}")
        print("=" * 70)
    elif args.json:
        result = evaluate(tk, profile=profile, sector=sector,
                          scoring_mode=scoring, lang=args.lang,
                          apply_bonuses=bonuses, verbose=False)
        print(json.dumps(result, indent=2, default=str))
    else:
        evaluate(tk, profile=profile, sector=sector,
                 scoring_mode=scoring, lang=args.lang,
                 apply_bonuses=bonuses)
