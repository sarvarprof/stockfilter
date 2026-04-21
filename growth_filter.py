"""
Growth Stock 10-Filter Framework Evaluator — FULLY AUTOMATED, FREE.

Reuses helpers from penny_filter.py. No paid APIs.
Data sources: yfinance, SEC EDGAR, Yahoo RSS, StockTwits, Reddit JSON.

Usage:
    python growth_filter.py TICKER
    python growth_filter.py TICKER --engine keyword|finbert
    python growth_filter.py TICKER --lang uz|en
"""

import sys
import re
import json
import argparse
from datetime import datetime, timedelta, timezone

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

from penny_filter import (
    http_get, sec_filings, get_news,
    DILUTION_FORMS, OFFERING_PATTERNS, match_any,
    _score_keyword, _score_finbert, _load_finbert,
)


# ============================================================
# i18n (Uzbek + English)
# ============================================================

MSGS = {
    "uz": {
        "pass": "🟢 PASS",
        "fail": "🔴 FAIL",
        "neu":  "🟡 NEUTRAL",
        "evaluating": "Baholanmoqda",
        "summary": "YAKUNIY NATIJA",
        "total": "JAMI BALL",
        "verdict": "XULOSA",
        "strong_buy": "🟢 STRONG BUY — Kuchli sotib olish signali",
        "buy":        "🟢 BUY — Sotib olish signali",
        "watch":      "🟡 WATCH — Kuzatish",
        "hard_pass":  "🔴 HARD PASS — Chetlab o'tish",
        "F1":  "F1  Sales/Mcap (≥10%)",
        "F2":  "F2  Institutsional egalik (>10%)",
        "F3":  "F3  Offering yo'q (3 oy)",
        "F4":  "F4  5+ ijobiy yangilik",
        "F5":  "F5  Haftalik 20-30%+ harakat",
        "F6":  "F6  Likvidlik",
        "F7":  "F7  Doimiy daromad o'sishi",
        "F8":  "F8  Analyst price target",
        "F9":  "F9  CEO tajribasi/faolligi",
        "F10": "F10 Ijtimoiy sentiment",
    },
    "en": {
        "pass": "🟢 PASS",
        "fail": "🔴 FAIL",
        "neu":  "🟡 NEUTRAL",
        "evaluating": "Evaluating",
        "summary": "SUMMARY",
        "total": "TOTAL PASS",
        "verdict": "VERDICT",
        "strong_buy": "🟢 STRONG BUY",
        "buy":        "🟢 BUY",
        "watch":      "🟡 WATCH",
        "hard_pass":  "🔴 HARD PASS",
        "F1":  "F1  Sales vs Market Cap (>=10%)",
        "F2":  "F2  Institutional ownership (>10%)",
        "F3":  "F3  No offering (3 months)",
        "F4":  "F4  5+ positive headlines",
        "F5":  "F5  Weekly 20-30%+ moves",
        "F6":  "F6  Liquidity",
        "F7":  "F7  Consistent sales growth",
        "F8":  "F8  Analyst price targets",
        "F9":  "F9  CEO experience/activity",
        "F10": "F10 Social sentiment",
    },
}


# ============================================================
# FILTERS F1–F10
# ============================================================

def g1_sales_mcap(info):
    """F1: Annual sales >= 10% of market cap."""
    sales = info.get("totalRevenue")
    mcap = info.get("marketCap")
    if not sales or not mcap:
        return None, "Sales/Mcap data unavailable"
    ratio = sales / mcap
    ok = ratio >= 0.10
    return ok, (f"Sales ${sales/1e6:.1f}M / Mcap ${mcap/1e6:.1f}M "
                f"= {ratio*100:.1f}% ({'>= 10%' if ok else '< 10%'})")


def g2_institutional(info):
    """F2: Institutional ownership > 10%."""
    inst = info.get("heldPercentInstitutions")
    if inst is None:
        return None, "Institutional ownership unknown"
    pct = inst * 100 if inst < 1.5 else inst
    ok = pct > 10
    return ok, f"Institutional = {pct:.1f}% ({'> 10%' if ok else '<= 10%'})"


def g3_no_offering(ticker, days=90):
    """F3: No offering in last 3 months."""
    filings = sec_filings(ticker, days=days)
    bad = [f for f in filings if f["form"].upper() in DILUTION_FORMS]
    for f in filings:
        if match_any(f"{f['form']} {f.get('desc','')}", OFFERING_PATTERNS):
            bad.append(f)
    if bad:
        return False, f"{len(bad)} offering filing(s) in {days}d ({bad[0]['form']})"
    return True, f"No offering in last {days}d"


def g4_positive_news(ticker, days=90, engine="auto"):
    """F4: 5+ positive headlines in recent news."""
    news = get_news(ticker, days=days)
    if not news:
        return None, "No headlines"

    chosen = engine
    if engine == "auto":
        chosen = "finbert" if _load_finbert() else "keyword"

    pos = 0
    for n in news:
        if chosen == "finbert":
            s = _score_finbert(n["title"]) or _score_keyword(n["title"])
        else:
            s = _score_keyword(n["title"])
        if s == "POS":
            pos += 1

    if pos >= 5:
        return True, f"{pos} positive headlines ({chosen})"
    if 2 <= pos <= 4:
        return None, f"{pos} positive headlines — mixed ({chosen})"
    return False, f"Only {pos} positive headlines ({chosen})"


def g5_price_moves(ticker):
    """F5: 2+ weekly 20-30%+ price moves in last 2 years."""
    try:
        hist = yf.Ticker(ticker).history(period="2y", interval="1wk")
    except Exception as e:
        return None, f"history err: {e}"
    if hist is None or hist.empty:
        return None, "No price history"
    closes = hist["Close"].dropna()
    count = 0
    for i in range(1, len(closes)):
        prev = closes.iloc[i - 1]
        cur = closes.iloc[i]
        if prev > 0:
            move = (cur - prev) / prev
            if move >= 0.20:  # 20%+ move (framework says 20-30%+)
                count += 1
    if count >= 2:
        return True, f"{count} weekly 20%+ moves in 2y"
    if count == 1:
        return None, "1 weekly 20%+ move (neutral)"
    return False, "No 20%+ weekly moves in 2y"


def g6_liquidity(info, ticker):
    """F6: Good liquidity — avg volume high + tight spread."""
    avg_vol = (info.get("averageVolume10days")
               or info.get("averageVolume"))
    bid = info.get("bid")
    ask = info.get("ask")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    notes = []
    score = 0
    total = 0

    if avg_vol is not None:
        total += 1
        notes.append(f"avgVol {avg_vol/1e6:.2f}M")
        if avg_vol >= 500_000:
            score += 1

    if bid and ask and price and price > 0:
        spread = (ask - bid) / price
        total += 1
        notes.append(f"spread {spread*100:.2f}%")
        if spread <= 0.01:  # < 1%
            score += 1
    else:
        # fallback: estimate from recent history
        try:
            h = yf.Ticker(ticker).history(period="1mo")
            if h is not None and not h.empty:
                # Use high-low % as rough spread proxy
                pass
        except Exception:
            pass

    if total == 0:
        return None, "Liquidity data unavailable"
    if score == total:
        return True, "; ".join(notes) + " -> good"
    if score >= 1:
        return None, "; ".join(notes) + " -> moderate"
    return False, "; ".join(notes) + " -> poor"


def g7_sales_growth(ticker):
    """F7: Consistent YoY/QoQ revenue growth."""
    t = yf.Ticker(ticker)

    # Annual (YoY)
    yoy_ok = None
    yoy_note = ""
    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            for idx in fin.index:
                if "total revenue" in str(idx).lower():
                    vals = fin.loc[idx].dropna().tolist()
                    # financials are newest-first
                    if len(vals) >= 2:
                        growths = []
                        for i in range(len(vals) - 1):
                            if vals[i + 1] > 0:
                                growths.append(
                                    (vals[i] - vals[i + 1]) / vals[i + 1])
                        if growths:
                            all_pos = all(g > 0 for g in growths[:2])
                            yoy_ok = all_pos
                            yoy_note = ("YoY: " +
                                        ", ".join(f"{g*100:+.0f}%"
                                                  for g in growths[:2]))
                    break
    except Exception as e:
        yoy_note = f"YoY err: {e}"

    # Quarterly (QoQ)
    qoq_ok = None
    qoq_note = ""
    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            for idx in qf.index:
                if "total revenue" in str(idx).lower():
                    vals = qf.loc[idx].dropna().tolist()
                    if len(vals) >= 3:
                        growths = []
                        for i in range(len(vals) - 1):
                            if vals[i + 1] > 0:
                                growths.append(
                                    (vals[i] - vals[i + 1]) / vals[i + 1])
                        if growths:
                            pos = sum(1 for g in growths[:3] if g > 0)
                            qoq_ok = pos >= 2
                            qoq_note = ("QoQ: " +
                                        ", ".join(f"{g*100:+.0f}%"
                                                  for g in growths[:3]))
                    break
    except Exception as e:
        qoq_note = f"QoQ err: {e}"

    if yoy_ok is None and qoq_ok is None:
        return None, "Revenue data unavailable"

    parts = [p for p in [yoy_note, qoq_note] if p]
    if yoy_ok or qoq_ok:
        mixed = (yoy_ok is False) or (qoq_ok is False)
        if mixed:
            return None, "; ".join(parts) + " -> mixed"
        return True, "; ".join(parts) + " -> consistent growth"
    return False, "; ".join(parts) + " -> declining"


def g8_analyst_target(info):
    """F8: Analyst consensus target > current price."""
    target = (info.get("targetMeanPrice")
              or info.get("targetMedianPrice"))
    price = (info.get("currentPrice")
             or info.get("regularMarketPrice"))
    n_analysts = info.get("numberOfAnalystOpinions") or 0
    if not target or not price:
        return None, "Analyst target unavailable"
    upside = (target - price) / price
    note = (f"Target ${target:.2f} vs price ${price:.2f} "
            f"= {upside*100:+.1f}% ({n_analysts} analysts)")
    if upside > 0.05:
        return True, note
    if abs(upside) <= 0.05:
        return None, note + " (±5%)"
    return False, note


def g9_ceo_profile(info, ticker):
    """F9: CEO experience + social activity."""
    officers = info.get("companyOfficers") or []
    ceo = None
    for o in officers:
        title = (o.get("title") or "").lower()
        if "ceo" in title or "chief executive" in title:
            ceo = o
            break
    if not ceo:
        return None, "CEO info unavailable"

    name = ceo.get("name", "?")
    age = ceo.get("age")
    pay = ceo.get("totalPay")

    experienced = age is not None and age >= 40
    parts = [f"CEO: {name}"]
    if age:
        parts.append(f"age {age}")
    if pay:
        parts.append(f"pay ${pay/1e6:.1f}M")

    # Activity check — mentions of CEO name in recent news
    news = get_news(ticker, days=180)
    first_name = name.split()[0] if name else ""
    last_name = name.split()[-1] if name else ""
    mentions = sum(1 for n in news
                   if last_name and last_name.lower() in n["title"].lower())
    parts.append(f"{mentions} news mentions (180d)")

    # Rough activity proxy: >=3 mentions = visible/active
    active = mentions >= 3

    if experienced and active:
        return True, "; ".join(parts) + " -> experienced + active"
    if experienced or active:
        return None, "; ".join(parts) + " -> partial"
    return False, "; ".join(parts) + " -> low profile"


def _stocktwits_sentiment(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        data = json.loads(http_get(url))
    except Exception as e:
        return None, 0, 0, 0
    msgs = data.get("messages", [])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    total = bull = bear = 0
    for m in msgs:
        try:
            dt = datetime.fromisoformat(
                m["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= cutoff:
            total += 1
            s = (m.get("entities", {}) or {}).get("sentiment")
            if s:
                lbl = (s.get("basic") or "").lower()
                if lbl == "bullish":
                    bull += 1
                elif lbl == "bearish":
                    bear += 1
    return (total, bull, bear)


def _reddit_mentions(ticker):
    """Reddit JSON search — no auth needed. Returns (posts_count, avg_score)."""
    url = (f"https://www.reddit.com/search.json?q=%24{ticker}"
           f"&sort=new&t=week&limit=25")
    try:
        data = json.loads(http_get(url, accept="application/json"))
    except Exception:
        return 0, 0.0
    posts = data.get("data", {}).get("children", [])
    if not posts:
        return 0, 0.0
    scores = [p["data"].get("score", 0) for p in posts]
    return len(posts), (sum(scores) / len(scores) if scores else 0.0)


def g10_social_sentiment(ticker):
    """F10: Social sentiment across StockTwits + Reddit."""
    st = _stocktwits_sentiment(ticker)
    rd_count, rd_avg_score = _reddit_mentions(ticker)

    total, bull, bear = (st if isinstance(st, tuple) and len(st) == 3
                         else (0, 0, 0))

    parts = [f"ST72h: {total} msgs ({bull}🐂/{bear}🐻)",
             f"Reddit 7d: {rd_count} posts (avg score {rd_avg_score:.1f})"]

    # Scoring:
    # PASS: active discussion AND bullish lean
    # NEUTRAL: some chatter
    # FAIL: silent or bearish
    active = (total >= 15) or (rd_count >= 5)
    bullish = (bull >= bear) and (rd_avg_score >= 0)

    if active and bullish:
        return True, "; ".join(parts) + " -> active+bullish"
    if active or (total + rd_count) > 0:
        return None, "; ".join(parts) + " -> mixed"
    return False, "; ".join(parts) + " -> silent"


# ============================================================
# WEIGHTED SCORING — multiple profiles, user-selectable per query
# ============================================================
# Each profile distributes 100 points across F1..F10 to match a
# trading style. `balanced` is the analyst default.

WEIGHT_PROFILES = {
    # Balanced analyst view (default)
    # Tier1=45 Tier2=30 Tier3=15 Tier4=10
    "balanced": {
        "F1": 13, "F2": 11, "F3": 14, "F4": 7,  "F5": 8,
        "F6": 9,  "F7": 18, "F8": 10, "F9": 6,  "F10": 4,
    },
    # Long-term growth investor: fundamentals above all
    "long_term": {
        "F1": 16, "F2": 14, "F3": 12, "F4": 4,  "F5": 3,
        "F6": 7,  "F7": 24, "F8": 12, "F9": 6,  "F10": 2,
    },
    # Swing trader: momentum + catalysts, weeks-to-months
    "swing": {
        "F1": 7,  "F2": 8,  "F3": 15, "F4": 12, "F5": 18,
        "F6": 12, "F7": 8,  "F8": 7,  "F9": 3,  "F10": 10,
    },
    # Risk-averse / capital preservation: dilution & liquidity vetoes
    "risk_averse": {
        "F1": 12, "F2": 14, "F3": 20, "F4": 5,  "F5": 4,
        "F6": 15, "F7": 15, "F8": 10, "F9": 3,  "F10": 2,
    },
    # Aggressive small-cap hunter: catch explosive movers
    "aggressive": {
        "F1": 6,  "F2": 6,  "F3": 12, "F4": 11, "F5": 20,
        "F6": 8,  "F7": 10, "F8": 6,  "F9": 6,  "F10": 15,
    },
}

for _pname, _pw in WEIGHT_PROFILES.items():
    assert sum(_pw.values()) == 100, f"Profile {_pname} != 100"

# Hard-fail (veto) rules — per profile.
# If any listed filter FAILS, cap the verdict regardless of score.
# Used to reflect "some signals should be vetoes, not just -points".
VETO_RULES = {
    "balanced":    {},                                  # no vetoes
    "long_term":   {"F7": "WATCH"},                    # no growth -> WATCH max
    "swing":       {},
    "risk_averse": {"F3": "WATCH", "F6": "WATCH"},     # dilution/illiquid -> WATCH max
    "aggressive":  {},
}


# ============================================================
# OPTION C — Sector-adaptive overlays
# ============================================================
# Each sector overlay is ADDED to the chosen base profile, then the
# result is renormalized to exactly 100. Deltas reflect what matters
# more/less for that sector vs. a generic growth stock.

SECTOR_OVERLAYS = {
    # Biotech/Pharma — pre-FDA cos often have no revenue; catalysts = everything
    "biotech": {
        "F1": -10, "F2": +2, "F3": +2, "F4": +4,  "F5": +2,
        "F6": -2,  "F7": -8, "F8": +6, "F9": +2,  "F10": +2,
    },
    # Tech / SaaS — growth above all; analyst coverage deep
    "tech": {
        "F1": -1, "F2": +1, "F3": 0,  "F4": 0,  "F5": 0,
        "F6": 0,  "F7": +5, "F8": +1, "F9": -2, "F10": -4,
    },
    # Energy / Commodity — cash flow > growth; institutional flows matter
    "energy": {
        "F1": +2, "F2": +4, "F3": -2, "F4": -2, "F5": +1,
        "F6": +2, "F7": -2, "F8": 0,  "F9": -2, "F10": -1,
    },
    # Financials / Fintech — regulatory cleanliness + institutional anchor
    "financial": {
        "F1": 0,  "F2": +5, "F3": +3, "F4": 0,  "F5": -2,
        "F6": +2, "F7": +1, "F8": +1, "F9": -4, "F10": -6,
    },
    # Consumer / Retail — sales growth decisive, liquidity for retail buzz cycles
    "consumer": {
        "F1": +2, "F2": 0,  "F3": 0,  "F4": +1, "F5": 0,
        "F6": +1, "F7": +3, "F8": 0,  "F9": -3, "F10": -4,
    },
    # Industrial / Defense — contracts (catalyst in F4), steady growth
    "industrial": {
        "F1": +2, "F2": +2, "F3": 0,  "F4": +3, "F5": -2,
        "F6": 0,  "F7": +2, "F8": +1, "F9": -4, "F10": -4,
    },
    # Pre-revenue / Early-stage — F1 & F7 are meaningless, narrative rules
    "pre_revenue": {
        "F1": -13, "F2": +3, "F3": +5, "F4": +3, "F5": +3,
        "F6": 0,   "F7": -18,"F8": +3, "F9": +6, "F10": +8,
    },
}


def _renormalize(weights):
    """Clamp to >=0, rescale to sum=100 exactly (rounded)."""
    w = {k: max(0, v) for k, v in weights.items()}
    s = sum(w.values())
    if s == 0:
        return {k: 10 for k in w}  # fallback equal-weight
    # Scale proportionally
    scaled = {k: v * 100.0 / s for k, v in w.items()}
    # Round and fix drift so total = 100
    rounded = {k: round(v) for k, v in scaled.items()}
    diff = 100 - sum(rounded.values())
    if diff != 0:
        # Apply drift to the key with largest fractional remainder
        fracs = sorted(scaled.items(),
                       key=lambda kv: (kv[1] - int(kv[1])),
                       reverse=(diff > 0))
        for k, _ in fracs[:abs(diff)]:
            rounded[k] += 1 if diff > 0 else -1
    return rounded


def resolve_weights(profile="balanced", sector_key=None):
    """Return final F1..F10 weights for given base profile + optional sector overlay."""
    base = dict(WEIGHT_PROFILES[profile])
    if sector_key and sector_key in SECTOR_OVERLAYS:
        overlay = SECTOR_OVERLAYS[sector_key]
        merged = {k: base[k] + overlay.get(k, 0) for k in base}
        return _renormalize(merged)
    return base


# Map yfinance sector/industry strings -> sector overlay key
SECTOR_KEYWORDS = {
    "biotech":     ["biotech", "biotechnology", "drug manufacturers",
                    "pharmaceutical", "life sciences", "therapeutic"],
    "tech":        ["technology", "software", "semiconductor",
                    "information technology", "internet", "saas",
                    "communication services"],
    "energy":      ["energy", "oil", "gas", "utilities", "pipeline",
                    "uranium", "solar", "renewable"],
    "financial":   ["financial", "bank", "insurance", "capital markets",
                    "asset management", "fintech"],
    "consumer":    ["consumer", "retail", "apparel", "restaurant",
                    "leisure", "travel", "media"],
    "industrial":  ["industrial", "aerospace", "defense",
                    "machinery", "construction", "transportation"],
}


def detect_sector(info):
    """Auto-detect sector overlay key from yfinance info. None if no match."""
    haystack = " ".join([
        (info.get("sector") or ""),
        (info.get("industry") or ""),
        (info.get("industryKey") or ""),
        (info.get("sectorKey") or ""),
    ]).lower()
    if not haystack.strip():
        return None

    # Pre-revenue override: no meaningful revenue -> pre_revenue overlay
    rev = info.get("totalRevenue") or 0
    mcap = info.get("marketCap") or 0
    if mcap and rev / max(mcap, 1) < 0.02:  # < 2% sales/mcap = effectively pre-rev
        # Still check if it's a biotech first — biotech overlay already handles this
        for kw in SECTOR_KEYWORDS["biotech"]:
            if kw in haystack:
                return "biotech"
        return "pre_revenue"

    for key, kws in SECTOR_KEYWORDS.items():
        if any(kw in haystack for kw in kws):
            return key
    return None


# ============================================================
# OPTION D — Scoring modes (NEUTRAL credit) + bonus rules
# ============================================================
# SCORING_MODES controls how NEUTRAL results are credited.
# Bonuses add flat points when a cluster of signals all PASS.

SCORING_MODES = {
    "strict":   {"neutral_credit": 0.30,
                 "desc": "NEUTRAL = 30% credit (penalize uncertainty)"},
    "normal":   {"neutral_credit": 0.50,
                 "desc": "NEUTRAL = 50% credit (default)"},
    "generous": {"neutral_credit": 0.70,
                 "desc": "NEUTRAL = 70% credit (give benefit of the doubt)"},
}

# Bonus rules: if all listed filters PASS, add `points` to total score.
# (Applied AFTER weighted scoring, capped so total cannot exceed 100.)
BONUS_RULES = [
    {"name": "Tier-1 sweep",  "requires": ["F1", "F3", "F7"], "points": 5},
    {"name": "Smart-money sweep", "requires": ["F2", "F8"],   "points": 3},
    {"name": "Momentum sweep", "requires": ["F4", "F5"],      "points": 2},
]


PROFILE_DESCRIPTIONS = {
    "balanced":    "Analyst default — Tier1 fundamentals 45% / momentum 15%",
    "long_term":   "Buy & hold 1-3y — fundamentals dominate (F7=24, F1=16)",
    "swing":       "Weeks-to-months — momentum/catalysts (F5=18, F4=12)",
    "risk_averse": "Capital preservation — dilution/liquidity vetoes (F3=20)",
    "aggressive":  "Explosive small-caps — social+moves (F5=20, F10=15)",
}

SECTOR_DESCRIPTIONS = {
    "auto":        "Auto-detect from yfinance sector/industry",
    "none":        "Disable sector overlay — use base profile only",
    "biotech":     "Pre-FDA biotech — revenue de-emphasized, catalysts up",
    "tech":        "Tech/SaaS — growth + analyst coverage emphasized",
    "energy":      "Energy/commodity — institutional flow + liquidity",
    "financial":   "Banks/fintech — regulatory + institutional anchor",
    "consumer":    "Retail/consumer — sales growth decisive",
    "industrial":  "Industrial/defense — contracts (F4) emphasized",
    "pre_revenue": "Early-stage — F1 & F7 zeroed, narrative + social up",
}


def profile_menu():
    """Interactive profile picker."""
    print("\nSelect weighting profile:")
    names = list(WEIGHT_PROFILES.keys())
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<12s} — {PROFILE_DESCRIPTIONS[n]}")
    while True:
        ans = input(f"Choice [1-{len(names)}, default 1]: ").strip()
        if not ans:
            return names[0]
        if ans.isdigit() and 1 <= int(ans) <= len(names):
            return names[int(ans) - 1]
        if ans in names:
            return ans


def sector_menu():
    """Interactive sector overlay picker."""
    print("\nSelect sector overlay (adjusts weights by sector):")
    names = list(SECTOR_DESCRIPTIONS.keys())  # auto, none, biotech, ...
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
    """Interactive scoring-mode picker."""
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


# ============================================================
# RUNNER
# ============================================================

def verdict(score, lang="en"):
    """Weighted score verdict (out of 100)."""
    m = MSGS[lang]
    if score >= 75:
        return m["strong_buy"]
    if score >= 60:
        return m["buy"]
    if score >= 45:
        return m["watch"]
    return m["hard_pass"]


VERDICT_RANK = {
    "🔴 HARD PASS": 0,
    "🔴 HARD PASS — Chetlab o'tish": 0,
    "🟡 WATCH": 1,
    "🟡 WATCH — Kuzatish": 1,
    "🟢 BUY": 2,
    "🟢 BUY — Sotib olish signali": 2,
    "🟢 STRONG BUY": 3,
    "🟢 STRONG BUY — Kuchli sotib olish signali": 3,
}


def apply_vetoes(verdict_str, fail_ids, profile, lang):
    """Cap verdict per profile's veto rules."""
    m = MSGS[lang]
    vetoes = VETO_RULES.get(profile, {})
    cap_level = None
    triggered = []
    for fid, cap in vetoes.items():
        if fid in fail_ids:
            triggered.append(f"{fid}->{cap}")
            level = 1 if cap == "WATCH" else 0
            if cap_level is None or level < cap_level:
                cap_level = level
    if cap_level is None:
        return verdict_str, triggered
    cap_verdict = m["watch"] if cap_level == 1 else m["hard_pass"]
    if VERDICT_RANK.get(verdict_str, 3) > cap_level:
        return cap_verdict, triggered
    return verdict_str, triggered


def evaluate(ticker,
             engine="auto",
             lang="en",
             profile="balanced",
             sector="auto",
             scoring_mode="normal",
             apply_bonuses=True,
             verbose=True):
    """
    Evaluate a ticker against the 10-filter framework.

    Args:
        ticker:         Stock symbol.
        engine:         F4 sentiment engine ('auto'|'keyword'|'finbert').
        lang:           'en' | 'uz' for output strings.
        profile:        Weighting profile key (see WEIGHT_PROFILES).
        sector:         'auto' (detect), 'none' (disable), or a SECTOR_OVERLAYS key.
        scoring_mode:   'strict' | 'normal' | 'generous' — NEUTRAL credit %.
        apply_bonuses:  If True, apply BONUS_RULES after scoring.
        verbose:        Print progress; set False for programmatic/web use.

    Returns dict (JSON-serializable except for raw `results` tuples).
    """
    ticker = ticker.upper().strip()
    m = MSGS[lang]

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        if verbose:
            print(f"[warn] info fetch: {e}")
        info = {}

    # Resolve sector overlay
    if sector == "auto":
        sector_key = detect_sector(info)
    elif sector in ("none", None, ""):
        sector_key = None
    else:
        sector_key = sector

    weights = resolve_weights(profile, sector_key)
    neutral_credit = SCORING_MODES[scoring_mode]["neutral_credit"]

    if verbose:
        header = (f"[profile: {profile} | sector: {sector_key or 'none'} | "
                  f"mode: {scoring_mode}]")
        print(f"\n{'='*70}\n  {m['evaluating']}: ${ticker}   "
              f"{header}\n{'='*70}")
        print(f"  Weights: " + ", ".join(f"{k}={v}"
                                         for k, v in weights.items()))

    filters = [
        ("F1",  m["F1"],  lambda: g1_sales_mcap(info)),
        ("F2",  m["F2"],  lambda: g2_institutional(info)),
        ("F3",  m["F3"],  lambda: g3_no_offering(ticker, 90)),
        ("F4",  m["F4"],  lambda: g4_positive_news(ticker, 90, engine)),
        ("F5",  m["F5"],  lambda: g5_price_moves(ticker)),
        ("F6",  m["F6"],  lambda: g6_liquidity(info, ticker)),
        ("F7",  m["F7"],  lambda: g7_sales_growth(ticker)),
        ("F8",  m["F8"],  lambda: g8_analyst_target(info)),
        ("F9",  m["F9"],  lambda: g9_ceo_profile(info, ticker)),
        ("F10", m["F10"], lambda: g10_social_sentiment(ticker)),
    ]

    results = []
    total_score = 0.0
    max_score = 0
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
        max_score += weight
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
            "note": note, "earned": round(earned, 2), "weight": weight,
            "ok": ok,
        })

    # Bonuses
    bonus_applied = []
    if apply_bonuses:
        for rule in BONUS_RULES:
            if all(fid in pass_ids for fid in rule["requires"]):
                total_score += rule["points"]
                bonus_applied.append(rule)
        # Cap at 100 (bonuses can't push beyond max)
        total_score = min(total_score, 100.0)

    raw_verdict = verdict(total_score, lang)
    final_verdict, triggered = apply_vetoes(
        raw_verdict, fail_ids, profile, lang)

    if verbose:
        print(f"\n{'='*70}\n  {m['summary']} — ${ticker}   "
              f"[profile: {profile} | sector: {sector_key or 'none'}]\n"
              f"{'='*70}")
        print(f"  {'Filter':<40s} {'Status':<12s} {'Score':>10s}")
        print("  " + "-" * 66)
        for r in results:
            print(f"  {r['name']:<40s} {r['status']:<12s} "
                  f"{r['earned']:>5.1f}/{r['weight']:<3d}")
        print("  " + "-" * 66)
        if bonus_applied:
            for b in bonus_applied:
                print(f"  + Bonus: {b['name']} (+{b['points']})")
        print(f"  {m['total']}: {total_score:.1f}/100   "
              f"(raw pass count: {pass_count}/10)")
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


def evaluate_all_scenarios(ticker, engine="auto", lang="en",
                           scoring_mode="normal"):
    """
    Run ALL profile/sector combinations for a ticker and return a list.
    Useful for web UI dropdowns / comparison tables.
    """
    results = []
    # First run auto-sector per profile (primary view)
    for profile in WEIGHT_PROFILES.keys():
        results.append(evaluate(ticker, engine=engine, lang=lang,
                                profile=profile, sector="auto",
                                scoring_mode=scoring_mode, verbose=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--engine", choices=["auto", "keyword", "finbert"],
                    default="auto",
                    help="F4 news sentiment engine")
    ap.add_argument("--lang", choices=["uz", "en"], default="en")
    ap.add_argument("--profile",
                    choices=list(WEIGHT_PROFILES.keys()) + ["ask"],
                    default="ask",
                    help="Weighting profile (default: ask interactively)")
    ap.add_argument("--sector",
                    choices=list(SECTOR_DESCRIPTIONS.keys()) + ["ask"],
                    default="auto",
                    help="Sector overlay (default: auto-detect)")
    ap.add_argument("--scoring",
                    choices=list(SCORING_MODES.keys()) + ["ask"],
                    default="normal",
                    help="NEUTRAL credit mode (default: normal = 50%)")
    ap.add_argument("--no-bonuses", action="store_true",
                    help="Disable bonus-signal rules")
    ap.add_argument("--compare-all", action="store_true",
                    help="Run all profiles (auto-sector) and compare")
    ap.add_argument("--json", action="store_true",
                    help="Emit single JSON result (for web/API use)")
    args = ap.parse_args()

    tk = args.ticker or input("Ticker: ").strip()

    # Resolve interactive options
    profile = args.profile if args.profile != "ask" else profile_menu()
    sector  = args.sector  if args.sector  != "ask" else sector_menu()
    scoring = args.scoring if args.scoring != "ask" else scoring_menu()
    bonuses = not args.no_bonuses

    if args.compare_all:
        print(f"\n{'='*70}\n  Comparing all profiles for ${tk.upper()}"
              f"  (sector={sector}, scoring={scoring})\n{'='*70}")
        summary = []
        for pname in WEIGHT_PROFILES.keys():
            print(f"\n\n>>>>> Profile: {pname} <<<<<")
            r = evaluate(tk, engine=args.engine, lang=args.lang,
                         profile=pname, sector=sector,
                         scoring_mode=scoring, apply_bonuses=bonuses)
            summary.append(r)
        print(f"\n{'='*70}\n  CROSS-PROFILE COMPARISON — ${tk.upper()}"
              f"\n{'='*70}")
        print(f"  {'Profile':<14s} {'Sector':<12s} {'Score':>7s}   Verdict")
        print("  " + "-" * 66)
        for r in summary:
            print(f"  {r['profile']:<14s} {str(r['sector'] or '-'):<12s} "
                  f"{r['score']:>5.1f}/100   {r['verdict']}")
        print("=" * 70)
    elif args.json:
        import json as _json
        result = evaluate(tk, engine=args.engine, lang=args.lang,
                          profile=profile, sector=sector,
                          scoring_mode=scoring, apply_bonuses=bonuses,
                          verbose=False)
        print(_json.dumps(result, indent=2, default=str))
    else:
        evaluate(tk, engine=args.engine, lang=args.lang,
                 profile=profile, sector=sector,
                 scoring_mode=scoring, apply_bonuses=bonuses)
