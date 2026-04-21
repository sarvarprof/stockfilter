"""
Automated F10 — Winner profile (no human input).

Signals:
  1. Float < 30M               -> yfinance
  2. Market cap < $20M         -> yfinance
  3. Hard catalyst             -> scan news/filings for gov/defense/contract terms
  4. Clean share structure     -> SEC filings: no S-1/S-3/424B/RS in last 180d +
                                   shares outstanding growth < 25% YoY
  5. High short interest       -> yfinance shortPercentOfFloat > 15%
  6. Clear sector narrative    -> reuses HOT_SECTOR_MAP
  7. Social buzz               -> StockTwits public watchlist/message volume
  8. Volume spike (5-20x avg)  -> yfinance recent vol vs 30-day avg

Usage:
    python auto_f10.py TICKER
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

from fetch_news import yf_news, yahoo_rss, sec_filings
from penny_filter import HOT_SECTOR_MAP

UA = "Mozilla/5.0 (penny-screener research contact@example.com)"


def http_get(url, accept="application/json"):
    req = Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------- 1 & 2: Float / Mcap ----------

def float_check(info):
    flt = info.get("floatShares") or info.get("sharesOutstanding")
    if not flt:
        return None, "Float unknown"
    ok = flt < 30_000_000
    return ok, f"Float {flt/1e6:.1f}M {'<' if ok else '>='} 30M"


def mcap_check(info):
    m = info.get("marketCap")
    if not m:
        return None, "Mcap unknown"
    ok = m < 20_000_000
    return ok, f"Mcap ${m/1e6:.1f}M {'<' if ok else '>='} $20M"


# ---------- 3: Hard catalyst ----------

CATALYST_KW = [
    # Government / defense
    "department of defense", "dod contract", "pentagon", "army", "navy",
    "air force", "space force", "darpa", "nasa", "government contract",
    "federal contract", "awarded contract", "task order", "idiq",
    # FDA / health
    "fda approval", "fda clearance", "breakthrough designation",
    "phase 3", "phase iii", "pdufa",
    # Deals
    "acquisition", "merger", "strategic partnership", "letter of intent",
    "memorandum of understanding", "mou signed",
    # Sector themes
    "ai partnership", "ai deployment", "quantum", "cybersecurity contract",
]


def catalyst_check(ticker, days=90):
    news = yf_news(ticker, days=days) or yahoo_rss(ticker, days=days)
    filings = sec_filings(ticker, days=days)
    blob = " ".join(n["title"] for n in news).lower()
    blob += " " + " ".join(f"{f['form']} {f['desc']}" for f in filings).lower()
    hits = [kw for kw in CATALYST_KW if kw in blob]
    ok = len(hits) > 0
    return ok, (f"Catalyst hits: {', '.join(hits[:3])}"
                if hits else "No catalyst keywords found")


# ---------- 4: Clean share structure ----------

def clean_structure(ticker, info, days=180):
    """No recent dilution filings + modest share count growth."""
    filings = sec_filings(ticker, days=days)
    dilution_forms = {"S-1", "S-3", "424B5", "424B4", "424B2", "FWP", "F-1", "F-3"}
    bad = [f for f in filings if f["form"].upper() in dilution_forms]

    # RS detection
    rs_hits = [f for f in filings
               if re.search(r"reverse|consolidation|split",
                            f["desc"].lower())]

    # Share count growth YoY (if available)
    share_growth = None
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            for idx in bs.index:
                if "share" in str(idx).lower() and "issued" in str(idx).lower():
                    vals = bs.loc[idx].dropna()
                    if len(vals) >= 2:
                        share_growth = (float(vals.iloc[0]) /
                                        float(vals.iloc[1]) - 1)
                    break
    except Exception:
        pass

    notes = []
    ok = True
    if bad:
        ok = False
        notes.append(f"{len(bad)} dilution filing(s) in {days}d "
                     f"({bad[0]['form']})")
    if rs_hits:
        ok = False
        notes.append(f"RS signal in filings")
    if share_growth is not None:
        notes.append(f"share count YoY: {share_growth*100:+.1f}%")
        if share_growth > 0.25:
            ok = False

    if not notes:
        notes.append(f"no dilution events in {days}d")
    return ok, "; ".join(notes)


# ---------- 5: Short interest ----------

def short_interest(info):
    sp = info.get("shortPercentOfFloat")
    if sp is None:
        sr = info.get("shortRatio")
        if sr is None:
            return None, "Short data unavailable"
        ok = sr >= 5
        return ok, f"Short ratio (days to cover) = {sr:.2f}"
    pct = sp * 100 if sp < 1 else sp
    ok = pct >= 15
    return ok, f"Short % of float = {pct:.1f}% ({'>= 15%' if ok else '< 15%'})"


# ---------- 6: Sector narrative ----------

def sector_narrative(info):
    sector = (info.get("sector") or "").strip()
    industry = (info.get("industry") or "").strip()
    summary = (info.get("longBusinessSummary") or "").strip()
    haystack = f" {sector} {industry} {summary} ".lower()
    matched = [lbl for lbl, kws in HOT_SECTOR_MAP.items()
               if any(kw in haystack for kw in kws)]
    ok = len(matched) > 0
    return ok, (f"Narrative: {', '.join(matched)}"
                if matched else f"No hot narrative ({sector})")


# ---------- 7: Social buzz (StockTwits) ----------

def social_buzz(ticker):
    """
    StockTwits public endpoint: no key required.
    Flags if >= 30 messages in last 24h (buzz threshold).
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        data = json.loads(http_get(url))
    except Exception as e:
        return None, f"StockTwits unavailable: {e}"

    msgs = data.get("messages", [])
    if not msgs:
        return False, "No StockTwits messages"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = 0
    bull = bear = 0
    for m in msgs:
        try:
            dt = datetime.fromisoformat(
                m["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= cutoff:
            recent += 1
            s = (m.get("entities", {}) or {}).get("sentiment")
            if s:
                lbl = (s.get("basic") or "").lower()
                if lbl == "bullish":
                    bull += 1
                elif lbl == "bearish":
                    bear += 1

    ok = recent >= 30 and bull >= bear
    return ok, (f"StockTwits 24h: {recent} msgs, "
                f"{bull} bull / {bear} bear")


# ---------- 8: Volume spike ----------

def volume_spike(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
    except Exception as e:
        return None, f"History fetch error: {e}"
    if hist is None or hist.empty or len(hist) < 30:
        return None, "Insufficient history"

    vols = hist["Volume"].dropna()
    avg30 = vols.iloc[-30:-1].mean()  # prior 30 days excluding today
    latest = vols.iloc[-1]
    if avg30 <= 0:
        return None, "No volume baseline"
    ratio = latest / avg30
    ok = 5 <= ratio <= 20
    return ok, (f"Today vol {latest/1e6:.2f}M vs 30d avg "
                f"{avg30/1e6:.2f}M = {ratio:.1f}x")


# ---------- Runner ----------

def evaluate_f10(ticker):
    ticker = ticker.upper().strip()
    print(f"\n=== Auto-F10 Winner Profile for ${ticker} ===\n")

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    checks = [
        ("Float < 30M",            lambda: float_check(info)),
        ("Mcap < $20M",            lambda: mcap_check(info)),
        ("Hard catalyst",          lambda: catalyst_check(ticker)),
        ("Clean share structure",  lambda: clean_structure(ticker, info)),
        ("High short interest",    lambda: short_interest(info)),
        ("Hot sector narrative",   lambda: sector_narrative(info)),
        ("Social buzz (24h)",      lambda: social_buzz(ticker)),
        ("Volume 5-20x avg",       lambda: volume_spike(ticker)),
    ]

    score = total = 0
    for name, fn in checks:
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = None, f"error: {e}"
        if ok is True:
            tag = "✅"
            score += 1
            total += 1
        elif ok is False:
            tag = "❌"
            total += 1
        else:
            tag = "🟡"
        print(f"  {tag} {name:26s}  {note}")

    pct = score / total if total else 0
    verdict = "PASS" if pct >= 0.60 else "FAIL"
    print("\n" + "=" * 60)
    print(f"F10 winner-profile score: {score}/{total} ({pct*100:.0f}%)")
    print(f"Verdict: {verdict}")
    print("=" * 60)
    return score, total, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    args = ap.parse_args()
    evaluate_f10(args.ticker)
