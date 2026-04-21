"""
Automated F9 — News sentiment + 2026 growth signal detection.

Tiers (auto-selects best available):
  1. Keyword scoring           (zero deps, always runs)
  2. FinBERT model             (pip install transformers torch)
  3. Claude API classification (set ANTHROPIC_API_KEY)

Usage:
    python auto_f9.py TICKER
    python auto_f9.py TICKER --engine keyword|finbert|claude
    python auto_f9.py TICKER --days 365
"""

import os
import sys
import re
import argparse
from datetime import datetime, timedelta, timezone

# Reuse news fetcher we already built
from fetch_news import yf_news, yahoo_rss, sec_filings


# ---------- Finance-specific keyword lexicon ----------

POSITIVE_KW = [
    # earnings / financials
    "beats", "beat estimates", "exceeds", "record revenue", "record quarter",
    "profitable", "profit", "strong growth", "strong results", "raised guidance",
    "upgraded", "upgrade", "buy rating", "outperform", "price target raised",
    # business
    "contract", "awarded", "partnership", "agreement", "acquisition",
    "fda approval", "cleared", "approved", "breakthrough", "patent",
    "launches", "launch", "expansion", "expands", "milestone",
    "order", "backlog", "selected", "wins", "deal",
    # momentum
    "surges", "soars", "rallies", "jumps", "climbs", "gains",
    "breakout", "all-time high",
]

NEGATIVE_KW = [
    # dilution / liquidity
    "offering", "dilution", "private placement", "at-the-market",
    "registered direct", "reverse split", "going concern",
    "bankruptcy", "restructuring", "delisting", "nasdaq notice",
    "deficiency", "non-compliance",
    # performance
    "misses", "missed estimates", "loss", "losses", "widens loss",
    "downgrade", "downgraded", "sell rating", "underperform",
    "guidance cut", "lowered guidance", "warning",
    # legal / regulatory
    "lawsuit", "sued", "investigation", "subpoena", "sec charges",
    "fraud", "class action", "fines",
    # operations
    "resigns", "resignation", "terminated", "layoffs",
    "recall", "delay", "delayed", "halts", "suspended",
    # price
    "plunges", "tumbles", "crashes", "slumps", "slides",
]

# 2026 growth / forward-looking signals
GROWTH_KW = [
    r"\b2026\b", r"\b2027\b", r"\bfy26\b", r"\bfy2026\b",
    r"\bforecast\b", r"\bguidance for 20(2[6-9])\b",
    r"\bprojected\b", r"\bexpects\b.*\bgrowth\b",
    r"\bramp(ing)?\b", r"\bscaling\b", r"\bpipeline\b",
    r"\btarget\b.*\b20(2[6-9])\b",
    r"\bmulti[\-\s]year\b", r"\blong[\-\s]term contract\b",
]


# ---------- Tier 1: keyword scoring ----------

def score_keyword(text):
    t = " " + text.lower() + " "
    pos = sum(1 for kw in POSITIVE_KW if kw in t)
    neg = sum(1 for kw in NEGATIVE_KW if kw in t)
    if pos > neg:
        label = "POS"
    elif neg > pos:
        label = "NEG"
    else:
        label = "NEU"
    return label, pos, neg


def has_growth_signal(text):
    t = text.lower()
    return any(re.search(p, t) for p in GROWTH_KW)


# ---------- Tier 2: FinBERT ----------

_finbert = None


def _load_finbert():
    global _finbert
    if _finbert is not None:
        return _finbert
    try:
        from transformers import pipeline
        _finbert = pipeline("sentiment-analysis",
                            model="ProsusAI/finbert",
                            truncation=True)
    except Exception as e:
        print(f"[warn] FinBERT unavailable: {e}")
        _finbert = False
    return _finbert


def score_finbert(text):
    pipe = _load_finbert()
    if not pipe:
        return None
    try:
        r = pipe(text[:512])[0]
        lbl = r["label"].upper()[:3]  # POS / NEG / NEU
        return lbl, float(r["score"])
    except Exception as e:
        print(f"[warn] FinBERT error: {e}")
        return None


# ---------- Tier 3: Claude API ----------

def score_claude(headlines):
    """One batch call to classify all headlines + detect 2026 growth."""
    try:
        import anthropic
    except ImportError:
        print("[warn] pip install anthropic")
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[warn] ANTHROPIC_API_KEY not set")
        return None

    client = anthropic.Anthropic()
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = (
        "Classify each headline for a penny-stock screener. "
        "Respond ONLY as CSV lines: idx,sentiment,growth2026\n"
        "  sentiment = POS | NEG | NEU\n"
        "  growth2026 = YES if mentions 2026+ growth/guidance/contracts, else NO\n\n"
        f"Headlines:\n{numbered}"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.content[0].text
    except Exception as e:
        print(f"[warn] Claude API error: {e}")
        return None

    results = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                idx = int(parts[0]) - 1
                results.append((idx, parts[1].upper()[:3],
                                parts[2].upper().startswith("Y")))
            except Exception:
                pass
    return results


# ---------- Runner ----------

def evaluate_f9(ticker, days=365, engine="auto"):
    ticker = ticker.upper().strip()
    print(f"\n=== Auto-F9 for ${ticker} (last {days}d, engine={engine}) ===\n")

    # Gather headlines
    news = yf_news(ticker, days=days) or yahoo_rss(ticker, days=days)
    filings = sec_filings(ticker, days=days)
    items = []
    for n in news:
        items.append((n.get("date", "?"), n["title"], "news"))
    for f in filings:
        items.append((f["date"], f"{f['form']}: {f['desc']}", "filing"))

    if not items:
        print("  (no headlines or filings)")
        return None, "No data"

    # Pick engine
    chosen = engine
    if engine == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            chosen = "claude"
        elif _load_finbert():
            chosen = "finbert"
        else:
            chosen = "keyword"
    print(f"  Using engine: {chosen}\n")

    pos = neg = neu = 0
    growth_hits = 0
    rows = []

    if chosen == "claude":
        texts = [t for _, t, _ in items]
        cls = score_claude(texts) or []
        cls_map = {i: (s, g) for i, s, g in cls}
        for i, (d, t, kind) in enumerate(items):
            s, g = cls_map.get(i, ("NEU", False))
            rows.append((d, kind, s, g, t))
    else:
        for d, t, kind in items:
            if chosen == "finbert":
                r = score_finbert(t)
                s = r[0] if r else score_keyword(t)[0]
            else:
                s, _, _ = score_keyword(t)
            g = has_growth_signal(t)
            rows.append((d, kind, s, g, t))

    for d, kind, s, g, t in rows:
        if s == "POS":
            pos += 1
        elif s == "NEG":
            neg += 1
        else:
            neu += 1
        if g:
            growth_hits += 1
        tag = {"POS": "🟢", "NEG": "🔴", "NEU": "⚪"}[s]
        gtag = " 📈2026" if g else ""
        print(f"  {tag} {d}  [{kind}]{gtag}  {t[:95]}")

    total = pos + neg + neu
    print("\n" + "=" * 60)
    print(f"Counts:  POS={pos}  NEG={neg}  NEU={neu}  (total {total})")
    print(f"2026 growth mentions: {growth_hits}")

    # Scoring rule (matches penny.txt framework)
    pos_ratio = pos / total if total else 0
    neg_ratio = neg / total if total else 0

    if pos_ratio >= 0.50 and growth_hits >= 1 and neg_ratio < 0.25:
        verdict = "✅ PASS — Mostly positive + 2026 growth signal"
        ok = True
    elif neg_ratio >= 0.50:
        verdict = "❌ FAIL — Mostly negative"
        ok = False
    else:
        verdict = "🟡 NEUTRAL — Mixed"
        ok = None

    print(verdict)
    print("=" * 60)
    return ok, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--engine",
                    choices=["auto", "keyword", "finbert", "claude"],
                    default="auto")
    args = ap.parse_args()
    evaluate_f9(args.ticker, days=args.days, engine=args.engine)
