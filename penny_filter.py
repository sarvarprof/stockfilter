"""
Penny Stock 10-Filter Framework Evaluator — FULLY AUTOMATED

Runs all 10 filters with no human interaction.
Data sources: yfinance, SEC EDGAR, Yahoo RSS, StockTwits, optional FinBERT/Claude.

Usage:
    python penny_filter.py TICKER
    python penny_filter.py TICKER --engine keyword|finbert|claude
    python penny_filter.py TICKER --days-news 365 --days-filings 30
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.parse import quote

try:
    import yfinance as yf
except ImportError:
    print("Install dependencies: pip install yfinance")
    sys.exit(1)

UA = "Mozilla/5.0 (penny-screener research contact@example.com)"


def http_get(url, accept="application/json"):
    req = Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


# ============================================================
# SHARED CONSTANTS
# ============================================================

HOT_SECTOR_MAP = {
    "Tech":     ["technology", "software", "semiconductor",
                 "information technology", "communication",
                 "internet", "cloud", "saas", "fintech"],
    "Defense":  ["defense", "aerospace", "military", "weapons",
                 "drone", "uav", "missile", "homeland security"],
    "Biotech":  ["biotech", "biotechnology", "pharmaceutical",
                 "drug manufacturers", "life sciences", "genomic",
                 "clinical", "therapeutic", "vaccine"],
    "AI":       ["artificial intelligence", " ai ", " ai.", " ai,",
                 "machine learning", "generative ai",
                 "llm", "neural network", "deep learning"],
    "Energy":   ["energy", "oil", "gas", "solar", "nuclear",
                 "uranium", "renewable", "lithium", "battery",
                 "hydrogen", "fuel cell"],
}

RS_PATTERNS = [
    r"\breverse\s+stock\s+split\b",
    r"\breverse\s+split\b",
    r"\bshare\s+consolidation\b",
    r"\b1[\-\s]for[\-\s]\d+\b",
    r"\b\d+[\-\s]for[\-\s]1\s+reverse\b",
]

OFFERING_PATTERNS = [
    r"\bpublic\s+offering\b",
    r"\bregistered\s+direct\s+offering\b",
    r"\bprivate\s+placement\b",
    r"\bat[\-\s]the[\-\s]market\s+offering\b",
    r"\bATM\s+offering\b",
    r"\bshelf\s+registration\b",
    r"\bS-3\b",
    r"\bsecondary\s+offering\b",
    r"\bpricing\s+of\b.*\boffering\b",
    r"\bunderwritten\s+offering\b",
    r"\bsale\s+of\s+common\s+stock\b",
]

DILUTION_FORMS = {"S-1", "S-3", "424B5", "424B4", "424B2",
                  "FWP", "F-1", "F-3"}

CATALYST_KW = [
    "department of defense", "dod contract", "pentagon", "army", "navy",
    "air force", "space force", "darpa", "nasa", "government contract",
    "federal contract", "awarded contract", "task order", "idiq",
    "fda approval", "fda clearance", "breakthrough designation",
    "phase 3", "phase iii", "pdufa",
    "acquisition", "merger", "strategic partnership", "letter of intent",
    "memorandum of understanding", "mou signed",
    "ai partnership", "ai deployment", "quantum", "cybersecurity contract",
]

POSITIVE_KW = [
    "beats", "beat estimates", "exceeds", "record revenue", "record quarter",
    "profitable", "profit", "strong growth", "strong results", "raised guidance",
    "upgraded", "upgrade", "buy rating", "outperform", "price target raised",
    "contract", "awarded", "partnership", "agreement", "acquisition",
    "fda approval", "cleared", "approved", "breakthrough", "patent",
    "launches", "launch", "expansion", "expands", "milestone",
    "order", "backlog", "selected", "wins", "deal",
    "surges", "soars", "rallies", "jumps", "climbs", "gains",
    "breakout", "all-time high",
]

NEGATIVE_KW = [
    "offering", "dilution", "private placement", "at-the-market",
    "registered direct", "reverse split", "going concern",
    "bankruptcy", "restructuring", "delisting", "nasdaq notice",
    "deficiency", "non-compliance",
    "misses", "missed estimates", "loss", "losses", "widens loss",
    "downgrade", "downgraded", "sell rating", "underperform",
    "guidance cut", "lowered guidance", "warning",
    "lawsuit", "sued", "investigation", "subpoena", "sec charges",
    "fraud", "class action", "fines",
    "resigns", "resignation", "terminated", "layoffs",
    "recall", "delay", "delayed", "halts", "suspended",
    "plunges", "tumbles", "crashes", "slumps", "slides",
]

GROWTH_KW = [
    r"\b2026\b", r"\b2027\b", r"\bfy26\b", r"\bfy2026\b",
    r"\bforecast\b", r"\bguidance for 20(2[6-9])\b",
    r"\bprojected\b", r"\bexpects\b.*\bgrowth\b",
    r"\bramp(ing)?\b", r"\bscaling\b", r"\bpipeline\b",
    r"\btarget\b.*\b20(2[6-9])\b",
    r"\bmulti[\-\s]year\b", r"\blong[\-\s]term contract\b",
]


# ============================================================
# SEC EDGAR + NEWS FETCHERS
# ============================================================

_cik_cache = {}


def sec_cik(ticker):
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]
    try:
        data = json.loads(http_get("https://www.sec.gov/files/company_tickers.json"))
    except Exception as e:
        print(f"[warn] CIK lookup failed: {e}")
        return None
    for _, row in data.items():
        if row.get("ticker", "").upper() == ticker:
            cik = str(row["cik_str"]).zfill(10)
            _cik_cache[ticker] = cik
            return cik
    return None


_filings_cache = {}


def sec_filings(ticker, days=180):
    """Return filings within `days` window."""
    key = (ticker.upper(), days)
    if key in _filings_cache:
        return _filings_cache[key]

    cik = sec_cik(ticker)
    if not cik:
        _filings_cache[key] = []
        return []
    try:
        data = json.loads(http_get(
            f"https://data.sec.gov/submissions/CIK{cik}.json"))
    except Exception as e:
        print(f"[warn] SEC filings fetch failed: {e}")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    descs = recent.get("primaryDocDescription", [])
    accs = recent.get("accessionNumber", [])
    items = recent.get("items", [])

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    out = []
    for i in range(len(forms)):
        try:
            d = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        if d < cutoff:
            continue
        acc = accs[i].replace("-", "")
        filing_url = (f"https://www.sec.gov/Archives/edgar/data/"
                      f"{int(cik)}/{acc}/")
        out.append({
            "date": dates[i],
            "form": forms[i],
            "desc": descs[i] if i < len(descs) else "",
            "items": items[i] if i < len(items) else "",
            "accession": accs[i],
            "url": filing_url,
        })
    _filings_cache[key] = out
    return out


def yf_news(ticker, days=365):
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[warn] yfinance news failed: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for n in news:
        content = n.get("content", n)
        title = content.get("title") or n.get("title") or ""
        pub = (content.get("pubDate")
               or content.get("providerPublishTime")
               or n.get("providerPublishTime"))
        link = (content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else content.get("link") or n.get("link") or "")
        try:
            if isinstance(pub, (int, float)):
                dt = datetime.fromtimestamp(pub, tz=timezone.utc)
            elif isinstance(pub, str):
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            else:
                dt = None
        except Exception:
            dt = None
        if dt and dt < cutoff:
            continue
        out.append({
            "date": dt.strftime("%Y-%m-%d") if dt else "n/a",
            "title": title,
            "url": link,
        })
    return out


def yahoo_rss(ticker, days=365):
    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?"
           f"s={quote(ticker)}&region=US&lang=en-US")
    try:
        xml = http_get(url, accept="application/rss+xml")
    except Exception as e:
        print(f"[warn] Yahoo RSS failed: {e}")
        return []
    items = re.findall(r"<item>(.*?)</item>", xml,
                       flags=re.DOTALL | re.IGNORECASE)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for it in items:
        title = re.search(r"<title>(.*?)</title>", it, re.DOTALL)
        link = re.search(r"<link>(.*?)</link>", it, re.DOTALL)
        pub = re.search(r"<pubDate>(.*?)</pubDate>", it, re.DOTALL)
        try:
            dt = (datetime.strptime(pub.group(1).strip(),
                                    "%a, %d %b %Y %H:%M:%S %z") if pub else None)
        except Exception:
            dt = None
        if dt and dt < cutoff:
            continue
        out.append({
            "date": dt.strftime("%Y-%m-%d") if dt else "n/a",
            "title": (title.group(1) if title else "").strip()
                .replace("<![CDATA[", "").replace("]]>", ""),
            "url": (link.group(1) if link else "").strip(),
        })
    return out


def get_news(ticker, days=365):
    return yf_news(ticker, days=days) or yahoo_rss(ticker, days=days)


def match_any(text, patterns):
    t = text.lower()
    return [p for p in patterns if re.search(p, t, re.IGNORECASE)]


def _latest_bs(df, keys):
    if df is None or df.empty:
        return None
    for k in keys:
        for idx in df.index:
            if k.lower() in str(idx).lower():
                try:
                    val = df.loc[idx].dropna()
                    if len(val):
                        return float(val.iloc[0])
                except Exception:
                    pass
    return None


# ============================================================
# FILTERS F1–F10
# ============================================================

def f1_float(info, ticker=None):
    """F1: Float < 50M"""
    flt = info.get("floatShares") or info.get("sharesOutstanding")
    if not flt and ticker:
        # EDGAR fallback: use most recent diluted shares
        try:
            import edgar_bridge as _eb
            shares_s = _eb.get_shares_series(ticker)
            if shares_s is not None and len(shares_s):
                flt = float(shares_s.iloc[-1])
        except Exception:
            pass
    if not flt:
        return None, "Float unknown"
    ok = flt < 50_000_000
    return ok, f"Float = {flt/1e6:.2f}M (<50M required)"


def f2_market_cap(info):
    """F2: Market Cap < $50M"""
    mcap = info.get("marketCap")
    if not mcap:
        return None, "Market cap unknown"
    ok = mcap < 50_000_000
    return ok, f"Market Cap = ${mcap/1e6:.2f}M (<$50M required)"


def f3_us_domiciled(info):
    """F3: US-domiciled"""
    country = (info.get("country") or "").strip()
    if not country:
        return None, "Country unknown"
    ok = country.lower() in ("united states", "usa", "us")
    return ok, f"Country = {country}"


def f4_reverse_split(ticker, days=60):
    """F4: No confirmed RS in next 2 weeks.
    Auto-detected: scans SEC filings + news for RS language."""
    filings = sec_filings(ticker, days=days)
    news = get_news(ticker, days=days)

    rs_filings = []
    for f in filings:
        blob = f"{f['form']} {f['desc']} {f['items']}"
        if match_any(blob, RS_PATTERNS):
            rs_filings.append(f)
        if f["form"].upper() == "8-K" and "3.03" in (f.get("items") or ""):
            rs_filings.append(f)

    rs_news = [n for n in news if match_any(n["title"], RS_PATTERNS)]

    if rs_filings or rs_news:
        src = []
        if rs_filings:
            src.append(f"{len(rs_filings)} filing(s)")
        if rs_news:
            src.append(f"{len(rs_news)} headline(s)")
        return False, f"RS signals: {', '.join(src)}"
    return True, f"No RS signals in last {days}d"


def f5_offering(ticker, days=30):
    """F5: No offering in past 1 month.
    Auto-detected: flags S-1/S-3/424B* filings + offering keywords."""
    filings = sec_filings(ticker, days=days)
    news = get_news(ticker, days=days)

    off_filings = []
    for f in filings:
        if f["form"].upper() in DILUTION_FORMS:
            off_filings.append(f)
            continue
        blob = f"{f['form']} {f['desc']}"
        if match_any(blob, OFFERING_PATTERNS):
            off_filings.append(f)

    off_news = [n for n in news if match_any(n["title"], OFFERING_PATTERNS)]

    if off_filings or off_news:
        src = []
        if off_filings:
            src.append(f"{len(off_filings)} filing(s) "
                       f"({off_filings[0]['form']})")
        if off_news:
            src.append(f"{len(off_news)} headline(s)")
        return False, f"Offering signals: {', '.join(src)}"
    return True, f"No offering signals in last {days}d"


def f6_sector(info):
    """F6: Hot sector: Tech / Defense / Biotech / AI / Energy"""
    sector = (info.get("sector") or "").strip()
    industry = (info.get("industry") or "").strip()
    summary = (info.get("longBusinessSummary") or "").strip()
    haystack = f" {sector} {industry} {summary} ".lower()
    matched = [label for label, kws in HOT_SECTOR_MAP.items()
               if any(kw in haystack for kw in kws)]
    if matched:
        return True, f"Hot sector match: {', '.join(matched)}"
    if not sector and not industry:
        return None, "No sector data"
    return False, f"Not a hot sector ({sector})"


def f7_spike_history(hist):
    """F7: 2+ weekly +60% spikes in last 2 years"""
    if hist is None or hist.empty:
        return None, "No price history"
    closes = hist["Close"].dropna()
    spikes = 0
    for i in range(1, len(closes)):
        prev = closes.iloc[i - 1]
        cur = closes.iloc[i]
        if prev > 0 and (cur - prev) / prev >= 0.60:
            spikes += 1
    ok = spikes >= 2
    return ok, f"{spikes} weekly 60%+ spikes in last 2y"


# ---------- F8 sub-checks ----------

def _cash_runway(ticker):
    # Try EDGAR first — audited, multi-year, reliable for micro-caps
    try:
        import edgar_bridge as _eb
        cdo = _eb.get_cash_debt_ocf(ticker)
        if cdo is not None:
            cash = cdo["cash"]
            ocf  = cdo["ocf"]
            annual_burn = None
            if ocf < 0:
                annual_burn = -ocf
            else:
                # cash-flow positive — also check FCF series trend
                fcf_s = _eb.get_fcf_series(ticker)
                if fcf_s is not None and len(fcf_s) and fcf_s.iloc[-1] < 0:
                    annual_burn = -float(fcf_s.iloc[-1])
            if cash is None or cash == 0:
                pass  # fall through to yfinance
            elif annual_burn is None or annual_burn <= 0:
                return True, f"Cash ${cash/1e6:.1f}M; cash-flow positive (EDGAR)"
            else:
                years = cash / annual_burn
                return years >= 2.0, (f"Cash ${cash/1e6:.1f}M / burn "
                                      f"${annual_burn/1e6:.1f}M/yr = {years:.2f}y (EDGAR)")
    except Exception:
        pass

    # yfinance fallback
    t = yf.Ticker(ticker)
    bs  = getattr(t, "balance_sheet", None)
    cf  = getattr(t, "cashflow", None)
    qcf = getattr(t, "quarterly_cashflow", None)

    cash = _latest_bs(bs, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ])
    ocf_a = _latest_bs(cf, ["Operating Cash Flow",
                             "Total Cash From Operating Activities"])
    ocf_q = _latest_bs(qcf, ["Operating Cash Flow",
                              "Total Cash From Operating Activities"])

    annual_burn = None
    if ocf_a is not None and ocf_a < 0:
        annual_burn = -ocf_a
    elif ocf_q is not None and ocf_q < 0:
        annual_burn = -ocf_q * 4

    if cash is None:
        return None, "Cash balance unavailable"
    if annual_burn is None or annual_burn <= 0:
        return True, f"Cash ${cash/1e6:.1f}M; cash-flow positive"
    years = cash / annual_burn
    return years >= 2.0, (f"Cash ${cash/1e6:.1f}M / burn "
                          f"${annual_burn/1e6:.1f}M/yr = {years:.2f}y")


def _insider_buying(ticker, days=90):
    cik = sec_cik(ticker)
    if not cik:
        return None, "CIK not found"
    try:
        data = json.loads(http_get(
            f"https://data.sec.gov/submissions/CIK{cik}.json"))
    except Exception as e:
        return None, f"SEC error: {e}"
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    form4s = []
    for i, f in enumerate(forms):
        if f != "4":
            continue
        try:
            d = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        if d >= cutoff:
            form4s.append((dates[i], accs[i]))

    if not form4s:
        return False, f"No Form 4 in last {days}d"

    buys = sells = 0
    for _, acc in form4s[:25]:
        acc_nd = acc.replace("-", "")
        base = (f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_nd}/")
        try:
            idx = http_get(base + "index.json")
            files = json.loads(idx).get("directory", {}).get("item", [])
            xml_name = next(
                (it["name"] for it in files
                 if it["name"].lower().endswith(".xml")), None)
            if not xml_name:
                continue
            xml = http_get(base + xml_name, accept="application/xml")
        except Exception:
            continue
        for c in re.findall(r"<transactionCode>([A-Z])</transactionCode>", xml):
            if c == "P":
                buys += 1
            elif c == "S":
                sells += 1

    ok = buys > 0 and buys >= sells
    return ok, f"Form 4 {days}d: {buys} buys / {sells} sells"


def _mgmt_changes(ticker, days=180):
    filings = sec_filings(ticker, days=days)
    hits = [f["date"] for f in filings
            if f["form"] == "8-K" and "5.02" in (f.get("items") or "")]
    if hits:
        return True, (f"{len(hits)} mgmt-change 8-K(s) "
                      f"({', '.join(hits[:2])})")
    return False, f"No 8-K 5.02 in last {days}d"


def f8_financial_health(ticker):
    """F8: Cash runway + insider buying + mgmt changes (>=2 of 3)"""
    results = []
    score = 0
    for name, fn in [("runway", lambda: _cash_runway(ticker)),
                     ("insider", lambda: _insider_buying(ticker, 90)),
                     ("mgmt",    lambda: _mgmt_changes(ticker, 180))]:
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = None, f"err: {e}"
        if ok is True:
            score += 1
        results.append(f"{name}={'Y' if ok else ('N' if ok is False else '?')}"
                       f"[{note}]")
    ok = score >= 2
    return ok, f"{score}/3: " + " | ".join(results)


# ---------- F9 sentiment ----------

def _score_keyword(text):
    t = " " + text.lower() + " "
    pos = sum(1 for kw in POSITIVE_KW if kw in t)
    neg = sum(1 for kw in NEGATIVE_KW if kw in t)
    if pos > neg:
        return "POS"
    if neg > pos:
        return "NEG"
    return "NEU"


def _has_growth(text):
    t = text.lower()
    return any(re.search(p, t) for p in GROWTH_KW)


_finbert = None


def _load_finbert():
    global _finbert
    if _finbert is not None:
        return _finbert
    try:
        from transformers import pipeline
        _finbert = pipeline("sentiment-analysis",
                            model="ProsusAI/finbert", truncation=True)
    except Exception:
        _finbert = False
    return _finbert


def _score_finbert(text):
    pipe = _load_finbert()
    if not pipe:
        return None
    try:
        return pipe(text[:512])[0]["label"].upper()[:3]
    except Exception:
        return None


def _score_claude(headlines):
    try:
        import anthropic
    except ImportError:
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    client = anthropic.Anthropic()
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = (
        "Classify each headline for a penny-stock screener. "
        "Respond ONLY as CSV lines: idx,sentiment,growth2026\n"
        "  sentiment = POS | NEG | NEU\n"
        "  growth2026 = YES if mentions 2026+ growth/guidance/contracts\n\n"
        f"Headlines:\n{numbered}"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.content[0].text
    except Exception as e:
        print(f"[warn] Claude error: {e}")
        return None
    results = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                results[int(parts[0]) - 1] = (
                    parts[1].upper()[:3],
                    parts[2].upper().startswith("Y"),
                )
            except Exception:
                pass
    return results


def f9_news(ticker, days=365, engine="auto"):
    """F9: Mostly positive news + 2026 growth signals"""
    news = get_news(ticker, days=days)
    filings = sec_filings(ticker, days=days)
    items = [(n.get("date", "?"), n["title"]) for n in news]
    items += [(f["date"], f"{f['form']}: {f['desc']}") for f in filings]
    if not items:
        return None, "No news/filings"

    chosen = engine
    if engine == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            chosen = "claude"
        elif _load_finbert():
            chosen = "finbert"
        else:
            chosen = "keyword"

    pos = neg = neu = 0
    growth_hits = 0
    if chosen == "claude":
        texts = [t for _, t in items]
        cls = _score_claude(texts) or {}
        for i, (_, t) in enumerate(items):
            s, g = cls.get(i, (_score_keyword(t), _has_growth(t)))
            if s == "POS":
                pos += 1
            elif s == "NEG":
                neg += 1
            else:
                neu += 1
            if g:
                growth_hits += 1
    else:
        for _, t in items:
            if chosen == "finbert":
                s = _score_finbert(t) or _score_keyword(t)
            else:
                s = _score_keyword(t)
            if s == "POS":
                pos += 1
            elif s == "NEG":
                neg += 1
            else:
                neu += 1
            if _has_growth(t):
                growth_hits += 1

    total = pos + neg + neu
    pos_r = pos / total if total else 0
    neg_r = neg / total if total else 0
    note = (f"{chosen}: POS={pos} NEG={neg} NEU={neu}, "
            f"2026 signals={growth_hits}")
    if pos_r >= 0.50 and growth_hits >= 1 and neg_r < 0.25:
        return True, note + " -> positive + growth"
    if neg_r >= 0.50:
        return False, note + " -> mostly negative"
    return None, note + " -> mixed"


# ---------- F10 sub-checks ----------

def _float_small(info):
    flt = info.get("floatShares") or info.get("sharesOutstanding")
    if not flt:
        return None, "Float unknown"
    return flt < 30_000_000, f"Float {flt/1e6:.1f}M vs 30M"


def _mcap_small(info):
    m = info.get("marketCap")
    if not m:
        return None, "Mcap unknown"
    return m < 20_000_000, f"Mcap ${m/1e6:.1f}M vs $20M"


def _catalyst(ticker, days=90):
    news = get_news(ticker, days=days)
    filings = sec_filings(ticker, days=days)
    blob = " ".join(n["title"] for n in news).lower()
    blob += " " + " ".join(f"{f['form']} {f['desc']}"
                           for f in filings).lower()
    hits = [kw for kw in CATALYST_KW if kw in blob]
    return bool(hits), (f"Catalysts: {', '.join(hits[:3])}"
                        if hits else "No catalyst keywords")


def _clean_structure(ticker, days=180):
    filings = sec_filings(ticker, days=days)
    bad = [f for f in filings if f["form"].upper() in DILUTION_FORMS]
    rs = [f for f in filings
          if re.search(r"reverse|consolidation",
                       (f.get("desc") or "").lower())]

    # EDGAR shares series: multi-year dilution CAGR (more reliable than yfinance BS row)
    share_growth = None
    share_src = ""
    try:
        import edgar_bridge as _eb
        shares_s = _eb.get_shares_series(ticker)
        if shares_s is not None and len(shares_s) >= 2:
            old = float(shares_s.iloc[-2])
            new = float(shares_s.iloc[-1])
            if old > 0:
                share_growth = new / old - 1
                share_src = " (EDGAR)"
    except Exception:
        pass

    if share_growth is None:
        try:
            bs = yf.Ticker(ticker).balance_sheet
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

    notes, ok = [], True
    if bad:
        ok = False
        notes.append(f"{len(bad)} dilution filing(s)")
    if rs:
        ok = False
        notes.append("RS signal")
    if share_growth is not None:
        notes.append(f"shares YoY: {share_growth*100:+.1f}%{share_src}")
        if share_growth > 0.25:
            ok = False
    if not notes:
        notes.append(f"clean in {days}d")
    return ok, "; ".join(notes)


def _short_interest(info):
    sp = info.get("shortPercentOfFloat")
    if sp is None:
        sr = info.get("shortRatio")
        if sr is None:
            return None, "Short data unavailable"
        return sr >= 5, f"Short ratio={sr:.2f}"
    pct = sp * 100 if sp < 1 else sp
    return pct >= 15, f"Short%Float={pct:.1f}%"


def _social_buzz(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        data = json.loads(http_get(url))
    except Exception as e:
        return None, f"StockTwits unavailable: {e}"
    msgs = data.get("messages", [])
    if not msgs:
        return False, "No ST messages"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = bull = bear = 0
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
    return (recent >= 30 and bull >= bear,
            f"ST 24h: {recent} msgs ({bull}🐂/{bear}🐻)")


def _volume_spike(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
    except Exception as e:
        return None, f"hist err: {e}"
    if hist is None or hist.empty or len(hist) < 30:
        return None, "Insufficient history"
    vols = hist["Volume"].dropna()
    avg30 = vols.iloc[-30:-1].mean()
    latest = vols.iloc[-1]
    if avg30 <= 0:
        return None, "No baseline"
    ratio = latest / avg30
    return (5 <= ratio <= 20,
            f"Today {latest/1e6:.2f}M vs avg {avg30/1e6:.2f}M = {ratio:.1f}x")


def f10_winner_profile(ticker, info):
    """F10: Winner profile — 8 signals, pass at >= 60% of applicable"""
    checks = [
        ("float<30M",   lambda: _float_small(info)),
        ("mcap<$20M",   lambda: _mcap_small(info)),
        ("catalyst",    lambda: _catalyst(ticker)),
        ("clean",       lambda: _clean_structure(ticker)),
        ("shortInt",    lambda: _short_interest(info)),
        ("narrative",   lambda: f6_sector(info)),
        ("social",      lambda: _social_buzz(ticker)),
        ("volSpike",    lambda: _volume_spike(ticker)),
    ]
    score = total = 0
    parts = []
    for name, fn in checks:
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = None, f"err: {e}"
        if ok is True:
            score += 1
            total += 1
            parts.append(f"✓{name}")
        elif ok is False:
            total += 1
            parts.append(f"✗{name}")
        else:
            parts.append(f"?{name}")
    pct = score / total if total else 0
    return pct >= 0.60, f"{score}/{total} ({pct*100:.0f}%): " + " ".join(parts)


# ============================================================
# WEIGHTED SCORING — analyst-tuned for PENNY STOCKS
# ============================================================
# Penny-stock math is dominated by capital destruction (dilution/RS),
# structure (float), and narrative (sector/news). Fundamentals barely
# matter because most penny stocks are pre-profit. Each profile sums
# to exactly 100.

WEIGHT_PROFILES = {
    # Analyst default — survival > structure > narrative > everything
    # Tier1 Survival=32  Tier2 Structure=21  Tier3 Narrative=19
    # Tier4 Behavior=16  Tier5 Health=7      Tier6 Hygiene=5
    "balanced": {
        "F1": 13, "F2": 8,  "F3": 5,  "F4": 15, "F5": 17,
        "F6": 10, "F7": 9,  "F8": 7,  "F9": 9,  "F10": 7,
    },
    # Pump-hunter — aggressive: explosive runners
    # Float, spike history, social, news catalysts carry the day
    "pump_hunter": {
        "F1": 18, "F2": 10, "F3": 3,  "F4": 12, "F5": 13,
        "F6": 11, "F7": 12, "F8": 3,  "F9": 9,  "F10": 9,
    },
    # Safe penny — prioritize survival; avoid destruction
    # Dilution/RS vetoes; US-domicile (fraud filter) matters more
    "safe_penny": {
        "F1": 9,  "F2": 7,  "F3": 10, "F4": 20, "F5": 22,
        "F6": 6,  "F7": 5,  "F8": 12, "F9": 5,  "F10": 4,
    },
    # Short-squeeze — tiny float + high SI + catalyst
    # F1 dominates; F10 (contains shortInt sub-signal) elevated
    "squeeze": {
        "F1": 22, "F2": 10, "F3": 3,  "F4": 13, "F5": 13,
        "F6": 7,  "F7": 8,  "F8": 3,  "F9": 9,  "F10": 12,
    },
    # Catalyst / news trader — react to announcements, hot sectors
    "catalyst": {
        "F1": 10, "F2": 6,  "F3": 4,  "F4": 13, "F5": 14,
        "F6": 17, "F7": 6,  "F8": 5,  "F9": 18, "F10": 7,
    },
}

for _pname, _pw in WEIGHT_PROFILES.items():
    assert sum(_pw.values()) == 100, f"Profile {_pname} != 100"


# Veto rules — per profile. Cap max verdict if any listed filter FAILS.
# For penny, dilution/RS vetoes matter even more than in growth stocks.
VETO_RULES = {
    "balanced":    {"F4": "WATCH", "F5": "WATCH"},     # soft caps
    "pump_hunter": {"F4": "WATCH"},                    # still avoid RS
    "safe_penny":  {"F4": "HARD_PASS", "F5": "HARD_PASS",
                    "F3": "WATCH"},                    # no compromise
    "squeeze":     {"F4": "WATCH", "F1": "WATCH"},     # need small float + no RS
    "catalyst":    {"F4": "WATCH"},
}


# ============================================================
# SECTOR OVERLAYS (penny-specific)
# ============================================================
# Deltas added to base profile, renormalized to 100.

SECTOR_OVERLAYS = {
    # Biotech — FDA catalysts > everything; revenue irrelevant
    "biotech": {
        "F1": -2, "F2": -1, "F3": +2, "F4": +2, "F5": +3,
        "F6": +2, "F7": -1, "F8": +4, "F9": +3, "F10": -2,
    },
    # Defense/Industrial — contract catalysts (news)
    "defense": {
        "F1": -2, "F2": 0,  "F3": +3, "F4": 0,  "F5": -1,
        "F6": +2, "F7": -2, "F8": +1, "F9": +5, "F10": -2,
    },
    # AI / Tech narrative — hot sector + social
    "ai_tech": {
        "F1": +2, "F2": +1, "F3": -1, "F4": -1, "F5": -2,
        "F6": +5, "F7": +1, "F8": -3, "F9": +1, "F10": +3,
    },
    # Energy/Uranium — cycle-driven, less retail-frothy
    "energy": {
        "F1": -2, "F2": +2, "F3": +2, "F4": 0,  "F5": -1,
        "F6": +2, "F7": -1, "F8": +3, "F9": +1, "F10": -3,
    },
    # Shell / SPAC / China RTO — fraud-risk overlay
    "shell": {
        "F1": -2, "F2": -1, "F3": +8, "F4": +3, "F5": +3,
        "F6": -3, "F7": -2, "F8": +2, "F9": -3, "F10": -3,
    },
    # Crypto / blockchain — pure narrative + social
    "crypto": {
        "F1": +2, "F2": -1, "F3": 0,  "F4": -2, "F5": -2,
        "F6": +4, "F7": +1, "F8": -4, "F9": +1, "F10": +5,
    },
}


# Map yfinance sector/industry -> overlay key
PENNY_SECTOR_KEYWORDS = {
    "biotech":  ["biotech", "biotechnology", "pharmaceutical",
                 "drug manufacturers", "life sciences", "therapeutic",
                 "clinical", "vaccine"],
    "defense":  ["defense", "aerospace", "military", "weapons",
                 "drone", "uav", "homeland security"],
    "ai_tech":  ["artificial intelligence", "software", "semiconductor",
                 "technology", "cloud", "saas", "machine learning"],
    "energy":   ["energy", "oil", "gas", "uranium", "solar", "nuclear",
                 "lithium", "battery", "hydrogen", "renewable"],
    "crypto":   ["crypto", "blockchain", "bitcoin", "digital asset",
                 "mining (industrial and commercial)"],
    "shell":    ["shell company", "blank check", "spac"],
}


def _renormalize_penny(weights):
    """Clamp to >=0, rescale to sum=100 (rounded exactly)."""
    w = {k: max(0, v) for k, v in weights.items()}
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
    """Return final F1..F10 weights for profile + optional sector overlay."""
    base = dict(WEIGHT_PROFILES[profile])
    if sector_key and sector_key in SECTOR_OVERLAYS:
        overlay = SECTOR_OVERLAYS[sector_key]
        merged = {k: base[k] + overlay.get(k, 0) for k in base}
        return _renormalize_penny(merged)
    return base


def detect_sector(info):
    """Auto-detect overlay key from yfinance info; None if no match."""
    haystack = " ".join([
        (info.get("sector") or ""),
        (info.get("industry") or ""),
        (info.get("industryKey") or ""),
        (info.get("longBusinessSummary") or ""),
    ]).lower()
    if not haystack.strip():
        return None
    for key, kws in PENNY_SECTOR_KEYWORDS.items():
        if any(kw in haystack for kw in kws):
            return key
    return None


# ============================================================
# SCORING MODES + BONUSES (penny-tuned)
# ============================================================

SCORING_MODES = {
    "strict":   {"neutral_credit": 0.30,
                 "desc": "NEUTRAL = 30% (penalize uncertainty)"},
    "normal":   {"neutral_credit": 0.50,
                 "desc": "NEUTRAL = 50% (default)"},
    "generous": {"neutral_credit": 0.70,
                 "desc": "NEUTRAL = 70% (benefit of the doubt)"},
}

# Penny-specific bonuses: reward high-conviction clusters
BONUS_RULES = [
    # Clean survival sweep — no dilution + no RS + US
    {"name": "Survival sweep",   "requires": ["F3", "F4", "F5"], "points": 5},
    # Explosive setup — tiny float + hot sector + historical spikes
    {"name": "Explosive setup",  "requires": ["F1", "F6", "F7"], "points": 5},
    # Catalyst confirmed — hot sector + positive news
    {"name": "Catalyst confirmed", "requires": ["F6", "F9"],     "points": 3},
    # Winner pattern — composite F10 + structure
    {"name": "Winner pattern",   "requires": ["F1", "F10"],      "points": 2},
]


VERDICT_RANK = {"🔴 HARD PASS": 0, "🟡 WATCH": 1,
                "🟢 BUY": 2, "🟢 STRONG BUY": 3}


PROFILE_DESCRIPTIONS = {
    "balanced":    "Analyst default — survival + structure + narrative balanced",
    "pump_hunter": "Aggressive — float/spike/social hunt explosive runners",
    "safe_penny":  "Risk-averse — hard vetoes on dilution/RS; US-only bias",
    "squeeze":     "Short-squeeze — tiny float + high SI + catalyst",
    "catalyst":    "News/event trader — react to hot sectors + announcements",
}

SECTOR_DESCRIPTIONS = {
    "auto":    "Auto-detect from yfinance sector/industry",
    "none":    "Disable sector overlay — use base profile only",
    "biotech": "Pre-FDA biotech — catalysts + cash runway up, revenue down",
    "defense": "Defense/industrial — contract catalysts (F9) emphasized",
    "ai_tech": "AI/tech narrative — hot sector + social up",
    "energy":  "Energy/commodity — institutional + domicile up, social down",
    "shell":   "Shell/SPAC/RTO — fraud guards (F3, F4, F5) hard-boosted",
    "crypto":  "Crypto/blockchain — narrative + social dominates",
}


# ============================================================
# INTERACTIVE PICKERS
# ============================================================

def profile_menu():
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
    print("\nSelect sector overlay:")
    names = list(SECTOR_DESCRIPTIONS.keys())
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<10s} — {SECTOR_DESCRIPTIONS[n]}")
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


# ============================================================
# RUNNER
# ============================================================

def verdict(score):
    """Weighted score -> verdict (out of 100)."""
    if score >= 75:
        return "🟢 STRONG BUY"
    if score >= 60:
        return "🟢 BUY"
    if score >= 45:
        return "🟡 WATCH"
    return "🔴 HARD PASS"


def apply_vetoes(raw_verdict, fail_ids, profile):
    """Cap verdict per profile's veto rules."""
    vetoes = VETO_RULES.get(profile, {})
    cap_level = None
    triggered = []
    for fid, cap in vetoes.items():
        if fid in fail_ids:
            triggered.append(f"{fid}->{cap}")
            level = 0 if cap == "HARD_PASS" else 1
            if cap_level is None or level < cap_level:
                cap_level = level
    if cap_level is None:
        return raw_verdict, triggered
    cap_verdict = "🟡 WATCH" if cap_level == 1 else "🔴 HARD PASS"
    if VERDICT_RANK.get(raw_verdict, 3) > cap_level:
        return cap_verdict, triggered
    return raw_verdict, triggered


def fetch_core(ticker):
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception as e:
        print(f"[warn] info fetch failed: {e}")
    hist = None
    try:
        hist = t.history(period="2y", interval="1wk")
    except Exception as e:
        print(f"[warn] history fetch failed: {e}")
    return info, hist


def evaluate(ticker,
             news_days=365,
             filing_days=30,
             f9_engine="auto",
             profile="balanced",
             sector="auto",
             scoring_mode="normal",
             apply_bonuses=True,
             verbose=True):
    """
    Evaluate ticker against the 10-filter penny framework.

    Args:
        ticker:         Symbol.
        news_days:      F9 news lookback.
        filing_days:    F5 offering lookback.
        f9_engine:      'auto'|'keyword'|'finbert'|'claude'.
        profile:        Weighting profile key.
        sector:         'auto' | 'none' | SECTOR_OVERLAYS key.
        scoring_mode:   'strict' | 'normal' | 'generous'.
        apply_bonuses:  Enable BONUS_RULES.
        verbose:        Print progress (False for programmatic/web use).

    Returns JSON-serializable dict.
    """
    ticker = ticker.upper().strip()
    info, hist = fetch_core(ticker)

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
        header = (f"[profile={profile} | sector={sector_key or 'none'} | "
                  f"mode={scoring_mode}]")
        print(f"\n{'='*70}\n  Evaluating ${ticker}   {header}\n{'='*70}")
        print(f"  Weights: " + ", ".join(f"{k}={v}"
                                         for k, v in weights.items()))

    filters = [
        ("F1",  "F1  Float < 50M",          lambda: f1_float(info, ticker)),
        ("F2",  "F2  Market Cap < $50M",    lambda: f2_market_cap(info)),
        ("F3",  "F3  US-Domiciled",         lambda: f3_us_domiciled(info)),
        ("F4",  "F4  No RS (2 weeks)",      lambda: f4_reverse_split(ticker, 60)),
        ("F5",  "F5  No offering (1 month)",
         lambda: f5_offering(ticker, filing_days)),
        ("F6",  "F6  Hot Sector",           lambda: f6_sector(info)),
        ("F7",  "F7  2+ weekly 60% spikes", lambda: f7_spike_history(hist)),
        ("F8",  "F8  Financial health/mgmt",
         lambda: f8_financial_health(ticker)),
        ("F9",  "F9  Positive news + growth",
         lambda: f9_news(ticker, news_days, f9_engine)),
        ("F10", "F10 Winner profile",
         lambda: f10_winner_profile(ticker, info)),
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
            status = "✅ PASS"
            earned = float(weight)
            pass_count += 1
            pass_ids.add(fid)
        elif ok is False:
            status = "❌ FAIL"
            earned = 0.0
            fail_ids.add(fid)
        else:
            status = "🟡 NEUTRAL"
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

    raw_verdict = verdict(total_score)
    final_verdict, triggered = apply_vetoes(raw_verdict, fail_ids, profile)

    if verbose:
        print(f"\n{'='*70}\n  SUMMARY for ${ticker}   "
              f"[profile={profile} | sector={sector_key or 'none'}]\n"
              f"{'='*70}")
        print(f"  {'Filter':<34s} {'Status':<14s} {'Score':>10s}")
        print("  " + "-" * 64)
        for r in results:
            print(f"  {r['name']:<34s} {r['status']:<14s} "
                  f"{r['earned']:>5.1f}/{r['weight']:<3d}")
        print("  " + "-" * 64)
        if bonus_applied:
            for b in bonus_applied:
                print(f"  + Bonus: {b['name']} (+{b['points']})")
        print(f"  TOTAL SCORE: {total_score:.1f}/100   "
              f"(raw pass count: {pass_count}/10)")
        if triggered:
            print(f"  ⚠  Veto triggered: {', '.join(triggered)}  "
                  f"(raw verdict was {raw_verdict})")
        print(f"  VERDICT:    {final_verdict}")
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
    """Run ALL profiles with auto-sector for web comparison views."""
    out = []
    for profile in WEIGHT_PROFILES.keys():
        out.append(evaluate(ticker, profile=profile, sector="auto",
                            verbose=False, **kwargs))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--news-days",    type=int, default=365)
    ap.add_argument("--filing-days",  type=int, default=30)
    ap.add_argument("--engine",
                    choices=["auto", "keyword", "finbert", "claude"],
                    default="auto",
                    help="F9 sentiment engine")
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
                    help="NEUTRAL credit mode (default: normal)")
    ap.add_argument("--no-bonuses", action="store_true",
                    help="Disable bonus-signal rules")
    ap.add_argument("--compare-all", action="store_true",
                    help="Run every profile (auto-sector) and compare")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON (for web/API consumers)")
    args = ap.parse_args()

    tk = args.ticker or input("Ticker: ").strip()

    profile = args.profile if args.profile != "ask" else profile_menu()
    sector  = args.sector  if args.sector  != "ask" else sector_menu()
    scoring = args.scoring if args.scoring != "ask" else scoring_menu()
    bonuses = not args.no_bonuses

    if args.compare_all:
        print(f"\n{'='*70}\n  Comparing all profiles for ${tk.upper()}"
              f"  (sector={sector}, mode={scoring})\n{'='*70}")
        summary = []
        for pname in WEIGHT_PROFILES.keys():
            print(f"\n\n>>>>> Profile: {pname} <<<<<")
            r = evaluate(tk, news_days=args.news_days,
                         filing_days=args.filing_days, f9_engine=args.engine,
                         profile=pname, sector=sector,
                         scoring_mode=scoring, apply_bonuses=bonuses)
            summary.append(r)
        print(f"\n{'='*70}\n  CROSS-PROFILE COMPARISON — ${tk.upper()}"
              f"\n{'='*70}")
        print(f"  {'Profile':<14s} {'Sector':<10s} {'Score':>8s}   Verdict")
        print("  " + "-" * 66)
        for r in summary:
            print(f"  {r['profile']:<14s} {str(r['sector'] or '-'):<10s} "
                  f"{r['score']:>5.1f}/100   {r['verdict']}")
        print("=" * 70)
    elif args.json:
        result = evaluate(tk, news_days=args.news_days,
                          filing_days=args.filing_days, f9_engine=args.engine,
                          profile=profile, sector=sector,
                          scoring_mode=scoring, apply_bonuses=bonuses,
                          verbose=False)
        print(json.dumps(result, indent=2, default=str))
    else:
        evaluate(tk, news_days=args.news_days,
                 filing_days=args.filing_days, f9_engine=args.engine,
                 profile=profile, sector=sector,
                 scoring_mode=scoring, apply_bonuses=bonuses)
