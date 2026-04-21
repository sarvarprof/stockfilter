"""
Fetch latest news + SEC filings for a ticker.
Auto-detects reverse-split and offering signals (F4/F5 filters).

Usage:
    python fetch_news.py TICKER
    python fetch_news.py TICKER --days 30
"""

import sys
import re
import argparse
import json
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.parse import quote

try:
    import yfinance as yf
except ImportError:
    yf = None

UA = "Mozilla/5.0 (penny-screener research contact@example.com)"

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


def http_get(url, accept="application/json"):
    req = Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------- SEC EDGAR ----------

def sec_cik(ticker):
    """Resolve ticker -> 10-digit CIK."""
    try:
        data = json.loads(http_get("https://www.sec.gov/files/company_tickers.json"))
    except Exception as e:
        print(f"[warn] SEC ticker lookup failed: {e}")
        return None
    t = ticker.upper()
    for _, row in data.items():
        if row.get("ticker", "").upper() == t:
            return str(row["cik_str"]).zfill(10)
    return None


def sec_filings(ticker, days=30):
    """Return recent filings within `days` window."""
    cik = sec_cik(ticker)
    if not cik:
        return []
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        data = json.loads(http_get(url))
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
        doc_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?"
                   f"action=getcompany&CIK={cik}&type={forms[i]}&dateb=&owner=include")
        filing_url = (f"https://www.sec.gov/Archives/edgar/data/"
                      f"{int(cik)}/{acc}/")
        out.append({
            "date": dates[i],
            "form": forms[i],
            "desc": descs[i] if i < len(descs) else "",
            "items": items[i] if i < len(items) else "",
            "url": filing_url,
        })
    return out


# ---------- News (yfinance + Yahoo RSS fallback) ----------

def yf_news(ticker, days=30):
    if yf is None:
        return []
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[warn] yfinance news failed: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for n in news:
        # yfinance news shape varies by version
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


def yahoo_rss(ticker, days=30):
    """Backup RSS feed."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(ticker)}&region=US&lang=en-US"
    try:
        xml = http_get(url, accept="application/rss+xml")
    except Exception as e:
        print(f"[warn] Yahoo RSS failed: {e}")
        return []

    items = re.findall(
        r"<item>(.*?)</item>", xml, flags=re.DOTALL | re.IGNORECASE)
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


# ---------- Signal detection ----------

def match_any(text, patterns):
    text_low = text.lower()
    hits = []
    for p in patterns:
        if re.search(p, text_low, re.IGNORECASE):
            hits.append(p)
    return hits


def scan(ticker, days=30):
    print(f"\n=== News + Filings for ${ticker.upper()} (last {days} days) ===\n")

    # --- Filings ---
    filings = sec_filings(ticker, days=days)
    rs_hits, off_hits = [], []

    print("--- SEC Filings ---")
    if not filings:
        print("  (no recent filings or CIK not found)")
    for f in filings:
        blob = f"{f['form']} {f['desc']} {f['items']}"
        # Form-based signals
        form = f["form"].upper()
        off_form = form in ("S-1", "S-3", "424B5", "424B4", "424B2",
                            "FWP", "F-1", "F-3")
        rs_form = "DEF 14A" in form or "PRE 14A" in form  # proxy may contain RS
        rs_text = match_any(blob, RS_PATTERNS)
        off_text = match_any(blob, OFFERING_PATTERNS)

        flags = []
        if off_form:
            flags.append("OFFERING(form)")
            off_hits.append(f)
        if rs_text:
            flags.append("REVERSE-SPLIT(text)")
            rs_hits.append(f)
        if off_text:
            flags.append("OFFERING(text)")
            off_hits.append(f)
        if form == "8-K" and f.get("items"):
            if "3.03" in f["items"]:  # material modification to rights
                flags.append("8-K item 3.03 (possible RS)")
        flag_s = f"  [{' | '.join(flags)}]" if flags else ""
        print(f"  {f['date']}  {form:10s}  {f['desc'][:60]:60s}{flag_s}")
        print(f"             {f['url']}")

    # --- News ---
    print("\n--- Headlines ---")
    news = yf_news(ticker, days=days) or yahoo_rss(ticker, days=days)
    if not news:
        print("  (no recent headlines found)")
    for n in news[:25]:
        title = n["title"]
        rs_text = match_any(title, RS_PATTERNS)
        off_text = match_any(title, OFFERING_PATTERNS)
        flags = []
        if rs_text:
            flags.append("RS")
            rs_hits.append(n)
        if off_text:
            flags.append("OFFERING")
            off_hits.append(n)
        tag = f"  [{'|'.join(flags)}]" if flags else ""
        print(f"  {n['date']}  {title[:90]}{tag}")
        if n["url"]:
            print(f"             {n['url']}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("AUTO-SIGNALS")
    print("=" * 60)
    if rs_hits:
        print(f"⚠️  REVERSE SPLIT signals found ({len(rs_hits)}) "
              f"— F4 likely FAIL. Verify effective date manually.")
    else:
        print("✅ No reverse-split signals detected in scanned window.")
    if off_hits:
        print(f"⚠️  OFFERING signals found ({len(off_hits)}) "
              f"— F5 likely FAIL within {days}-day window.")
    else:
        print(f"✅ No offering signals detected in last {days} days.")
    print("=" * 60)

    return {"filings": filings, "news": news,
            "rs_hits": rs_hits, "off_hits": off_hits}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--days", type=int, default=30,
                    help="Lookback window in days (default 30)")
    args = ap.parse_args()
    scan(args.ticker, days=args.days)
