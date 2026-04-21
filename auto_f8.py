"""
Automated F8 signals (no human input):
  - Cash runway (yrs)       -> yfinance balance sheet + cash flow
  - Insider buying (90d)    -> SEC EDGAR Form 4 filings
  - Management changes (6m) -> SEC EDGAR 8-K item 5.02

Usage:
    python auto_f8.py TICKER
"""

import sys
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

UA = "Mozilla/5.0 (penny-screener research contact@example.com)"


def http_get(url, accept="application/json"):
    req = Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------- helpers ----------

def sec_cik(ticker):
    try:
        data = json.loads(http_get("https://www.sec.gov/files/company_tickers.json"))
    except Exception as e:
        print(f"[warn] CIK lookup failed: {e}")
        return None
    t = ticker.upper()
    for _, row in data.items():
        if row.get("ticker", "").upper() == t:
            return str(row["cik_str"]).zfill(10)
    return None


def _latest(df, keys):
    """Return the most recent numeric value for any matching row label."""
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


# ---------- 1. Cash runway ----------

def cash_runway(ticker):
    t = yf.Ticker(ticker)
    bs = getattr(t, "balance_sheet", None)
    cf = getattr(t, "cashflow", None)
    qcf = getattr(t, "quarterly_cashflow", None)

    cash = _latest(bs, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ])
    # annual operating cash flow (negative for cash-burning co)
    ocf_annual = _latest(cf, [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
    ])
    # fallback: quarterly * 4
    ocf_q = _latest(qcf, [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
    ])

    annual_burn = None
    if ocf_annual is not None and ocf_annual < 0:
        annual_burn = -ocf_annual
    elif ocf_q is not None and ocf_q < 0:
        annual_burn = -ocf_q * 4

    if cash is None:
        return None, "Cash balance unavailable"
    if annual_burn is None or annual_burn <= 0:
        return True, (f"Cash ${cash/1e6:.1f}M; company is cash-flow positive "
                      f"or burn unknown -> runway OK")

    years = cash / annual_burn
    ok = years >= 2.0
    return ok, (f"Cash ${cash/1e6:.1f}M / burn ${annual_burn/1e6:.1f}M/yr "
                f"= {years:.2f} yrs runway")


# ---------- 2. Insider buying ----------

def insider_buying(ticker, days=90):
    """
    Scan Form 4 filings in the last `days`.
    Purchases have transactionCode = 'P'.
    Returns (ok, note).
    """
    cik = sec_cik(ticker)
    if not cik:
        return None, "CIK not found"

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        data = json.loads(http_get(url))
    except Exception as e:
        return None, f"SEC fetch error: {e}"

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
        if d < cutoff:
            continue
        form4s.append((dates[i], accs[i]))

    if not form4s:
        return False, f"No Form 4 filings in last {days}d"

    # Fetch each Form 4 XML and check for open-market purchase (code 'P')
    buys, sells = 0, 0
    for date, acc in form4s[:25]:  # cap for speed
        acc_nodash = acc.replace("-", "")
        idx_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?"
                   f"action=getcompany&CIK={cik}&type=4&dateb=&owner=include")
        # Try primary doc index
        base = (f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_nodash}/")
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

        codes = re.findall(r"<transactionCode>([A-Z])</transactionCode>", xml)
        for c in codes:
            if c == "P":
                buys += 1
            elif c == "S":
                sells += 1

    ok = buys > 0 and buys >= sells
    return ok, (f"Form 4 last {days}d: {buys} purchase(s), {sells} sale(s)")


# ---------- 3. Management changes ----------

def management_changes(ticker, days=180):
    """
    Scan 8-K filings for item 5.02 (Departure/Election of Directors/Officers).
    Flag any recent filing = management change event.
    """
    cik = sec_cik(ticker)
    if not cik:
        return None, "CIK not found"

    try:
        data = json.loads(http_get(
            f"https://data.sec.gov/submissions/CIK{cik}.json"))
    except Exception as e:
        return None, f"SEC fetch error: {e}"

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items = recent.get("items", [])

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    hits = []
    for i, f in enumerate(forms):
        if f != "8-K":
            continue
        try:
            d = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        if d < cutoff:
            continue
        it = items[i] if i < len(items) else ""
        if "5.02" in it:
            hits.append(dates[i])

    if hits:
        return True, (f"{len(hits)} mgmt-change 8-K(s) in last {days}d: "
                      f"{', '.join(hits[:3])} (NOTE: sentiment not auto-judged)")
    return False, f"No 8-K item 5.02 in last {days}d"


# ---------- Runner ----------

def evaluate_f8(ticker):
    ticker = ticker.upper().strip()
    print(f"\n=== Auto-F8 for ${ticker} ===\n")

    checks = [
        ("Cash runway >= 2y",      lambda: cash_runway(ticker)),
        ("Insider buying (90d)",   lambda: insider_buying(ticker, 90)),
        ("Mgmt changes (180d)",    lambda: management_changes(ticker, 180)),
    ]
    score = 0
    for name, fn in checks:
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = None, f"error: {e}"
        if ok is True:
            tag = "✅"
            score += 1
        elif ok is False:
            tag = "❌"
        else:
            tag = "🟡"
        print(f"  {tag} {name:22s}  {note}")

    print("\n" + "=" * 50)
    verdict = "PASS" if score >= 2 else "FAIL"
    print(f"F8 auto-score: {score}/3  -> {verdict}")
    print("=" * 50)
    return score, verdict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        tk = input("Ticker: ").strip()
    else:
        tk = sys.argv[1]
    evaluate_f8(tk)
