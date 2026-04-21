# Stock Screener — Penny / Growth / Value

Web-based 10-filter stock screener that auto-routes a ticker to the right
framework by market cap.

| Band | Market cap | Framework |
|---|---|---|
| nano + micro | `< $300M` | penny |
| small + mid  | `$300M – $2B` | growth |
| large + mega | `>= $2B` | value |

Each framework has multiple weighting **profiles** (e.g. `balanced`,
`pump_hunter`, `garp`, `deep_value`) and **sector overlays**. A single
"Run" produces verdicts for every profile of the resolved framework plus a
per-filter breakdown.

## Stack

- Python 3.11+
- Flask + gunicorn
- yfinance (price/financials), SEC EDGAR (filings), Yahoo RSS (news)
- FinBERT (`ProsusAI/finbert`) for headline sentiment
- Optional Anthropic Claude for richer news classification
- 10-min in-process TTL cache for ticker data

## Local dev

    python -m venv venv
    source venv/bin/activate    # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:5000/

CLI:

    python screener.py AAPL
    python screener.py SOUN --framework penny --profile pump_hunter
    python screener.py NVDA --compare-all-frameworks --json

## Deploy

See [deploy/DEPLOY.md](deploy/DEPLOY.md) for the full Contabo + nginx +
certbot walkthrough.

## Files

- [app.py](app.py) — Flask web server
- [screener.py](screener.py) — unified market-cap dispatcher
- [penny_filter.py](penny_filter.py), [growth_filter.py](growth_filter.py),
  [value_filter.py](value_filter.py) — the three frameworks
- [cache.py](cache.py) — TTL cache for yfinance/SEC
- [templates/index.html](templates/index.html) — single-page UI
- [deploy/](deploy/) — production configs (nginx, systemd, gunicorn)
