# Deploying stock.modulgroup.uz on Contabo

A step-by-step walkthrough for a fresh Ubuntu 22.04/24.04 Contabo VPS.

Assumes:

- You have SSH root access to the VPS
- The public IP of the VPS is known (let's call it `VPS_IP`)
- You can edit DNS for `modulgroup.uz`
- The repo has been pushed to GitHub at `https://github.com/<you>/<repo>.git`

---

## 1. DNS — point the subdomain at the VPS

In your DNS provider for `modulgroup.uz`, add an **A record**:

    Host:  stock
    Type:  A
    Value: <VPS_IP>
    TTL:   300

Wait for propagation:

    dig +short stock.modulgroup.uz

should return `<VPS_IP>` before you run certbot later.

---

## 2. Server — base install

SSH in as root, then:

    apt update && apt upgrade -y
    apt install -y python3.11 python3.11-venv python3-pip \
                   nginx git ufw certbot python3-certbot-nginx \
                   build-essential

    # Firewall
    ufw allow OpenSSH
    ufw allow 'Nginx Full'
    ufw --force enable

---

## 3. App user + clone

    # Use www-data (nginx default). If you prefer a dedicated user:
    #   useradd --system --shell /usr/sbin/nologin --home /opt/stock-screener stock
    mkdir -p /opt/stock-screener
    cd /opt
    git clone https://github.com/<you>/<repo>.git stock-screener
    cd stock-screener

    chown -R www-data:www-data /opt/stock-screener

---

## 4. Python venv + dependencies

    cd /opt/stock-screener
    sudo -u www-data python3.11 -m venv venv
    sudo -u www-data ./venv/bin/pip install --upgrade pip

    # Core + production server
    sudo -u www-data ./venv/bin/pip install -r requirements.txt

> **Size note:** `transformers` + `torch` is ~2 GB. If your VPS has
> < 4 GB RAM or you don't want FinBERT, drop those two lines from
> `requirements.txt` before installing. The screener will auto-fall-back
> to the keyword engine.

Pre-download FinBERT into the cache so first request is fast:

    sudo -u www-data ./venv/bin/python -c \
      "from transformers import pipeline; \
       pipeline('sentiment-analysis', model='ProsusAI/finbert')"

---

## 5. Environment file (secrets)

The app loads `/opt/stock-screener/.env` at startup (via `python-dotenv`,
also exported by systemd's `EnvironmentFile=` directive).

    cp /opt/stock-screener/.env.example /opt/stock-screener/.env
    nano /opt/stock-screener/.env
    # Fill in:
    #   SECRET_KEY=<random 64-char hex; generate with: python -c "import secrets; print(secrets.token_hex(32))">
    #   FRED_API_KEY=<get free at https://fredaccount.stlouisfed.org/>
    #   EDGAR_USER_AGENT=stockscreener you@example.com
    # Optional:
    #   ANTHROPIC_API_KEY=sk-ant-...

    chown www-data:www-data /opt/stock-screener/.env
    chmod 600 /opt/stock-screener/.env   # owner-only read

---

## 6. systemd service

    cp /opt/stock-screener/deploy/stock-screener.service \
       /etc/systemd/system/stock-screener.service

    systemctl daemon-reload
    systemctl enable --now stock-screener
    systemctl status stock-screener    # should be 'active (running)'
    journalctl -u stock-screener -f    # follow logs

Quick sanity check:

    curl -s http://127.0.0.1:8000/api/config | head -c 200

---

## 6. nginx + TLS

    cp /opt/stock-screener/deploy/nginx.conf \
       /etc/nginx/sites-available/stock.modulgroup.uz
    ln -sf /etc/nginx/sites-available/stock.modulgroup.uz \
           /etc/nginx/sites-enabled/stock.modulgroup.uz
    nginx -t
    systemctl reload nginx

Issue the TLS certificate (requires the DNS A record to be live):

    certbot --nginx -d stock.modulgroup.uz \
            --agree-tos -m you@example.com --redirect

certbot auto-rewrites the nginx config to add the 443 block and set up a
renewal timer. Verify:

    systemctl list-timers | grep certbot

---

## 7. Visit

    https://stock.modulgroup.uz

---

## Updating / deploying a new version

    cd /opt/stock-screener
    sudo -u www-data git pull
    sudo -u www-data ./venv/bin/pip install -r requirements.txt
    systemctl restart stock-screener

---

## Troubleshooting

- **502 from nginx**: `systemctl status stock-screener` →
  `journalctl -u stock-screener -n 100`
- **Slow first request**: FinBERT is loading (~30 s cold start). Gunicorn
  config sets `preload_app = True` so subsequent requests are warm.
- **Rate-limited by yfinance/SEC**: Two-layer cache.
  1. `cache.py` — 10-min in-memory TTL on yfinance.Ticker + SEC/news helpers
     (per-process, lost on restart).
  2. `db_cache.py` — persistent SQLite (default `cache/api_cache.db`) for the
     6 main API endpoints. TTLs: valuation 6 h, screen* 1 h, insider 12 h,
     superinvestors 24 h. Clients can bypass with `?fresh=1`.
  - Inspect:    `curl http://127.0.0.1:8000/api/cache/stats`
  - Purge old:  `curl -XPOST http://127.0.0.1:8000/api/cache/purge`
  - Wipe one:   `curl -XPOST -H 'Content-Type: application/json' \
                   -d '{"ticker":"AAPL"}' http://127.0.0.1:8000/api/cache/clear`
  - For heavier traffic switch to redis-backed caching (drop-in replacement
    in `db_cache.py`).
- **`/api/cache/stats` shows `db_path` outside `/opt/stock-screener`**:
  Set `TRADING_CACHE_DB=/opt/stock-screener/cache/api_cache.db` in `.env`
  and ensure the dir is writable: `mkdir -p /opt/stock-screener/cache &&
  chown www-data:www-data /opt/stock-screener/cache`.
- **Out of memory**: `workers = 1` in `gunicorn.conf.py`, or drop
  `transformers`/`torch` from requirements.
