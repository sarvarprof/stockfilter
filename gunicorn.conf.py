"""
gunicorn.conf.py — Production WSGI config.

Run:
    gunicorn -c gunicorn.conf.py app:app
"""

import multiprocessing

# Network
bind = "127.0.0.1:8000"   # nginx proxies to this

# Workers
# FinBERT pipelines are heavy (~1 GB RSS each). Keep workers small.
# 2 workers is a good default for a 2-4 GB VPS.
workers = 2
worker_class = "sync"
threads = 2
timeout = 120            # yfinance+SEC calls can be slow
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "-"          # stdout (systemd journal)
errorlog = "-"
loglevel = "info"

# Misc
proc_name = "stock-screener"
preload_app = True       # load FinBERT once before forking
