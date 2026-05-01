"""
app.py — Flask web UI for the unified stock screener.

Endpoints:
    GET  /                → single-page UI
    GET  /api/config      → dropdown options (profiles, sectors, etc.)
    POST /api/screen      → run one framework on one ticker
    POST /api/screen-all  → run ALL three frameworks (compare view)

Run locally for preview:
    python app.py
then open http://127.0.0.1:5000/
"""

from __future__ import annotations

import hashlib
import os
import sys as _sys
import traceback
from datetime import timedelta
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

# Load .env into os.environ before any module reads env vars (FRED_API_KEY, etc.)
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, ".env"))
except ImportError:
    pass  # dotenv is optional; systemd EnvironmentFile= can also supply vars

import penny_filter
import growth_filter
import value_filter
import screener
import cache as ticker_cache
import insider as insider_mod
import db_cache

# darkpool/optionsflow lazy-import matplotlib only inside their CLI dashboard
# functions, so the web API path doesn't require matplotlib/seaborn.
import darkpool
import optionsflow
import io
import pandas as pd

_sys.path.insert(0, os.path.join(HERE, "valuator"))
from valuator import StockValuator

# Install the TTL cache (10 min) before any evaluator runs.
# Wraps yfinance.Ticker + the SEC/news helpers.
ticker_cache.install(ttl_seconds=600)

# ── Persistent cache TTLs (seconds) ───────────────────────────────────────
TTL_VALUATION      = 6  * 3600   # fundamentals stable intraday
TTL_SCREEN         = 1  * 3600   # news sentiment is the freshness driver
TTL_INSIDER        = 12 * 3600   # Form 4 — refresh twice/day
TTL_SUPERINVESTORS = 24 * 3600   # 13F filed quarterly


def _bypass_cache() -> bool:
    """Return True if the request explicitly asked to bypass the cache."""
    qf = (request.args.get("fresh") or "").lower()
    if qf in ("1", "true", "yes"):
        return True
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if body.get("fresh") in (True, 1, "1", "true", "yes"):
            return True
    return False


app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "templates"),
    static_folder=os.path.join(HERE, "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

_SECRET = os.getenv("SECRET_KEY")
if not _SECRET:
    # Dev-only fallback: ephemeral random key. Sessions don't survive restarts.
    # Production MUST set SECRET_KEY in .env.
    import secrets as _secrets
    _SECRET = _secrets.token_hex(32)
    import warnings as _warn
    _warn.warn("SECRET_KEY not set — using ephemeral key. Set in .env for production.")
app.secret_key = _SECRET
app.permanent_session_lifetime = timedelta(days=30)

_PASSWORD_HASH = hashlib.sha256(b"tradingssF_").hexdigest()


@app.before_request
def require_login():
    if request.endpoint in ("login", "logout", "static"):
        return
    if not session.get("authenticated"):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        pw = request.form.get("password", "")
        remember = bool(request.form.get("remember"))
        if hashlib.sha256(pw.encode()).hexdigest() == _PASSWORD_HASH:
            session.permanent = remember
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Incorrect password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Config exposed to the frontend (drives the dropdowns)
# ---------------------------------------------------------------------------

FRAMEWORK_META = {
    "penny":  {
        "label":    "Penny / Micro-cap",
        "cap_hint": f"< ${screener.PENNY_MAX/1e6:.0f}M",
        "profiles": list(penny_filter.WEIGHT_PROFILES.keys()),
        "sectors":  list(penny_filter.SECTOR_OVERLAYS.keys()),
    },
    "growth": {
        "label":    "Growth / Small-Mid-cap",
        "cap_hint": f"${screener.PENNY_MAX/1e6:.0f}M – ${screener.GROWTH_MAX/1e9:.0f}B",
        "profiles": list(growth_filter.WEIGHT_PROFILES.keys()),
        "sectors":  list(growth_filter.SECTOR_OVERLAYS.keys()),
    },
    "value":  {
        "label":    "Value / Large-cap",
        "cap_hint": f"≥ ${screener.GROWTH_MAX/1e9:.0f}B",
        "profiles": list(value_filter.WEIGHT_PROFILES.keys()),
        "sectors":  list(value_filter.SECTOR_OVERLAYS.keys()),
    },
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    return jsonify({
        "frameworks":    FRAMEWORK_META,
        "scoring_modes": ["strict", "normal", "generous"],
        "engines":       ["auto", "keyword", "finbert", "claude"],
        "langs":         ["en", "uz"],
        "bands": {
            "penny_max":  screener.PENNY_MAX,
            "growth_max": screener.GROWTH_MAX,
        },
    })


def _build_kwargs(payload):
    """Translate frontend payload into screen() kwargs."""
    return dict(
        framework=     payload.get("framework", "auto"),
        profile=       payload.get("profile", "balanced"),
        sector=        payload.get("sector", "auto"),
        scoring_mode=  payload.get("scoring_mode", "normal"),
        apply_bonuses= bool(payload.get("apply_bonuses", True)),
        verbose=       False,
        lang=          payload.get("lang", "en"),
        engine=        payload.get("engine", "auto"),
        news_days=     payload.get("news_days") or None,
        filing_days=   payload.get("filing_days") or None,
    )


@app.route("/api/screen", methods=["POST"])
def api_screen():
    payload = request.get_json(force=True) or {}
    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        kwargs = _build_kwargs(payload)
        if not _bypass_cache():
            cached = db_cache.get("screen", ticker, kwargs)
            if cached is not None:
                return jsonify({"ok": True, "data": cached, "cached": True})
        result = screener.screen(ticker, **kwargs)
        db_cache.set("screen", ticker, kwargs, result, TTL_SCREEN)
        return jsonify({"ok": True, "data": result, "cached": False})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


@app.route("/api/compare-profiles", methods=["POST"])
def api_compare_profiles():
    payload = request.get_json(force=True) or {}
    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        kwargs = _build_kwargs(payload)
        framework = kwargs.pop("framework", "auto")
        kwargs.pop("profile", None)
        cache_params = {"framework": framework, **kwargs}
        if not _bypass_cache():
            cached = db_cache.get("compare-profiles", ticker, cache_params)
            if cached is not None:
                return jsonify({"ok": True, "data": cached, "cached": True,
                                "cache": ticker_cache.stats()})
        result = screener.compare_all_profiles(ticker, framework=framework,
                                               **kwargs)
        db_cache.set("compare-profiles", ticker, cache_params, result, TTL_SCREEN)
        return jsonify({"ok": True, "data": result, "cached": False,
                        "cache": ticker_cache.stats()})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


@app.route("/api/superinvestors/<ticker>")
def api_superinvestors(ticker):
    ticker = ticker.upper().strip()
    try:
        if not _bypass_cache():
            cached = db_cache.get("superinvestors", ticker, None)
            if cached is not None:
                return jsonify({"ok": True, "data": cached, "cached": True})
        result = insider_mod.get_superinvestors(ticker)
        db_cache.set("superinvestors", ticker, None, result, TTL_SUPERINVESTORS)
        return jsonify({"ok": True, "data": result, "cached": False})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/insider/<ticker>")
def api_insider(ticker):
    ticker = ticker.upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        if not _bypass_cache():
            cached = db_cache.get("insider", ticker, None)
            if cached is not None:
                return jsonify({"ok": True, "data": cached, "cached": True})
        result = insider_mod.get_insider_summary(ticker)
        db_cache.set("insider", ticker, None, result, TTL_INSIDER)
        return jsonify({"ok": True, "data": result, "cached": False})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


_SECTOR_PE = {
    "Technology": 28, "Healthcare": 22, "Consumer Cyclical": 20,
    "Financial Services": 13, "Utilities": 16, "Real Estate": 30,
    "Consumer Defensive": 18, "Industrials": 18, "Basic Materials": 14,
    "Energy": 12, "Communication Services": 22,
}
_SECTOR_EV = {
    "Technology": 20, "Healthcare": 18, "Utilities": 10,
    "Energy": 7, "Financial Services": 10, "Real Estate": 14,
}
_SECTOR_PB = {
    "Financial Services": 1.3, "Real Estate": 1.5, "Utilities": 1.5,
}


def _simple_valuation(ticker: str) -> dict | None:
    """
    Fallback: yfinance-only fair-value estimates when the full EDGAR/FRED
    valuator fails. Returns a dict in the same shape as StockValuator.report()
    so renderValuation() in the frontend works without changes.
    """
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None

    price = (info.get("currentPrice") or info.get("regularMarketPrice")
             or info.get("navPrice") or 0)
    if not price:
        return None

    sector   = info.get("sector") or ""
    pe_tgt   = _SECTOR_PE.get(sector, 16)
    ev_tgt   = _SECTOR_EV.get(sector, 12)
    pb_tgt   = _SECTOR_PB.get(sector, 2.5)

    def _v(fair, p):
        if not fair or not p:
            return "N/A"
        r = fair / p
        return "Undervalued" if r >= 1.20 else ("Fairly Valued" if r >= 0.90 else "Overvalued")

    methods: dict = {}

    # 1. Trailing P/E
    eps = info.get("trailingEps")
    if eps and eps > 0:
        fair = round(eps * pe_tgt, 2)
        methods["pe_trailing"] = {
            "name": f"Trailing P/E ({pe_tgt}×)",
            "value": fair, "verdict": _v(fair, price),
            "reliability": "Medium",
            "note": f"Trailing EPS ${eps:.2f} × {pe_tgt}× sector P/E",
        }

    # 2. Forward P/E
    fwd_eps = info.get("forwardEps")
    if fwd_eps and fwd_eps > 0:
        fair = round(fwd_eps * pe_tgt, 2)
        methods["pe_forward"] = {
            "name": f"Forward P/E ({pe_tgt}×)",
            "value": fair, "verdict": _v(fair, price),
            "reliability": "Medium",
            "note": f"Forward EPS ${fwd_eps:.2f} × {pe_tgt}×",
        }

    # 3. Analyst consensus target
    target     = info.get("targetMeanPrice") or info.get("targetMedianPrice")
    n_analysts = info.get("numberOfAnalystOpinions") or 0
    if target and target > 0:
        methods["analyst_target"] = {
            "name": "Analyst Target (mean)",
            "value": round(target, 2), "verdict": _v(target, price),
            "reliability": "Medium" if n_analysts >= 5 else "Low",
            "note": f"{n_analysts} analyst{'s' if n_analysts != 1 else ''}",
        }

    # 4. Price / Book
    book = info.get("bookValue")
    if book and book > 0:
        fair = round(book * pb_tgt, 2)
        methods["price_book"] = {
            "name": f"Price/Book ({pb_tgt}×)",
            "value": fair, "verdict": _v(fair, price),
            "reliability": "Low",
            "note": f"Book ${book:.2f} × {pb_tgt}×",
        }

    # 5. EV / EBITDA
    ebitda = info.get("ebitda")
    shares = info.get("sharesOutstanding")
    debt   = info.get("totalDebt") or 0
    cash   = info.get("totalCash") or 0
    if ebitda and ebitda > 0 and shares and shares > 0:
        equity_val = ebitda * ev_tgt - debt + cash
        fair = round(equity_val / shares, 2) if equity_val > 0 else None
        if fair:
            methods["ev_ebitda"] = {
                "name": f"EV/EBITDA ({ev_tgt}×)",
                "value": fair, "verdict": _v(fair, price),
                "reliability": "Medium",
                "note": f"EBITDA × {ev_tgt}× − net debt per share",
            }

    if not methods:
        return None

    verdicts = [m["verdict"] for m in methods.values() if m["verdict"] != "N/A"]
    n_u = sum(1 for v in verdicts if v == "Undervalued")
    n_f = sum(1 for v in verdicts if v == "Fairly Valued")
    n_o = sum(1 for v in verdicts if v == "Overvalued")
    consensus = ("Undervalued"   if n_u > n_o and n_u >= n_f else
                 "Overvalued"    if n_o > n_u and n_o > n_f  else
                 "Fairly Valued")

    return {
        "current_price":    price,
        "consensus":        consensus,
        "consensus_counts": {"undervalued": n_u, "fairly_valued": n_f, "overvalued": n_o},
        "methods":          methods,
        "assumptions":      {},
        "macro":            {},
        "data_warnings":    ["Simple estimates only — EDGAR/FRED valuation unavailable for this ticker"],
        "assumption_notes": [],
        "simple_mode":      True,
    }


@app.route("/api/valuation/<ticker>")
def api_valuation(ticker):
    ticker = ticker.upper().strip()
    try:
        if not _bypass_cache():
            cached = db_cache.get("valuation", ticker, None)
            if cached is not None:
                return jsonify({"ok": True, "data": cached, "cached": True})

        if not os.environ.get("EDGAR_USER_AGENT"):
            os.environ["EDGAR_USER_AGENT"] = "stockscreener saidrakhmonov94@gmail.com"

        rep = None
        primary_err = None
        try:
            v = StockValuator(ticker)
            rep = v.report()
            # Serialize the pandas DataFrame sensitivity table to JSON
            sens_df = rep.pop("sensitivity", None)
            if sens_df is not None and not sens_df.empty:
                rep["sensitivity"] = {
                    "wacc_labels": [f"{r:.2%}" for r in sens_df.index],
                    "tg_labels":   [f"{c:.2%}" for c in sens_df.columns],
                    "values":      sens_df.round(2).values.tolist(),
                }
        except Exception as e:
            primary_err = e

        if rep is None:
            rep = _simple_valuation(ticker)
            if rep is None:
                return jsonify({
                    "ok": False,
                    "error": str(primary_err) if primary_err else "No data available",
                }), 500
            if primary_err:
                rep["data_warnings"] = (
                    [f"Full valuation error: {primary_err}"] +
                    rep.get("data_warnings", [])
                )

        db_cache.set("valuation", ticker, None, rep, TTL_VALUATION)
        return jsonify({"ok": True, "data": rep, "cached": False})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear():
    payload = request.get_json(silent=True) or {}
    ticker_cache.clear()
    deleted = 0
    if payload.get("persistent", True):
        ticker_arg = (payload.get("ticker") or "").strip() or None
        ns_arg     = (payload.get("namespace") or "").strip() or None
        deleted = db_cache.invalidate(namespace=ns_arg, ticker=ticker_arg)
    return jsonify({"ok": True,
                    "memory": ticker_cache.stats(),
                    "persistent_deleted": deleted,
                    "persistent": db_cache.stats()})


@app.route("/api/cache/stats")
def api_cache_stats():
    return jsonify({"memory": ticker_cache.stats(),
                    "persistent": db_cache.stats()})


@app.route("/api/cache/purge", methods=["POST"])
def api_cache_purge():
    """Delete only expired rows from the persistent cache."""
    n = db_cache.purge_expired()
    return jsonify({"ok": True, "purged": n, "persistent": db_cache.stats()})


@app.route("/api/screen-all", methods=["POST"])
def api_screen_all():
    payload = request.get_json(force=True) or {}
    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        kwargs = _build_kwargs(payload)
        kwargs.pop("framework", None)  # screen_all runs all three
        if not _bypass_cache():
            cached = db_cache.get("screen-all", ticker, kwargs)
            if cached is not None:
                return jsonify({"ok": True, "data": cached, "cached": True})
        result = screener.screen_all(ticker, **kwargs)
        db_cache.set("screen-all", ticker, kwargs, result, TTL_SCREEN)
        return jsonify({"ok": True, "data": result, "cached": False})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


# ---------------------------------------------------------------------------
# Dark Pool & Options Flow analyzers — CSV upload endpoints
# ---------------------------------------------------------------------------

# Required CSV columns for each analyzer. We validate before handing off so
# the user gets a clear error rather than a deep stack trace.
_DARKPOOL_COLS = {"date", "time", "symbol", "size", "price", "premium"}
_OPTIONS_COLS  = {"date", "time", "symbol", "expiry", "strike", "put_call",
                  "side", "spot", "size", "price", "premium",
                  "sweep_block_split", "volume", "open_int", "conds"}


def _read_uploaded_csv(field_name: str) -> pd.DataFrame:
    """Read a CSV file from a multipart upload field. Raises ValueError on issues."""
    if field_name not in request.files:
        raise ValueError(f"No file uploaded under field '{field_name}'")
    f = request.files[field_name]
    if not f or f.filename == "":
        raise ValueError("Empty file upload")
    if not f.filename.lower().endswith(".csv"):
        raise ValueError("File must be a .csv")
    raw = f.read()
    if not raw:
        raise ValueError("Uploaded file is empty")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")


@app.route("/api/darkpool", methods=["POST"])
def api_darkpool():
    try:
        df = _read_uploaded_csv("file")
        missing = _DARKPOOL_COLS - set(df.columns)
        if missing:
            return jsonify({
                "ok": False,
                "error": f"CSV missing required columns: {sorted(missing)}",
            }), 400
        result = darkpool.analyze_for_api(df)
        return jsonify({"ok": True, "data": result})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


@app.route("/api/optionsflow", methods=["POST"])
def api_optionsflow():
    try:
        df = _read_uploaded_csv("file")
        missing = _OPTIONS_COLS - set(df.columns)
        if missing:
            return jsonify({
                "ok": False,
                "error": f"CSV missing required columns: {sorted(missing)}",
            }), 400
        result = optionsflow.analyze_for_api(df)
        return jsonify({"ok": True, "data": result})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


if __name__ == "__main__":
    # Preview-only: bind to localhost, debug off so the Jinja template is
    # loaded once and errors surface as JSON.
    app.run(host="127.0.0.1", port=5000, debug=False)
