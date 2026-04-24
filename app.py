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
import traceback
from datetime import timedelta
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import penny_filter
import growth_filter
import value_filter
import screener
import cache as ticker_cache
import insider as insider_mod

# Install the TTL cache (10 min) before any evaluator runs.
# Wraps yfinance.Ticker + the SEC/news helpers.
ticker_cache.install(ttl_seconds=600)


HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "templates"),
    static_folder=os.path.join(HERE, "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

app.secret_key = os.getenv("SECRET_KEY", "stk-scr-s3cr3t-xk8m3p9r2v5w1z4")
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
        result = screener.screen(ticker, **kwargs)
        return jsonify({"ok": True, "data": result})
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
        result = screener.compare_all_profiles(ticker, framework=framework,
                                               **kwargs)
        return jsonify({"ok": True, "data": result,
                        "cache": ticker_cache.stats()})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


@app.route("/api/insider/<ticker>")
def api_insider(ticker):
    ticker = ticker.upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        result = insider_mod.get_insider_summary(ticker)
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear():
    ticker_cache.clear()
    return jsonify({"ok": True, "cache": ticker_cache.stats()})


@app.route("/api/cache/stats")
def api_cache_stats():
    return jsonify(ticker_cache.stats())


@app.route("/api/screen-all", methods=["POST"])
def api_screen_all():
    payload = request.get_json(force=True) or {}
    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        kwargs = _build_kwargs(payload)
        kwargs.pop("framework", None)  # screen_all runs all three
        result = screener.screen_all(ticker, **kwargs)
        return jsonify({"ok": True, "data": result})
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
