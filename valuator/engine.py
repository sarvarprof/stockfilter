"""
Valuation engine.

Methodological fixes vs. the original:

  • DCF, Monte Carlo, and reverse DCF now all use the same 10-year horizon
    with a 5-year fade so results are directly comparable
  • Monte Carlo uses lognormal growth (bounded below by -100%) instead of
    raw normal sampling
  • g and r are sampled with positive correlation (~0.4): companies in
    higher-growth regimes tend to face higher discount rates
  • Buffett method discounts terminal value at cost of equity, not WACC
    (it's an equity-holder framework)
  • Verdict bands scale by Monte Carlo dispersion, not arbitrary ±30%
  • Sensitivity table for (WACC × terminal_g) is included in the report
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from data import TickerData, _row
from assumptions import Assumptions

log = logging.getLogger(__name__)


# ─── Two-stage DCF: 5y explicit growth → 5y linear fade → terminal ─────────

def two_stage_dcf(
    fcf0: float,
    growth_rate: float,
    wacc: float,
    terminal_g: float,
    years: int = 10,
    fade_start: int = 5,
    shares: float = 1.0,
) -> float | None:
    """
    Standard two-stage DCF, returns per-share intrinsic value.

    Years 1..fade_start: grow at `growth_rate`
    Years fade_start+1..years: linearly fade from `growth_rate` to `terminal_g`
    Plus a Gordon terminal value at year `years`.
    """
    if fcf0 <= 0 or wacc <= terminal_g or shares <= 0:
        return None

    pv = 0.0
    fcf = fcf0

    for t in range(1, fade_start + 1):
        fcf *= (1 + growth_rate)
        pv += fcf / (1 + wacc) ** t

    fade_years = years - fade_start
    for i in range(1, fade_years + 1):
        alpha = i / fade_years
        g = growth_rate * (1 - alpha) + terminal_g * alpha
        fcf *= (1 + g)
        pv += fcf / (1 + wacc) ** (fade_start + i)

    tv = fcf * (1 + terminal_g) / (wacc - terminal_g)
    pv += tv / (1 + wacc) ** years
    return pv / shares


# ─── Monte Carlo: lognormal growth, correlated rates, 10-year horizon ──────

def monte_carlo_dcf(
    fcf0: float,
    shares: float,
    growth_mean: float,
    growth_sigma: float,
    wacc_mean: float,
    wacc_sigma: float,
    terminal_g: float,
    iterations: int = 5000,
    correlation: float = 0.4,
    years: int = 10,
    fade_start: int = 5,
    seed: int | None = None,
) -> dict:
    """
    Monte Carlo DCF with proper distributional choices.

    - Growth is sampled as ln(1+g) ~ N, so g > -1 always (lognormal).
    - WACC and growth are correlated via Cholesky decomposition; positive
      correlation reflects the empirical link between growth regimes and
      discount rates (high growth → higher rates).
    - Horizon matches the deterministic DCF (10y with 5y fade) for
      comparability.
    """
    if fcf0 <= 0 or shares <= 0:
        return {"error": "Invalid inputs"}

    rng = np.random.default_rng(seed)

    # Build correlated draws via Cholesky
    cov = np.array([[1.0, correlation], [correlation, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((iterations, 2))
    z = z @ L.T

    # Lognormal growth: log(1+g) ~ N(mu, sigma)
    mu_g = np.log(1 + growth_mean)
    sigma_g = growth_sigma
    growth_draws = np.exp(mu_g + sigma_g * z[:, 0]) - 1

    # WACC: clip to keep positive and above terminal_g
    wacc_draws = wacc_mean + wacc_sigma * z[:, 1]
    wacc_draws = np.clip(wacc_draws, terminal_g + 0.005, 0.30)

    fade_years = years - fade_start
    values = np.empty(iterations)

    for k in range(iterations):
        g, r = growth_draws[k], wacc_draws[k]
        pv = 0.0
        fcf = fcf0
        for t in range(1, fade_start + 1):
            fcf *= (1 + g)
            pv += fcf / (1 + r) ** t
        for i in range(1, fade_years + 1):
            alpha = i / fade_years
            year_g = g * (1 - alpha) + terminal_g * alpha
            fcf *= (1 + year_g)
            pv += fcf / (1 + r) ** (fade_start + i)
        tv = fcf * (1 + terminal_g) / (r - terminal_g)
        pv += tv / (1 + r) ** years
        values[k] = pv / shares

    arr = values
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "p05":  float(np.percentile(arr, 5)),
        "p10":  float(np.percentile(arr, 10)),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p90":  float(np.percentile(arr, 90)),
        "p95":  float(np.percentile(arr, 95)),
    }


# ─── Buffett method: discount at cost of EQUITY, not WACC ──────────────────

def buffett_valuation(
    eps0: float,
    growth_rate: float,
    terminal_g: float,
    cost_of_equity: float,
    future_pe: float,
    margin_of_safety: float,
    years: int = 10,
) -> tuple[float, float] | tuple[None, None]:
    """
    Project EPS with linear growth fade, apply exit P/E, discount to PV
    at the cost of equity (not WACC — this is an equity-holder framework).
    """
    if eps0 <= 0 or cost_of_equity <= terminal_g:
        return None, None

    eps = eps0
    for t in range(1, years + 1):
        alpha = (t - 1) / max(years - 1, 1)
        g = growth_rate * (1 - alpha) + terminal_g * alpha
        eps *= (1 + g)

    iv = (eps * future_pe) / (1 + cost_of_equity) ** years
    buy = iv * (1 - margin_of_safety)
    return iv, buy


# ─── GGM, Graham, EV/EBITDA, Reverse DCF ───────────────────────────────────

def gordon_growth(div: float, div_growth: float, cost_of_equity: float) -> float | str:
    if div <= 0:
        return "N/A (No Dividend)"
    if cost_of_equity <= div_growth:
        return "N/A (g > r)"
    return div * (1 + div_growth) / (cost_of_equity - div_growth)


def graham_number(eps: float, bvps: float) -> float | None:
    if eps <= 0 or bvps <= 0:
        return None
    return (22.5 * eps * bvps) ** 0.5


def ev_ebitda_value(
    ebitda: float,
    multiple: float,
    debt: float,
    cash: float,
    shares: float,
) -> float | None:
    if ebitda <= 0 or shares <= 0:
        return None
    equity_value = multiple * ebitda - debt + cash
    if equity_value <= 0:
        return None
    return equity_value / shares


def monte_carlo_from_posterior(
    fcf0: float,
    shares: float,
    posterior,                  # GrowthPosterior from growth_pool
    wacc_mean: float,
    wacc_sigma: float,
    terminal_g: float,
    iterations: int = 5000,
    correlation: float = 0.4,
    years: int = 10,
    fade_start: int = 5,
    seed: int | None = None,
) -> dict:
    """
    MC using the Bayesian posterior instead of point (mean, sigma).
    Draws growth from the posterior distribution directly.
    """
    return monte_carlo_dcf(
        fcf0=fcf0,
        shares=shares,
        growth_mean=posterior.mu,
        growth_sigma=posterior.sigma,
        wacc_mean=wacc_mean,
        wacc_sigma=wacc_sigma,
        terminal_g=terminal_g,
        iterations=iterations,
        correlation=correlation,
        years=years,
        fade_start=fade_start,
        seed=seed,
    )


def reverse_dcf(
    fcf0: float,
    shares: float,
    current_price: float,
    wacc: float,
    terminal_g: float,
    years: int = 10,
    fade_start: int = 5,
) -> float | None:
    """Binary-search the FCF growth rate that justifies the current price."""
    if fcf0 <= 0 or shares <= 0 or current_price <= 0:
        return None

    def fv(g):
        v = two_stage_dcf(fcf0, g, wacc, terminal_g, years, fade_start, shares)
        return v if v is not None else -1e18

    lo, hi = -0.30, 1.50
    if fv(lo) > current_price or fv(hi) < current_price:
        return None  # price not bracketed
    for _ in range(80):
        mid = (lo + hi) / 2
        if fv(mid) < current_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─── Sensitivity table ─────────────────────────────────────────────────────

def sensitivity_table(
    fcf0: float,
    shares: float,
    growth_rate: float,
    wacc_base: float,
    terminal_g_base: float,
    wacc_offsets: tuple = (-0.02, -0.01, 0.0, 0.01, 0.02),
    g_offsets: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01),
    years: int = 10,
    fade_start: int = 5,
) -> pd.DataFrame:
    """
    2D sensitivity grid of intrinsic value vs. (WACC, terminal_g).
    Rows are WACC, columns are terminal_g.
    """
    rows = {}
    for dw in wacc_offsets:
        wacc = wacc_base + dw
        row = {}
        for dg in g_offsets:
            tg = terminal_g_base + dg
            if wacc <= tg:
                row[round(tg, 4)] = float("nan")
                continue
            v = two_stage_dcf(fcf0, growth_rate, wacc, tg, years, fade_start, shares)
            row[round(tg, 4)] = round(v, 2) if v is not None else float("nan")
        rows[round(wacc, 4)] = row
    df = pd.DataFrame(rows).T
    df.index.name = "WACC"
    df.columns.name = "terminal_g"
    return df


# ─── Verdict bands scaled by MC dispersion ─────────────────────────────────

def adaptive_verdict(
    intrinsic: float | None,
    current_price: float,
    mc_p25: float | None = None,
    mc_p75: float | None = None,
    mc_p50: float | None = None,
    use_mc_distribution: bool = False,
) -> str:
    """
    Verdict bands. By default, uses scaled ratio bands (works for any method).

    When use_mc_distribution=True AND the MC quartiles are provided, compares
    the price directly against the MC distribution. This is only appropriate
    for methods that are themselves DCFs — applying it to GGM, Graham, or
    EV/EBITDA would compare apples to oranges (they're on different scales
    and use different inputs).
    """
    if intrinsic is None or current_price <= 0:
        return "N/A"

    if use_mc_distribution and None not in (mc_p25, mc_p50, mc_p75):
        if current_price < mc_p25:
            return "Undervalued"
        if current_price < mc_p50:
            return "Slight Upside"
        if current_price <= mc_p75:
            return "Fairly Valued"
        return "Overvalued"

    # Default: ratio bands against the method's own intrinsic value
    r = intrinsic / current_price
    if r >= 1.30:  return "Undervalued"
    if r >= 1.10:  return "Slight Upside"
    if r >= 0.90:  return "Fairly Valued"
    if r >= 0.70:  return "Slight Downside"
    return "Overvalued"


# ─── Dividend growth helper ────────────────────────────────────────────────

def historical_div_growth(div_history: pd.Series | None, default: float = 0.03) -> float:
    if div_history is None or len(div_history) < 8:
        return default
    annual = div_history.resample("YE").sum()
    annual = annual[annual > 0].dropna()
    if len(annual) < 3:
        return default
    n = len(annual) - 1
    start, end = float(annual.iloc[0]), float(annual.iloc[-1])
    if start <= 0:
        return default
    g = (end / start) ** (1 / n) - 1
    return float(np.clip(g, 0.0, 0.15))
