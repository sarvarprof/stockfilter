"""
growth_pool.py — Bayesian growth-rate pooling (Stage 3).

Each growth-rate source produces a (mean, sigma) estimate.
We combine them using inverse-variance weighting to get a posterior
(mu, sigma) that drives the Monte Carlo instead of a point estimate.

Sources and their uncertainty models:

  1. SGR = ROE × retention
       sigma from bootstrap-resampling historical ROE over available years.
       Wide prior (sigma_floor=0.04) since retention policy can change.

  2. ROIC × reinvestment rate
       sigma from bootstrap over available ROIC observations.
       Moderate prior (sigma_floor=0.03).

  3. Historical EPS CAGR
       sigma from jackknife (drop-one) resampling of the EPS series.
       Narrow prior (sigma_floor=0.02) — actual history is the tightest signal.

  4. Historical FCF CAGR (log returns)
       sigma = std(log returns) / sqrt(n-1) — standard error of the mean.
       Moderate prior (sigma_floor=0.025).

  5. Analyst consensus
       sigma from implied dispersion: analysts disagree by ~30% of their
       own estimate on average (empirical from I/B/E/S studies). So
       sigma = 0.30 × |mean_analyst_g|, floored at 0.03.

Posterior:
  mu_post = sum(mu_i / sigma_i^2) / sum(1 / sigma_i^2)
  sigma_post = 1 / sqrt(sum(1 / sigma_i^2))

This is a conjugate Gaussian update — conceptually, each source is a
noisy measurement of the "true" long-run growth rate, and we weight them
by their precision (1/sigma^2).

References:
  Gelman et al. (2013) BDA3 §2.2 — normal-normal conjugate update
  Bradshaw (2011) — analyst forecast dispersion ≈ 30% of estimate
  Damodaran (2012) Investment Valuation, Ch. 11 — growth estimation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import TickerData
from assumptions import cagr, winsorize_pct_change

log = logging.getLogger(__name__)


# ─── Per-source distribution ───────────────────────────────────────────────

@dataclass
class GrowthSource:
    name: str
    mean: float
    sigma: float
    weight: float = 0.0        # filled in by the pooler
    precision: float = 0.0     # 1/sigma^2, filled in by the pooler
    available: bool = True

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive (got {self.sigma} for {self.name})")
        self.precision = 1.0 / (self.sigma ** 2)


@dataclass
class GrowthPosterior:
    """Posterior distribution over the true growth rate."""
    mu: float
    sigma: float
    sources: list[GrowthSource] = field(default_factory=list)
    n_sources: int = 0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n samples from the posterior (normal approximation)."""
        return rng.normal(self.mu, self.sigma, size=n)


# ─── Bootstrap helpers ─────────────────────────────────────────────────────

def _bootstrap_cagr_sigma(series: pd.Series, n_boot: int = 500,
                           rng: np.random.Generator | None = None) -> float:
    """
    Bootstrap standard deviation of the CAGR estimate.
    Each resample draws len(series) observations with replacement, then
    computes CAGR from the resulting pseudo-series' min and max index.
    This approximates the sampling distribution of the CAGR.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    s = series.dropna().sort_index()
    s = s[s > 0]
    if len(s) < 3:
        return 0.10  # not enough data
    vals = s.values
    n = len(vals)
    cadrs = []
    for _ in range(n_boot):
        boot = rng.choice(vals, size=n, replace=True)
        boot_s = np.sort(boot)
        g = (boot_s[-1] / boot_s[0]) ** (1.0 / (n - 1)) - 1.0
        if -0.5 < g < 1.5:
            cadrs.append(g)
    if len(cadrs) < 10:
        return 0.10
    return float(np.std(cadrs))


def _jackknife_cagr_sigma(series: pd.Series) -> float:
    """
    Jackknife (leave-one-out) standard error of the CAGR.
    More stable than bootstrap for small samples.
    """
    s = series.dropna().sort_index()
    s = s[s > 0]
    n = len(s)
    if n < 4:
        return 0.08
    full_cagr = cagr(s)
    if full_cagr is None:
        return 0.08
    jk_cagrs = []
    for i in range(n):
        sub = s.drop(s.index[i])
        g = cagr(sub)
        if g is not None and -0.5 < g < 1.5:
            jk_cagrs.append(g)
    if len(jk_cagrs) < 3:
        return 0.08
    # Jackknife variance estimator: (n-1)/n * sum((g_i - g_mean)^2)
    g_mean = np.mean(jk_cagrs)
    jk_var = (n - 1) / n * np.sum((np.array(jk_cagrs) - g_mean) ** 2)
    return float(np.sqrt(max(jk_var, 1e-6)))


def _fcf_log_return_sigma(fcf: pd.Series) -> float:
    """Standard error of mean log-return: sigma_lr / sqrt(n-1)."""
    lr = winsorize_pct_change(fcf)
    if len(lr) < 2:
        return 0.10
    return float(lr.std() / np.sqrt(max(len(lr) - 1, 1)))


# ─── Source builders ────────────────────────────────────────────────────────

def _source_sgr(roe: float | None, payout: float | None,
                td: TickerData) -> GrowthSource | None:
    """SGR = ROE × (1 - payout). Sigma from bootstrap on historical ROE."""
    if roe is None:
        return None
    payout = max(0.0, min(float(payout or 0.0), 1.0))
    sgr = float(roe) * (1.0 - payout)
    sgr = float(np.clip(sgr, 0.01, 0.20))

    # Estimate sigma from variation in ROE over time
    # Use net_income / equity if available, else just widen the prior
    sigma = 0.05  # default: SGR is a noisy signal
    if td.income_stmt is not None and td.balance_sheet is not None:
        from data import _row
        ni = _row(td.income_stmt, "net_income")
        eq = _row(td.balance_sheet, "stockholders_equity")
        if ni is not None and eq is not None:
            common = ni.index.intersection(eq.index)
            if len(common) >= 3:
                roe_series = (ni.loc[common] / eq.loc[common]).dropna()
                roe_series = roe_series[(roe_series > -0.5) & (roe_series < 0.5)]
                if len(roe_series) >= 3:
                    sigma = float(np.clip(roe_series.std() * (1 - payout), 0.02, 0.12))
    return GrowthSource(name="SGR (ROE×retention)", mean=sgr, sigma=max(sigma, 0.04))


def _source_roic_reinvest(roic: float | None,
                           g_roic: float | None,
                           td: TickerData) -> GrowthSource | None:
    """ROIC × reinvestment. Sigma from variation in ROIC over time."""
    if roic is None or g_roic is None:
        return None
    g = float(np.clip(g_roic, 0.0, 0.30))

    # Sigma: variation in ROIC over time
    sigma = 0.04
    if td.income_stmt is not None and td.balance_sheet is not None:
        from data import _row
        from assumptions import _roic
        # Approximate ROIC series from operating income / (assets - curr_liab - cash)
        op = _row(td.income_stmt, "operating_income")
        ta = _row(td.balance_sheet, "total_assets")
        if op is not None and ta is not None:
            common = op.index.intersection(ta.index)
            if len(common) >= 3:
                roic_vals = []
                for d in common:
                    ic = float(ta.loc[d]) * 0.7  # rough: ~70% of assets is invested capital
                    if ic > 0:
                        roic_vals.append(float(op.loc[d]) / ic)
                if len(roic_vals) >= 3:
                    sigma = float(np.clip(np.std(roic_vals) * 0.5, 0.02, 0.10))
    return GrowthSource(name="ROIC×reinvestment", mean=g, sigma=max(sigma, 0.03))


def _source_historical_eps(td: TickerData) -> GrowthSource | None:
    """Historical EPS CAGR with jackknife standard error."""
    s = td.eps_series
    if s is None or len(s) < 3:
        return None
    g = cagr(s)
    if g is None:
        return None
    g = float(np.clip(g, -0.20, 0.50))
    sigma = _jackknife_cagr_sigma(s)
    return GrowthSource(name="Historical EPS CAGR", mean=g, sigma=max(sigma, 0.02))


def _source_historical_fcf(td: TickerData) -> GrowthSource | None:
    """Historical FCF CAGR with log-return standard error."""
    fcf = td.fcf_series
    if fcf is None or len(fcf) < 3:
        return None
    pos = fcf[fcf > 0]
    g = cagr(pos)
    if g is None:
        return None
    g = float(np.clip(g, -0.20, 0.50))
    sigma = _fcf_log_return_sigma(fcf)
    return GrowthSource(name="Historical FCF CAGR", mean=g, sigma=max(sigma, 0.025))


def _source_analyst(analyst_g: float | None) -> GrowthSource | None:
    """
    Analyst consensus. Sigma ≈ 30% of the absolute estimate (Bradshaw 2011).
    Floor at 3% because even tightly-agreed estimates carry model risk.
    """
    if analyst_g is None:
        return None
    g = float(np.clip(analyst_g, -0.10, 0.50))
    sigma = max(0.30 * abs(g), 0.03)
    return GrowthSource(name="Analyst consensus", mean=g, sigma=sigma)


# ─── Pooler ────────────────────────────────────────────────────────────────

def pool_growth_sources(sources: list[GrowthSource]) -> GrowthPosterior:
    """
    Inverse-variance weighted posterior (normal-normal conjugate update).

    mu_post  = Σ(mu_i / sigma_i^2) / Σ(1 / sigma_i^2)
    sigma_post = 1 / sqrt(Σ(1/sigma_i^2))
    """
    available = [s for s in sources if s.available]
    if not available:
        return GrowthPosterior(mu=0.06, sigma=0.05, sources=sources, n_sources=0)

    total_precision = sum(s.precision for s in available)
    mu_post = sum(s.mean * s.precision for s in available) / total_precision
    sigma_post = 1.0 / np.sqrt(total_precision)

    # Assign weights for display
    for s in available:
        s.weight = s.precision / total_precision

    return GrowthPosterior(
        mu=float(mu_post),
        sigma=float(sigma_post),
        sources=available,
        n_sources=len(available),
    )


def build_growth_posterior(
    td: TickerData,
    roe: float | None = None,
    payout: float | None = None,
    roic: float | None = None,
    g_roic_reinvest: float | None = None,
    analyst_g: float | None = None,
) -> GrowthPosterior:
    """
    Main entry point. Builds all available sources and pools them.
    Returns a GrowthPosterior whose .mu and .sigma drive the MC.
    """
    sources: list[GrowthSource] = []

    s_sgr = _source_sgr(roe, payout, td)
    if s_sgr:
        sources.append(s_sgr)

    s_roic = _source_roic_reinvest(roic, g_roic_reinvest, td)
    if s_roic:
        sources.append(s_roic)

    s_eps = _source_historical_eps(td)
    if s_eps:
        sources.append(s_eps)

    s_fcf = _source_historical_fcf(td)
    if s_fcf:
        sources.append(s_fcf)

    s_analyst = _source_analyst(analyst_g)
    if s_analyst:
        sources.append(s_analyst)

    posterior = pool_growth_sources(sources)
    log.info("Growth posterior: mu=%.3f sigma=%.3f (%d sources)",
             posterior.mu, posterior.sigma, posterior.n_sources)
    return posterior


def format_posterior_table(p: GrowthPosterior) -> str:
    """Human-readable table of source contributions."""
    lines = [
        f"{'Source':<28} {'Mean':>7} {'Sigma':>7} {'Weight':>7}",
        "─" * 52,
    ]
    for s in p.sources:
        lines.append(
            f"  {s.name:<26} {s.mean*100:>6.1f}% {s.sigma*100:>6.1f}% {s.weight*100:>6.1f}%"
        )
    lines += [
        "─" * 52,
        f"  {'Posterior (IV-weighted)':<26} {p.mu*100:>6.1f}% {p.sigma*100:>6.1f}%",
    ]
    return "\n".join(lines)
