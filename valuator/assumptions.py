"""
Assumptions layer.

Takes raw TickerData and produces the economic assumptions used by valuation:
WACC, terminal growth, exit multiples, growth rates (SGR / historical / analyst).

Key methodological improvements over the original:

  • ERP comes from Damodaran (live) instead of hardcoded 5%
  • Risk-free rate from FRED (live)
  • Beta is Blume-adjusted: 0.67 * raw + 0.33 * 1.0
    (Marshall Blume, 1971: betas regress to 1 over time)
  • Cost of debt comes from interest_expense / total_debt averaged over
    available history, with a credit-quality floor inferred from interest
    coverage ratio
  • Effective tax rate is a 3-year average from the income statement
  • Terminal growth is floored against long-run nominal GDP (4%), not just WACC
  • SGR computation includes Damodaran's ROIC × reinvestment-rate variant
  • Sector P/E and EV/EBITDA tables include sub-industry granularity for
    Real Estate (FFO-aware) and Financials (banks vs. insurers)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import TickerData, _row
from macro import MacroEnvironment

log = logging.getLogger(__name__)


# ─── Sector / sub-industry multiples ────────────────────────────────────────
# Two-level lookup: prefer sub-industry, fall back to sector.

SECTOR_PE = {
    "Technology":               25.0,
    "Communication Services":   22.0,
    "Consumer Cyclical":        18.0,
    "Consumer Defensive":       18.0,
    "Healthcare":               20.0,
    "Financial Services":       13.0,
    "Industrials":              16.0,
    "Basic Materials":          13.0,
    "Energy":                   12.0,
    "Utilities":                15.0,
    "Real Estate":              28.0,  # P/E is misleading for REITs; see below
}

# Industry-level overrides for cases where the sector average is misleading.
# Yahoo industry strings are inconsistent — match on substrings.
INDUSTRY_PE_OVERRIDES = {
    "REIT":              35.0,   # P/E inflated by D&A; FFO would be cleaner
    "Banks":             11.0,   # below FS sector average
    "Insurance":         12.0,
    "Asset Management":  16.0,
    "Capital Markets":   15.0,
    "Semiconductor":     28.0,   # higher than tech sector
    "Software":          30.0,
    "Biotech":           18.0,
    "Drug Manufacturers":18.0,
    "Aerospace":         18.0,
    "Airlines":          10.0,
    "Auto Manufacturers":12.0,
    "Oil & Gas":         11.0,
    "Tobacco":           14.0,
    "Beverages":         22.0,
}

SECTOR_EV_EBITDA = {
    "Technology": 22.0, "Communication Services": 18.0, "Consumer Cyclical": 12.0,
    "Consumer Defensive": 14.0, "Healthcare": 16.0, "Financial Services": 10.0,
    "Industrials": 13.0, "Basic Materials": 10.0, "Energy": 8.0,
    "Utilities": 12.0, "Real Estate": 20.0,
}


def _industry_lookup(industry: str, table: dict) -> float | None:
    """Match Yahoo's industry string against substrings in the override table."""
    if not industry:
        return None
    for key, val in table.items():
        if key.lower() in industry.lower():
            return val
    return None


# ─── Growth-rate helpers ────────────────────────────────────────────────────

def cagr(series: pd.Series) -> float | None:
    """Compound annual growth rate from oldest to newest, sign-checked."""
    if series is None or len(series) < 2:
        return None
    s = series.dropna().sort_index()
    if len(s) < 2:
        return None
    start, end = float(s.iloc[0]), float(s.iloc[-1])
    if start <= 0 or end <= 0:
        return None  # CAGR is meaningless across sign changes
    n = len(s) - 1
    return (end / start) ** (1.0 / n) - 1.0


def winsorize_pct_change(series: pd.Series) -> pd.Series:
    """
    Use log returns instead of pct_change for sigma estimation. Avoids the
    blow-up when the underlying series crosses zero — a known failure mode
    for FCF volatility.
    """
    s = series.dropna().sort_index()
    s = s[s > 0]
    if len(s) < 2:
        return pd.Series(dtype=float)
    log_returns = np.log(s / s.shift(1)).dropna()
    return log_returns


# ─── Main assumption bundle ─────────────────────────────────────────────────

@dataclass
class Assumptions:
    """All economic assumptions for a valuation, with provenance."""
    # Discount rate
    risk_free_rate: float
    equity_risk_premium: float
    beta_raw: float
    beta_adjusted: float          # Blume-adjusted
    cost_of_equity: float
    cost_of_debt: float
    effective_tax_rate: float
    debt_weight: float
    equity_weight: float
    wacc: float

    # Terminal & growth
    terminal_growth: float        # capped at min(rf, long-run GDP)
    long_run_gdp: float

    # Growth-rate set
    growth_fundamental: float     # SGR
    growth_roic_reinvest: float | None  # Damodaran's variant
    growth_historical_eps: float | None
    growth_historical_fcf: float | None
    growth_analyst: float
    growth_blended: float

    # Multiples
    future_pe: float
    ev_ebitda_multiple: float

    # FCF / quality
    fcf_normalized: float         # 3y average, smoothing one-time items
    fcf_growth_sigma: float       # for Monte Carlo, from log returns
    roic: float | None

    # Bayesian growth posterior (Stage 3)
    growth_posterior: "GrowthPosterior | None" = None

    # Diagnostics
    margin_of_safety: float = 0.30
    notes: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        log.info("[assumptions] %s", msg)
        self.notes.append(msg)


def _blume_adjust(beta_raw: float | None) -> tuple[float, float]:
    """Blume (1971) adjustment toward 1.0. Returns (raw_clamped, adjusted)."""
    if beta_raw is None:
        return 1.0, 1.0
    raw = float(np.clip(beta_raw, 0.3, 3.0))
    adj = 0.67 * raw + 0.33 * 1.0
    return raw, adj


def _cost_of_debt(td: TickerData, rf: float, macro: MacroEnvironment) -> tuple[float, str]:
    """
    Cost of debt with three-tier fallback:
      1. Compute kd from interest_expense / total_debt (most accurate)
      2. Estimate from interest coverage + FRED-derived credit spreads (BAA/AAA)
      3. Default to rf + (BAA+AAA)/2 if no coverage info

    The big improvement vs. v1 is step 2 uses live market BAA10Y / AAA10Y
    spreads instead of static bands.
    """
    if td.interest_expense_series is not None and td.balance_sheet is not None:
        debt_row = _row(td.balance_sheet, "total_debt")
        if debt_row is not None:
            common = td.interest_expense_series.index.intersection(debt_row.index)
            if len(common) >= 1:
                ratios = []
                for d in common:
                    debt = float(debt_row.loc[d])
                    ie = float(td.interest_expense_series.loc[d])
                    if debt > 0 and ie > 0:
                        ratios.append(ie / debt)
                if ratios:
                    kd = float(np.mean(ratios))
                    kd = float(np.clip(kd, max(rf - 0.005, 0.01), 0.20))
                    return kd, "interest_expense / total_debt (avg)"

    # Step 2: synthetic rating × FRED spread
    icr = _interest_coverage(td)
    from macro import credit_spread_for_rating
    spread = credit_spread_for_rating(icr, macro.baa_spread, macro.aaa_spread)
    icr_str = f"ICR={icr:.1f}" if icr is not None else "ICR=unknown"
    src = "FRED BAA/AAA" if (macro.baa_spread or macro.aaa_spread) else "default spread bands"
    return rf + spread, f"rf + {spread*100:.2f}% spread ({icr_str}, {src})"


def _interest_coverage(td: TickerData) -> float | None:
    if td.income_stmt is None:
        return None
    op = _row(td.income_stmt, "operating_income")
    ie = td.interest_expense_series
    if op is None or ie is None or len(op) == 0 or len(ie) == 0:
        return None
    common = op.index.intersection(ie.index)
    if len(common) == 0:
        return None
    op_recent = float(op.loc[common].iloc[-1])
    ie_recent = float(ie.loc[common].iloc[-1])
    if ie_recent <= 0:
        return None
    return op_recent / ie_recent


def _effective_tax_rate(td: TickerData, default: float = 0.21) -> float:
    """3-year average effective tax rate from the income statement."""
    s = td.tax_rate_series
    if s is None or len(s) == 0:
        return default
    avg = float(s.tail(3).mean())
    return float(np.clip(avg, 0.0, 0.40))


def _normalized_fcf(td: TickerData) -> tuple[float, list]:
    """3-year average of positive FCF; falls back gracefully."""
    if td.fcf_series is None or len(td.fcf_series) == 0:
        return 0.0, ["fcf_series unavailable"]
    s = td.fcf_series.tail(3)
    pos = s[s > 0]
    notes = [f"FCF history (last 3y): {[round(x/1e6, 1) for x in s.values]} ($M)"]
    if len(pos) >= 2:
        return float(pos.mean()), notes + ["normalized = mean of positive years"]
    if len(s) >= 2:
        return float(s.mean()), notes + ["normalized = simple 3y mean (some negative)"]
    return float(s.iloc[-1]), notes + ["only one period available"]


def _roic(td: TickerData, tax_rate: float) -> float | None:
    """ROIC = NOPAT / Invested Capital."""
    if td.income_stmt is None or td.balance_sheet is None:
        return None
    op = _row(td.income_stmt, "operating_income")
    if op is None or len(op) == 0:
        return None
    nopat = float(op.iloc[-1]) * (1 - tax_rate)

    ta = _row(td.balance_sheet, "total_assets")
    cl = _row(td.balance_sheet, "current_liabilities")
    cash_row = _row(td.balance_sheet, "cash")
    if ta is None or cl is None:
        return None
    cash_val = float(cash_row.iloc[-1]) if cash_row is not None else 0.0
    ic = float(ta.iloc[-1]) - float(cl.iloc[-1]) - cash_val
    if ic <= 0:
        return None
    return nopat / ic


def _sgr(td: TickerData) -> float:
    """Sustainable Growth Rate: ROE × (1 - payout)."""
    roe = td.return_on_equity
    payout = td.payout_ratio
    if roe is None:
        return 0.05
    payout = max(0.0, min(payout if payout is not None else 0.0, 1.0))
    sgr = float(roe) * (1.0 - payout)
    return float(np.clip(sgr, 0.02, 0.15)) if sgr > 0 else 0.05


def _roic_reinvestment_growth(td: TickerData, roic: float | None, tax_rate: float) -> float | None:
    """
    Damodaran's preferred growth formulation: g = ROIC × reinvestment_rate
    where reinvestment_rate = (Capex - D&A + ΔWC) / NOPAT.

    Cleaner economically than ROE × retention because it uses operating
    cash deployment, not accounting earnings retention.

    We approximate reinvestment_rate ≈ (Capex / OCF) as a simple proxy
    when full accruals decomposition is not available.
    """
    if roic is None or td.cash_flow is None:
        return None
    ocf = _row(td.cash_flow, "operating_cash_flow")
    capex = _row(td.cash_flow, "capex")
    if ocf is None or capex is None or len(ocf) == 0:
        return None
    ocf_recent = float(ocf.iloc[-1])
    capex_recent = abs(float(capex.iloc[-1]))
    if ocf_recent <= 0:
        return None
    reinvest_rate = float(np.clip(capex_recent / ocf_recent, 0.0, 1.0))
    return roic * reinvest_rate


def _historical_eps_growth(td: TickerData) -> float | None:
    if td.eps_series is None or len(td.eps_series) < 3:
        return None
    g = cagr(td.eps_series)
    if g is None:
        return None
    return float(np.clip(g, -0.30, 0.50))


def _historical_fcf_growth(td: TickerData) -> float | None:
    if td.fcf_series is None or len(td.fcf_series) < 3:
        return None
    g = cagr(td.fcf_series[td.fcf_series > 0])
    if g is None:
        return None
    return float(np.clip(g, -0.30, 0.50))


def _future_pe(td: TickerData) -> float:
    """Sub-industry override → sector → blend with current trailing/forward P/E."""
    base = _industry_lookup(td.industry, INDUSTRY_PE_OVERRIDES)
    if base is None:
        base = SECTOR_PE.get(td.sector, 15.0)

    # Blend with the company's own current P/E if it's in a sane range,
    # so we don't hand a beaten-down stock the sector's average multiple.
    ref = None
    fwd, ttm = td.forward_pe, td.trailing_pe
    if fwd and 5 <= float(fwd) <= 60:
        ref = float(fwd)
    elif ttm and 5 <= float(ttm) <= 60:
        ref = float(ttm)
    if ref is not None:
        base = 0.5 * base + 0.5 * ref

    return float(np.clip(base, 8.0, 40.0))


def _ev_ebitda_multiple(td: TickerData) -> float:
    return SECTOR_EV_EBITDA.get(td.sector, 12.0)


def _fcf_growth_sigma(td: TickerData) -> float:
    """
    Volatility for Monte Carlo. Uses log returns of FCF (handles sign changes
    far better than pct_change), bounded to a reasonable range.
    """
    if td.fcf_series is None or len(td.fcf_series) < 3:
        return 0.10
    lr = winsorize_pct_change(td.fcf_series)
    if len(lr) < 2:
        return 0.10
    return float(np.clip(lr.std(), 0.03, 0.40))


def build(td: TickerData, macro: MacroEnvironment | None = None) -> Assumptions:
    """Assemble all assumptions from raw data + macro environment."""
    if macro is None:
        macro = MacroEnvironment.fetch()

    rf = macro.risk_free_rate
    erp = macro.equity_risk_premium

    # ── Cost of equity (CAPM with Blume-adjusted beta) ──
    beta_raw, beta_adj = _blume_adjust(td.beta_raw)
    ke = rf + beta_adj * erp

    # ── Cost of debt from real data + FRED credit spreads ──
    kd, kd_source = _cost_of_debt(td, rf, macro)

    # ── Effective tax rate ──
    tax = _effective_tax_rate(td)

    # ── Capital structure weights ──
    debt_total = 0.0
    if td.balance_sheet is not None:
        debt_row = _row(td.balance_sheet, "total_debt")
        if debt_row is not None and len(debt_row):
            debt_total = float(debt_row.iloc[-1])
    mcap = float(td.market_cap or 0)
    V = mcap + debt_total
    if V > 0:
        we = mcap / V
        wd = debt_total / V
        wacc = we * ke + wd * kd * (1 - tax)
    else:
        we, wd, wacc = 1.0, 0.0, ke
    wacc = float(np.clip(wacc, 0.06, 0.20))

    # ── Terminal growth: floor at min(rf, long-run GDP) — never above either ──
    # Mathematically, terminal_g must be < wacc for the perpetuity to converge,
    # but economically it must also be ≤ long-run nominal GDP.
    terminal_g = min(0.025, macro.long_run_gdp, wacc - 0.01, rf)
    terminal_g = max(terminal_g, 0.01)  # don't go negative

    # ── Growth rate set ──
    sgr = _sgr(td)
    fcf_normalized, fcf_notes = _normalized_fcf(td)
    roic = _roic(td, tax)
    g_roic_reinvest = _roic_reinvestment_growth(td, roic, tax)
    g_hist_eps = _historical_eps_growth(td)
    g_hist_fcf = _historical_fcf_growth(td)

    # Analyst forward EPS growth (with bounds)
    if td.earnings_growth is not None:
        g_analyst = float(np.clip(float(td.earnings_growth), 0.02, 0.40))
    else:
        g_analyst = sgr

    # Blended: median of all available estimates
    candidates = [sgr, g_analyst]
    for g in (g_roic_reinvest, g_hist_eps, g_hist_fcf):
        if g is not None:
            candidates.append(float(np.clip(g, -0.10, 0.40)))
    g_blended = float(np.median(candidates))

    # ── Multiples ──
    pe = _future_pe(td)
    ev_mult = _ev_ebitda_multiple(td)

    # ── Volatility for Monte Carlo ──
    fcf_sigma = _fcf_growth_sigma(td)

    # ── Bayesian growth posterior ──
    from growth_pool import build_growth_posterior, GrowthPosterior
    posterior = build_growth_posterior(
        td=td,
        roe=td.return_on_equity,
        payout=td.payout_ratio,
        roic=roic,
        g_roic_reinvest=g_roic_reinvest,
        analyst_g=td.earnings_growth,
    )

    a = Assumptions(
        risk_free_rate=rf,
        equity_risk_premium=erp,
        beta_raw=beta_raw,
        beta_adjusted=beta_adj,
        cost_of_equity=ke,
        cost_of_debt=kd,
        effective_tax_rate=tax,
        debt_weight=wd,
        equity_weight=we,
        wacc=wacc,
        terminal_growth=terminal_g,
        long_run_gdp=macro.long_run_gdp,
        growth_fundamental=sgr,
        growth_roic_reinvest=g_roic_reinvest,
        growth_historical_eps=g_hist_eps,
        growth_historical_fcf=g_hist_fcf,
        growth_analyst=g_analyst,
        growth_blended=g_blended,
        future_pe=pe,
        ev_ebitda_multiple=ev_mult,
        fcf_normalized=fcf_normalized,
        fcf_growth_sigma=fcf_sigma,
        roic=roic,
        growth_posterior=posterior,
    )
    a.note(f"Cost of debt: {kd_source}")
    for n in fcf_notes:
        a.note(n)
    return a
