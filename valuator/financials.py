"""
financials.py — Sector-specific valuation models (Stage 4).

Why standard FCFF DCF fails for financial firms:
  - Debt is raw material, not financing. Working capital is the business.
  - Capital expenditure, depreciation, and D&A ratios are meaningless.
  - Regulatory capital constraints drive growth capacity, not ROIC.
  - Damodaran: "you cannot value a bank the same way you value a steel company."

What we build:

  ┌─────────────┬───────────────────────────────────────────────────────┐
  │ Subsector   │ Model                                                 │
  ├─────────────┼───────────────────────────────────────────────────────┤
  │ Banks       │ 2-stage Excess Return Model (Damodaran Ch.21)         │
  │             │  IV = BV + PV(excess returns, high growth)            │
  │             │       + terminal excess return / (ke - g)             │
  ├─────────────┼───────────────────────────────────────────────────────┤
  │ REITs       │ P/AFFO (AFFO = FFO - capex/maintenance) + NAV-proxy  │
  │             │  AFFO = Net Income + D&A - capex - straight-line rent │
  ├─────────────┼───────────────────────────────────────────────────────┤
  │ Insurers    │ DDM with adjusted payout (flagged as limited fidelity)│
  │             │  Embedded-value modeling is not viable from XBRL.     │
  ├─────────────┼───────────────────────────────────────────────────────┤
  │ Asset Mgrs  │ Standard FCFF DCF (they're close to industrials)      │
  └─────────────┴───────────────────────────────────────────────────────┘

References:
  Damodaran (2009) "Valuing Financial Service Firms"
  Damodaran Investment Valuation 3rd ed., Chapters 21-22
  NAREIT AFFO definition (whitepaper, 2003)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd

from data import TickerData, _row

log = logging.getLogger(__name__)


# ─── Subsector classification ──────────────────────────────────────────────

class FinancialSubsector(Enum):
    BANK          = auto()   # commercial banks, savings institutions (SIC 6000-6199)
    INSURER       = auto()   # life, P&C, reinsurance (SIC 6300-6399)
    REIT          = auto()   # real estate investment trusts (SIC 6500+, or "reit" in name)
    ASSET_MANAGER = auto()   # investment advisors, brokers (SIC 6200-6299, 6700s)
    DIVERSIFIED   = auto()   # conglomerates, holding companies
    NOT_FINANCIAL = auto()   # not a financial firm


# Yahoo Finance sector strings and industry substrings that identify sub-types
_BANK_INDUSTRIES = {"bank", "savings", "thrift", "credit union", "commercial bank",
                    "mortgage", "lending", "financial holding"}
_INSURER_INDUSTRIES = {"insurance", "life insurance", "property", "casualty",
                       "reinsurance", "surety"}
_REIT_INDUSTRIES = {"reit", "real estate investment trust"}
_AM_INDUSTRIES = {"asset management", "capital markets", "investment advisor",
                  "brokerage", "wealth management", "exchange"}
_SIC_BANKS = range(6000, 6200)
_SIC_BROKERS = range(6200, 6300)
_SIC_INSURANCE = range(6300, 6400)
_SIC_REAL_ESTATE = range(6500, 6600)
_SIC_HOLDING = range(6700, 6800)


def classify(td: TickerData) -> FinancialSubsector:
    """Route a TickerData to the right financial subsector model."""
    sector = (td.sector or "").lower()
    industry = (td.industry or "").lower()

    if "financial" not in sector and "real estate" not in sector:
        return FinancialSubsector.NOT_FINANCIAL

    # SIC-based routing (EDGAR source, reliable)
    # SIC stored in td.industry when loaded from EDGAR (e.g. "Electronic Computers [3571]")
    # Alternatively check the raw sector/industry strings
    for kw in _REIT_INDUSTRIES:
        if kw in industry or kw in sector:
            return FinancialSubsector.REIT
    if "real estate" in sector and "reit" not in industry:
        # Plain real estate (not REIT): treat as industrial for now
        return FinancialSubsector.NOT_FINANCIAL

    for kw in _BANK_INDUSTRIES:
        if kw in industry:
            return FinancialSubsector.BANK

    for kw in _INSURER_INDUSTRIES:
        if kw in industry:
            return FinancialSubsector.INSURER

    for kw in _AM_INDUSTRIES:
        if kw in industry:
            return FinancialSubsector.ASSET_MANAGER

    return FinancialSubsector.DIVERSIFIED


# ─── Result container ──────────────────────────────────────────────────────

@dataclass
class FinancialValuation:
    subsector: FinancialSubsector
    method_name: str
    intrinsic_value_per_share: float | None
    buy_price: float | None           # with margin of safety
    reliability: str                  # "High" / "Moderate" / "Low" / "Flag"
    notes: list[str]
    inputs: dict                      # for transparency / debugging


# ─── Bank: 2-stage Excess Return Model ────────────────────────────────────

def _bank_excess_return(
    bvps: float,            # book value per share (equity)
    roe_high: float,        # ROE in high-growth stage
    roe_stable: float,      # ROE in terminal stage (usually ke or near it)
    g_high: float,          # EPS/BV growth in high-growth stage
    g_stable: float,        # terminal growth rate
    ke: float,              # cost of equity
    years_high: int = 5,
    margin_of_safety: float = 0.25,
) -> tuple[float | None, dict]:
    """
    Damodaran excess-return model for banks (equity-only, not WACC).

    Value = BV₀ + Σ [BV_t-1 × (ROE_t - ke)] / (1+ke)^t   [high-growth stage]
                 + [BV_high × (ROE_stable - ke) / (ke - g_stable)] / (1+ke)^years_high

    In stable stage ROE should converge toward ke; we fade linearly.
    Excess returns in terminal value → zero when ROE = ke exactly.
    """
    if ke <= 0 or ke <= g_stable:
        return None, {"error": "ke must be > g_stable"}
    if bvps <= 0:
        return None, {"error": "Book value must be positive"}

    pv_excess = 0.0
    bv = bvps
    stage_log = []

    for t in range(1, years_high + 1):
        # Fade ROE from high → stable linearly
        alpha = t / years_high
        roe_t = roe_high * (1 - alpha) + roe_stable * alpha
        # Each year BV grows at g_high
        bv_start = bv
        bv *= (1 + g_high)
        excess = bv_start * (roe_t - ke)
        pv = excess / (1 + ke) ** t
        pv_excess += pv
        stage_log.append({"year": t, "roe": round(roe_t, 4),
                           "excess": round(excess, 4), "pv": round(pv, 4)})

    # Terminal value of excess returns
    bv_terminal = bv
    excess_terminal = bv_terminal * (roe_stable - ke)
    if abs(ke - g_stable) < 1e-6:
        tv = 0.0
    else:
        tv = excess_terminal / (ke - g_stable)
    pv_tv = tv / (1 + ke) ** years_high

    iv = bvps + pv_excess + pv_tv
    buy = iv * (1 - margin_of_safety)

    inputs = {
        "bvps": bvps, "roe_high": roe_high, "roe_stable": roe_stable,
        "g_high": g_high, "g_stable": g_stable, "ke": ke,
        "pv_excess_returns": round(pv_excess, 4),
        "pv_terminal": round(pv_tv, 4),
        "stage_log": stage_log,
    }
    return (iv if iv > 0 else None), inputs


def value_bank(td: TickerData, ke: float, g_stable: float,
               margin_of_safety: float = 0.25) -> FinancialValuation:
    """Build all inputs from TickerData and run the excess-return model."""
    notes = []

    # Book value per share
    bvps = td.book_value_per_share
    if bvps is None:
        # Try to derive from balance sheet
        if td.balance_sheet is not None and td.shares_outstanding:
            eq = _row(td.balance_sheet, "stockholders_equity")
            if eq is not None and len(eq):
                bvps = float(eq.iloc[-1]) / td.shares_outstanding

    if bvps is None or bvps <= 0:
        return FinancialValuation(
            subsector=FinancialSubsector.BANK, method_name="Bank Excess Return",
            intrinsic_value_per_share=None, buy_price=None,
            reliability="Flag", notes=["Book value unavailable"],
            inputs={},
        )

    # ROE: 3-year average from income stmt (more stable than single-year)
    roe_hist = _derive_historical_roe(td)
    roe_high = roe_hist if roe_hist is not None else (td.return_on_equity or 0.10)
    roe_high = float(np.clip(roe_high, -0.05, 0.30))

    # Stable ROE: converge toward ke (economically: no bank earns excess
    # returns in perpetuity under competition)
    roe_stable = float(np.clip(ke * 1.05, ke, roe_high))  # slight premium above ke

    # Growth: use earnings growth or SGR
    g_high = float(np.clip(
        td.earnings_growth or (roe_high * 0.4),  # 40% retention assumption
        0.0, 0.20
    ))

    reliability = "High" if roe_hist is not None else "Moderate"
    if td.income_stmt is None:
        reliability = "Low"
        notes.append("No income statement — used yfinance ROE fallback")

    iv, inputs = _bank_excess_return(
        bvps=bvps, roe_high=roe_high, roe_stable=roe_stable,
        g_high=g_high, g_stable=g_stable, ke=ke,
        margin_of_safety=margin_of_safety,
    )
    inputs["bvps_source"] = "balance_sheet" if td.balance_sheet is not None else "info"
    buy = iv * (1 - margin_of_safety) if iv is not None else None

    return FinancialValuation(
        subsector=FinancialSubsector.BANK,
        method_name="Bank 2-Stage Excess Return (Damodaran Ch.21)",
        intrinsic_value_per_share=iv,
        buy_price=buy,
        reliability=reliability,
        notes=notes,
        inputs=inputs,
    )


def _derive_historical_roe(td: TickerData) -> float | None:
    """3-year average ROE from net income / book equity."""
    if td.income_stmt is None or td.balance_sheet is None:
        return None
    ni = _row(td.income_stmt, "net_income")
    eq = _row(td.balance_sheet, "stockholders_equity")
    if ni is None or eq is None:
        return None
    common = ni.index.intersection(eq.index)
    if len(common) < 2:
        return None
    roe_series = (ni.loc[common] / eq.loc[common]).tail(3).dropna()
    roe_series = roe_series[roe_series.between(-0.20, 0.40)]
    return float(roe_series.mean()) if len(roe_series) >= 1 else None


# ─── REIT: AFFO + NAV proxy ────────────────────────────────────────────────

# Sector-average AFFO multiples (current as of 2025; update annually)
# Source: NAREIT T-Tracker, Green Street
REIT_SECTOR_AFFO_MULTIPLES = {
    "industrial reit":     22.0,
    "office reit":         12.0,
    "retail reit":         14.0,
    "residential reit":    25.0,
    "healthcare reit":     16.0,
    "data center reit":    30.0,
    "storage reit":        22.0,
    "net lease":           16.0,
    "diversified reit":    18.0,
    "default":             18.0,
}

# GAAP depreciation add-back tags used in XBRL for D&A
DEPRECIATION_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
]


def _reit_affo_multiple(industry: str) -> float:
    """Lookup P/AFFO multiple by REIT sub-type."""
    ind = (industry or "").lower()
    for key, mult in REIT_SECTOR_AFFO_MULTIPLES.items():
        if key.split()[0] in ind:
            return mult
    return REIT_SECTOR_AFFO_MULTIPLES["default"]


def _derive_affo(td: TickerData) -> tuple[float | None, str]:
    """
    AFFO = Net Income + D&A - Recurring Capex - Straight-Line Rent Adjustments

    In practice without full REIT disclosures (which aren't fully in XBRL):
    AFFO ≈ Net Income + D&A - Maintenance Capex
    We approximate Maintenance Capex as 15% of total assets (NAREIT guidance
    for maintenance on mature portfolios).

    EDGAR note: 'RealEstateInvestmentPropertyNet' and the depreciation
    add-back come from XBRL facts when available.
    """
    if td.income_stmt is None:
        return None, "income statement unavailable"

    ni = _row(td.income_stmt, "net_income")
    if ni is None or len(ni) == 0:
        return None, "net income unavailable"
    net_income = float(ni.iloc[-1])

    # D&A add-back — from cash flow statement first
    da = 0.0
    if td.cash_flow is not None:
        # Depreciation is often in the cash flow as an add-back
        for tag in ["Depreciation And Amortization", "Depreciation",
                    "Depreciation Depletion And Amortization"]:
            row = _row(td.cash_flow, "dna") if tag == "dna" else None
            if row is not None and len(row):
                da = abs(float(row.iloc[-1]))
                break
        # Rough fallback: try from info
    if da == 0.0 and td.balance_sheet is not None:
        ta = _row(td.balance_sheet, "total_assets")
        if ta is not None and len(ta):
            # Rough: buildings typically depreciate at ~3% per year
            da = float(ta.iloc[-1]) * 0.03

    # Maintenance capex: NAREIT guidance ≈ 10-15% of gross assets
    maint_capex = 0.0
    if td.balance_sheet is not None:
        ta = _row(td.balance_sheet, "total_assets")
        if ta is not None and len(ta):
            maint_capex = float(ta.iloc[-1]) * 0.12  # 12% midpoint

    affo = net_income + da - maint_capex
    shares = td.shares_outstanding or 1
    affo_per_share = affo / shares

    method = f"NI({net_income/1e9:.2f}B) + D&A({da/1e9:.2f}B) - MaintCapex({maint_capex/1e9:.2f}B)"
    return affo_per_share if affo > 0 else None, method


def value_reit(td: TickerData, ke: float,
               margin_of_safety: float = 0.20) -> FinancialValuation:
    """P/AFFO-based REIT valuation."""
    notes = []
    affo_ps, method_note = _derive_affo(td)

    if affo_ps is None or affo_ps <= 0:
        notes.append(f"AFFO unavailable or negative ({method_note})")
        # Fall back to dividend yield valuation
        if td.current_dividend and td.current_dividend > 0:
            div_yield_value = td.current_dividend / ke if ke > 0 else None
            if div_yield_value:
                notes.append("Fallback: Div/ke (Gordon perpetuity)")
                return FinancialValuation(
                    subsector=FinancialSubsector.REIT,
                    method_name="REIT Dividend Yield (Fallback)",
                    intrinsic_value_per_share=div_yield_value,
                    buy_price=div_yield_value * (1 - margin_of_safety),
                    reliability="Low",
                    notes=notes,
                    inputs={"dividend": td.current_dividend, "ke": ke},
                )
        return FinancialValuation(
            subsector=FinancialSubsector.REIT,
            method_name="REIT P/AFFO",
            intrinsic_value_per_share=None, buy_price=None,
            reliability="Flag", notes=notes, inputs={}
        )

    affo_mult = _reit_affo_multiple(td.industry)
    iv = affo_ps * affo_mult
    buy = iv * (1 - margin_of_safety)
    notes.append(f"AFFO/share: ${affo_ps:.2f}  ({method_note})")
    notes.append(f"P/AFFO multiple: {affo_mult}× (sector: {td.industry})")

    return FinancialValuation(
        subsector=FinancialSubsector.REIT,
        method_name="REIT P/AFFO (NAREIT-adjusted)",
        intrinsic_value_per_share=iv,
        buy_price=buy,
        reliability="Moderate",   # always moderate — XBRL lacks full AFFO disclosures
        notes=notes,
        inputs={"affo_per_share": affo_ps, "multiple": affo_mult,
                "method": method_note},
    )


# ─── Insurer: DDM with flag ────────────────────────────────────────────────

def value_insurer(td: TickerData, ke: float, g_stable: float,
                  margin_of_safety: float = 0.25) -> FinancialValuation:
    """
    Insurers: Dividend Discount Model with a strong reliability flag.

    The correct model is embedded-value (EV) using VIF + ANAV. This requires
    disclosures (value-in-force, new business value, risk discount rate)
    that are not available in XBRL filings. We fall back to DDM on dividends
    with an explicit "Low" reliability warning.
    """
    notes = [
        "⚠ Insurer: proper valuation uses Embedded Value (VIF + ANAV).",
        "  XBRL filings don't contain EV disclosures. DDM is a rough proxy.",
        "  Treat this output with scepticism — get the IR supplemental package.",
    ]

    div = td.current_dividend or 0.0
    if div <= 0 or ke <= g_stable:
        notes.append("Dividend unavailable or ke ≤ g — cannot compute DDM.")
        return FinancialValuation(
            subsector=FinancialSubsector.INSURER,
            method_name="Insurer DDM (⚠ Low fidelity — no EV data)",
            intrinsic_value_per_share=None, buy_price=None,
            reliability="Flag", notes=notes, inputs={},
        )

    iv = div * (1 + g_stable) / (ke - g_stable)
    buy = iv * (1 - margin_of_safety)
    return FinancialValuation(
        subsector=FinancialSubsector.INSURER,
        method_name="Insurer DDM (⚠ Low fidelity — no EV data)",
        intrinsic_value_per_share=iv,
        buy_price=buy,
        reliability="Low",
        notes=notes,
        inputs={"dividend": div, "ke": ke, "g_stable": g_stable},
    )


# ─── Router ────────────────────────────────────────────────────────────────

def value_financial(
    td: TickerData,
    ke: float,
    g_stable: float,
    margin_of_safety: float = 0.25,
) -> FinancialValuation | None:
    """
    Route to the correct financial valuation model.
    Returns None for non-financial firms (caller uses standard DCF).
    """
    sub = classify(td)

    if sub == FinancialSubsector.NOT_FINANCIAL:
        return None

    if sub == FinancialSubsector.BANK:
        return value_bank(td, ke=ke, g_stable=g_stable,
                          margin_of_safety=margin_of_safety)

    if sub == FinancialSubsector.REIT:
        return value_reit(td, ke=ke, margin_of_safety=margin_of_safety)

    if sub == FinancialSubsector.INSURER:
        return value_insurer(td, ke=ke, g_stable=g_stable,
                             margin_of_safety=margin_of_safety)

    if sub == FinancialSubsector.ASSET_MANAGER:
        # Asset managers are close to industrials — standard FCFF DCF is fine
        # We note it but don't override
        return FinancialValuation(
            subsector=FinancialSubsector.ASSET_MANAGER,
            method_name="Asset Manager: Standard DCF applicable",
            intrinsic_value_per_share=None, buy_price=None,
            reliability="High",
            notes=["Asset managers have predictable AUM-based revenue — FCFF DCF is appropriate."],
            inputs={},
        )

    return FinancialValuation(
        subsector=FinancialSubsector.DIVERSIFIED,
        method_name="Diversified Financial: Sum-of-parts recommended",
        intrinsic_value_per_share=None, buy_price=None,
        reliability="Low",
        notes=["Diversified financials (holding companies, conglomerates) should be "
               "valued via sum-of-parts. Single-model IV is misleading."],
        inputs={},
    )
