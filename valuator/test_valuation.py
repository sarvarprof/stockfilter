"""
Tests for the valuation refactor.

Uses synthetic data so we can verify the math independently of yfinance.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import engine
from assumptions import Assumptions, _blume_adjust, cagr, winsorize_pct_change
from data import TickerData
from macro import MacroEnvironment


def assert_close(a, b, tol=1e-2, label=""):
    if a is None or b is None:
        ok = a == b
    elif isinstance(a, str) or isinstance(b, str):
        ok = a == b
    else:
        ok = abs(a - b) <= tol
    status = "✓" if ok else "✗"
    print(f"  {status} {label}: got {a}, expected ≈ {b}")
    if not ok:
        sys.exit(1)


def assert_true(cond, label):
    status = "✓" if cond else "✗"
    print(f"  {status} {label}")
    if not cond:
        sys.exit(1)


# ─── Tests for engine math ─────────────────────────────────────────────────

def test_two_stage_dcf_known_value():
    """
    Hand-calculable case: zero growth in stage 1, immediate fade to terminal_g.
    With fcf=100, growth=0, fade_start=5, terminal_g=0.025, wacc=0.10, years=10:
      - Years 1-5: FCF stays at 100 (since growth=0)
      - Years 6-10: linear fade from 0% to 2.5%, so g = 0.5%, 1.0%, 1.5%, 2.0%, 2.5%
      - TV = FCF_10 × 1.025 / (0.10 - 0.025)
    """
    fcf0, g, wacc, tg = 100.0, 0.0, 0.10, 0.025
    v = engine.two_stage_dcf(fcf0, g, wacc, tg, years=10, fade_start=5, shares=1.0)
    # Replicate by hand
    pv = 0.0
    fcf = fcf0
    for t in range(1, 6):
        fcf *= (1 + g)
        pv += fcf / (1 + wacc) ** t
    for i in range(1, 6):
        alpha = i / 5
        gy = g * (1 - alpha) + tg * alpha
        fcf *= (1 + gy)
        pv += fcf / (1 + wacc) ** (5 + i)
    tv = fcf * (1 + tg) / (wacc - tg)
    pv += tv / (1 + wacc) ** 10
    assert_close(v, pv, tol=1e-6, label="DCF matches hand calculation")


def test_dcf_returns_none_on_invalid():
    assert_close(engine.two_stage_dcf(0, 0.05, 0.10, 0.025, shares=1), None,
                 label="DCF returns None on zero FCF")
    assert_close(engine.two_stage_dcf(100, 0.05, 0.02, 0.025, shares=1), None,
                 label="DCF returns None when wacc <= terminal_g")


def test_terminal_growth_below_wacc():
    """Convergence requires wacc > terminal_g."""
    v = engine.two_stage_dcf(100, 0.05, 0.08, 0.025, shares=1)
    assert_true(v is not None and v > 0, "DCF positive when wacc > tg")


def test_buffett_uses_cost_of_equity():
    """Same EPS at higher discount rate → lower IV. Pure consistency check."""
    iv1, _ = engine.buffett_valuation(eps0=5, growth_rate=0.08, terminal_g=0.025,
                                       cost_of_equity=0.08, future_pe=18, margin_of_safety=0.3)
    iv2, _ = engine.buffett_valuation(eps0=5, growth_rate=0.08, terminal_g=0.025,
                                       cost_of_equity=0.12, future_pe=18, margin_of_safety=0.3)
    assert_true(iv2 < iv1, "Higher cost of equity → lower Buffett IV")


def test_monte_carlo_lognormal_no_negative_growth_explosion():
    """Lognormal sampling means we never get g < -100% (which would crash)."""
    out = engine.monte_carlo_dcf(
        fcf0=1000, shares=10, growth_mean=0.10, growth_sigma=0.30,
        wacc_mean=0.10, wacc_sigma=0.02, terminal_g=0.025,
        iterations=1000, seed=42,
    )
    assert_true("error" not in out, "MC completes without error on high σ")
    assert_true(out["p05"] < out["p50"] < out["p95"], "MC distribution properly ordered")
    # With heavy lognormal, mean should exceed median
    assert_true(out["mean"] >= out["p50"] * 0.95,
                "Lognormal MC: mean ≈ or > median (right-skewed)")


def test_monte_carlo_correlation_works():
    """When growth is high, WACC samples should be biased high too."""
    # Smoke test: just confirm correlated draws don't crash
    out = engine.monte_carlo_dcf(
        fcf0=1000, shares=10, growth_mean=0.10, growth_sigma=0.10,
        wacc_mean=0.10, wacc_sigma=0.02, terminal_g=0.025,
        iterations=2000, correlation=0.4, seed=7,
    )
    assert_true("error" not in out, "Correlated MC runs cleanly")


def test_reverse_dcf_consistency():
    """The implied growth should reproduce the price within a small tolerance."""
    fcf0, shares, wacc, tg = 1_000_000, 10_000, 0.09, 0.025
    # Pick a price corresponding to g=0.07
    target_iv = engine.two_stage_dcf(fcf0, 0.07, wacc, tg, shares=shares)
    implied = engine.reverse_dcf(fcf0, shares, target_iv, wacc, tg)
    assert_close(implied, 0.07, tol=1e-3, label="Reverse DCF recovers known g")


def test_sensitivity_table_shape():
    df = engine.sensitivity_table(
        fcf0=1000, shares=10, growth_rate=0.07,
        wacc_base=0.10, terminal_g_base=0.025,
    )
    assert_true(df.shape == (5, 5), "5×5 sensitivity grid")
    # Diagonal should be monotonic: lower WACC + higher terminal_g → higher value
    assert_true(df.iloc[0, -1] > df.iloc[-1, 0],
                "Top-right (low WACC, high g) > bottom-left")


def test_blume_adjust():
    raw, adj = _blume_adjust(1.5)
    assert_close(adj, 0.67 * 1.5 + 0.33, tol=1e-6, label="Blume(1.5) = 1.34")
    raw, adj = _blume_adjust(None)
    assert_close(adj, 1.0, label="Blume(None) = 1.0")
    raw, adj = _blume_adjust(5.0)  # should clip to 3.0
    assert_close(raw, 3.0, label="Beta clamped to 3.0")


def test_cagr():
    s = pd.Series([100, 110, 121], index=pd.date_range("2021", periods=3, freq="YE"))
    assert_close(cagr(s), 0.10, tol=1e-6, label="CAGR(100→121) over 2y = 10%")
    # CAGR is endpoint-only by definition; negative endpoint should return None.
    assert_close(cagr(pd.Series([100, 50, -10])), None,
                 label="CAGR returns None when endpoint is negative")


def test_winsorize_pct_change_handles_signs():
    """The original code used .pct_change() which blows up on sign changes."""
    s = pd.Series([100, 50, 75, 60, 90])
    lr = winsorize_pct_change(s)
    assert_true(len(lr) > 0, "Log-return series populated")
    assert_true(lr.std() < 5.0, "Log return std bounded (no inf blow-up)")


def test_adaptive_verdict_uses_mc():
    # Price below P25 → undervalued
    v = engine.adaptive_verdict(100, current_price=80,
                                 mc_p25=85, mc_p50=100, mc_p75=120,
                                 use_mc_distribution=True)
    assert_close(v, "Undervalued", label="Price < P25 → Undervalued")
    # Price above P75
    v = engine.adaptive_verdict(100, current_price=130,
                                 mc_p25=85, mc_p50=100, mc_p75=120,
                                 use_mc_distribution=True)
    assert_close(v, "Overvalued", label="Price > P75 → Overvalued")
    # Price in IQR
    v = engine.adaptive_verdict(100, current_price=110,
                                 mc_p25=85, mc_p50=100, mc_p75=120,
                                 use_mc_distribution=True)
    assert_close(v, "Fairly Valued", label="Price in IQR → Fairly Valued")
    # Without MC flag, falls back to ratio bands
    v = engine.adaptive_verdict(100, current_price=80)  # ratio = 1.25
    assert_close(v, "Slight Upside", label="No MC → ratio band (1.25 → Slight Upside)")


# ─── Integration test with synthetic TickerData ────────────────────────────

def make_synthetic_ticker() -> TickerData:
    """A clean synthetic ticker that exercises the whole pipeline."""
    dates = pd.date_range("2021-09-30", periods=4, freq="YE")  # 4 annual periods

    income_stmt = pd.DataFrame({
        d: {
            "Operating Income":   2000 * (1.10 ** i),
            "Net Income":         1500 * (1.10 ** i),
            "Diluted EPS":        2.0 * (1.10 ** i),
            "Diluted Average Shares": 750,
            "Tax Provision":      400 * (1.10 ** i),
            "Pretax Income":      1900 * (1.10 ** i),
            "Interest Expense":   100,
            "Total Revenue":      10_000 * (1.08 ** i),
            "EBITDA":             3000 * (1.10 ** i),
        }
        for i, d in enumerate(dates)
    })

    balance_sheet = pd.DataFrame({
        d: {
            "Total Assets":         15_000,
            "Current Liabilities":  3_000,
            "Total Debt":           2_000,
            "Cash And Cash Equivalents": 1_000,
            "Stockholders Equity":  8_000,
        }
        for d in dates
    })

    cash_flow = pd.DataFrame({
        d: {
            "Operating Cash Flow":  1800 * (1.09 ** i),
            "Capital Expenditure": -500 * (1.05 ** i),
        }
        for i, d in enumerate(dates)
    })

    td = TickerData(
        ticker="TEST",
        current_price=50.0,
        shares_outstanding=750.0,
        market_cap=37_500.0,
        beta_raw=1.2,
        trailing_eps=2.66,
        forward_eps=2.93,
        book_value_per_share=8.0,
        current_dividend=0.50,
        payout_ratio=0.20,
        return_on_equity=0.20,
        earnings_growth=0.10,
        trailing_pe=18.8,
        forward_pe=17.0,
        sector="Technology",
        industry="Software—Application",
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
    )

    # Manually run the derive step
    from data import _derive_series
    _derive_series(td)
    return td


def test_full_pipeline_synthetic():
    print("\n[integration] Full pipeline on synthetic ticker:")
    td = make_synthetic_ticker()

    assert_true(td.fcf_series is not None and len(td.fcf_series) >= 3,
                "FCF series derived (3+ periods)")
    assert_true(td.eps_series is not None and len(td.eps_series) >= 3,
                "EPS series derived from Diluted EPS row")
    assert_true(td.tax_rate_series is not None,
                "Tax rate series derived from Tax Provision / Pretax Income")
    assert_true(td.interest_expense_series is not None,
                "Interest expense series extracted")

    # Build assumptions with a fixed macro env (no network)
    macro = MacroEnvironment(risk_free_rate=0.043, equity_risk_premium=0.0423,
                             long_run_gdp=0.04)
    from assumptions import build
    a = build(td, macro)

    # Sanity-check the assumption values
    assert_true(0.07 <= a.wacc <= 0.13,
                f"WACC in plausible range (got {a.wacc:.3f})")
    assert_true(a.beta_adjusted < a.beta_raw,
                f"Blume pulled beta toward 1: {a.beta_raw:.2f}→{a.beta_adjusted:.2f}")
    assert_true(a.terminal_growth <= macro.long_run_gdp,
                f"Terminal g floored at GDP ({a.terminal_growth:.3f} ≤ {macro.long_run_gdp})")
    assert_true(a.terminal_growth < a.wacc,
                "Terminal g < WACC (perpetuity converges)")
    assert_true(a.fcf_normalized > 0,
                f"FCF normalized positive (got {a.fcf_normalized:.0f})")
    assert_true(a.roic is not None and a.roic > 0,
                f"ROIC computed (got {a.roic:.3f})")
    # Tax rate from data should be ≈ 400/1900 = 0.21
    assert_close(a.effective_tax_rate, 0.21, tol=0.005,
                 label="Tax rate ≈ Tax/Pretax = 21%")
    # Cost of debt: 100/2000 = 5%
    assert_close(a.cost_of_debt, 0.05, tol=0.005,
                 label="Cost of debt ≈ IE/Debt = 5%")

    # Run a DCF and confirm we get a positive number
    iv = engine.two_stage_dcf(a.fcf_normalized, a.growth_blended, a.wacc,
                               a.terminal_growth, shares=td.shares_outstanding)
    assert_true(iv is not None and iv > 0, f"DCF produces positive IV (${iv:.2f})")


# ─── Run all tests ────────────────────────────────────────────────────────

def run():
    tests = [
        ("two_stage_dcf hand calc",         test_two_stage_dcf_known_value),
        ("DCF returns None on invalid",     test_dcf_returns_none_on_invalid),
        ("Terminal g < WACC required",      test_terminal_growth_below_wacc),
        ("Buffett uses cost of equity",     test_buffett_uses_cost_of_equity),
        ("MC lognormal no explosion",       test_monte_carlo_lognormal_no_negative_growth_explosion),
        ("MC correlation runs",             test_monte_carlo_correlation_works),
        ("Reverse DCF round-trip",          test_reverse_dcf_consistency),
        ("Sensitivity table shape",         test_sensitivity_table_shape),
        ("Blume beta adjustment",           test_blume_adjust),
        ("CAGR helper",                     test_cagr),
        ("Log-return σ no blow-up",         test_winsorize_pct_change_handles_signs),
        ("Adaptive verdict uses MC IQR",    test_adaptive_verdict_uses_mc),
        ("Full synthetic pipeline",         test_full_pipeline_synthetic),
    ]
    for name, fn in tests:
        print(f"\n[{name}]")
        fn()
    print(f"\n{'='*60}\nAll {len(tests)} tests passed ✓\n{'='*60}")


if __name__ == "__main__":
    run()
