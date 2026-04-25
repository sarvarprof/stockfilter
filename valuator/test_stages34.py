"""Tests for Stage 3 (Bayesian growth pooling) and Stage 4 (financials)."""
from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

# fresh test cache
_TMP = Path(tempfile.mkdtemp(prefix="val_s34_test_"))
os.environ["VALUATION_CACHE"] = str(_TMP)

import numpy as np
import pandas as pd

from growth_pool import (
    GrowthSource, GrowthPosterior,
    pool_growth_sources,
    build_growth_posterior,
    _bootstrap_cagr_sigma,
    _jackknife_cagr_sigma,
    _fcf_log_return_sigma,
    format_posterior_table,
)
from financials import (
    FinancialSubsector, classify, value_bank, value_reit, value_insurer,
    value_financial, _bank_excess_return, _derive_historical_roe, _derive_affo,
)
from data import TickerData, _derive_series


# ── helpers ─────────────────────────────────────────────────────────────────

def assert_close(a, b, tol=1e-3, label=""):
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


def make_series(values, freq="YE"):
    idx = pd.date_range("2020-12-31", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3: Growth pooling
# ════════════════════════════════════════════════════════════════════════════

def test_inverse_variance_weighting():
    """Precise source dominates with low sigma."""
    print("\n[Inverse-variance weighting: precise source dominates]")
    s1 = GrowthSource("A", mean=0.10, sigma=0.20)   # imprecise
    s2 = GrowthSource("B", mean=0.06, sigma=0.02)   # precise
    p = pool_growth_sources([s1, s2])
    # Precision: 1/0.04 = 25, 1/0.0004 = 2500 → s2 has 100× more weight
    assert_true(abs(p.mu - 0.06) < 0.005,
                f"Posterior dominated by precise source ({p.mu:.4f} ≈ 0.06)")
    assert_true(p.sigma < s1.sigma and p.sigma < s2.sigma,
                "Posterior sigma < either individual sigma")


def test_equal_sigma_is_simple_average():
    """Equal uncertainty → straight average."""
    print("\n[Equal sigma → arithmetic mean]")
    sources = [GrowthSource("A", mean=0.04, sigma=0.05),
               GrowthSource("B", mean=0.10, sigma=0.05),
               GrowthSource("C", mean=0.07, sigma=0.05)]
    p = pool_growth_sources(sources)
    assert_close(p.mu, 0.07, label="Posterior = mean(0.04, 0.10, 0.07) = 0.07")


def test_posterior_sigma_narrower_with_more_sources():
    """Adding a consistent source always reduces posterior uncertainty."""
    print("\n[More sources → narrower posterior sigma]")
    s1 = [GrowthSource("A", mean=0.08, sigma=0.04)]
    s2 = s1 + [GrowthSource("B", mean=0.07, sigma=0.04)]
    s3 = s2 + [GrowthSource("C", mean=0.09, sigma=0.04)]
    p1 = pool_growth_sources(s1)
    p2 = pool_growth_sources(s2)
    p3 = pool_growth_sources(s3)
    assert_true(p3.sigma < p2.sigma < p1.sigma,
                f"sigma: {p1.sigma:.4f} > {p2.sigma:.4f} > {p3.sigma:.4f}")


def test_empty_sources_returns_default():
    print("\n[Empty source list → default posterior]")
    p = pool_growth_sources([])
    assert_true(p.n_sources == 0, "n_sources = 0")
    assert_true(p.mu > 0, "Default mu positive")


def test_jackknife_sigma_stable():
    """Jackknife sigma should be well-defined for clean series."""
    print("\n[Jackknife sigma stable on clean series]")
    s = make_series([100, 112, 125, 140, 157])  # ~12% growth
    sigma = _jackknife_cagr_sigma(s)
    assert_true(0.0 < sigma < 0.15,
                f"Jackknife sigma in [0, 15%] (got {sigma:.4f})")


def test_bootstrap_sigma_wider_for_volatile():
    """More volatile series → higher bootstrap sigma."""
    print("\n[Bootstrap sigma: volatile series > stable series]")
    stable = make_series([100, 105, 110, 116, 122])
    volatile = make_series([100, 50, 200, 80, 190])
    sig_s = _bootstrap_cagr_sigma(stable.clip(lower=0.01))
    sig_v = _bootstrap_cagr_sigma(volatile.clip(lower=0.01))
    assert_true(sig_v > sig_s,
                f"Volatile sigma ({sig_v:.3f}) > stable sigma ({sig_s:.3f})")


def test_fcf_log_return_sigma():
    """Log-return sigma should decrease as SE of the mean."""
    print("\n[FCF log-return SE decreases with more data]")
    short = make_series([100, 120, 110, 130])
    long_ = make_series([100, 112, 125, 140, 157, 172, 190, 208])
    s_short = _fcf_log_return_sigma(short)
    s_long = _fcf_log_return_sigma(long_)
    assert_true(s_short > s_long,
                f"Shorter series has higher SE ({s_short:.4f} vs {s_long:.4f})")


def test_full_posterior_with_synthetic_ticker():
    """build_growth_posterior wires correctly end-to-end."""
    print("\n[Full posterior on synthetic ticker]")
    dates = pd.date_range("2020-12-31", periods=5, freq="YE")
    td = TickerData(
        ticker="TEST",
        shares_outstanding=100.0,
        return_on_equity=0.18,
        payout_ratio=0.30,
        earnings_growth=0.10,
    )
    td.income_stmt = pd.DataFrame({
        d: {"Net Income": 150 * 1.1**i, "Diluted EPS": 1.5 * 1.1**i,
            "Diluted Average Shares": 100, "Pretax Income": 185 * 1.1**i,
            "Tax Provision": 35 * 1.1**i, "Operating Income": 200 * 1.1**i}
        for i, d in enumerate(dates)
    })
    td.balance_sheet = pd.DataFrame({
        d: {"Stockholders Equity": 800, "Total Assets": 2000,
            "Current Liabilities": 300, "Total Debt": 500,
            "Cash And Cash Equivalents": 100}
        for d in dates
    })
    td.cash_flow = pd.DataFrame({
        d: {"Operating Cash Flow": 180 * 1.09**i,
            "Capital Expenditure": -50 * 1.05**i}
        for i, d in enumerate(dates)
    })
    _derive_series(td)

    p = build_growth_posterior(
        td=td, roe=0.18, payout=0.30,
        roic=0.15, g_roic_reinvest=0.06, analyst_g=0.10,
    )
    assert_true(p.n_sources >= 3, f"At least 3 sources (got {p.n_sources})")
    assert_true(0.04 < p.mu < 0.20, f"Posterior mu in plausible range ({p.mu:.3f})")
    assert_true(0.01 < p.sigma < 0.15, f"Posterior sigma reasonable ({p.sigma:.3f})")

    # Weights should sum to 1
    total_w = sum(s.weight for s in p.sources)
    assert_close(total_w, 1.0, tol=1e-6, label="Weights sum to 1.0")

    # Format table — just check it doesn't crash
    table = format_posterior_table(p)
    assert_true("Posterior" in table, "Table contains Posterior row")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 4: Financial sector models
# ════════════════════════════════════════════════════════════════════════════

def make_bank_ticker():
    dates = pd.date_range("2021-12-31", periods=4, freq="YE")
    td = TickerData(
        ticker="BANK", sector="Financial Services",
        industry="Banks—Commercial",
        shares_outstanding=500_000_000,
        book_value_per_share=40.0,
        return_on_equity=0.12,
        payout_ratio=0.40,
        current_dividend=1.20,
        earnings_growth=0.06,
        current_price=55.0,
    )
    td.income_stmt = pd.DataFrame({
        d: {"Net Income": 3_000e6 * 1.06**i, "Diluted EPS": 6.0 * 1.06**i,
            "Pretax Income": 3_600e6 * 1.06**i, "Tax Provision": 600e6 * 1.06**i}
        for i, d in enumerate(dates)
    })
    td.balance_sheet = pd.DataFrame({
        d: {"Stockholders Equity": 20_000e6, "Total Assets": 200_000e6,
            "Current Liabilities": 50_000e6, "Total Debt": 30_000e6,
            "Cash And Cash Equivalents": 5_000e6}
        for d in dates
    })
    _derive_series(td)
    return td


def make_reit_ticker():
    td = TickerData(
        ticker="REIT", sector="Real Estate",
        industry="REIT—Industrial",
        shares_outstanding=200_000_000,
        book_value_per_share=25.0,
        current_dividend=2.00,
        current_price=40.0,
    )
    td.income_stmt = pd.DataFrame({
        pd.Timestamp("2024-12-31"): {
            "Net Income": 500e6,
            "Pretax Income": 600e6,
            "Tax Provision": 100e6,
        }
    })
    td.balance_sheet = pd.DataFrame({
        pd.Timestamp("2024-12-31"): {
            "Total Assets": 5_000e6,
            "Stockholders Equity": 4_500e6,
            "Current Liabilities": 200e6,
            "Total Debt": 300e6,
            "Cash And Cash Equivalents": 100e6,
        }
    })
    _derive_series(td)
    return td


def make_insurer_ticker():
    td = TickerData(
        ticker="INS", sector="Financial Services",
        industry="Insurance—Life",
        shares_outstanding=300_000_000,
        current_dividend=1.50,
        current_price=50.0,
    )
    return td


# ── Classify ──────────────────────────────────────────────────────────────

def test_classification():
    print("\n[classify() routes to correct subsector]")
    td_bank = make_bank_ticker()
    td_reit = make_reit_ticker()
    td_ins = make_insurer_ticker()
    td_tech = TickerData(ticker="T", sector="Technology", industry="Software")

    assert_close(classify(td_bank).name, "BANK", label="Bank ticker classified correctly")
    assert_close(classify(td_reit).name, "REIT", label="REIT ticker classified correctly")
    assert_close(classify(td_ins).name, "INSURER", label="Insurer classified correctly")
    assert_close(classify(td_tech).name, "NOT_FINANCIAL", label="Tech is not financial")


# ── Bank excess return model ──────────────────────────────────────────────

def test_bank_excess_return_formula():
    """Hand-verify year 1 excess return and terminal value."""
    print("\n[Bank excess return formula correctness]")
    bvps, roe_h, roe_s, g, ke = 40.0, 0.14, 0.10, 0.07, 0.10

    iv, inputs = _bank_excess_return(
        bvps=bvps, roe_high=roe_h, roe_stable=roe_s,
        g_high=g, g_stable=0.03, ke=ke, years_high=5,
    )
    # Year 1: BV_0 * (ROE_1 - ke) discounted
    # ROE_1 fades: roe_h*(1-1/5) + roe_s*(1/5) = 0.14*0.8 + 0.10*0.2 = 0.132
    roe_1 = roe_h * 0.8 + roe_s * 0.2
    excess_1 = bvps * (roe_1 - ke)
    pv_1 = excess_1 / (1 + ke)
    assert_close(inputs["stage_log"][0]["pv"], pv_1, tol=0.001,
                 label=f"Year 1 PV excess matches hand calc ({pv_1:.4f})")

    # When ROE exactly = ke, should still produce a valid (potentially low) value
    iv_zero, _ = _bank_excess_return(
        bvps=40.0, roe_high=0.10, roe_stable=0.10,
        g_high=0.05, g_stable=0.03, ke=0.10, years_high=5,
    )
    assert_true(iv_zero is not None and iv_zero > 0,
                f"ROE=ke gives zero excess but still valid (BV=IV): got {iv_zero:.2f}")


def test_bank_valuation_end_to_end():
    print("\n[Bank valuation end-to-end]")
    td = make_bank_ticker()
    result = value_bank(td, ke=0.10, g_stable=0.025, margin_of_safety=0.25)

    assert_close(result.subsector.name, "BANK", label="Subsector = BANK")
    assert_true(result.intrinsic_value_per_share is not None,
                "IV per share computed")
    assert_true(result.intrinsic_value_per_share > 0, "IV > 0")
    assert_close(result.buy_price,
                 result.intrinsic_value_per_share * 0.75,
                 label="Buy price = IV × (1 - 25% MOS)")
    assert_true(result.reliability in ("High", "Moderate"),
                f"Reliability reasonable (got {result.reliability})")

    # Historical ROE derived from balance sheet
    roe = _derive_historical_roe(td)
    assert_close(roe, 0.15, tol=0.02, label="Historical ROE ≈ 15%")


def test_bank_negative_bvps_handled():
    print("\n[Bank: negative book value returns Flag]")
    td = make_bank_ticker()
    td.book_value_per_share = -5.0  # negative equity
    td.balance_sheet = None
    result = value_bank(td, ke=0.10, g_stable=0.025)
    assert_close(result.reliability, "Flag",
                 label="Negative BVPS flagged (not crashed)")


# ── REIT model ────────────────────────────────────────────────────────────

def test_reit_affo_derivation():
    print("\n[REIT AFFO derivation]")
    td = make_reit_ticker()
    affo_ps, note = _derive_affo(td)
    # NI = 500M, DA ≈ 5000M*0.03 = 150M, MaintCapex ≈ 5000M*0.12 = 600M
    # AFFO = 500+150-600 = 50M → per share = 50M / 200M = $0.25
    assert_true(affo_ps is not None, "AFFO per share computed")
    assert_close(affo_ps, 0.25, tol=0.05, label="AFFO/share ≈ $0.25")


def test_reit_valuation_end_to_end():
    print("\n[REIT P/AFFO valuation end-to-end]")
    td = make_reit_ticker()
    result = value_reit(td, ke=0.06, margin_of_safety=0.20)
    assert_close(result.subsector.name, "REIT", label="Subsector = REIT")
    assert_true(result.intrinsic_value_per_share is not None, "IV computed")
    # AFFO/sh ≈ $0.25, industrial REIT multiple ≈ 22 → IV ≈ $5.50
    # (small because NI is small relative to asset base in this toy)
    assert_true(result.intrinsic_value_per_share > 0, "IV > 0")
    assert_close(result.reliability, "Moderate", label="Reliability = Moderate")


# ── Insurer model ─────────────────────────────────────────────────────────

def test_insurer_ddm_and_flag():
    print("\n[Insurer: DDM computed + flag message]")
    td = make_insurer_ticker()
    result = value_insurer(td, ke=0.09, g_stable=0.025)
    # IV = 1.50 * 1.025 / (0.09 - 0.025) = 1.5375 / 0.065 ≈ $23.65
    assert_close(result.intrinsic_value_per_share, 1.50 * 1.025 / (0.09 - 0.025),
                 tol=0.01, label="DDM IV ≈ $23.65")
    assert_close(result.reliability, "Low", label="Reliability = Low (flag)")
    assert_true(any("Embedded Value" in n for n in result.notes),
                "Warning note mentions Embedded Value")


def test_insurer_no_dividend_returns_flag():
    print("\n[Insurer with no dividend → Flag]")
    td = make_insurer_ticker()
    td.current_dividend = None
    result = value_insurer(td, ke=0.09, g_stable=0.025)
    assert_true(result.intrinsic_value_per_share is None,
                "No dividend → no IV (flagged)")
    assert_close(result.reliability, "Flag", label="Reliability = Flag")


# ── Router ───────────────────────────────────────────────────────────────

def test_router_returns_none_for_non_financial():
    print("\n[Router returns None for non-financial sector]")
    td = TickerData(ticker="MSFT", sector="Technology", industry="Software")
    result = value_financial(td, ke=0.09, g_stable=0.025)
    assert_true(result is None, "Non-financial returns None (caller uses DCF)")


def test_router_dispatches_correctly():
    print("\n[Router dispatches bank / REIT / insurer correctly]")
    bank = value_financial(make_bank_ticker(), ke=0.10, g_stable=0.025)
    reit = value_financial(make_reit_ticker(), ke=0.06, g_stable=0.025)
    ins = value_financial(make_insurer_ticker(), ke=0.09, g_stable=0.025)
    assert_close(bank.subsector.name, "BANK",    label="Router → bank")
    assert_close(reit.subsector.name, "REIT",    label="Router → REIT")
    assert_close(ins.subsector.name,  "INSURER", label="Router → insurer")


# ── MC integration with posterior ────────────────────────────────────────

def test_mc_uses_posterior():
    """When posterior sigma > point sigma, MC distribution should be wider."""
    print("\n[MC: posterior-based wider than point-estimate MC]")
    from engine import monte_carlo_dcf, monte_carlo_from_posterior
    from growth_pool import GrowthPosterior, GrowthSource

    fcf0, shares, wacc, tg = 1_000_000, 10_000, 0.09, 0.025

    # Narrow point MC
    mc_point = monte_carlo_dcf(fcf0, shares, growth_mean=0.08,
                                growth_sigma=0.02, wacc_mean=wacc,
                                wacc_sigma=0.01, terminal_g=tg, seed=1)

    # Wide posterior MC
    wide_post = GrowthPosterior(mu=0.08, sigma=0.12,
                                sources=[GrowthSource("A", 0.08, 0.12)],
                                n_sources=1)
    wide_post.sources[0].weight = 1.0
    mc_post = monte_carlo_from_posterior(fcf0, shares, wide_post, wacc, 0.01, tg, seed=1)

    spread_point = mc_point["p90"] - mc_point["p10"]
    spread_post = mc_post["p90"] - mc_post["p10"]
    assert_true(spread_post > spread_point,
                f"Posterior MC IQR wider: {spread_post:.0f} > {spread_point:.0f}")


# ─── Runner ─────────────────────────────────────────────────────────────────

def run():
    stage3 = [
        ("IV-weighting: precise source dominates",  test_inverse_variance_weighting),
        ("Equal sigma → arithmetic mean",            test_equal_sigma_is_simple_average),
        ("More sources → narrower sigma",            test_posterior_sigma_narrower_with_more_sources),
        ("Empty sources → default posterior",        test_empty_sources_returns_default),
        ("Jackknife sigma stable",                   test_jackknife_sigma_stable),
        ("Bootstrap sigma: volatile > stable",       test_bootstrap_sigma_wider_for_volatile),
        ("Log-return SE decreases with more data",   test_fcf_log_return_sigma),
        ("Full posterior on synthetic ticker",       test_full_posterior_with_synthetic_ticker),
        ("MC: posterior wider than point MC",        test_mc_uses_posterior),
    ]
    stage4 = [
        ("classify() routes correctly",              test_classification),
        ("Bank excess return formula",               test_bank_excess_return_formula),
        ("Bank valuation end-to-end",                test_bank_valuation_end_to_end),
        ("Bank negative BV → Flag",                  test_bank_negative_bvps_handled),
        ("REIT AFFO derivation",                     test_reit_affo_derivation),
        ("REIT P/AFFO valuation",                    test_reit_valuation_end_to_end),
        ("Insurer DDM + flag",                       test_insurer_ddm_and_flag),
        ("Insurer no dividend → Flag",               test_insurer_no_dividend_returns_flag),
        ("Router → None for non-financial",          test_router_returns_none_for_non_financial),
        ("Router dispatches bank/REIT/insurer",      test_router_dispatches_correctly),
    ]
    all_tests = stage3 + stage4
    print(f"\n{'═'*60}\nRunning Stage 3 ({len(stage3)} tests) + Stage 4 ({len(stage4)} tests)\n{'═'*60}")
    for name, fn in all_tests:
        fn()
    import shutil
    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{'='*60}\nAll {len(all_tests)} tests passed ✓\n{'='*60}")


if __name__ == "__main__":
    run()
