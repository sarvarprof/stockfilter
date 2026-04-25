"""
Top-level orchestrator and CLI report.

Flow: TickerData (data.py + edgar.py) → Assumptions (assumptions.py)
      → Methods (engine.py) → structured Report dict + console output.

Data source priority:
  1. EDGAR for fundamentals (when EDGAR_USER_AGENT env var is set and
     ticker is a SEC filer)
  2. yfinance for market data (price, beta, market cap) always
  3. yfinance for fundamentals as fallback (ADRs, foreign listings)
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

import pandas as pd

import data as data_layer
import engine
from assumptions import Assumptions, build as build_assumptions
from data import TickerData, _row
from macro import MacroEnvironment

log = logging.getLogger(__name__)


class StockValuator:
    """Top-level façade. Loads data, builds assumptions, runs all methods."""

    def __init__(self, ticker: str, macro: MacroEnvironment | None = None,
                 user_agent: str | None = None, prefer_edgar: bool = True):
        self.ticker = ticker
        self.macro = macro or MacroEnvironment.fetch()

        # Try EDGAR-primary loader if a User-Agent is available; else yfinance only
        ua = user_agent or os.environ.get("EDGAR_USER_AGENT")
        if prefer_edgar and ua:
            try:
                import edgar
                self.data: TickerData = edgar.load(ticker, user_agent=ua)
            except Exception as e:
                log.warning("EDGAR-primary loader failed (%s); using yfinance only", e)
                self.data = data_layer.load(ticker)
        else:
            self.data = data_layer.load(ticker)

        if not self.data.shares_outstanding or self.data.shares_outstanding <= 0:
            raise ValueError(f"{ticker}: missing share count, cannot value")
        self.assumptions: Assumptions = build_assumptions(self.data, self.macro)

    # ─────────────────────────────────────────────────────────── methods ──

    def dcf(self, growth_rate: float) -> float | None:
        return engine.two_stage_dcf(
            fcf0=self.assumptions.fcf_normalized,
            growth_rate=growth_rate,
            wacc=self.assumptions.wacc,
            terminal_g=self.assumptions.terminal_growth,
            shares=self.data.shares_outstanding,
        )

    def buffett(self, growth_rate: float) -> tuple:
        # Use forward EPS if available; else trailing
        eps = self.data.forward_eps if (self.data.forward_eps and self.data.forward_eps > 0) \
              else self.data.trailing_eps
        if not eps or eps <= 0:
            return None, None
        return engine.buffett_valuation(
            eps0=float(eps),
            growth_rate=growth_rate,
            terminal_g=self.assumptions.terminal_growth,
            cost_of_equity=self.assumptions.cost_of_equity,  # ← key fix
            future_pe=self.assumptions.future_pe,
            margin_of_safety=self.assumptions.margin_of_safety,
        )

    def gordon(self) -> float | str:
        div = self.data.current_dividend or 0
        div_g = engine.historical_div_growth(self.data.dividends)
        return engine.gordon_growth(
            div=div,
            div_growth=div_g,
            cost_of_equity=self.assumptions.cost_of_equity,
        )

    def graham(self) -> float | None:
        eps = self.data.trailing_eps or 0
        bvps = self.data.book_value_per_share or 0
        return engine.graham_number(eps, bvps)

    def ev_ebitda(self) -> float | None:
        ebitda = self.data.ebitda_point
        if ebitda is None:
            # Fall back to info.ebitda via a fresh ticker call
            try:
                import yfinance as yf
                info = yf.Ticker(self.ticker).info
                ebitda = info.get("ebitda")
            except Exception:
                ebitda = None
        if not ebitda:
            return None

        debt = 0.0
        cash = 0.0
        if self.data.balance_sheet is not None:
            d = _row(self.data.balance_sheet, "total_debt")
            c = _row(self.data.balance_sheet, "cash")
            if d is not None and len(d): debt = float(d.iloc[-1])
            if c is not None and len(c): cash = float(c.iloc[-1])

        return engine.ev_ebitda_value(
            ebitda=float(ebitda),
            multiple=self.assumptions.ev_ebitda_multiple,
            debt=debt,
            cash=cash,
            shares=self.data.shares_outstanding,
        )

    def monte_carlo(self) -> dict:
        a = self.assumptions
        wacc = a.wacc
        if a.growth_posterior is not None:
            return engine.monte_carlo_from_posterior(
                fcf0=a.fcf_normalized,
                shares=self.data.shares_outstanding,
                posterior=a.growth_posterior,
                wacc_mean=wacc,
                wacc_sigma=max(0.005, wacc * 0.10),
                terminal_g=a.terminal_growth,
            )
        # fallback: plain MC with point estimates
        return engine.monte_carlo_dcf(
            fcf0=a.fcf_normalized,
            shares=self.data.shares_outstanding,
            growth_mean=a.growth_blended,
            growth_sigma=a.fcf_growth_sigma,
            wacc_mean=wacc,
            wacc_sigma=max(0.005, wacc * 0.10),
            terminal_g=a.terminal_growth,
        )

    def reverse_dcf_implied(self) -> float | None:
        return engine.reverse_dcf(
            fcf0=self.assumptions.fcf_normalized,
            shares=self.data.shares_outstanding,
            current_price=self.data.current_price or 0,
            wacc=self.assumptions.wacc,
            terminal_g=self.assumptions.terminal_growth,
        )

    def sensitivity(self, growth_rate: float | None = None) -> pd.DataFrame:
        if growth_rate is None:
            growth_rate = self.assumptions.growth_blended
        return engine.sensitivity_table(
            fcf0=self.assumptions.fcf_normalized,
            shares=self.data.shares_outstanding,
            growth_rate=growth_rate,
            wacc_base=self.assumptions.wacc,
            terminal_g_base=self.assumptions.terminal_growth,
        )

    # ───────────────────────────────────────────────────────── full run ──

    def report(self) -> dict[str, Any]:
        a = self.assumptions
        p = self.data.current_price or 0

        # Run all methods
        dcf_f = self.dcf(a.growth_fundamental)
        dcf_a = self.dcf(a.growth_analyst)
        dcf_b = self.dcf(a.growth_blended)
        bear_g = max(a.terminal_growth, a.growth_fundamental * 0.4)
        bull_g = min(a.growth_analyst * 1.5, 0.40)
        dcf_bear = self.dcf(bear_g)
        dcf_bull = self.dcf(bull_g)

        iv_f, buy_f = self.buffett(a.growth_fundamental)
        iv_a, buy_a = self.buffett(a.growth_analyst)

        ggm = self.gordon()
        ggm_val = ggm if isinstance(ggm, (int, float)) else None
        gn = self.graham()
        ev_v = self.ev_ebitda()
        mc = self.monte_carlo()
        implied = self.reverse_dcf_implied()
        sens = self.sensitivity()

        # Adaptive verdict: use MC IQR for DCF-family, ratio bands for others
        mc_args = {}
        if "error" not in mc:
            mc_args = {"mc_p25": mc["p25"], "mc_p50": mc["p50"], "mc_p75": mc["p75"]}

        def verdict_dcf(v):
            return engine.adaptive_verdict(v, p, use_mc_distribution=True, **mc_args)

        def verdict_other(v):
            return engine.adaptive_verdict(v, p)

        # Probabilistic 25/50/25
        if dcf_b is not None:
            bear_v = dcf_bear if dcf_bear is not None else dcf_b * 0.7
            bull_v = dcf_bull if dcf_bull is not None else dcf_b * 1.3
            prob = bear_v * 0.25 + dcf_b * 0.5 + bull_v * 0.25
        else:
            prob = None

        methods = {
            "dcf_fundamental":   {"name": "DCF — Fundamental (SGR)", "value": dcf_f,
                                  "verdict": verdict_dcf(dcf_f), "growth_used": a.growth_fundamental},
            "dcf_analyst":       {"name": "DCF — Analyst Consensus", "value": dcf_a,
                                  "verdict": verdict_dcf(dcf_a), "growth_used": a.growth_analyst},
            "dcf_blended":       {"name": "DCF — Blended Growth", "value": dcf_b,
                                  "verdict": verdict_dcf(dcf_b), "growth_used": a.growth_blended},
            "buffett_fundamental": {"name": "Buffett IV — Fundamental", "value": iv_f,
                                    "buy_price": buy_f, "verdict": verdict_other(iv_f),
                                    "growth_used": a.growth_fundamental},
            "buffett_analyst":   {"name": "Buffett IV — Analyst", "value": iv_a,
                                  "buy_price": buy_a, "verdict": verdict_other(iv_a),
                                  "growth_used": a.growth_analyst},
            "probabilistic":     {"name": "Probabilistic DCF", "value": prob,
                                  "verdict": verdict_dcf(prob)},
            "ggm":               {"name": "Gordon Growth Model", "value": ggm_val,
                                  "note": ggm if isinstance(ggm, str) else None,
                                  "verdict": verdict_other(ggm_val)},
            "graham_number":     {"name": "Graham Number", "value": gn,
                                  "verdict": verdict_other(gn)},
            "ev_ebitda":         {"name": "EV/EBITDA (Sector)", "value": ev_v,
                                  "verdict": verdict_other(ev_v)},
        }
        # Financial subsector routing (Stage 4)
        from financials import value_financial, FinancialSubsector
        fin_val = value_financial(
            td=self.data,
            ke=a.cost_of_equity,
            g_stable=a.terminal_growth,
            margin_of_safety=a.margin_of_safety,
        )

        if fin_val is not None and fin_val.intrinsic_value_per_share is not None:
            methods[f"sector_specific"] = {
                "name": fin_val.method_name,
                "value": fin_val.intrinsic_value_per_share,
                "buy_price": fin_val.buy_price,
                "verdict": verdict_other(fin_val.intrinsic_value_per_share),
                "reliability": fin_val.reliability,
                "notes": fin_val.notes,
            }

        if "error" not in mc:
            methods["monte_carlo"] = {
                "name": "Monte Carlo DCF (10y, lognormal)", **mc,
                "value": mc["p50"], "verdict": verdict_dcf(mc["p50"]),
            }

        # Consensus tally
        verdicts = [m["verdict"] for m in methods.values() if m["verdict"] != "N/A"]
        uv = sum(1 for v in verdicts if "Under" in v or "Upside" in v)
        ov = sum(1 for v in verdicts if "Over" in v or "Downside" in v)
        fv = len(verdicts) - uv - ov
        consensus = ("Undervalued" if uv > ov and uv >= fv
                     else "Overvalued" if ov > uv and ov >= fv
                     else "Fairly Valued")

        # ROIC vs WACC
        roic_signal = None
        if a.roic is not None:
            spread = a.roic - a.wacc
            roic_signal = ("Strong moat (ROIC >> WACC)" if spread >= 0.10
                           else "Solid returns (ROIC > WACC)" if spread >= 0.03
                           else "Adequate (ROIC ≈ WACC)" if spread >= 0.0
                           else "Value-destroying (ROIC < WACC)")

        return {
            "ticker": self.ticker,
            "current_price": p,
            "implied_growth": implied,
            "financial_subsector": fin_val.subsector.name if fin_val else None,
            "financial_reliability": fin_val.reliability if fin_val else None,
            "consensus": consensus,
            "consensus_counts": {"undervalued": uv, "overvalued": ov, "fairly_valued": fv},
            "macro": {
                "rf": a.risk_free_rate,
                "erp": a.equity_risk_premium,
                "long_run_gdp": a.long_run_gdp,
                "source_rf": self.macro.source_rf,
                "source_erp": self.macro.source_erp,
            },
            "assumptions": {
                "wacc": round(a.wacc, 4),
                "cost_of_equity": round(a.cost_of_equity, 4),
                "cost_of_debt": round(a.cost_of_debt, 4),
                "tax_rate": round(a.effective_tax_rate, 4),
                "beta_raw": round(a.beta_raw, 2),
                "beta_adjusted": round(a.beta_adjusted, 2),
                "equity_weight": round(a.equity_weight, 3),
                "debt_weight": round(a.debt_weight, 3),
                "terminal_g": round(a.terminal_growth, 4),
                "future_pe": round(a.future_pe, 1),
                "margin_of_safety": a.margin_of_safety,
                "growth_fundamental_sgr": round(a.growth_fundamental, 4),
                "growth_roic_reinvest": round(a.growth_roic_reinvest, 4) if a.growth_roic_reinvest else None,
                "growth_historical_eps": round(a.growth_historical_eps, 4) if a.growth_historical_eps else None,
                "growth_historical_fcf": round(a.growth_historical_fcf, 4) if a.growth_historical_fcf else None,
                "growth_analyst": round(a.growth_analyst, 4),
                "growth_blended": round(a.growth_blended, 4),
                "growth_posterior": {
                    "mu": round(a.growth_posterior.mu, 4),
                    "sigma": round(a.growth_posterior.sigma, 4),
                    "n_sources": a.growth_posterior.n_sources,
                    "sources": [
                        {"name": s.name, "mean": round(s.mean, 4),
                         "sigma": round(s.sigma, 4), "weight": round(s.weight, 4)}
                        for s in a.growth_posterior.sources
                    ],
                } if a.growth_posterior else None,
                "fcf_normalized": round(a.fcf_normalized, 0),
                "fcf_growth_sigma": round(a.fcf_growth_sigma, 4),
                "roic": round(a.roic, 4) if a.roic is not None else None,
                "roic_signal": roic_signal,
            },
            "methods": methods,
            "sensitivity": sens,
            "data_warnings": self.data.warnings,
            "assumption_notes": a.notes,
        }


# ───────────────────────────────────────────────────────────────── CLI ──

def print_report(rep: dict) -> None:
    p = rep["current_price"]
    a = rep["assumptions"]
    m = rep["macro"]
    print(f"{'='*72}")
    print(f"  VALUATION REPORT: {rep['ticker']}  |  Price: ${p:.2f}")
    print(f"{'='*72}")
    print(f"Macro: rf {m['rf']*100:.2f}% ({m['source_rf']})  "
          f"ERP {m['erp']*100:.2f}% ({m['source_erp']})")
    print(f"Discount: WACC {a['wacc']*100:.2f}%  ke {a['cost_of_equity']*100:.2f}%  "
          f"kd {a['cost_of_debt']*100:.2f}%  tax {a['tax_rate']*100:.1f}%")
    print(f"Beta: raw {a['beta_raw']:.2f} → Blume-adj {a['beta_adjusted']:.2f}  |  "
          f"weights: E {a['equity_weight']:.0%} / D {a['debt_weight']:.0%}")
    print(f"Terminal g: {a['terminal_g']*100:.2f}%  Exit P/E: {a['future_pe']:.1f}")

    print(f"\nGrowth rates considered:")
    print(f"  SGR (ROE×retention) ........ {a['growth_fundamental_sgr']*100:5.1f}%")
    if a['growth_roic_reinvest'] is not None:
        print(f"  ROIC × Reinvestment Rate ... {a['growth_roic_reinvest']*100:5.1f}%")
    if a['growth_historical_eps'] is not None:
        print(f"  Historical EPS CAGR ........ {a['growth_historical_eps']*100:5.1f}%")
    if a['growth_historical_fcf'] is not None:
        print(f"  Historical FCF CAGR ........ {a['growth_historical_fcf']*100:5.1f}%")
    print(f"  Analyst forward consensus .. {a['growth_analyst']*100:5.1f}%")
    print(f"  Blended (median) ........... {a['growth_blended']*100:5.1f}%  ← used as point")

    p = rep["assumptions"].get("growth_posterior")
    if p and p.get("n_sources", 0) > 0:
        print(f"\nGrowth posterior (Bayesian IV-weighted, {p['n_sources']} sources):")
        hdr = f"  {'Source':<28} {'Mean':>7} {'Sigma':>7} {'Weight':>7}"
        print(hdr)
        print("  " + "─" * 52)
        for s in p["sources"]:
            print(f"  {s['name']:<28} {s['mean']*100:>6.1f}% {s['sigma']*100:>6.1f}%"
                  f" {s['weight']*100:>6.1f}%")
        print("  " + "─" * 52)
        print(f"  {'Posterior':<28} {p['mu']*100:>6.1f}% {p['sigma']*100:>6.1f}%")

    if a['roic'] is not None:
        print(f"\nROIC {a['roic']*100:.1f}%  →  {a['roic_signal']}")
    if rep['implied_growth'] is not None:
        print(f"Reverse DCF: market is pricing in {rep['implied_growth']*100:.1f}% perpetual FCF growth")
    print(f"FCF (3y avg): ${a['fcf_normalized']:,.0f}   σ(log returns): {a['fcf_growth_sigma']*100:.1f}%")

    print(f"\n{'─'*72}\nMethod                                         Value     Buy   [growth] → Verdict")
    print('─'*72)
    for key, mthd in rep["methods"].items():
        v = mthd.get("value")
        v_str = f"${v:>9.2f}" if v is not None else f"  {'N/A':>9}"
        buy = f"  ${mthd['buy_price']:>6.2f}" if mthd.get("buy_price") else " " * 9
        g = f" [{mthd['growth_used']*100:>4.1f}%]" if mthd.get("growth_used") else " " * 8
        print(f"  {mthd['name']:<42} {v_str}{buy}{g} → {mthd['verdict']}")

    if "monte_carlo" in rep["methods"]:
        mc = rep["methods"]["monte_carlo"]
        print(f"\n  Monte Carlo distribution: P10 ${mc['p10']:.2f}  "
              f"P25 ${mc['p25']:.2f}  P50 ${mc['p50']:.2f}  "
              f"P75 ${mc['p75']:.2f}  P90 ${mc['p90']:.2f}")

    print(f"\nSensitivity (intrinsic value vs WACC × terminal_g):")
    sens = rep["sensitivity"]
    sens_str = sens.to_string(float_format=lambda x: f"${x:>7.2f}" if pd.notna(x) else "    N/A")
    for line in sens_str.split("\n"):
        print(f"  {line}")

    c = rep["consensus_counts"]
    print(f"\nConsensus: {rep['consensus']}  "
          f"({c['undervalued']} under / {c['fairly_valued']} fair / {c['overvalued']} over)")

    if rep.get("financial_subsector") and rep["financial_subsector"] != "NOT_FINANCIAL":
        rel = rep.get("financial_reliability", "")
        rel_str = f"  [reliability: {rel}]" if rel else ""
        print(f"\n⚑  Financial firm detected: {rep['financial_subsector']}{rel_str}")
        # Print any flag notes from the sector-specific model
        if "sector_specific" in rep.get("methods", {}):
            for note in rep["methods"]["sector_specific"].get("notes", []):
                print(f"   {note}")
        print(f"\n⚠ Data warnings:")
        for w in rep["data_warnings"]:
            print(f"  • {w}")
    print('='*72)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    v = StockValuator(ticker)
    print_report(v.report())
