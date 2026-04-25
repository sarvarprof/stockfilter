"""Demo: full report for synthetic ticker so we can see the output format."""
import logging
import sys

sys.path.insert(0, ".")

from test_valuation import make_synthetic_ticker
from macro import MacroEnvironment
from assumptions import build
from valuator import StockValuator, print_report
from data import TickerData


# Monkey-patch the data loader so StockValuator uses synthetic data
import data as data_layer
_real_load = data_layer.load
def _fake_load(_ticker: str) -> TickerData:
    td = make_synthetic_ticker()
    td.ticker = _ticker  # so the report header looks right
    return td
data_layer.load = _fake_load


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    macro = MacroEnvironment(
        risk_free_rate=0.043,
        equity_risk_premium=0.0423,
        long_run_gdp=0.04,
    )
    v = StockValuator("DEMO", macro=macro)
    print_report(v.report())
