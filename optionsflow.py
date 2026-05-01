import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import os
import warnings
warnings.filterwarnings('ignore')

# matplotlib/seaborn are only used by the CLI dashboard. They are imported
# lazily inside generate_options_dashboard() so the Flask web UI can use
# this module's analyze_for_api() without those plotting dependencies.


# =====================================================================
# PARSERS & CLASSIFIERS
# =====================================================================

def parse_premium(prem_str):
    """Convert '$4.5M', '$800K', '$1.2B' strings to float."""
    if pd.isna(prem_str):
        return 0.0
    s = str(prem_str).replace('$', '').replace(',', '').upper().strip()
    if 'B' in s:
        return float(s.replace('B', '')) * 1_000_000_000
    if 'M' in s:
        return float(s.replace('M', '')) * 1_000_000
    if 'K' in s:
        return float(s.replace('K', '')) * 1_000
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_conditions(conds_str):
    """Extract boolean flags from the conds field."""
    s = str(conds_str).lower() if not pd.isna(conds_str) else ''
    return {
        'is_opening':  'opening' in s,
        'is_unusual':  'unusual' in s or 'highly_un' in s,
        'is_highly_unusual': 'highly_un' in s,
    }


def classify_aggressor(side, put_call):
    """
    Map (side, put_call) to a directional sentiment and numeric score.

    Score convention:
      +1.0  = strongly bullish
       0.5  = mildly bullish
       0.0  = neutral
      -0.5  = mildly bearish
      -1.0  = strongly bearish
    """
    side = str(side).lower().strip()
    pc   = str(put_call).lower().strip()

    mapping = {
        # Calls
        ('ask',   'call'): ('Bullish',        +1.0),
        ('above', 'call'): ('Strongly Bullish',+1.0),
        ('bid',   'call'): ('Bearish',         -0.5),   # covered call / close
        ('below', 'call'): ('Bearish',         -0.5),
        ('mid',   'call'): ('Neutral',          0.0),
        # Puts
        ('ask',   'put'):  ('Bearish',         -1.0),
        ('above', 'put'):  ('Strongly Bearish',-1.0),
        ('bid',   'put'):  ('Bullish',         +0.5),   # put write
        ('below', 'put'):  ('Bullish',         +0.5),
        ('mid',   'put'):  ('Neutral',          0.0),
    }
    sentiment, score = mapping.get((side, pc), ('Neutral', 0.0))
    return sentiment, score


def classify_dte_bucket(trade_date, expiry_date):
    """Bucket a trade by its days-to-expiry."""
    dte = (expiry_date - trade_date).days
    if dte <= 0:
        return '0DTE', dte
    elif dte <= 7:
        return 'Weekly (1-7d)', dte
    elif dte <= 30:
        return 'Monthly (8-30d)', dte
    elif dte <= 90:
        return 'Near-Term (31-90d)', dte
    elif dte <= 180:
        return 'Mid-Term (91-180d)', dte
    else:
        return 'LEAPS (180d+)', dte


def classify_execution_type(sweep_block_split):
    """Weight execution type for urgency/conviction scoring."""
    val = str(sweep_block_split).lower().strip()
    weights = {'sweep': 1.2, 'block': 1.1, 'split': 1.0}
    return val, weights.get(val, 1.0)


def classify_moneyness(spot, strike, put_call):
    """Return ITM / ATM / OTM and distance as % from spot."""
    if spot == 0:
        return 'Unknown', 0.0
    pct = (strike - spot) / spot * 100
    pc = str(put_call).lower()

    if pc == 'call':
        if pct > 2:   label = 'OTM'
        elif pct < -2: label = 'ITM'
        else:          label = 'ATM'
    else:  # put
        if pct < -2:  label = 'OTM'
        elif pct > 2:  label = 'ITM'
        else:          label = 'ATM'

    return label, round(pct, 2)


# =====================================================================
# CORE METRICS
# =====================================================================

def compute_breakeven(strike, premium_per_contract, put_call):
    """Return break-even price at expiry."""
    pc = str(put_call).lower()
    if pc == 'call':
        return round(strike + premium_per_contract, 2)
    else:
        return round(strike - premium_per_contract, 2)


def compute_bull_score(row):
    """
    Composite bull score for a single trade (0–100 scale after normalization).

    Factors:
      - Sentiment score            (−1 to +1)
      - Opening position bonus     (+0.3)
      - Unusual activity bonus     (+0.2)
      - Sweep urgency weight       (1.0–1.2×)
      - LEAPS conviction bonus     (+0.2)
      - V/OI ratio bonus           (capped at +0.3)
      - Premium size               (log-scaled, relative)
    """
    score = row['sentiment_score']

    if row['is_opening']:
        score += 0.3 * np.sign(score) if score != 0 else 0.1
    if row['is_unusual']:
        score += 0.2 * np.sign(score) if score != 0 else 0.1
    if row['is_highly_unusual']:
        score += 0.1 * np.sign(score) if score != 0 else 0.05

    score *= row['exec_weight']

    if row['dte_bucket'] == 'LEAPS (180d+)' and score > 0:
        score += 0.2

    voi = min(row['voi_ratio'], 10) / 10 * 0.3
    score += voi * np.sign(score) if score != 0 else 0

    return round(score, 4)


# =====================================================================
# WHALE / ACCUMULATION DETECTION
# =====================================================================

def detect_whale_accumulation(df, min_trades=3, min_total_premium=250_000):
    """
    Group by (symbol, expiry, strike, put_call) to find positions being
    built in multiple prints — a classic institutional accumulation signature.
    """
    grouped = (
        df.groupby(['symbol', 'expiry_dt', 'strike', 'put_call'])
        .agg(
            trade_count=('premium_val', 'count'),
            total_premium=('premium_val', 'sum'),
            avg_price=('price', 'mean'),
            avg_spot=('spot', 'mean'),
            net_bull_score=('bull_score', 'sum'),
            is_opening_any=('is_opening', 'any'),
            is_unusual_any=('is_unusual', 'any'),
            first_seen=('datetime', 'min'),
            last_seen=('datetime', 'max'),
            exec_types=('exec_type', lambda x: '/'.join(sorted(set(x)))),
        )
        .reset_index()
    )

    whales = grouped[
        (grouped['trade_count'] >= min_trades) &
        (grouped['total_premium'] >= min_total_premium)
    ].copy()

    whales['breakeven'] = whales.apply(
        lambda r: compute_breakeven(r['strike'], r['avg_price'], r['put_call']), axis=1
    )
    whales['required_move_pct'] = (
        (whales['breakeven'] - whales['avg_spot']) / whales['avg_spot'] * 100
    ).round(2)
    whales['direction'] = whales['net_bull_score'].apply(
        lambda s: 'Bullish' if s > 0 else ('Bearish' if s < 0 else 'Neutral')
    )

    return whales.sort_values('total_premium', ascending=False)


# =====================================================================
# AGGREGATION HELPERS
# =====================================================================

def compute_daily_net_flow(df):
    """Net bullish vs bearish premium per day."""
    df['bull_premium'] = df.apply(
        lambda r: r['premium_val'] if r['sentiment_score'] > 0 else 0, axis=1
    )
    df['bear_premium'] = df.apply(
        lambda r: r['premium_val'] if r['sentiment_score'] < 0 else 0, axis=1
    )
    daily = df.groupby(df['datetime'].dt.normalize()).agg(
        bull_premium=('bull_premium', 'sum'),
        bear_premium=('bear_premium', 'sum'),
        total_premium=('premium_val', 'sum'),
        trade_count=('premium_val', 'count'),
    ).reset_index()
    daily.rename(columns={'datetime': 'date'}, inplace=True)
    daily['net_flow'] = daily['bull_premium'] - daily['bear_premium']
    daily['cp_ratio'] = (daily['bull_premium'] / daily['bear_premium'].replace(0, np.nan)).round(2)
    return daily


def compute_strike_heatmap(df):
    """
    Build a (strike × expiry_label) pivot of total premium for the heatmap.
    Only include calls on the bullish side for a long-position view.
    """
    calls = df[df['put_call'] == 'call'].copy()
    calls['expiry_label'] = calls['expiry_dt'].dt.strftime('%b %d %Y')
    pivot = (
        calls.groupby(['strike', 'expiry_label'])['premium_val']
        .sum()
        .unstack(fill_value=0)
        / 1e6
    )
    # Sort strikes ascending, expiries chronologically
    expiry_order = (
        calls.groupby('expiry_label')['expiry_dt']
        .first()
        .sort_values()
        .index.tolist()
    )
    pivot = pivot.reindex(columns=[c for c in expiry_order if c in pivot.columns])
    return pivot


# =====================================================================
# VISUALIZATION DASHBOARD
# =====================================================================

def generate_options_dashboard(df, daily_flow, whale_df, pivot_heatmap, symbol):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    print("Generating Options Flow Dashboard...")

    # Style
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor':   '#161b22',
        'text.color':       '#e6edf3',
        'axes.labelcolor':  '#e6edf3',
        'xtick.color':      '#8b949e',
        'ytick.color':      '#8b949e',
        'axes.edgecolor':   '#30363d',
        'grid.color':       '#21262d',
        'axes.titlecolor':  '#e6edf3',
    })
    sns.set_theme(style='darkgrid')

    BULL  = '#2ea043'
    BEAR  = '#f85149'
    GOLD  = '#d29922'
    BLUE  = '#388bfd'
    PURP  = '#a371f7'
    TEAL  = '#39d353'

    fig = plt.figure(figsize=(24, 28))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f'{symbol} — Options Flow Analysis Dashboard',
                 fontsize=22, fontweight='bold', color='white', y=0.995)
    gs = fig.add_gridspec(4, 2, hspace=0.48, wspace=0.32)

    # ------------------------------------------------------------------
    # 1. Daily Net Premium Flow (full width, top)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(daily_flow))
    width = 0.38
    bars_bull = ax1.bar(x - width/2, daily_flow['bull_premium'] / 1e6,
                        width=width, label='Bullish Premium', color=BULL, alpha=0.9)
    bars_bear = ax1.bar(x + width/2, daily_flow['bear_premium'] / 1e6,
                        width=width, label='Bearish Premium', color=BEAR, alpha=0.9)

    # Net flow line
    ax1b = ax1.twinx()
    ax1b.plot(x, daily_flow['net_flow'] / 1e6, color=GOLD, linewidth=2.5,
              marker='D', markersize=6, label='Net Flow', zorder=5)
    ax1b.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax1b.set_ylabel('Net Flow ($M)', color=GOLD, fontsize=10)
    ax1b.tick_params(axis='y', colors=GOLD)

    ax1.set_xticks(x)
    ax1.set_xticklabels(daily_flow['date'].dt.strftime('%b %d'), rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Premium ($ Millions)', fontsize=11)
    ax1.set_title('Daily Bullish vs Bearish Premium Flow', fontsize=14, fontweight='bold')

    # C/P ratio labels
    for xi, (_, row) in zip(x, daily_flow.iterrows()):
        if pd.notna(row['cp_ratio']):
            ax1.text(xi, max(row['bull_premium'], row['bear_premium']) / 1e6 + 0.5,
                     f"C/P {row['cp_ratio']:.1f}x",
                     ha='center', fontsize=7, color=GOLD)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.4, fontsize=9)

    # ------------------------------------------------------------------
    # 2. Call Premium Heatmap (strike × expiry)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, :])
    if not pivot_heatmap.empty:
        # Limit to top 20 strikes by total premium
        top_strikes = pivot_heatmap.sum(axis=1).nlargest(20).index
        heatmap_data = pivot_heatmap.loc[top_strikes].sort_index(ascending=False)
        sns.heatmap(
            heatmap_data,
            ax=ax2,
            cmap='YlOrRd',
            linewidths=0.5,
            linecolor='#0d1117',
            fmt='.1f',
            annot=True,
            annot_kws={'size': 8},
            cbar_kws={'label': 'Premium ($M)', 'shrink': 0.8},
        )
        ax2.set_title('Call Premium Heatmap — Strike × Expiry ($M)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Expiry', fontsize=10)
        ax2.set_ylabel('Strike', fontsize=10)
        ax2.tick_params(axis='x', rotation=30)
    else:
        ax2.text(0.5, 0.5, 'Insufficient call data for heatmap',
                 transform=ax2.transAxes, ha='center', va='center', color='white')

    # ------------------------------------------------------------------
    # 3. DTE Bucket Breakdown
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    dte_order = ['0DTE', 'Weekly (1-7d)', 'Monthly (8-30d)',
                 'Near-Term (31-90d)', 'Mid-Term (91-180d)', 'LEAPS (180d+)']
    dte_palette = {
        '0DTE':              '#ff6b6b',
        'Weekly (1-7d)':     '#ffa07a',
        'Monthly (8-30d)':   '#ffd700',
        'Near-Term (31-90d)':BULL,
        'Mid-Term (91-180d)':BLUE,
        'LEAPS (180d+)':     PURP,
    }
    dte_bull = df[df['sentiment_score'] > 0].groupby('dte_bucket')['premium_val'].sum() / 1e6
    dte_bear = df[df['sentiment_score'] < 0].groupby('dte_bucket')['premium_val'].sum() / 1e6

    dte_x = [b for b in dte_order if b in dte_bull.index or b in dte_bear.index]
    xi = np.arange(len(dte_x))
    w = 0.4
    ax3.bar(xi - w/2, [dte_bull.get(b, 0) for b in dte_x], width=w, label='Bullish', color=BULL, alpha=0.9)
    ax3.bar(xi + w/2, [dte_bear.get(b, 0) for b in dte_x], width=w, label='Bearish', color=BEAR, alpha=0.9)
    ax3.set_xticks(xi)
    ax3.set_xticklabels(dte_x, rotation=25, ha='right', fontsize=8)
    ax3.set_title('Premium by DTE Bucket', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Premium ($M)', fontsize=10)
    ax3.legend(fontsize=9, framealpha=0.4)

    # ------------------------------------------------------------------
    # 4. Execution Type (Sweep / Block / Split)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    exec_bull = df[df['sentiment_score'] > 0].groupby('exec_type')['premium_val'].sum() / 1e6
    exec_bear = df[df['sentiment_score'] < 0].groupby('exec_type')['premium_val'].sum() / 1e6
    exec_types = sorted(set(exec_bull.index) | set(exec_bear.index))
    xi2 = np.arange(len(exec_types))
    ax4.bar(xi2 - w/2, [exec_bull.get(t, 0) for t in exec_types], width=w, label='Bullish', color=BULL, alpha=0.9)
    ax4.bar(xi2 + w/2, [exec_bear.get(t, 0) for t in exec_types], width=w, label='Bearish', color=BEAR, alpha=0.9)
    ax4.set_xticks(xi2)
    ax4.set_xticklabels([t.title() for t in exec_types], fontsize=10)
    ax4.set_title('Premium by Execution Type', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Premium ($M)', fontsize=10)
    ax4.legend(fontsize=9, framealpha=0.4)

    # ------------------------------------------------------------------
    # 5. Unusual / Opening Activity Scatter (V/OI vs Premium)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[3, 0])
    unusual_df = df[df['voi_ratio'] > 1].copy()
    if not unusual_df.empty:
        colors_sc = unusual_df['sentiment_score'].apply(
            lambda s: BULL if s > 0 else (BEAR if s < 0 else '#8b949e')
        )
        sizes_sc = (unusual_df['premium_val'] / unusual_df['premium_val'].max() * 300).clip(lower=20)
        sc = ax5.scatter(
            unusual_df['voi_ratio'].clip(upper=20),
            unusual_df['premium_val'] / 1e6,
            c=colors_sc, s=sizes_sc, alpha=0.7,
            edgecolors='white', linewidth=0.5
        )
        ax5.axvline(5, color=GOLD, linestyle='--', alpha=0.5, linewidth=1, label='V/OI = 5')
        # Annotate top outliers
        for _, row in unusual_df.nlargest(5, 'premium_val').iterrows():
            ax5.annotate(
                f"{row['strike']}{row['put_call'][0].upper()} {row['expiry_dt'].strftime('%b%y')}",
                (min(row['voi_ratio'], 20), row['premium_val'] / 1e6),
                xytext=(5, 5), textcoords='offset points', fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#333', alpha=0.7)
            )
    ax5.set_title('Unusual Activity: V/OI Ratio vs Premium', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Volume / Open Interest Ratio (capped at 20)', fontsize=9)
    ax5.set_ylabel('Premium ($M)', fontsize=10)
    ax5.legend(fontsize=9, framealpha=0.4)

    # Proxy legend for colors
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=BULL, markersize=8, label='Bullish'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=BEAR, markersize=8, label='Bearish'),
    ]
    ax5.legend(handles=legend_els, fontsize=9, framealpha=0.4)

    # ------------------------------------------------------------------
    # 6. Top Whale Accumulations (horizontal bar)
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[3, 1])
    if not whale_df.empty:
        top_whales = whale_df.head(10).copy()
        top_whales['label'] = top_whales.apply(
            lambda r: f"${r['strike']:.0f}{r['put_call'][0].upper()} "
                      f"{r['expiry_dt'].strftime('%b%y')} "
                      f"({r['trade_count']}x)",
            axis=1
        )
        bar_colors = [BULL if d == 'Bullish' else BEAR for d in top_whales['direction']]
        bars_w = ax6.barh(
            range(len(top_whales)), top_whales['total_premium'] / 1e6,
            color=bar_colors, alpha=0.9
        )
        ax6.set_yticks(range(len(top_whales)))
        ax6.set_yticklabels(top_whales['label'], fontsize=8)
        ax6.invert_yaxis()
        ax6.set_xlabel('Total Accumulated Premium ($M)', fontsize=9)
        ax6.set_title('Top Whale Accumulations', fontsize=12, fontweight='bold')
        for bar, val in zip(bars_w, top_whales['total_premium'] / 1e6):
            ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'${val:.1f}M', va='center', fontsize=7.5, color='white')
    else:
        ax6.text(0.5, 0.5, 'No whale accumulation detected',
                 transform=ax6.transAxes, ha='center', va='center', color='white')

    output_file = f'{symbol}_options_flow.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Dashboard saved to {output_file}")
    plt.show()


# =====================================================================
# MAIN ANALYSIS
# =====================================================================

def analyze_options_flow(csv_filepath):
    print(f"\nLoading options flow data from {csv_filepath}...")
    df = pd.read_csv(csv_filepath)

    # --- 1. Parse ---
    df['premium_val'] = df['premium'].apply(parse_premium)
    df['datetime']    = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df['trade_date']  = df['datetime'].dt.normalize()
    df['expiry_dt']   = pd.to_datetime(df['expiry'], format='%m/%d/%Y')
    df['strike']      = pd.to_numeric(df['strike'], errors='coerce')
    df['spot']        = pd.to_numeric(df['spot'], errors='coerce')
    df['price']       = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['size']        = pd.to_numeric(df['size'], errors='coerce')
    df['volume']      = pd.to_numeric(df['volume'], errors='coerce')
    df['open_int']    = pd.to_numeric(df['open_int'], errors='coerce')

    # --- 2. Classify ---
    sentiments = df.apply(lambda r: classify_aggressor(r['side'], r['put_call']), axis=1)
    df['sentiment'],     df['sentiment_score'] = zip(*sentiments)

    dte_info = df.apply(lambda r: classify_dte_bucket(r['trade_date'].date(), r['expiry_dt'].date()), axis=1)
    df['dte_bucket'], df['dte'] = zip(*dte_info)

    exec_info = df['sweep_block_split'].apply(classify_execution_type)
    df['exec_type'], df['exec_weight'] = zip(*exec_info)

    moneyness = df.apply(lambda r: classify_moneyness(r['spot'], r['strike'], r['put_call']), axis=1)
    df['moneyness'], df['otm_pct'] = zip(*moneyness)

    cond_flags = df['conds'].apply(parse_conditions)
    df['is_opening']        = [c['is_opening'] for c in cond_flags]
    df['is_unusual']        = [c['is_unusual'] for c in cond_flags]
    df['is_highly_unusual'] = [c['is_highly_unusual'] for c in cond_flags]

    # --- 3. Core metrics ---
    df['voi_ratio'] = (df['volume'] / df['open_int'].replace(0, np.nan)).fillna(0).round(3)
    df['breakeven'] = df.apply(
        lambda r: compute_breakeven(r['strike'], r['price'], r['put_call']), axis=1
    )
    df['required_move_pct'] = ((df['breakeven'] - df['spot']) / df['spot'] * 100).round(2)
    df['bull_score'] = df.apply(compute_bull_score, axis=1)

    symbol = df['symbol'].iloc[0]

    # --- 4. Aggregate ---
    daily_flow    = compute_daily_net_flow(df)
    pivot_heatmap = compute_strike_heatmap(df)
    whale_df      = detect_whale_accumulation(df)

    # --- 5. Overall stats ---
    total_prem    = df['premium_val'].sum()
    bull_prem     = df[df['sentiment_score'] > 0]['premium_val'].sum()
    bear_prem     = df[df['sentiment_score'] < 0]['premium_val'].sum()
    net_flow      = bull_prem - bear_prem
    cp_ratio      = bull_prem / bear_prem if bear_prem > 0 else float('inf')
    unusual_prem  = df[df['is_unusual']]['premium_val'].sum()
    opening_prem  = df[df['is_opening']]['premium_val'].sum()
    leaps_bull    = df[(df['dte_bucket'] == 'LEAPS (180d+)') & (df['sentiment_score'] > 0)]['premium_val'].sum()

    # Dominant bias
    if cp_ratio >= 2.0:      overall = 'STRONGLY BULLISH'
    elif cp_ratio >= 1.3:    overall = 'BULLISH'
    elif cp_ratio >= 0.77:   overall = 'NEUTRAL / MIXED'
    elif cp_ratio >= 0.5:    overall = 'BEARISH'
    else:                    overall = 'STRONGLY BEARISH'

    # ----------------------------------------------------------------
    # PRINT REPORT
    # ----------------------------------------------------------------

    W = 80

    print("\n" + "=" * W)
    print(f"  OPTIONS FLOW ANALYSIS — {symbol}")
    print("=" * W)
    print(f"  Total Premium Traded  : ${total_prem/1e6:>10.2f}M")
    print(f"  Bullish Premium       : ${bull_prem/1e6:>10.2f}M  ({bull_prem/total_prem*100:.1f}%)")
    print(f"  Bearish Premium       : ${bear_prem/1e6:>10.2f}M  ({bear_prem/total_prem*100:.1f}%)")
    print(f"  Net Flow (Bull−Bear)  : ${net_flow/1e6:>+10.2f}M")
    print(f"  Call/Put Ratio        : {cp_ratio:>10.2f}x")
    print(f"  Unusual Activity $    : ${unusual_prem/1e6:>10.2f}M  ({unusual_prem/total_prem*100:.1f}%)")
    print(f"  Opening Position $    : ${opening_prem/1e6:>10.2f}M  ({opening_prem/total_prem*100:.1f}%)")
    print(f"  LEAPS Bullish $       : ${leaps_bull/1e6:>10.2f}M")
    print(f"\n  --> OVERALL BIAS      : {overall}")

    print("\n" + "=" * W)
    print("  DAILY NET FLOW SUMMARY")
    print("=" * W)
    print(f"  {'Date':<12} {'Bull $M':>9} {'Bear $M':>9} {'Net $M':>9} {'C/P':>6} {'Trades':>7}")
    print("  " + "-" * 55)
    for _, r in daily_flow.iterrows():
        cp_str = f"{r['cp_ratio']:.2f}x" if pd.notna(r['cp_ratio']) else "  N/A"
        print(f"  {str(r['date'].date()):<12} "
              f"{r['bull_premium']/1e6:>9.1f} "
              f"{r['bear_premium']/1e6:>9.1f} "
              f"{r['net_flow']/1e6:>+9.1f} "
              f"{cp_str:>6} "
              f"{int(r['trade_count']):>7}")

    print("\n" + "=" * W)
    print("  WHALE ACCUMULATION — REPEAT STRIKE BUILDS")
    print("=" * W)
    if not whale_df.empty:
        print(f"  {'Strike+Exp':<22} {'Dir':>9} {'Trades':>7} {'Total $M':>9} "
              f"{'Avg Px':>8} {'BE':>8} {'Req Move':>10} {'Exec':<12}")
        print("  " + "-" * 92)
        for _, r in whale_df.head(15).iterrows():
            label = f"${r['strike']:.0f}{r['put_call'][0].upper()} {r['expiry_dt'].strftime('%b%y')}"
            flag = ' [OP]' if r['is_opening_any'] else ''
            flag += ' [UNQ]' if r['is_unusual_any'] else ''
            print(f"  {label:<22} {r['direction']:>9} {r['trade_count']:>7} "
                  f"{r['total_premium']/1e6:>9.2f} "
                  f"{r['avg_price']:>8.2f} "
                  f"{r['breakeven']:>8.2f} "
                  f"{r['required_move_pct']:>+9.1f}% "
                  f"{r['exec_types']:<12}{flag}")
    else:
        print("  No whale accumulation detected with current thresholds.")

    print("\n" + "=" * W)
    print("  TOP UNUSUAL / HIGH V-OI TRADES  (V/OI > 3)")
    print("=" * W)
    unusual = df[df['voi_ratio'] > 3].nlargest(15, 'premium_val')
    if not unusual.empty:
        print(f"  {'Date':<11} {'Strike+Exp':<22} {'Side':<8} {'$M':>7} {'V/OI':>7} {'Moneyness':<10} {'BE':>8}")
        print("  " + "-" * 78)
        for _, r in unusual.iterrows():
            label = f"${r['strike']:.0f}{r['put_call'][0].upper()} {r['expiry_dt'].strftime('%b%y')}"
            print(f"  {str(r['trade_date'].date()):<11} {label:<22} "
                  f"{r['sentiment']:<8} "
                  f"{r['premium_val']/1e6:>7.3f} "
                  f"{r['voi_ratio']:>7.1f} "
                  f"{r['moneyness']:<10} "
                  f"{r['breakeven']:>8.2f}")
    else:
        print("  No trades with V/OI > 3 found.")

    print("\n" + "=" * W)
    print("  LEAPS POSITIONS  (>180 DTE) — LONG-TERM CONVICTION")
    print("=" * W)
    leaps = df[df['dte_bucket'] == 'LEAPS (180d+)'].sort_values('premium_val', ascending=False)
    if not leaps.empty:
        print(f"  {'Date':<11} {'Strike+Exp':<26} {'Side':<18} {'$M':>7} "
              f"{'DTE':>5} {'Req Move':>10} {'UNQ':>5}")
        print("  " + "-" * 85)
        for _, r in leaps.head(20).iterrows():
            label = f"${r['strike']:.0f}{r['put_call'][0].upper()} {r['expiry_dt'].strftime('%b %d %Y')}"
            unq = 'YES' if r['is_unusual'] else ''
            print(f"  {str(r['trade_date'].date()):<11} {label:<26} "
                  f"{r['sentiment']:<18} "
                  f"{r['premium_val']/1e6:>7.3f} "
                  f"{int(r['dte']):>5} "
                  f"{r['required_move_pct']:>+9.1f}% "
                  f"{unq:>5}")
    else:
        print("  No LEAPS trades found.")

    print("\n" + "=" * W)
    print("  LONG POSITION TRADE IDEAS  (highest bull scores)")
    print("=" * W)
    long_ideas = df[
        (df['sentiment_score'] > 0) &
        (df['premium_val'] >= 50_000)
    ].nlargest(10, 'bull_score')
    if not long_ideas.empty:
        print(f"  {'Date':<11} {'Strike+Exp':<26} {'Sentiment':<22} {'Bull Score':>10} "
              f"{'$M':>7} {'Exec':<8} {'Flags'}")
        print("  " + "-" * 95)
        for _, r in long_ideas.iterrows():
            label  = f"${r['strike']:.0f}{r['put_call'][0].upper()} {r['expiry_dt'].strftime('%b %d %Y')}"
            flags  = []
            if r['is_opening']:        flags.append('OPEN')
            if r['is_unusual']:        flags.append('UNQ')
            if r['is_highly_unusual']: flags.append('HIGH-UNQ')
            if r['voi_ratio'] > 5:     flags.append(f"V/OI={r['voi_ratio']:.1f}")
            print(f"  {str(r['trade_date'].date()):<11} {label:<26} "
                  f"{r['sentiment']:<22} {r['bull_score']:>10.4f} "
                  f"{r['premium_val']/1e6:>7.3f} "
                  f"{r['exec_type']:<8} "
                  f"{', '.join(flags)}")
    print("=" * W + "\n")

    # --- 6. Dashboard ---
    generate_options_dashboard(df, daily_flow, whale_df, pivot_heatmap, symbol)

    return df


# =====================================================================
# API ENTRY POINT (used by Flask web UI — no print, no plot)
# =====================================================================

def _safe_num(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (f != f) else f
    return v


def _enrich_options_dataframe(df):
    """Apply all classification & metric columns to df (mutates and returns)."""
    df['premium_val'] = df['premium'].apply(parse_premium)
    df['datetime']    = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df['trade_date']  = df['datetime'].dt.normalize()
    df['expiry_dt']   = pd.to_datetime(df['expiry'], format='%m/%d/%Y')
    df['strike']      = pd.to_numeric(df['strike'], errors='coerce')
    df['spot']        = pd.to_numeric(df['spot'], errors='coerce')
    df['price']       = pd.to_numeric(
        df['price'].astype(str).str.replace('$', '').str.replace(',', ''),
        errors='coerce'
    )
    df['size']        = pd.to_numeric(df['size'], errors='coerce')
    df['volume']      = pd.to_numeric(df['volume'], errors='coerce')
    df['open_int']    = pd.to_numeric(df['open_int'], errors='coerce')

    sentiments = df.apply(lambda r: classify_aggressor(r['side'], r['put_call']), axis=1)
    df['sentiment'], df['sentiment_score'] = zip(*sentiments)

    dte_info = df.apply(
        lambda r: classify_dte_bucket(r['trade_date'].date(), r['expiry_dt'].date()), axis=1
    )
    df['dte_bucket'], df['dte'] = zip(*dte_info)

    exec_info = df['sweep_block_split'].apply(classify_execution_type)
    df['exec_type'], df['exec_weight'] = zip(*exec_info)

    moneyness = df.apply(
        lambda r: classify_moneyness(r['spot'], r['strike'], r['put_call']), axis=1
    )
    df['moneyness'], df['otm_pct'] = zip(*moneyness)

    cond_flags = df['conds'].apply(parse_conditions)
    df['is_opening']        = [c['is_opening'] for c in cond_flags]
    df['is_unusual']        = [c['is_unusual'] for c in cond_flags]
    df['is_highly_unusual'] = [c['is_highly_unusual'] for c in cond_flags]

    df['voi_ratio'] = (df['volume'] / df['open_int'].replace(0, np.nan)).fillna(0).round(3)
    df['breakeven'] = df.apply(
        lambda r: compute_breakeven(r['strike'], r['price'], r['put_call']), axis=1
    )
    df['required_move_pct'] = ((df['breakeven'] - df['spot']) / df['spot'] * 100).round(2)
    df['bull_score'] = df.apply(compute_bull_score, axis=1)

    return df


def analyze_for_api(df_raw):
    """Run full options-flow analysis and return JSON-safe dict for the web UI."""
    df = _enrich_options_dataframe(df_raw.copy())
    symbol = str(df['symbol'].iloc[0]) if not df.empty else ''

    daily_flow_df = compute_daily_net_flow(df)
    whale_df      = detect_whale_accumulation(df)

    total_prem    = float(df['premium_val'].sum())
    bull_prem     = float(df[df['sentiment_score'] > 0]['premium_val'].sum())
    bear_prem     = float(df[df['sentiment_score'] < 0]['premium_val'].sum())
    neutral_prem  = float(df[df['sentiment_score'] == 0]['premium_val'].sum())
    net_flow      = bull_prem - bear_prem
    cp_ratio      = (bull_prem / bear_prem) if bear_prem > 0 else None
    unusual_prem  = float(df[df['is_unusual']]['premium_val'].sum())
    opening_prem  = float(df[df['is_opening']]['premium_val'].sum())
    leaps_bull    = float(df[(df['dte_bucket'] == 'LEAPS (180d+)') & (df['sentiment_score'] > 0)]['premium_val'].sum())

    if cp_ratio is None:
        overall = 'BULLISH' if bull_prem > 0 else 'NEUTRAL / MIXED'
    elif cp_ratio >= 2.0:    overall = 'STRONGLY BULLISH'
    elif cp_ratio >= 1.3:    overall = 'BULLISH'
    elif cp_ratio >= 0.77:   overall = 'NEUTRAL / MIXED'
    elif cp_ratio >= 0.5:    overall = 'BEARISH'
    else:                    overall = 'STRONGLY BEARISH'

    # Daily flow rows
    daily_flow = []
    for _, r in daily_flow_df.iterrows():
        daily_flow.append({
            'date':           str(r['date'].date()),
            'bull_premium':   float(r['bull_premium']),
            'bear_premium':   float(r['bear_premium']),
            'total_premium':  float(r['total_premium']),
            'net_flow':       float(r['net_flow']),
            'cp_ratio':       _safe_num(r['cp_ratio']),
            'trade_count':    int(r['trade_count']),
        })

    # Whales
    whales = []
    for _, r in whale_df.head(20).iterrows():
        whales.append({
            'strike':           float(r['strike']),
            'expiry':           r['expiry_dt'].strftime('%Y-%m-%d'),
            'put_call':         str(r['put_call']),
            'direction':        str(r['direction']),
            'trade_count':      int(r['trade_count']),
            'total_premium':    float(r['total_premium']),
            'avg_price':        float(r['avg_price']),
            'avg_spot':         float(r['avg_spot']),
            'breakeven':        float(r['breakeven']),
            'required_move_pct':float(r['required_move_pct']),
            'is_opening_any':   bool(r['is_opening_any']),
            'is_unusual_any':   bool(r['is_unusual_any']),
            'exec_types':       str(r['exec_types']),
            'first_seen':       r['first_seen'].isoformat(),
            'last_seen':        r['last_seen'].isoformat(),
        })

    # LEAPS
    leaps_rows = []
    leaps_df = df[df['dte_bucket'] == 'LEAPS (180d+)'].sort_values('premium_val', ascending=False)
    for _, r in leaps_df.head(25).iterrows():
        leaps_rows.append({
            'date':             str(r['trade_date'].date()),
            'strike':           float(r['strike']),
            'expiry':           r['expiry_dt'].strftime('%Y-%m-%d'),
            'put_call':         str(r['put_call']),
            'side':             str(r['side']),
            'sentiment':        str(r['sentiment']),
            'sentiment_score':  float(r['sentiment_score']),
            'premium_val':      float(r['premium_val']),
            'dte':              int(r['dte']),
            'required_move_pct':float(r['required_move_pct']),
            'is_unusual':       bool(r['is_unusual']),
            'exec_type':        str(r['exec_type']),
        })

    # Unusual trades (V/OI > 3)
    unusual_rows = []
    unusual_df = df[df['voi_ratio'] > 3].nlargest(20, 'premium_val')
    for _, r in unusual_df.iterrows():
        unusual_rows.append({
            'date':         str(r['trade_date'].date()),
            'strike':       float(r['strike']),
            'expiry':       r['expiry_dt'].strftime('%Y-%m-%d'),
            'put_call':     str(r['put_call']),
            'sentiment':    str(r['sentiment']),
            'premium_val':  float(r['premium_val']),
            'voi_ratio':    float(r['voi_ratio']),
            'moneyness':    str(r['moneyness']),
            'breakeven':    float(r['breakeven']),
        })

    # Long-position ideas
    long_ideas = []
    li_df = df[(df['sentiment_score'] > 0) & (df['premium_val'] >= 50_000)].nlargest(15, 'bull_score')
    for _, r in li_df.iterrows():
        flags = []
        if r['is_opening']:        flags.append('OPEN')
        if r['is_unusual']:        flags.append('UNQ')
        if r['is_highly_unusual']: flags.append('HIGH-UNQ')
        if r['voi_ratio'] > 5:     flags.append(f"V/OI={r['voi_ratio']:.1f}")
        long_ideas.append({
            'date':             str(r['trade_date'].date()),
            'strike':           float(r['strike']),
            'expiry':           r['expiry_dt'].strftime('%Y-%m-%d'),
            'put_call':         str(r['put_call']),
            'sentiment':        str(r['sentiment']),
            'bull_score':       float(r['bull_score']),
            'premium_val':      float(r['premium_val']),
            'exec_type':        str(r['exec_type']),
            'flags':            flags,
            'required_move_pct':float(r['required_move_pct']),
            'breakeven':        float(r['breakeven']),
        })

    # DTE breakdown (bull/bear)
    dte_order = ['0DTE', 'Weekly (1-7d)', 'Monthly (8-30d)',
                 'Near-Term (31-90d)', 'Mid-Term (91-180d)', 'LEAPS (180d+)']
    dte_breakdown = []
    for b in dte_order:
        g = df[df['dte_bucket'] == b]
        if len(g) == 0:
            continue
        dte_breakdown.append({
            'bucket':         b,
            'trades':         int(len(g)),
            'bull_premium':   float(g[g['sentiment_score'] > 0]['premium_val'].sum()),
            'bear_premium':   float(g[g['sentiment_score'] < 0]['premium_val'].sum()),
            'total_premium':  float(g['premium_val'].sum()),
        })

    # Execution-type breakdown
    exec_breakdown = []
    for et in df['exec_type'].unique():
        g = df[df['exec_type'] == et]
        exec_breakdown.append({
            'exec_type':      str(et),
            'trades':         int(len(g)),
            'bull_premium':   float(g[g['sentiment_score'] > 0]['premium_val'].sum()),
            'bear_premium':   float(g[g['sentiment_score'] < 0]['premium_val'].sum()),
            'total_premium':  float(g['premium_val'].sum()),
        })

    # Top strikes by total premium (calls only — long-position view)
    calls = df[df['put_call'] == 'call']
    top_strikes = []
    if not calls.empty:
        ts = (
            calls.groupby('strike')
            .agg(trades=('premium_val', 'count'),
                 total_premium=('premium_val', 'sum'),
                 avg_spot=('spot', 'mean'))
            .nlargest(15, 'total_premium')
            .reset_index()
        )
        for _, r in ts.iterrows():
            top_strikes.append({
                'strike':          float(r['strike']),
                'trades':          int(r['trades']),
                'total_premium':   float(r['total_premium']),
                'avg_spot':        float(r['avg_spot']),
                'distance_pct':    float((r['strike'] - r['avg_spot']) / r['avg_spot'] * 100),
            })

    return {
        'symbol': symbol,
        'summary': {
            'total_premium':    total_prem,
            'bull_premium':     bull_prem,
            'bear_premium':     bear_prem,
            'neutral_premium':  neutral_prem,
            'net_flow':         net_flow,
            'cp_ratio':         cp_ratio,
            'unusual_premium':  unusual_prem,
            'opening_premium':  opening_prem,
            'leaps_bullish':    leaps_bull,
            'overall_bias':     overall,
            'trade_count':      int(len(df)),
            'date_range': {
                'start': df['datetime'].min().isoformat() if not df.empty else None,
                'end':   df['datetime'].max().isoformat() if not df.empty else None,
            }
        },
        'daily_flow':       daily_flow,
        'whales':           whales,
        'leaps':            leaps_rows,
        'unusual_trades':   unusual_rows,
        'long_ideas':       long_ideas,
        'dte_breakdown':    dte_breakdown,
        'exec_breakdown':   exec_breakdown,
        'top_strikes':      top_strikes,
    }


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("=== Options Flow Analyzer — Long Position Intelligence ===")

    user_input = input("Enter the CSV file name (e.g. 'my_flow'): ").strip('\'" ')

    if not user_input.lower().endswith('.csv'):
        file_name = user_input + '.csv'
    else:
        file_name = user_input

    if not os.path.exists(file_name):
        print(f"\n[ERROR] File not found: '{file_name}'")
        print("Ensure the file is in the same folder as this script.")
    else:
        try:
            result_df = analyze_options_flow(file_name)
        except pd.errors.EmptyDataError:
            print("\n[ERROR] The CSV file is empty.")
        except KeyError as e:
            print(f"\n[ERROR] Missing required column: {e}")
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Unexpected error: {e}")
            traceback.print_exc()
