import pandas as pd
import yfinance as yf
from datetime import datetime, time as dt_time
import os
import numpy as np

# matplotlib/seaborn are only used by the CLI dashboard. They are imported
# lazily inside generate_enhanced_dashboard() so the Flask web UI can use
# this module's analyze_for_api() without those plotting dependencies.


# =====================================================================
# PARSERS & CLASSIFIERS
# =====================================================================

def parse_premium(prem_str):
    """Converts string premiums like '$4.5M' or '$800K' to float."""
    if pd.isna(prem_str):
        return 0
    prem_str = str(prem_str).replace('$', '').replace(',', '').upper().strip()
    if 'M' in prem_str:
        return float(prem_str.replace('M', '')) * 1_000_000
    elif 'K' in prem_str:
        return float(prem_str.replace('K', '')) * 1_000
    return float(prem_str)


def classify_session(dt):
    """Classify a trade into its market session."""
    t = dt.time()
    if t < dt_time(9, 30):
        return 'Pre-Market'
    elif t < dt_time(16, 0):
        return 'Regular'
    elif t == dt_time(16, 0):
        return 'At-Close'
    else:
        return 'After-Hours'


def classify_tier(premium_val):
    """Bucket a print by dollar size."""
    if premium_val >= 50_000_000:
        return 'Mega (>$50M)'
    elif premium_val >= 10_000_000:
        return 'Large ($10-50M)'
    elif premium_val >= 4_000_000:
        return 'Medium ($4-10M)'
    else:
        return 'Small (<$4M)'


def simplify_verdict(v):
    if 'Bullish' in v:
        return 'Bullish'
    if 'Bearish' in v:
        return 'Bearish'
    return 'Neutral'


# =====================================================================
# CLUSTER DETECTION
# =====================================================================

def detect_clusters(df, time_window_minutes=5, price_tolerance_pct=0.02):
    """
    Mark trades as belonging to a cluster when 3+ prints occur within
    time_window_minutes of each other at prices within price_tolerance_pct.
    Cluster ID = -1 means the trade is not part of any cluster.
    """
    df = df.sort_values('datetime').reset_index(drop=True)
    cluster_id = [-1] * len(df)
    cluster_counter = 0

    for i in range(len(df)):
        if cluster_id[i] != -1:
            continue
        members = [i]
        for j in range(i + 1, len(df)):
            time_diff = (df.loc[j, 'datetime'] - df.loc[i, 'datetime']).total_seconds() / 60
            if time_diff > time_window_minutes:
                break
            price_diff = abs(df.loc[j, 'price'] - df.loc[i, 'price']) / df.loc[i, 'price']
            if price_diff <= price_tolerance_pct:
                members.append(j)

        if len(members) >= 3:
            for idx in members:
                cluster_id[idx] = cluster_counter
            cluster_counter += 1

    df['cluster_id'] = cluster_id
    return df, cluster_counter


# =====================================================================
# KEY INSTITUTIONAL PRICE LEVELS
# =====================================================================

def find_key_levels(df, min_occurrences=3, price_tolerance_pct=0.02):
    """
    Identify price levels where multiple dark pool prints clustered.
    These represent institutional support/resistance or standing orders.
    """
    prices = df['price'].values
    premiums = df['premium_val'].values
    datetimes = df['datetime'].values
    used = np.zeros(len(prices), dtype=bool)
    levels = []

    for i in range(len(prices)):
        if used[i]:
            continue
        nearby_mask = np.abs(prices - prices[i]) / prices[i] <= price_tolerance_pct
        count = nearby_mask.sum()
        if count >= min_occurrences:
            used[nearby_mask] = True
            levels.append({
                'price': float(np.mean(prices[nearby_mask])),
                'count': int(count),
                'total_premium': float(np.sum(premiums[nearby_mask])),
                'dates': sorted(set(
                    pd.Timestamp(d).date() for d in datetimes[nearby_mask]
                ))
            })

    return sorted(levels, key=lambda x: x['total_premium'], reverse=True)


# =====================================================================
# VWAP SENTIMENT
# =====================================================================

def vwap_verdict(price, vwap):
    """Compare dark pool execution price to daily VWAP approximation."""
    if vwap is None or vwap == 0:
        return 'Unknown'
    ratio = (price - vwap) / vwap
    if ratio <= -0.01:
        return 'Bullish (Below VWAP)'
    elif ratio >= 0.01:
        return 'Bearish (Above VWAP)'
    else:
        return 'Neutral (At VWAP)'


# =====================================================================
# ROLLING ACCUMULATION
# =====================================================================

def calculate_rolling_accumulation(df, window_days=3):
    """Return daily and rolling N-day premium totals."""
    daily = df.groupby(df['datetime'].dt.normalize())['premium_val'].sum().sort_index()
    rolling = daily.rolling(window=window_days, min_periods=1).sum()
    return daily, rolling


# =====================================================================
# FORWARD RETURNS
# =====================================================================

def calculate_forward_returns(df, market_data, windows=(1, 3, 5)):
    """
    For each trading day with dark pool activity, compute forward returns
    at 1, 3, and 5 days out. Shows whether large accumulation days
    predict subsequent price moves.
    """
    sym = df['symbol'].iloc[0]
    hist = market_data.get(sym)
    if hist is None:
        return pd.DataFrame()

    daily = (
        df.groupby(df['datetime'].dt.normalize())['premium_val']
        .sum()
        .reset_index()
        .rename(columns={'datetime': 'date'})
    )

    results = []
    for _, row in daily.iterrows():
        date, premium = row['date'], row['daily_premium'] if 'daily_premium' in row else row['premium_val']
        available = hist[hist.index <= date]
        future = hist[hist.index > date]
        if available.empty:
            continue
        entry_price = available.iloc[-1]['Close']
        rec = {'date': date, 'daily_premium': premium, 'entry_price': entry_price}
        for w in windows:
            if len(future) >= w:
                rec[f'fwd_{w}d_return'] = (future.iloc[w - 1]['Close'] - entry_price) / entry_price * 100
            else:
                rec[f'fwd_{w}d_return'] = None
        results.append(rec)

    return pd.DataFrame(results)


# =====================================================================
# ENHANCED DASHBOARD
# =====================================================================

def generate_enhanced_dashboard(df, fwd_returns_df, daily_premium, rolling_premium, key_levels, num_clusters):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    print("Generating Enhanced Visualizations...")

    df = df.sort_values('datetime').reset_index(drop=True)
    df['sentiment'] = df['verdict'].apply(simplify_verdict)
    df['cumulative_premium'] = df['premium_val'].cumsum()

    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.facecolor'] = '#1a1a2e'
    plt.rcParams['axes.facecolor'] = '#16213e'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    colors = {'Bullish': '#2ca02c', 'Bearish': '#d62728', 'Neutral': '#7f7f7f'}
    tier_colors = {
        'Mega (>$50M)':   '#d62728',
        'Large ($10-50M)': '#ff7f0e',
        'Medium ($4-10M)': '#2ca02c',
        'Small (<$4M)':   '#1f77b4',
    }
    session_colors = {
        'Pre-Market':  '#9467bd',
        'Regular':     '#2ca02c',
        'At-Close':    '#d62728',
        'After-Hours': '#ff7f0e',
    }
    symbol = df['symbol'].iloc[0] if not df.empty else "Stock"

    fig = plt.figure(figsize=(22, 26))
    fig.suptitle(f'{symbol} — Enhanced Dark Pool Analysis', fontsize=20, fontweight='bold',
                 color='white', y=0.99)
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.32)

    # ------------------------------------------------------------------
    # 1. Price & Capital Flow (with cluster highlights)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    for sentiment_label in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment_label]
        bubble_sizes = subset['premium_val'] / (df['premium_val'].max() / 800)
        bubble_sizes = bubble_sizes.clip(lower=30)
        ax1.scatter(subset['datetime'], subset['price'],
                    s=bubble_sizes, alpha=0.65, label=sentiment_label,
                    color=colors.get(sentiment_label, 'blue'),
                    edgecolors='white', linewidth=0.8)

    clustered = df[df['cluster_id'] >= 0]
    if not clustered.empty:
        ax1.scatter(clustered['datetime'], clustered['price'],
                    s=250, facecolors='none', edgecolors='yellow',
                    linewidth=2, label=f'Cluster ({num_clusters} detected)', zorder=5)

    for _, row in df.nlargest(3, 'premium_val').iterrows():
        ax1.annotate(
            f"${row['premium_val']/1e6:.1f}M",
            (row['datetime'], row['price']),
            xytext=(8, 10), textcoords='offset points', fontsize=9,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333', alpha=0.8)
        )

    ax1.set_title(f'{symbol} Dark Pool Prints: Price & Capital Flow', fontsize=13, fontweight='bold', color='white')
    ax1.set_ylabel('Execution Price ($)', fontsize=11)
    ax1.legend(title='Sentiment', loc='upper left', framealpha=0.5)

    # ------------------------------------------------------------------
    # 2. Cumulative Capital Flow
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['datetime'], df['cumulative_premium'] / 1e6,
             color='#4c72b0', linewidth=2, marker='o', markersize=2)
    ax2.fill_between(df['datetime'], df['cumulative_premium'] / 1e6, color='#4c72b0', alpha=0.2)
    ax2.set_title('Cumulative Capital Flow', fontsize=12, fontweight='bold', color='white')
    ax2.set_ylabel('Total Premium ($ Millions)', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)

    # ------------------------------------------------------------------
    # 3. Rolling 3-Day Accumulation Rate
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(rolling_premium.index, rolling_premium.values / 1e6, color='#e377c2', alpha=0.85)
    ax3.set_title('3-Day Rolling Accumulation Rate', fontsize=12, fontweight='bold', color='white')
    ax3.set_ylabel('3-Day Premium ($ Millions)', fontsize=10)
    ax3.tick_params(axis='x', rotation=45)

    # ------------------------------------------------------------------
    # 4. Session Breakdown
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    session_order = ['Pre-Market', 'Regular', 'At-Close', 'After-Hours']
    session_summary = df.groupby('session')['premium_val'].sum() / 1e6
    session_summary = session_summary.reindex([s for s in session_order if s in session_summary.index])
    bars4 = ax4.bar(session_summary.index, session_summary.values,
                    color=[session_colors.get(s, 'gray') for s in session_summary.index])
    ax4.set_title('Premium by Market Session', fontsize=12, fontweight='bold', color='white')
    ax4.set_ylabel('Premium ($ Millions)', fontsize=10)
    for bar, val in zip(bars4, session_summary.values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + session_summary.max() * 0.02,
                 f'${val:.0f}M', ha='center', fontsize=9, fontweight='bold', color='white')

    # ------------------------------------------------------------------
    # 5. Print Size Tier
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    tier_order = ['Mega (>$50M)', 'Large ($10-50M)', 'Medium ($4-10M)', 'Small (<$4M)']
    tier_summary = df.groupby('tier')['premium_val'].sum() / 1e6
    tier_summary = tier_summary.reindex([t for t in tier_order if t in tier_summary.index])
    bars5 = ax5.bar(range(len(tier_summary)), tier_summary.values,
                    color=[tier_colors.get(t, 'gray') for t in tier_summary.index])
    ax5.set_xticks(range(len(tier_summary)))
    ax5.set_xticklabels(tier_summary.index, rotation=20, ha='right', fontsize=8)
    ax5.set_title('Premium by Print Size Tier', fontsize=12, fontweight='bold', color='white')
    ax5.set_ylabel('Premium ($ Millions)', fontsize=10)
    for bar, val in zip(bars5, tier_summary.values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + tier_summary.max() * 0.02,
                 f'${val:.0f}M', ha='center', fontsize=9, fontweight='bold', color='white')

    # ------------------------------------------------------------------
    # 6. Forward Returns vs Daily Premium
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[3, :])
    if not fwd_returns_df.empty:
        valid = fwd_returns_df.dropna(subset=['fwd_1d_return'])
        if not valid.empty:
            sc = ax6.scatter(
                valid['date'], valid['fwd_1d_return'],
                c=valid['daily_premium'] / 1e6,
                cmap='RdYlGn',
                s=[max(p / 500_000, 30) for p in valid['daily_premium']],
                alpha=0.85, edgecolors='white', linewidth=0.8
            )
            cb = plt.colorbar(sc, ax=ax6)
            cb.set_label('Daily Dark Pool Premium ($M)', color='white')
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
            ax6.axhline(0, color='white', linestyle='--', alpha=0.4, linewidth=1)
            ax6.set_title('Daily Dark Pool Premium vs Next-Day Price Return (%)', fontsize=12, fontweight='bold', color='white')
            ax6.set_ylabel('1-Day Forward Return (%)', fontsize=10)
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'Forward return data not available', transform=ax6.transAxes,
                     ha='center', va='center', fontsize=12, color='white')
            ax6.set_title('Forward Return Analysis', fontsize=12, fontweight='bold', color='white')
    else:
        ax6.text(0.5, 0.5, 'Forward return data not available', transform=ax6.transAxes,
                 ha='center', va='center', fontsize=12, color='white')
        ax6.set_title('Forward Return Analysis', fontsize=12, fontweight='bold', color='white')

    output_file = f'{symbol}_darkpool_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Chart saved to {output_file}")
    plt.show()


# =====================================================================
# MAIN ANALYSIS
# =====================================================================

def analyze_dark_pool(csv_filepath):
    print(f"\nLoading data from {csv_filepath}...")
    df = pd.read_csv(csv_filepath)

    # --- 1. Parse & classify ---
    df['premium_val'] = df['premium'].apply(parse_premium)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df['session'] = df['datetime'].apply(classify_session)
    df['tier'] = df['premium_val'].apply(classify_tier)

    # --- 2. Cluster detection ---
    print("Detecting institutional block clusters...")
    df, num_clusters = detect_clusters(df)

    # --- 3. Fetch market data ---
    print("Fetching bulk historical market data from Yahoo Finance...")
    unique_symbols = df['symbol'].unique()
    min_date = df['datetime'].min() - pd.Timedelta(days=7)
    max_date = df['datetime'].max() + pd.Timedelta(days=10)

    market_data = {}
    for sym in unique_symbols:
        ticker = yf.Ticker(sym)
        hist = ticker.history(start=min_date.strftime('%Y-%m-%d'), end=max_date.strftime('%Y-%m-%d'))
        if not hist.empty:
            hist.index = hist.index.tz_localize(None).normalize()
            market_data[sym] = hist

    # --- 4. Generate verdicts (range + VWAP) ---
    verdicts, ranges, vwap_verdicts, vwap_vals = [], [], [], []

    for _, row in df.iterrows():
        sym, dt, price = row['symbol'], row['datetime'].normalize(), row['price']
        hist = market_data.get(sym)

        if hist is not None and not hist.empty:
            available = hist[hist.index <= dt]
            if not available.empty:
                bar = available.iloc[-1]
                high, low, close = bar['High'], bar['Low'], bar['Close']

                # Range-based verdict (original)
                ranges.append(f"${low:.2f} - ${high:.2f}")
                rng = high - low
                if rng == 0:
                    verdicts.append("Neutral")
                else:
                    pct = (price - low) / rng
                    if pct >= 0.66:
                        verdicts.append("Bullish (Near High/Ask)")
                    elif pct <= 0.33:
                        verdicts.append("Bearish (Near Low/Bid)")
                    else:
                        verdicts.append("Neutral (Mid-Range)")

                # VWAP approximation: (H + L + C) / 3
                vwap = (high + low + close) / 3
                vwap_vals.append(round(vwap, 2))
                vwap_verdicts.append(vwap_verdict(price, vwap))
            else:
                verdicts.append("Unknown (Date too far back)")
                ranges.append("N/A")
                vwap_verdicts.append("Unknown")
                vwap_vals.append(None)
        else:
            verdicts.append("Unknown (No Ticker Data)")
            ranges.append("N/A")
            vwap_verdicts.append("Unknown")
            vwap_vals.append(None)

    df['daily_range'] = ranges
    df['verdict'] = verdicts
    df['vwap_approx'] = vwap_vals
    df['vwap_verdict'] = vwap_verdicts

    # --- 5. Key price levels ---
    print("Detecting key institutional price levels...")
    key_levels = find_key_levels(df)

    # --- 6. Rolling accumulation ---
    daily_premium, rolling_premium = calculate_rolling_accumulation(df)

    # --- 7. Forward returns ---
    print("Calculating forward returns...")
    fwd_returns_df = calculate_forward_returns(df, market_data)

    # --- 8. Aggregate verdicts ---
    bullish_vol = df[df['verdict'].str.contains('Bullish', na=False)]['premium_val'].sum()
    bearish_vol = df[df['verdict'].str.contains('Bearish', na=False)]['premium_val'].sum()
    neutral_vol = df[df['verdict'].str.contains('Neutral', na=False)]['premium_val'].sum()
    total_vol = bullish_vol + bearish_vol + neutral_vol

    overall = "Neutral / Mixed"
    if bullish_vol > bearish_vol * 1.5:
        overall = "Strongly Bullish Accumulation"
    elif bearish_vol > bullish_vol * 1.5:
        overall = "Strongly Bearish Distribution"
    elif bullish_vol > bearish_vol:
        overall = "Slightly Bullish Accumulation"
    elif bearish_vol > bullish_vol:
        overall = "Slightly Bearish Distribution"

    # ----------------------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------------------

    print("\n" + "=" * 90)
    print("INDIVIDUAL TRADE VERDICTS")
    print("=" * 90)
    display_cols = ['date', 'time', 'symbol', 'price', 'premium', 'tier', 'session', 'daily_range', 'verdict', 'vwap_verdict']
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 65)
    print("OVERALL VERDICT SUMMARY")
    print("=" * 65)
    print(f"  Total Bullish Premium : ${bullish_vol:>15,.0f}  ({bullish_vol/total_vol*100:.1f}%)")
    print(f"  Total Bearish Premium : ${bearish_vol:>15,.0f}  ({bearish_vol/total_vol*100:.1f}%)")
    print(f"  Total Neutral Premium : ${neutral_vol:>15,.0f}  ({neutral_vol/total_vol*100:.1f}%)")
    print(f"  --> FINAL VERDICT     : {overall}")

    print("\n" + "=" * 65)
    print("SESSION BREAKDOWN")
    print("=" * 65)
    sess = df.groupby('session').agg(
        trades=('premium_val', 'count'),
        total_M=('premium_val', lambda x: x.sum() / 1e6)
    )
    print(sess.to_string())

    print("\n" + "=" * 65)
    print(f"INSTITUTIONAL CLUSTERS DETECTED: {num_clusters}")
    print("=" * 65)
    if num_clusters > 0:
        for cid in range(num_clusters):
            ct = df[df['cluster_id'] == cid]
            span_min = (ct['datetime'].max() - ct['datetime'].min()).total_seconds() / 60
            print(f"  Cluster {cid+1:>2}:  {len(ct):>3} trades | "
                  f"${ct['premium_val'].sum()/1e6:>7.1f}M total | "
                  f"Avg price ${ct['price'].mean():>7.2f} | "
                  f"Span {span_min:>5.1f} min | "
                  f"{ct['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}")
    else:
        print("  No clusters found with current parameters.")

    print("\n" + "=" * 65)
    print("KEY INSTITUTIONAL PRICE LEVELS  (support/resistance)")
    print("=" * 65)
    for lvl in key_levels[:10]:
        dates_str = ', '.join(str(d) for d in lvl['dates'][:3])
        if len(lvl['dates']) > 3:
            dates_str += f" +{len(lvl['dates'])-3} more"
        print(f"  ${lvl['price']:>7.2f}  |  {lvl['count']:>3} prints  |  "
              f"${lvl['total_premium']/1e6:>7.1f}M  |  {dates_str}")

    print("\n" + "=" * 65)
    print("PRINT SIZE TIER BREAKDOWN")
    print("=" * 65)
    tier_order = ['Mega (>$50M)', 'Large ($10-50M)', 'Medium ($4-10M)', 'Small (<$4M)']
    tier_df = df.groupby('tier').agg(
        count=('premium_val', 'count'),
        total_M=('premium_val', lambda x: x.sum() / 1e6),
        avg_M=('premium_val', lambda x: x.mean() / 1e6)
    ).reindex([t for t in tier_order if t in df['tier'].unique()])
    print(tier_df.to_string())

    if not fwd_returns_df.empty:
        print("\n" + "=" * 75)
        print("FORWARD RETURN ANALYSIS  (top 10 highest-volume days)")
        print("=" * 75)
        print(f"  {'Date':<12} {'Premium ($M)':>13}  {'1D Ret':>8}  {'3D Ret':>8}  {'5D Ret':>8}")
        print("  " + "-" * 55)
        for _, r in fwd_returns_df.nlargest(10, 'daily_premium').iterrows():
            def fmt_ret(key):
                v = r.get(key)
                return f"{v:+.2f}%" if v is not None else "  N/A  "
            print(f"  {str(r['date']):<12} ${r['daily_premium']/1e6:>12.1f}  "
                  f"{fmt_ret('fwd_1d_return'):>8}  "
                  f"{fmt_ret('fwd_3d_return'):>8}  "
                  f"{fmt_ret('fwd_5d_return'):>8}")

    print("=" * 75 + "\n")

    # --- Generate charts ---
    generate_enhanced_dashboard(df, fwd_returns_df, daily_premium, rolling_premium, key_levels, num_clusters)

    return df


# =====================================================================
# API ENTRY POINT (used by Flask web UI — no print, no plot)
# =====================================================================

def _enrich_dataframe(df):
    """Apply all analytical columns to df. Returns (df, num_clusters, market_data, key_levels, daily_premium, rolling_premium, fwd_returns_df)."""
    df['premium_val'] = df['premium'].apply(parse_premium)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df['session'] = df['datetime'].apply(classify_session)
    df['tier'] = df['premium_val'].apply(classify_tier)

    df, num_clusters = detect_clusters(df)

    unique_symbols = df['symbol'].unique()
    min_date = df['datetime'].min() - pd.Timedelta(days=7)
    max_date = df['datetime'].max() + pd.Timedelta(days=10)

    market_data = {}
    for sym in unique_symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=min_date.strftime('%Y-%m-%d'),
                                  end=max_date.strftime('%Y-%m-%d'))
            if not hist.empty:
                hist.index = hist.index.tz_localize(None).normalize()
                market_data[sym] = hist
        except Exception:
            pass

    verdicts, ranges, vwap_verdicts, vwap_vals = [], [], [], []
    for _, row in df.iterrows():
        sym, dt, price = row['symbol'], row['datetime'].normalize(), row['price']
        hist = market_data.get(sym)
        if hist is not None and not hist.empty:
            available = hist[hist.index <= dt]
            if not available.empty:
                bar = available.iloc[-1]
                high, low, close = bar['High'], bar['Low'], bar['Close']
                ranges.append(f"${low:.2f} - ${high:.2f}")
                rng = high - low
                if rng == 0:
                    verdicts.append("Neutral")
                else:
                    pct = (price - low) / rng
                    if pct >= 0.66:
                        verdicts.append("Bullish (Near High/Ask)")
                    elif pct <= 0.33:
                        verdicts.append("Bearish (Near Low/Bid)")
                    else:
                        verdicts.append("Neutral (Mid-Range)")
                vwap = (high + low + close) / 3
                vwap_vals.append(round(float(vwap), 2))
                vwap_verdicts.append(vwap_verdict(price, vwap))
            else:
                verdicts.append("Unknown (Date too far back)")
                ranges.append("N/A")
                vwap_verdicts.append("Unknown")
                vwap_vals.append(None)
        else:
            verdicts.append("Unknown (No Ticker Data)")
            ranges.append("N/A")
            vwap_verdicts.append("Unknown")
            vwap_vals.append(None)

    df['daily_range'] = ranges
    df['verdict'] = verdicts
    df['vwap_approx'] = vwap_vals
    df['vwap_verdict'] = vwap_verdicts

    key_levels = find_key_levels(df)
    daily_premium, rolling_premium = calculate_rolling_accumulation(df)
    fwd_returns_df = calculate_forward_returns(df, market_data)

    return df, num_clusters, market_data, key_levels, daily_premium, rolling_premium, fwd_returns_df


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


def analyze_for_api(df_raw):
    """Run full dark-pool analysis and return JSON-safe dict for the web UI."""
    df, num_clusters, _, key_levels, daily_premium, rolling_premium, fwd_returns_df = _enrich_dataframe(df_raw.copy())

    bullish = float(df[df['verdict'].str.contains('Bullish', na=False)]['premium_val'].sum())
    bearish = float(df[df['verdict'].str.contains('Bearish', na=False)]['premium_val'].sum())
    neutral = float(df[df['verdict'].str.contains('Neutral', na=False)]['premium_val'].sum())
    total   = bullish + bearish + neutral

    overall = "Neutral / Mixed"
    if bullish > bearish * 1.5:
        overall = "Strongly Bullish Accumulation"
    elif bearish > bullish * 1.5:
        overall = "Strongly Bearish Distribution"
    elif bullish > bearish:
        overall = "Slightly Bullish Accumulation"
    elif bearish > bullish:
        overall = "Slightly Bearish Distribution"

    # Trades
    trades = []
    for _, r in df.iterrows():
        trades.append({
            'datetime':    r['datetime'].isoformat(),
            'date':        str(r['datetime'].date()),
            'time':        r['datetime'].strftime('%H:%M:%S'),
            'symbol':      str(r['symbol']),
            'price':       _safe_num(r['price']),
            'size':        _safe_num(r['size']),
            'premium_str': str(r['premium']),
            'premium_val': _safe_num(r['premium_val']),
            'tier':        str(r['tier']),
            'session':     str(r['session']),
            'daily_range': str(r['daily_range']),
            'verdict':     str(r['verdict']),
            'vwap_verdict':str(r['vwap_verdict']),
            'cluster_id':  int(r['cluster_id']),
        })

    sessions = []
    for s, g in df.groupby('session'):
        sessions.append({
            'session': str(s),
            'trades':  int(len(g)),
            'total_premium': float(g['premium_val'].sum()),
        })

    tier_order = ['Mega (>$50M)', 'Large ($10-50M)', 'Medium ($4-10M)', 'Small (<$4M)']
    tiers = []
    for t in tier_order:
        g = df[df['tier'] == t]
        if len(g):
            tiers.append({
                'tier':    t,
                'count':   int(len(g)),
                'total_premium': float(g['premium_val'].sum()),
                'avg_premium':   float(g['premium_val'].mean()),
            })

    clusters = []
    for cid in range(num_clusters):
        ct = df[df['cluster_id'] == cid]
        clusters.append({
            'id':            int(cid + 1),
            'trades':        int(len(ct)),
            'total_premium': float(ct['premium_val'].sum()),
            'avg_price':     float(ct['price'].mean()),
            'span_minutes':  float((ct['datetime'].max() - ct['datetime'].min()).total_seconds() / 60),
            'first_seen':    ct['datetime'].min().isoformat(),
        })

    levels = [{
        'price':         float(l['price']),
        'count':         int(l['count']),
        'total_premium': float(l['total_premium']),
        'dates':         [str(d) for d in l['dates']],
    } for l in key_levels[:15]]

    daily_flow = [{
        'date':    str(d.date()),
        'premium': float(p),
    } for d, p in daily_premium.items()]

    rolling = [{
        'date':                str(d.date()),
        'rolling_3d_premium':  float(p),
    } for d, p in rolling_premium.items()]

    fwd = []
    if not fwd_returns_df.empty:
        for _, r in fwd_returns_df.iterrows():
            fwd.append({
                'date':           str(r['date'].date()) if hasattr(r['date'], 'date') else str(r['date']),
                'daily_premium':  float(r['daily_premium']),
                'entry_price':    _safe_num(r.get('entry_price')),
                'fwd_1d_return':  _safe_num(r.get('fwd_1d_return')),
                'fwd_3d_return':  _safe_num(r.get('fwd_3d_return')),
                'fwd_5d_return':  _safe_num(r.get('fwd_5d_return')),
            })

    return {
        'symbol': str(df['symbol'].iloc[0]) if not df.empty else '',
        'summary': {
            'total_premium':    total,
            'bullish_premium':  bullish,
            'bearish_premium':  bearish,
            'neutral_premium':  neutral,
            'bullish_pct':      (bullish / total * 100) if total else 0,
            'bearish_pct':      (bearish / total * 100) if total else 0,
            'neutral_pct':      (neutral / total * 100) if total else 0,
            'overall_verdict':  overall,
            'trade_count':      int(len(df)),
            'num_clusters':     int(num_clusters),
            'date_range': {
                'start': df['datetime'].min().isoformat(),
                'end':   df['datetime'].max().isoformat(),
            }
        },
        'trades':               trades,
        'session_breakdown':    sessions,
        'tier_breakdown':       tiers,
        'clusters':             clusters,
        'key_levels':           levels,
        'daily_flow':           daily_flow,
        'rolling_accumulation': rolling,
        'forward_returns':      fwd,
    }


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("=== Dark Pool Print Analyzer & Visualizer (Enhanced) ===")

    user_input = input("Enter the CSV file name (e.g. 'my_data'): ").strip('\'" ')

    if not user_input.lower().endswith('.csv'):
        file_name = user_input + '.csv'
    else:
        file_name = user_input

    if not os.path.exists(file_name):
        print(f"\n[ERROR] Could not find the file: '{file_name}'")
        print("Ensure the file name is spelled correctly and is in the same folder as this script.")
    else:
        try:
            final_dataframe = analyze_dark_pool(file_name)
        except pd.errors.EmptyDataError:
            print("\n[ERROR] The provided CSV file is empty.")
        except KeyError as e:
            print(f"\n[ERROR] The CSV is missing a required column. Details: {e}")
        except Exception as e:
            import traceback
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            traceback.print_exc()
