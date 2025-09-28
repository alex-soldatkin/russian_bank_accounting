# %%
"""
Journey bands with start/end markers (standalone)

Hotfix for duplicate-column melt bug:
- When `total_assets` is both in the variables chunk **and** in the base columns,
  Pandas will materialize **duplicate column names** in `sub` and then
  `sub['total_assets']` becomes a *DataFrame* (→ "Cannot set a DataFrame ...").
- Fix: **deduplicate** `keep_cols` and create a separate `size_assets` used only
  for bubble sizes & quartiles. `total_assets` remains available for plotting as
  a variable without entering `id_vars`.

What this script does
- For each selected variable, computes **within-month asset quartiles** of banks
- Plots translucent background **bubbles** (one per bank-month), size ~ log(total_assets)
- Overlays **quartile bands** (P25–P75) and **median lines** per quartile through time
- Adds **start (green)** / **end (red)** markers for highlighted banks
- Facets by variable, with ≤ 9 facets per page; saves HTML + SVG per page

Assumptions
- Input parquet: data/HOORRAAH_final_banking_indicators_preprocessed.parquet
- Columns present: DT (timestamp), REGN (bank id), total_assets, variables listed below
- Mixed-sign amounts are allowed (e.g., state_loans < 0)

Tune the PARAMETERS section to pick variables, groups, highlight mode, etc.

New in this update:
- Added **per-variable winsorization on the viz scale** (default 0.5–99.5%) to tame extreme outliers that were blowing up the y-axis.
- Dropped scientific-power tick formatting to avoid unreadable exponents.
- Exposed key aesthetics (alphas, size range) via parameters for quick tweaks.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from lets_plot import LetsPlot
from lets_plot import (
    ggplot, aes, geom_point, geom_line, geom_ribbon, facet_wrap,
    labs, ggsize, ggsave, theme, element_text, scale_y_continuous,
    scale_x_datetime, scale_size, scale_y_log10, geom_smooth, geom_abline
)

LetsPlot.setup_html()

# ------------------------------------------------------------------
# PARAMETERS (edit here)
# ------------------------------------------------------------------
# Paths
DATA_PATH = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
OUT_HTML = 'visualisations/html'
OUT_SVG  = 'visualisations/svg'

# Variables to consider (will auto-filter to those present)
PREFERRED_VARS = [
    # Percent / ratio metrics (if available)
    'ROA', 'ROE', 'NIM', 'npl_ratio', 'llp_to_loans_ratio', 'coverage_ratio',
    'loan_to_deposit_ratio',
    # Core financials (amounts)
    'total_assets', 'total_passives', 'total_equity', 'total_liabilities',
    'state_equity_pct', 'state_loans', 'individual_loans', 'company_loans',
    'total_loans', 'npl_amount', 'provision_amount', 'total_deposits',
    'total_liquid_assets', 'interest_income', 'operating_income',
    'interest_expense', 'operating_expense', 'net_interest_income',
    'net_income_amount',
]

# Faceting / paging
MAX_FACETS_PER_PAGE = 2

# Highlight mode: 'none' | 'top_n_assets' | 'regn_list'
HIGHLIGHT_MODE = 'top_n_assets'
TOP_N_BANKS = 30
REGN_LIST: List[int] = []   # used if HIGHLIGHT_MODE == 'regn_list'

# Bubble styling
POINT_ALPHA = 0.03      # from 0.10
POINT_MIN_SIZE = 0.6    # from 0.1 (too tiny → noisy stipple)
POINT_MAX_SIZE = 4.5    # from 6.0

# Band/line styling
RIBBON_ALPHA = 0.28     # from 0.15 (bands more legible)
MEDIAN_LINE_WIDTH = 0.7 # from 1.0 (lines less dominant)

# Bubbles for outliers

OUTLIER_HIGH_Q = 0.99  # from 0.999
OUTLIER_LOW_Q  = 0.01  # from 0.001

# Winsorization for value clipping (per variable, viz scale)
CLIP_ENABLED = True
LOW_Q, HIGH_Q = 0.05, 0.95   # from 0.005, 0.995

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def to_percent_series(s: pd.Series) -> Tuple[pd.Series, bool]:
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return s, False
    p95 = float(np.nanpercentile(np.abs(s.dropna()), 95))
    return (s * 100.0, True) if p95 <= 1.5 else (s, False)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def normalize_sizes(x: pd.Series, smin=POINT_MIN_SIZE, smax=POINT_MAX_SIZE) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=x.index)
    lx = np.log1p(np.maximum(x, 0))
    lo, hi = float(np.nanpercentile(lx.dropna(), 1)), float(np.nanpercentile(lx.dropna(), 99))
    if not np.isfinite(hi - lo) or hi == lo:
        return pd.Series(np.clip(np.full_like(lx, (smin + smax) / 2), smin, smax), index=x.index)
    z = (lx - lo) / (hi - lo)
    return pd.Series(smin + z * (smax - smin), index=x.index)


def compute_viz_clip_bounds(df_long_viz: pd.DataFrame, low_q: float, high_q: float) -> Dict[str, Tuple[float, float]]:
    """Per-variable quantile bounds on the **viz-scale** values."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for v, s in df_long_viz.groupby('variable')['value_viz']:
        arr = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if arr.empty:
            continue
        lo = float(np.nanquantile(arr, low_q))
        hi = float(np.nanquantile(arr, high_q))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        bounds[v] = (lo, hi)
    return bounds

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
ensure_dirs(OUT_HTML, OUT_SVG)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")

print(f"Loading {DATA_PATH} ...")
df = pd.read_parquet(DATA_PATH)
print("Rows:", len(df))

if 'DT' not in df.columns:
    raise ValueError("Expected column DT (timestamp)")

df['DT'] = pd.to_datetime(df['DT'])

# Keep only preferred vars that exist
available = [c for c in PREFERRED_VARS if c in df.columns]
print("Variables available:", available)

# Coerce to numeric
for c in available + ['total_assets']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Grouping
percent_vars = [v for v in available if v in {'ROA','ROE','NIM'} or v.endswith('_pct')]
ratio_vars   = [v for v in available if v in {'npl_ratio','llp_to_loans_ratio','coverage_ratio','loan_to_deposit_ratio'} and v not in percent_vars]
amount_vars  = [v for v in available if v not in set(percent_vars) | set(ratio_vars)]

print("Groups -> percents:", percent_vars)
print("Groups -> ratios:", ratio_vars)
print("Groups -> amounts:", amount_vars)

# Highlight set
if HIGHLIGHT_MODE == 'regn_list':
    highlight_regn = set(REGN_LIST)
elif HIGHLIGHT_MODE == 'top_n_assets':
    agg = (df[['REGN','total_assets']].dropna().groupby('REGN')['total_assets']
           .max().sort_values(ascending=False))
    highlight_regn = set(agg.head(TOP_N_BANKS).index.tolist())
else:
    highlight_regn = set()

print(f"Highlight mode: {HIGHLIGHT_MODE}; banks highlighted: {len(highlight_regn)}")

# ------------------------------------------------------------------
# Prep per-variable journey data
# ------------------------------------------------------------------
ALL_VARS = percent_vars + ratio_vars + amount_vars
pages: List[List[str]] = [ALL_VARS[i:i+MAX_FACETS_PER_PAGE] for i in range(0, len(ALL_VARS), MAX_FACETS_PER_PAGE)]

for page_idx, var_chunk in enumerate(pages, start=1):
    # Build working frame
    base_cols = ['DT','REGN','total_assets']
    keep_cols = base_cols + var_chunk
    # Deduplicate keep_cols while preserving order
    keep_cols = list(dict.fromkeys(keep_cols))
    sub = df[keep_cols].copy()

    # Safe sizing column (Series even if names duplicate elsewhere)
    sub['size_assets'] = sub['total_assets'] if 'total_assets' in sub.columns else np.nan

    # Melt: id_vars exclude total_assets; it will be melted if in var_chunk
    long = pd.melt(
        sub,
        id_vars=['DT','REGN','size_assets'],
        value_vars=var_chunk,
        var_name='variable',
        value_name='value'
    )

    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    long = long.replace([np.inf, -np.inf], np.nan)

    # Transform per group
    def transform_series(var: str, s: pd.Series) -> Tuple[pd.Series, str]:
        if var in amount_vars:
            return pd.Series(signed_log(s.to_numpy()), index=s.index), 'signed_log'
        if var in percent_vars:
            s2, _ = to_percent_series(s)
            return s2, 'percent'
        return s, 'linear'

    out = []
    kind_map: Dict[str, str] = {}
    for v in var_chunk:
        s = long.loc[long['variable'].eq(v), 'value']
        tv, kind = transform_series(v, s)
        kind_map[v] = kind
        tmp = long.loc[long['variable'].eq(v)].copy()
        tmp['value_viz'] = tv.values
        out.append(tmp)
    long_viz = pd.concat(out, ignore_index=True).dropna(subset=['value_viz'])

    # Clip on the viz scale to prevent a few extremes from wrecking axes
    if CLIP_ENABLED and not long_viz.empty:
        bmap = compute_viz_clip_bounds(long_viz, LOW_Q, HIGH_Q)
        if bmap:
            long_viz['lo'] = long_viz['variable'].map({k: v[0] for k, v in bmap.items()})
            long_viz['hi'] = long_viz['variable'].map({k: v[1] for k, v in bmap.items()})
            long_viz = long_viz[(long_viz['value_viz'] >= long_viz['lo']) & (long_viz['value_viz'] <= long_viz['hi'])]
            long_viz = long_viz.drop(columns=['lo','hi'])

    # Quartiles by month using size_assets
    m = long_viz.dropna(subset=['size_assets']).copy()
    # truncate m to outliers only 
    m = m[(m['value_viz'] < m['value_viz'].quantile(OUTLIER_HIGH_Q)) & (m['value_viz'] > m['value_viz'].quantile(OUTLIER_LOW_Q))]
    def month_quartiles(g: pd.DataFrame) -> pd.Series:
        x = g['size_assets']
        try:
            q = pd.qcut(x.rank(method='first'), 4, labels=['Q1','Q2','Q3','Q4'])
        except Exception:
            q = pd.Series(['Q2'] * len(g), index=g.index)
        return q
    m['quartile'] = (m.groupby(['DT']).apply(month_quartiles).reset_index(level=0, drop=True))

    # Bubble size
    m['size_pt'] = normalize_sizes(m['size_assets'])

    # Bands & medians
    ag = (m.groupby(['variable','quartile','DT'])['value_viz']
            .agg(p25=lambda s: np.nanpercentile(s, 25),
                 p50=lambda s: np.nanpercentile(s, 50),
                 p75=lambda s: np.nanpercentile(s, 75))
            .reset_index())

    PRETTY = {
        'NIM':'NIM','ROA':'ROA','ROE':'ROE','coverage_ratio':'Coverage ratio',
        'llp_to_loans_ratio':'LLP ÷ Loans','loan_to_deposit_ratio':'Loans ÷ Deposits',
        'total_assets':'Total assets','total_passives':'Total liabilities','total_equity':'Total equity',
        'total_liabilities':'Total liabilities','state_equity_pct':'State equity %','state_loans':'Loans to state',
        'individual_loans':'Retail loans','company_loans':'Company loans','npl_amount':'NPL',
        'provision_amount':'Loss provisions','total_deposits':'Total deposits','total_liquid_assets':'Total liquid assets',
        'interest_income':'Interest income','operating_income':'Operating income','interest_expense':'Interest expense',
        'operating_expense':'Operating expense','net_interest_income':'Net interest income','net_income_amount':'Net income',
        'npl_ratio':'NPL ratio'
    }

    def facet_label(v: str) -> str:
        base = PRETTY.get(v, v)
        k = kind_map.get(v, 'linear')
        if k == 'signed_log':
            return f"{base}\n[signed_log]"
        if k == 'percent':
            return f"{base} [%]"
        return f"{base}"

    m['facet'] = m['variable'].map(facet_label)
    ag['facet'] = ag['variable'].map(facet_label)

    # Facet order by median IQR over time
    iqr_tmp = (ag.groupby('facet').apply(lambda g: np.nanmedian(g['p75'] - g['p25']))
                 .sort_values(ascending=False))
    facet_order = iqr_tmp.index.tolist()
    m['facet']  = pd.Categorical(m['facet'], categories=facet_order, ordered=True)
    ag['facet'] = pd.Categorical(ag['facet'], categories=facet_order, ordered=True)

    # Highlights
    if highlight_regn:
        hv = m[m['REGN'].isin(highlight_regn)].copy().sort_values(['REGN','variable','DT'])
        first_idx = hv.groupby(['REGN','variable']).head(1).index
        last_idx  = hv.groupby(['REGN','variable']).tail(1).index
        starts = hv.loc[first_idx, ['DT','variable','value_viz','facet']].copy()
        ends   = hv.loc[last_idx,  ['DT','variable','value_viz','facet']].copy()
    else:
        starts = pd.DataFrame(columns=['DT','variable','value_viz','facet'])
        ends   = pd.DataFrame(columns=['DT','variable','value_viz','facet'])

    # Plot
    p = (
        ggplot()
        + geom_point(aes(x='DT', y='value_viz', size='size_pt', color='quartile'), data=m, alpha=POINT_ALPHA)
        # + geom_ribbon(aes(x='DT', ymin='p25', ymax='p75', fill='quartile'), data=ag, alpha=RIBBON_ALPHA)
        + geom_line(aes(x='DT', y='p50', color='quartile', group='quartile'), data=ag, size=MEDIAN_LINE_WIDTH)
        + (geom_point(aes(x='DT', y='value_viz'), data=starts, color='green', size=3.5) if len(starts) else None)
        + (geom_point(aes(x='DT', y='value_viz'), data=ends, color='red', size=3.5) if len(ends) else None)
        # optionally connect start/end with dashed line
        + (geom_line(aes(x='DT', y='value_viz', group='REGN'), data=hv[hv['REGN'].isin(highlight_regn)], color='gray', size=0.4, linetype='dashed') if len(hv) > 0 else None)
        + (geom_smooth(aes(x='DT', y='value_viz'), data=m, method='loess', color='black', size=1.0, se=False)
           if len(m) > 10 else None)
        + facet_wrap('facet', scales='free_y', labwidth=24)
        + scale_x_datetime()
        # + scale_y_log10()
        + scale_size(range=[POINT_MIN_SIZE, POINT_MAX_SIZE])
        + labs(x='Time', y='Value', color='Asset quartile', fill='Asset quartile', size='Bank size')
        + theme(
            plot_title   = element_text(size=16),
            axis_title   = element_text(size=10),
            axis_text_x  = element_text(size=8),
            axis_text_y  = element_text(size=8),
            strip_text   = element_text(size=9),
            legend_text  = element_text(size=8),
            legend_title = element_text(size=9),
                    )
        + ggsize(1400, 900)
    )

    base = f"journey_bands_page{page_idx}"
    ggsave(p, path=OUT_HTML, filename=f"{base}.html")
    # ggsave(p, path=OUT_SVG,  filename=f"{base}.png")
    print(f"Saved: {os.path.join(OUT_HTML, base + '.html')}")

print("Done.")
