# %%
"""
Quarterly violin plots (with dropout bubbles) using Lets-Plot
----------------------------------------------------------------
- Mirrors the data plumbing and viz-scale transforms from the box/"journey"
  scripts you use (percent handling, signed_log for amounts, winsorization).
- Groups by **quarter** on the X-axis and draws **violin plots** per variable.
- Adds translucent **dropout bubbles** sized by how many banks have their last
  observation in that quarter (same definition as the boxplot script), layered
  *behind* the violins for context.
- Optional quantile lines on the violins.

Output: HTML pages under `visualisations/html/violin_quarter_page<N>.html`.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from lets_plot import LetsPlot
from lets_plot import (
    ggplot, aes, geom_violin, geom_point, facet_wrap,
    labs, ggsize, ggsave, theme, element_text, scale_size, scale_x_discrete
)

LetsPlot.setup_html()

# ------------------------------------------------------------------
# PARAMETERS (edit here)
# ------------------------------------------------------------------
DATA_PATH = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
OUT_HTML  = 'visualisations/html'

# Variables to consider (auto-filter to those present)
PREFERRED_VARS = [
    # Ratios / %
    'ROA','ROE','NIM','npl_ratio','llp_to_loans_ratio','coverage_ratio',
    'loan_to_deposit_ratio','state_equity_pct',
    # Amounts
    'total_assets','total_passives','total_equity','total_liabilities',
    'state_loans','individual_loans','company_loans','total_loans',
    'npl_amount','provision_amount','total_deposits','total_liquid_assets',
    'interest_income','operating_income','interest_expense','operating_expense',
    'net_interest_income','net_income_amount',
]

# Faceting / paging
MAX_FACETS_PER_PAGE = 2

# Violin styling
VIOLIN_ALPHA       = 0.30
VIOLIN_WIDTH_SCALE = 'area'   # 'area' | 'count' | 'width'
VIOLIN_TRIM        = True
VIOLIN_SHOW_HALF   = 0        # -1 (left), 0 (full), 1 (right)
SHOW_QUANTILES     = True
QUANTILES          = [0.25, 0.5, 0.75]

# Background dropout bubble styling
BUBBLE_MIN_SIZE    = 2.0
BUBBLE_MAX_SIZE    = 18.0
BUBBLE_ALPHA       = 0.22
BUBBLE_COLOR       = 'red'

# Value clipping on viz scale (winsorization) per variable
CLIP_ENABLED       = True
LOW_Q, HIGH_Q      = 0.05, 0.95

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def to_percent_series(s: pd.Series) -> Tuple[pd.Series, bool]:
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return s, False
    p95 = float(np.nanpercentile(np.abs(s.dropna()), 95))
    return (s * 100.0, True) if p95 <= 1.5 else (s, False)


def compute_viz_clip_bounds(df_long_viz: pd.DataFrame,
                             low_q: float, high_q: float) -> Dict[str, Tuple[float, float]]:
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


def normalize_sizes(x: pd.Series, smin=BUBBLE_MIN_SIZE, smax=BUBBLE_MAX_SIZE) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=x.index)
    lx = np.log1p(np.maximum(x, 0))
    lo, hi = float(np.nanpercentile(lx.dropna(), 1)), float(np.nanpercentile(lx.dropna(), 99))
    if not np.isfinite(hi - lo) or hi == lo:
        return pd.Series(np.clip(np.full_like(lx, (smin + smax) / 2), smin, smax), index=x.index)
    z = (lx - lo) / (hi - lo)
    return pd.Series(smin + z * (smax - smin), index=x.index)

# ------------------------------------------------------------------
# Load & prep
# ------------------------------------------------------------------
ensure_dirs(OUT_HTML)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")

print(f"Loading {DATA_PATH} ...")
DF = pd.read_parquet(DATA_PATH)
print('Rows:', len(DF))

if 'DT' not in DF.columns or 'REGN' not in DF.columns:
    raise ValueError('Expected columns DT (timestamp) and REGN (bank id)')

DF['DT'] = pd.to_datetime(DF['DT'])
if 'total_assets' in DF.columns:
    DF['total_assets'] = pd.to_numeric(DF['total_assets'], errors='coerce')

available = [c for c in PREFERRED_VARS if c in DF.columns]
print('Variables available:', available)
for c in available:
    DF[c] = pd.to_numeric(DF[c], errors='coerce')

percent_vars = [v for v in available if v in {'ROA','ROE','NIM'} or v.endswith('_pct')]
ratio_vars   = [v for v in available if v in {'npl_ratio','llp_to_loans_ratio','coverage_ratio','loan_to_deposit_ratio'} and v not in percent_vars]
amount_vars  = [v for v in available if v not in set(percent_vars) | set(ratio_vars)]

print('Groups -> percents:', percent_vars)
print('Groups -> ratios:', ratio_vars)
print('Groups -> amounts:', amount_vars)

ALL_VARS: List[str] = percent_vars + ratio_vars + amount_vars
pages: List[List[str]] = [ALL_VARS[i:i+MAX_FACETS_PER_PAGE] for i in range(0, len(ALL_VARS), MAX_FACETS_PER_PAGE)]

# Quarter keys & dropout counts
DF['quarter'] = DF['DT'].dt.to_period('Q')
DF['quarter_str'] = DF['quarter'].astype(str)

last_quarter = (
    DF.dropna(subset=['REGN', 'DT'])
      .groupby('REGN')['DT']
      .max()
      .dt.to_period('Q')
      .astype(str)
      .rename('quarter_str')
      .to_frame()
)
DROP_COUNTS = last_quarter.groupby('quarter_str').size().rename('dropouts').reset_index()

# ------------------------------------------------------------------
# Iterate pages
# ------------------------------------------------------------------
for page_idx, var_chunk in enumerate(pages, start=1):
    keep_cols = ['DT','REGN','total_assets','quarter','quarter_str'] + var_chunk
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in DF.columns]))
    sub = DF[keep_cols].copy()

    long = pd.melt(
        sub,
        id_vars=['DT','REGN','total_assets','quarter','quarter_str'],
        value_vars=var_chunk,
        var_name='variable',
        value_name='value'
    )
    long['value'] = pd.to_numeric(long['value'], errors='coerce').replace([np.inf,-np.inf], np.nan)

    # Transform per variable to visualization scale
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

    if CLIP_ENABLED and not long_viz.empty:
        bmap = compute_viz_clip_bounds(long_viz, LOW_Q, HIGH_Q)
        if bmap:
            long_viz['lo'] = long_viz['variable'].map({k: v[0] for k, v in bmap.items()})
            long_viz['hi'] = long_viz['variable'].map({k: v[1] for k, v in bmap.items()})
            long_viz = long_viz[(long_viz['value_viz'] >= long_viz['lo']) & (long_viz['value_viz'] <= long_viz['hi'])]
            long_viz = long_viz.drop(columns=['lo','hi'])

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
        return base

    long_viz['facet'] = long_viz['variable'].map(facet_label)

    # Median per quarter for bubble y placement
    med = (
        long_viz.groupby(['variable','facet','quarter_str'])['value_viz']
                .median().rename('y_med').reset_index()
    )

    bubble_df = (
        med.merge(DROP_COUNTS, on='quarter_str', how='left')
           .fillna({'dropouts': 0})
    )
    bubble_df['size_pt'] = normalize_sizes(bubble_df['dropouts'])

    # Quarter order
    all_q = (
        long_viz[['quarter_str','quarter']].drop_duplicates()
                .sort_values('quarter')
    )
    q_order = all_q['quarter_str'].tolist()

    base = ggplot() + scale_x_discrete(limits=q_order)

    # Background dropout bubbles
    if not bubble_df.empty:
        base = base + geom_point(
            aes(x='quarter_str', y='y_med', size='size_pt'),
            data=bubble_df,
            color=BUBBLE_COLOR,
            alpha=BUBBLE_ALPHA,
            show_legend=False,
        ) + scale_size(range=[BUBBLE_MIN_SIZE, BUBBLE_MAX_SIZE])

    # Violin plots per quarter
    base = base + geom_violin(
        aes(x='quarter_str', y='value_viz', group='quarter_str'),
        data=long_viz,
        alpha=VIOLIN_ALPHA,
        scale=VIOLIN_WIDTH_SCALE,
        trim=VIOLIN_TRIM,
        show_half=VIOLIN_SHOW_HALF,
        quantiles=QUANTILES,
        quantile_lines=SHOW_QUANTILES,
    )

    plot = (
        base
        + facet_wrap('facet', scales='free_y', labwidth=24)
        + labs(x='Quarter', y='Value', title='Quarterly Violin Plots with Dropout Bubbles',
               subtitle='Bubble size = number of banks with final observation in that quarter')
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

    out_name = f"violin_quarter_page{page_idx}.html"
    ggsave(plot, path=OUT_HTML, filename=out_name)
    print(f"Saved: {os.path.join(OUT_HTML, out_name)}")

print('Done.')
