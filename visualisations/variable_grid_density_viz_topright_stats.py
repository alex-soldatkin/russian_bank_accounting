# %%
"""
Variable grid density with robust mixed-sign handling
- Ensures preferred columns are coerced to numeric regardless of dtype
- Uses *symmetric* winsorization for mixed-sign variables (e.g., `state_loans`)
  so negative values (loans from the state) are not clipped away
- Chooses transforms more robustly:
    * mixed-sign  -> 'signed_log'
    * non-negative & highly skewed -> 'log1p'
    * otherwise -> 'linear'
"""

import os
import numpy as np
import pandas as pd
from lets_plot import LetsPlot
from lets_plot import (
    ggplot, aes, geom_density, facet_wrap, labs, ggsize, ggsave, geom_label,
    scale_x_continuous, scale_y_continuous, theme, element_text,
    sampling_random_stratified, sampling_systematic
)

LetsPlot.setup_html()

# Paths
preprocessed_data_path = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
output_dir_html = 'visualisations/html'
output_dir_svg  = 'visualisations/svg'

os.makedirs(output_dir_html, exist_ok=True)
os.makedirs(output_dir_svg,  exist_ok=True)

# --- Load ---
if not os.path.exists(preprocessed_data_path):
    raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_data_path}")

print(f"Loading preprocessed data from {preprocessed_data_path}...")
df = pd.read_parquet(preprocessed_data_path)
print("Rows loaded:", len(df))

# --- Variables ---
preferred = [
    # Ratios & logs could be added here if desired
    # 'ROA', 'ROE', 'NIM', 'npl_ratio', 'llp_to_loans_ratio', 'coverage_ratio', 'loan_to_deposit_ratio', 'log_total_assets', 'total_assets', 'total_loans', 'total_deposits', 'net_income_amount'

    # Core financials
    "total_assets", "total_passives", "total_equity", "total_liabilities",
    "state_equity_pct", "state_loans", "individual_loans", "company_loans",
    "total_loans", "npl_amount", "provision_amount", "total_deposits",
    "total_liquid_assets", "interest_income", "operating_income",
    "interest_expense", "operating_expense", "net_interest_income",
    "net_income_amount",
]

# Force numeric (Parquet may yield 'string' or 'string[pyarrow]' dtypes)
for c in preferred:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Select available variables up to a cap (keep order from 'preferred')
max_vars = 19
selected = [c for c in preferred if c in df.columns][:max_vars]

# If fewer than max_vars, pad with other numeric columns to reach variety
if len(selected) < max_vars:
    numeric_others = [c for c in df.select_dtypes(include=[np.number]).columns if c not in selected]
    selected.extend(numeric_others[: (max_vars - len(selected))])

print("Selected variables for grid plots:", selected)

# --- Prepare long frame ---
id_cols = ['DT', 'REGN', 'form']
keep_cols = [c for c in id_cols if c in df.columns] + selected

df_sub = df[keep_cols].copy()

melted = df_sub.melt(
    id_vars=[c for c in id_cols if c in df_sub.columns],
    value_vars=selected,
    var_name='variable',
    value_name='value'
)

melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
melted.replace([np.inf, -np.inf], np.nan, inplace=True)
melted.dropna(subset=['value'], inplace=True)

# --- Robust per-variable clipping ---
LOW_Q, HIGH_Q = 0.005, 0.995

def winsor_bounds(series: pd.Series, low_q=LOW_Q, high_q=HIGH_Q):
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    s_min, s_max = np.nanmin(s), np.nanmax(s)
    # Mixed-sign -> symmetric bound on |s| to keep both negative and positive tails
    if (s_min < 0) and (s_max > 0):
        h = np.nanquantile(np.abs(s), high_q)
        if not np.isfinite(h) or h == 0:
            h = np.nanmax(np.abs(s))
        return (-h, h)
    # Otherwise use standard quantile bounds
    lo = np.nanquantile(s, low_q)
    hi = np.nanquantile(s, high_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = s_min, s_max
    return (lo, hi)

bounds = {}
for v in selected:
    b = winsor_bounds(df[v])
    if b is not None:
        bounds[v] = b

# Filter rows inside bounds
melted = melted[melted['variable'].isin(bounds.keys())].copy()
melted['lo'] = melted['variable'].map({k: v[0] for k, v in bounds.items()})
melted['hi'] = melted['variable'].map({k: v[1] for k, v in bounds.items()})
mask = (melted['value'] >= melted['lo']) & (melted['value'] <= melted['hi'])
melted_clip = melted[mask].copy()

# Drop variables that became empty post-clipping
cnts = melted_clip['variable'].value_counts()
keep_vars = cnts.index.tolist()
melted_clip = melted_clip[melted_clip['variable'].isin(keep_vars)].copy()

print("Rows before clipping:", len(melted), "Rows after clipping:", len(melted_clip))
print("Variables retained:", sorted(set(melted_clip['variable'])))

# --- Transform decision ---

def choose_transform(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 'linear'
    s_min, s_max = np.nanmin(s), np.nanmax(s)
    # Mixed-sign -> signed_log by default (helps with wide dynamic range around 0)
    if (s_min < 0) and (s_max > 0):
        return 'signed_log'
    # Non-negative: decide by robust spread (p99.5 vs p50)
    if s_min >= 0:
        p995 = np.nanpercentile(s, 99.5)
        p50  = np.nanpercentile(s, 50)
        ratio = (p995 + 1e-12) / (p50 + 1e-12)
        return 'log1p' if ratio > 50 else 'linear'
    # All negative (rare here) -> signed_log too
    return 'signed_log'

transform_map = {}
for v in sorted(set(melted_clip['variable'])):
    transform_map[v] = choose_transform(df[v])

# Build transformed copy for plotting
melted_viz = melted_clip.copy()

def apply_transform(val, t):
    if t == 'log1p':
        return np.log1p(np.maximum(val, 0))
    if t == 'signed_log':
        return np.sign(val) * np.log1p(np.abs(val))
    return val

melted_viz['transform'] = melted_viz['variable'].map(transform_map)
melted_viz['value_viz'] = [apply_transform(v, t) for v, t in zip(melted_viz['value'], melted_viz['transform'])]

# Pretty labels
pretty_labels = {
    'NIM': 'NIM',
    'ROA': 'ROA',
    'ROE': 'ROE',
    'coverage_ratio': 'Coverage ratio',
    'llp_to_loans_ratio': 'LLP ÷ Loans',
    'loan_to_deposit_ratio': 'Loans ÷ Deposits',
    'log_total_assets': 'log(Total assets)',
    'total_assets': 'Total assets',
    'total_deposits': 'Total deposits',
    'total_loans': 'Total loans',
    'net_income_amount': 'Net income',
    'npl_ratio': 'NPL ratio',
    'total_passives': 'Total liabilities',
    'total_equity': 'Total equity',
    'total_liabilities': 'Total liabilities',
    'state_equity_pct': 'State equity %',
    'state_loans': 'Loans to state',
    'individual_loans': 'Retail loans',
    'company_loans': 'Company loans',
    'npl_amount': 'NPL',
    'provision_amount': 'Loss provisions',
    'total_liquid_assets': 'Total liquid assets',
    'interest_income': 'Interest income',
    'operating_income': 'Operating income',
    'interest_expense': 'Interest expense',
    'operating_expense': 'Operating expense',
    'net_interest_income': 'Net interest income',
}

melted_viz['pretty'] = melted_viz['variable'].map(pretty_labels).fillna(melted_viz['variable'])
melted_viz['facet']  = melted_viz['pretty'] + '\n[' + melted_viz['transform'] + ']'

# --- Sampling budget ---
N_TOTAL_BUDGET = 300_000
n_facets = melted_viz['facet'].nunique()
N_PER_FACET  = max(4_000, N_TOTAL_BUDGET // max(n_facets, 1))

sampler = sampling_random_stratified(N_PER_FACET, min_subsample=1_000) + sampling_systematic(N_TOTAL_BUDGET)

# --- Order facets by IQR (viz-scale) ---
iqr = (melted_viz.groupby('facet')['value_viz'].quantile([0.75, 0.25]).unstack())
facet_order = (iqr[0.75] - iqr[0.25]).sort_values(ascending=False).index.tolist()
melted_viz['facet'] = pd.Categorical(melted_viz['facet'], categories=facet_order, ordered=True)

# --- Per-facet stats for label placement (top-right heuristic) ---
g = melted_viz.groupby('facet')['value_viz']
q = g.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).unstack()
q.columns = ['q05','q10','q25','q50','q75','q90','q95']

stats = pd.DataFrame({
    'facet': q.index,
    'n': g.size().values,
    'zeros_pct': g.apply(lambda s: (s == 0).mean() * 100.0).values,
    'q05': q['q05'].values, 'q10': q['q10'].values,
    'q25': q['q25'].values, 'q50': q['q50'].values, 'q75': q['q75'].values,
    'q90': q['q90'].values, 'q95': q['q95'].values
})

stats['iqr'] = stats['q75'] - stats['q25']
robust_sigma = np.clip(stats['iqr'] / 1.349, 1e-6, None)
y_peak = 1.0 / (np.sqrt(2*np.pi) * robust_sigma)
y_max_est = y_peak * 1.6

stats_tr = stats.copy()
stats_tr['x_pos'] = stats_tr['q05'] + 0.85 * (stats_tr['q95'] - stats_tr['q05'])
stats_tr['y_pos'] = 0.93 * y_max_est

stats_tr['label'] = stats_tr.apply(
    lambda r: f"n={int(r['n']):,}\nzeros={r['zeros_pct']:.1f}%\nmedian={r['q50']:.3g}\nIQR={r['iqr']:.3g}",
    axis=1
)

# --- Theme ---
SMALL = theme(
    plot_title   = element_text(size=16),
    axis_title   = element_text(size=10),
    axis_text_x  = element_text(size=8),
    axis_text_y  = element_text(size=8),
    strip_text   = element_text(size=9),
    legend_text  = element_text(size=8),
    legend_title = element_text(size=9),
    exponent_format='pow',
)

# --- Plot ---
p_density_viz_topright = (
    ggplot(melted_viz, aes(x='value_viz', group='facet'))
    + geom_density(
        sampling=sampler,
        n=256, trim=True,
        quantiles=[0.25, 0.5, 0.75],
        quantile_lines=True,
        tooltips='none'
      )
    + geom_label(
        aes(x='x_pos', y='y_pos', label='label'),
        data=stats_tr,
        hjust=16, vjust=1,
        size=3, alpha=0.10, fill='gray92',
        nudge_x=0.3
      )
    + facet_wrap('facet', scales='free', labwidth=24)
    + scale_x_continuous(expand=[0.06, 0])
    + scale_y_continuous(expand=[0.10, 0])
    + labs(x='Value (viz scale)', y='Density')
    + SMALL
    + ggsize(1400, 900)
)

# --- Save ---
html_name = 'variable_grid_density_viz_topright_stats_RAW_STATS.html'
svg_name  = 'variable_grid_density_viz_topright_RAW_STATS.svg'

ggsave(p_density_viz_topright, path=output_dir_html, filename=html_name)
ggsave(p_density_viz_topright, path=output_dir_svg,  filename=svg_name)

print(f"Saved density grid with top-right stats to {os.path.join(output_dir_html, html_name)}")
print(f"Saved density grid with top-right stats to {os.path.join(output_dir_svg, svg_name)}")

# --- Quick debug: verify negatives survive for state_loans ---
if 'state_loans' in selected:
    s_all = pd.to_numeric(df['state_loans'], errors='coerce')
    s_all = s_all.replace([np.inf, -np.inf], np.nan).dropna()
    s_post = melted_viz.loc[melted_viz['variable'].eq('state_loans'), 'value']
    print("state_loans stats -> all(min, max):", float(np.nanmin(s_all)), float(np.nanmax(s_all)))
    if not s_post.empty:
        print("state_loans post-clip -> min:", float(np.nanmin(s_post)), "max:", float(np.nanmax(s_post)),
              "neg_share:", float((s_post < 0).mean()))
    else:
        print("state_loans disappeared post-clip (no rows)")
