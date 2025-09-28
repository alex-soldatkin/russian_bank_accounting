# %%
"""
Grouped variable density grids with group-specific transforms
- Amounts  -> **signed_log** (x-scale) per variable
- Percents -> keep data as % (0–1 auto-converted to 0–100); **y-scale log10**
- Ratios   -> keep as-is (linear x); **y-scale log10**

Other features:
- ≤ 9 facets per grid page (chunks if needed)
- Robust winsorization; symmetric for mixed-sign amounts (negatives retained)
- Sampling, facet ordering by IQR (viz-scale), and stat cards
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from lets_plot import LetsPlot
from lets_plot import (
    ggplot, aes, geom_density, facet_wrap, labs, ggsize, ggsave, geom_label,
    scale_x_continuous, scale_y_continuous, scale_y_log10, theme, element_text,
    sampling_random_stratified, sampling_systematic
)

LetsPlot.setup_html()

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
preprocessed_data_path = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
output_dir_html = 'visualisations/html'
output_dir_svg  = 'visualisations/svg'

os.makedirs(output_dir_html, exist_ok=True)
os.makedirs(output_dir_svg,  exist_ok=True)

if not os.path.exists(preprocessed_data_path):
    raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_data_path}")

print(f"Loading preprocessed data from {preprocessed_data_path}...")
df = pd.read_parquet(preprocessed_data_path)
print("Rows loaded:", len(df))

# ------------------------------------------------------------------
# Variables & pretty labels
# ------------------------------------------------------------------
preferred = [
    # Percent / ratio metrics (include if present)
    'ROA', 'ROE', 'NIM', 'npl_ratio', 'llp_to_loans_ratio', 'coverage_ratio',
    'loan_to_deposit_ratio', 
    # 'log_total_assets',

    # Core financials (amounts)
    "total_assets", "total_passives", "total_equity", "total_liabilities",
    "state_equity_pct", "state_loans", "individual_loans", "company_loans",
    "total_loans", "npl_amount", "provision_amount", "total_deposits",
    "total_liquid_assets", "interest_income", "operating_income",
    "interest_expense", "operating_expense", "net_interest_income",
    "net_income_amount",
]

pretty_labels: Dict[str, str] = {
    'NIM': 'NIM', 'ROA': 'ROA', 'ROE': 'ROE',
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

# Coerce preferred to numeric where present
for c in preferred:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Keep only available variables
available = [c for c in preferred if c in df.columns]

# --- Grouping rules ---
percent_vars = [v for v in available if v in {'ROA','ROE','NIM'} or v.endswith('_pct')]
ratio_vars   = [v for v in available if v in {'npl_ratio','llp_to_loans_ratio','coverage_ratio','loan_to_deposit_ratio'}]
amount_vars  = [v for v in available if v not in set(percent_vars) | set(ratio_vars)]

print("Groups -> percents:", percent_vars)
print("Groups -> ratios:", ratio_vars)
print("Groups -> amounts:", amount_vars)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
LOW_Q, HIGH_Q = 0.005, 0.995


def winsor_bounds(series: pd.Series, symmetric_if_mixed: bool = False) -> Tuple[float, float] | None:
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
    if symmetric_if_mixed and (s_min < 0) and (s_max > 0):
        h = float(np.nanquantile(np.abs(s), HIGH_Q))
        if not np.isfinite(h) or h == 0:
            h = float(np.nanmax(np.abs(s)))
        return (-h, h)
    lo = float(np.nanquantile(s, LOW_Q))
    hi = float(np.nanquantile(s, HIGH_Q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = s_min, s_max
    return (lo, hi)


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


# Global viz settings
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

N_TOTAL_BUDGET = 300_000
id_cols = ['DT', 'REGN', 'form']


def make_group_plots(group_name: str, vars_list: List[str], transform_kind: str):
    if not vars_list:
        print(f"[skip] {group_name}: no variables present")
        return

    # Split into chunks of at most 9 variables
    chunk_size = 9
    chunks = [vars_list[i:i+chunk_size] for i in range(0, len(vars_list), chunk_size)]

    for page, chunk in tqdm(enumerate(chunks, start=1), desc=f"Group: {group_name}", unit="page"):
        keep_cols = [c for c in id_cols if c in df.columns] + chunk
        df_sub = df[keep_cols].copy()
        melted = df_sub.melt(
            id_vars=[c for c in id_cols if c in df_sub.columns],
            value_vars=chunk,
            var_name='variable',
            value_name='value'
        )
        melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
        melted.replace([np.inf, -np.inf], np.nan, inplace=True)
        melted.dropna(subset=['value'], inplace=True)

        # Bounds: symmetric for amounts (to keep negatives), regular otherwise
        bounds: Dict[str, Tuple[float, float]] = {}
        for v in chunk:
            sym = (transform_kind == 'amounts')
            b = winsor_bounds(df[v], symmetric_if_mixed=sym)
            if b is not None:
                bounds[v] = b

        melted = melted[melted['variable'].isin(bounds.keys())].copy()
        melted['lo'] = melted['variable'].map({k: v[0] for k, v in bounds.items()})
        melted['hi'] = melted['variable'].map({k: v[1] for k, v in bounds.items()})
        mask = (melted['value'] >= melted['lo']) & (melted['value'] <= melted['hi'])
        melted_clip = melted[mask].copy()

        # Build transformed copy for plotting
        melted_viz = melted_clip.copy()

        transforms: Dict[str, str] = {}
        mult_map: Dict[str, float] = {}

        if transform_kind == 'amounts':
            transforms = {v: 'signed_log' for v in chunk}
            melted_viz['value_viz'] = signed_log(melted_viz['value'].to_numpy())  # type: ignore
        elif transform_kind == 'percents':
            # Keep as %; convert 0..1 -> 0..100 when appropriate per variable
            for v in chunk:
                s = pd.to_numeric(df[v], errors='coerce')
                p95 = float(np.nanpercentile(np.abs(s.dropna()), 95)) if s.notna().any() else 0
                mult_map[v] = 100.0 if p95 <= 1.5 else 1.0
                transforms[v] = 'percent'
            melted_viz['value_viz'] = [val * mult_map[var] for val, var in zip(melted_viz['value'], melted_viz['variable'])]
        else:  # ratios
            transforms = {v: 'linear' for v in chunk}
            melted_viz['value_viz'] = melted_viz['value']

        # Labels
        def facet_label(v: str) -> str:
            base = pretty_labels.get(v, v)
            t = transforms.get(v, 'linear')
            if t == 'signed_log':
                return f"{base}\n[signed_log]"
            if t == 'percent':
                return f"{base}\n[%]"
            return f"{base}\n[linear]"

        melted_viz['facet'] = [facet_label(v) for v in melted_viz['variable']]

        # Sampling budget (per page)
        n_facets = melted_viz['facet'].nunique()
        N_PER_FACET = max(4_000, N_TOTAL_BUDGET // max(n_facets, 1))
        sampler = sampling_random_stratified(N_PER_FACET, min_subsample=1_000) + sampling_systematic(N_TOTAL_BUDGET)

        # Order facets by IQR on viz-scale
        iqr = (melted_viz.groupby('facet')['value_viz'].quantile([0.75, 0.25]).unstack())
        facet_order = (iqr[0.75] - iqr[0.25]).sort_values(ascending=False).index.tolist()
        melted_viz['facet'] = pd.Categorical(melted_viz['facet'], categories=facet_order, ordered=True)

        # Stats for labels (top-right)
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

        # Build plot
        xlab = {
            'amounts': 'Value (signed_log)',
            'percents': 'Value (%)',
            'ratios': 'Value (ratio)'
        }[transform_kind]

        p = (
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
            # y-scale per group (log10 for percents & ratios only)
            + (scale_y_log10() if transform_kind in {'percents','ratios'} else scale_y_continuous(expand=[0.10, 0]))
            + labs(x=xlab, y='Density')
            + SMALL
            + ggsize(1400, 900)
        )

        # Save
        base = f"grid_density_{group_name}_page{page}"
        ggsave(p, path=output_dir_html, filename=f"{base}.html")
        ggsave(p, path=output_dir_svg,  filename=f"{base}.svg")
        print(f"Saved: {os.path.join(output_dir_html, base + '.html')}")


# ------------------------------------------------------------------
# Run for each group
# ------------------------------------------------------------------
make_group_plots('amounts', amount_vars, 'amounts')
make_group_plots('percents', percent_vars, 'percents')
make_group_plots('ratios', ratio_vars, 'ratios')

# --- Quick integrity check for state_loans ---
if 'state_loans' in amount_vars:
    s_all = pd.to_numeric(df['state_loans'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    print("state_loans -> min,max:", float(np.nanmin(s_all)), float(np.nanmax(s_all)), "neg_share:", float((s_all < 0).mean()))
