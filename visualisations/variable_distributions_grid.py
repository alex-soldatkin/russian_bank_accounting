import os
import numpy as np
import pandas as pd
from lets_plot import *
from lets_plot import ggplot, aes, geom_histogram, geom_density, facet_wrap, labs, ggsize, ggsave

LetsPlot.setup_html()

# Paths
preprocessed_data_path = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
output_dir_html = 'visualisations/html'
os.makedirs(output_dir_html, exist_ok=True)

if not os.path.exists(preprocessed_data_path):
    print(f"Error: Preprocessed data file not found at {preprocessed_data_path}")
else:
    print(f"Loading preprocessed data from {preprocessed_data_path}...")
    df = pd.read_parquet(preprocessed_data_path)
    n_rows = len(df)
    print("Rows loaded:", n_rows)

    preferred = [
        'ROA', 'ROE', 'NIM', 'npl_ratio', 'llp_to_loans_ratio',
        'coverage_ratio', 'loan_to_deposit_ratio', 'log_total_assets',
        'total_assets', 'total_loans', 'total_deposits', 'net_income_amount'
    ]

    # Force numeric where possible (Parquet decimals can arrive as 'object')
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors='ignore')

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    selected = [c for c in preferred if c in numeric_columns]
    max_vars = 12
    if len(selected) < max_vars:
        for c in numeric_columns:
            if c not in selected:
                selected.append(c)
            if len(selected) >= max_vars:
                break

    print("Selected variables for grid plots:", selected)

    id_cols = ['DT', 'REGN', 'form']
    to_keep = [c for c in id_cols if c in df.columns] + selected
    df_sub = df[to_keep].copy()

    # Build a long frame and make sure value is numeric float
    melted = df_sub.melt(
        id_vars=[c for c in id_cols if c in df_sub.columns],
        value_vars=selected,
        var_name='variable',
        value_name='value'
    )
    melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
    melted = melted.replace([np.inf, -np.inf], np.nan).dropna(subset=['value'])

    # --- Robust per-variable clipping (guards against all-NaN or constant cols) ---
    low_q, high_q = 0.005, 0.995
    bounds = {}
    for v in selected:
        s = pd.to_numeric(df[v], errors='coerce')
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.size == 0:
            continue
        lo = np.nanquantile(s, low_q)
        hi = np.nanquantile(s, high_q)
        # Fallbacks if quantiles are weird
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(s), np.nanmax(s)
        if np.isfinite(lo) and np.isfinite(hi):
            bounds[v] = (lo, hi)

    # Keep only variables with usable bounds
    melted = melted[melted['variable'].isin(bounds.keys())].copy()
    melted['lo'] = melted['variable'].map({k: v[0] for k, v in bounds.items()})
    melted['hi'] = melted['variable'].map({k: v[1] for k, v in bounds.items()})
    mask = (melted['value'] >= melted['lo']) & (melted['value'] <= melted['hi'])
    melted_clip = melted[mask].copy()

    # Drop variables that ended up empty post-clipping (prevents blank facets)
    cnts = melted_clip['variable'].value_counts()
    keep_vars = cnts.index.tolist()
    melted_clip = melted_clip[melted_clip['variable'].isin(keep_vars)].copy()

    print("Rows before clipping:", len(melted), "Rows after clipping:", len(melted_clip))
    print("Variables retained:", sorted(set(melted_clip['variable'])))

    # 1) Histograms (counts)
    p_hist_grid = (
        ggplot(melted_clip, aes(x='value'))
        + geom_histogram(aes(y='..count..'), bins=60, fill='skyblue', color='black', alpha=0.7)
        + facet_wrap(facets='variable', scales='free')
        + labs(title='Grid: Histograms (clipped 0.5%-99.5%)', x='Value', y='Count')
        + ggsize(1400, 900)
    )
    ggsave(p_hist_grid, path=output_dir_html, filename='variable_grid_histograms.html')
    print(f"Saved histograms grid to {os.path.join(output_dir_html, 'variable_grid_histograms.html')}")

    # 2) Density
    p_density_grid = (
        ggplot(melted_clip, aes(x='value'))
        + geom_density(aes(y='..density..'), fill='lightcoral', alpha=0.6)
        + facet_wrap(facets='variable', scales='free')
        + labs(title='Grid: Density Plots (clipped 0.5%-99.5%)', x='Value', y='Density')
        + ggsize(1400, 900)
    )
    ggsave(p_density_grid, path=output_dir_html, filename='variable_grid_density.html')
    print(f"Saved density grid to {os.path.join(output_dir_html, 'variable_grid_density.html')}")

    # 3) (Optional) Overlay version: histogram + density in one figure
    p_hist_with_density = (
        ggplot(melted_clip, aes(x='value'))
        + geom_histogram(aes(y='..density..'), bins=60, fill='skyblue', color='black', alpha=0.5)
        + geom_density()
        + facet_wrap(facets='variable', scales='free')
        + labs(title='Grid: Histogram + Density (clipped 0.5%-99.5%)', x='Value', y='Density')
        + ggsize(1400, 900)
    )
    ggsave(p_hist_with_density, path=output_dir_html, filename='variable_grid_histograms_with_density.html')
    print(f"Saved overlay grid to {os.path.join(output_dir_html, 'variable_grid_histograms_with_density.html')}")

    # 4) Signed-log histograms for high-dynamic-range variables
    signed_log_vars = []
    for var in keep_vars:
        col = pd.to_numeric(df[var], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(col) < 10:
            continue
        q10, q90 = np.nanquantile(col, 0.1), np.nanquantile(col, 0.9)
        if (abs(q90) / (abs(q10) + 1e-12) > 1e3) or (col.abs().max() / (col.abs().median() + 1e-12) > 1e3):
            signed_log_vars.append(var)

    if signed_log_vars:
        print("Variables selected for signed-log:", signed_log_vars)
        df_sl = df_sub[[*(c for c in id_cols if c in df_sub.columns), *signed_log_vars]].copy()
        for var in signed_log_vars:
            s = pd.to_numeric(df_sl[var], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df_sl[f'{var}_signed_log'] = np.sign(s) * np.log1p(np.abs(s))
        sl_cols = [f'{v}_signed_log' for v in signed_log_vars]
        melted_sl = df_sl.melt(
            id_vars=[c for c in id_cols if c in df_sl.columns],
            value_vars=sl_cols,
            var_name='variable',
            value_name='value'
        ).dropna(subset=['value'])
        p_sl = (
            ggplot(melted_sl, aes(x='value'))
            + geom_histogram(bins=80, fill='seagreen', color='black')
            + facet_wrap(facets='variable', scales='free')
            + labs(title='Signed-Log Histograms for High-Dynamic-Range Variables',
                   x='signed_log(1+|value|)', y='Count')
            + ggsize(1200, 700)
        )
        ggsave(p_sl, path=output_dir_html, filename='variable_signed_log_histograms.html')
        print(f"Saved signed-log histograms to {os.path.join(output_dir_html, 'variable_signed_log_histograms.html')}")
    else:
        print("No variables required signed-log transform plotting based on heuristic.")

    print("All grid visualisations saved to:", output_dir_html)
