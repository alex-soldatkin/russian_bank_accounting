import os
import numpy as np
import pandas as pd
from lets_plot import *

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

    if 'ROA' not in df.columns:
        print("Column 'ROA' not found in dataframe. Available columns:", df.columns.tolist())
    else:
        # Basic diagnostics
        roa = df['ROA'].replace([np.inf, -np.inf], np.nan).dropna()
        print("ROA count:", len(roa))
        print("ROA min/max:", roa.min(), roa.max())
        print("ROA summary:")
        print(roa.describe())

        # Identify reasonable plotting range by percentiles to avoid extreme outliers dominating the visualization
        low_q, high_q = 0.005, 0.995
        p_low, p_high = roa.quantile([low_q, high_q]).values
        print(f"Using percentile clip range: {low_q*100:.2f}th -> {high_q*100:.2f}th = {p_low} .. {p_high}")

        df_clip = df[(df['ROA'] >= p_low) & (df['ROA'] <= p_high)].copy()
        print("Rows after clipping to percentiles:", len(df_clip))

        # 1) Histogram on clipped ROA (removes extreme outliers that distort binning)
        p_roa_clipped = (
            ggplot(df_clip, aes(x='ROA'))
            + geom_histogram(bins=120, fill='skyblue', color='black')
            + labs(title=f'Distribution of ROA (clipped {low_q*100:.1f}%-{high_q*100:.1f}%)', x='ROA', y='Count')
            + xlim(p_low, p_high)
            + ggsize(900, 550)
        )
        ggsave(p_roa_clipped, path=output_dir_html, filename='roa_histogram_clipped.html')
        print(f"Saved clipped histogram to {os.path.join(output_dir_html, 'roa_histogram_clipped.html')}")

        # 2) Density plot on clipped ROA
        p_roa_density_clipped = (
            ggplot(df_clip, aes(x='ROA'))
            + geom_density(fill='lightcoral', alpha=0.6)
            + labs(title=f'ROA Density (clipped {low_q*100:.1f}%-{high_q*100:.1f}%)', x='ROA', y='Density')
            + xlim(p_low, p_high)
            + ggsize(900, 550)
        )
        ggsave(p_roa_density_clipped, path=output_dir_html, filename='roa_density_plot_clipped.html')
        print(f"Saved clipped density plot to {os.path.join(output_dir_html, 'roa_density_plot_clipped.html')}")

        # 3) Signed log-transform to visualise wide dynamic range while preserving sign
        #    sign(x) * log1p(abs(x)) is useful for values that can be negative and span many orders of magnitude
        roa_signed_log = np.sign(df['ROA']) * np.log1p(np.abs(df['ROA'].replace([np.inf, -np.inf], np.nan).fillna(0)))
        df['ROA_signed_log'] = roa_signed_log

        # For plotting, restrict to the central 99% of the transformed values to avoid tiny number of extremes
        ts_low, ts_high = df['ROA_signed_log'].quantile([0.005, 0.995]).values
        df_log_clip = df[(df['ROA_signed_log'] >= ts_low) & (df['ROA_signed_log'] <= ts_high)].copy()
        print("Rows after signed-log clipping:", len(df_log_clip))

        p_roa_signed_log = (
            ggplot(df_log_clip, aes(x='ROA_signed_log'))
            + geom_histogram(bins=120, fill='seagreen', color='black')
            + labs(title='Distribution of Signed-Log(1+|ROA|)', x='signed_log(1+|ROA|)', y='Count')
            + ggsize(900, 550)
        )
        ggsave(p_roa_signed_log, path=output_dir_html, filename='roa_histogram_signed_log.html')
        print(f"Saved signed-log histogram to {os.path.join(output_dir_html, 'roa_histogram_signed_log.html')}")

        # 4) (Optional) Full-range histogram with very wide bins for transparency (keeps original data but coarse)
        # Use broader binning so the plot remains legible even in presence of extremes
        p_roa_full = (
            ggplot(df, aes(x='ROA'))
            + geom_histogram(bins=80, fill='lightgrey', color='black')
            + labs(title='ROA Histogram (full range, coarse bins)', x='ROA', y='Count')
            + ggsize(900, 550)
        )
        ggsave(p_roa_full, path=output_dir_html, filename='roa_histogram_full_range.html')
        print(f"Saved full-range histogram to {os.path.join(output_dir_html, 'roa_histogram_full_range.html')}")

        print("All ROA visualizations generated and saved in", output_dir_html)
