import pandas as pd
from lets_plot import *
import os

LetsPlot.setup_html()

# Define the path to the preprocessed Parquet file
preprocessed_data_path = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
output_dir_html = 'visualisations/html'

# Ensure output directory exists
os.makedirs(output_dir_html, exist_ok=True)

# Check if the file exists
if not os.path.exists(preprocessed_data_path):
    print(f"Error: Preprocessed data file not found at {preprocessed_data_path}")
else:
    print(f"Loading preprocessed data from {preprocessed_data_path}...")
    df = pd.read_parquet(preprocessed_data_path)

    # Convert 'DT' to datetime objects to ensure proper time series plotting
    df['DT'] = pd.to_datetime(df['DT'])

    # Aggregate data to show overall trend (e.g., sum of total_assets per month)
    # This might be too simplistic given the REGN column, but for an initial overall view, it works.
    # For individual bank trends, we'd facet by REGN.
    df_agg = df.groupby('DT')[['total_assets', 'log_total_assets']].sum().reset_index()

    # Create an interactive line chart for total_assets over time
    p_assets = (
        ggplot(df_agg, aes(x='DT', y='total_assets'))
        + geom_line(color='blue')
        + labs(title='Total Assets Over Time (Aggregated)', x='Date', y='Total Assets')
        + ggsize(1000, 500)
    )
    # Save as HTML
    ggsave(p_assets, path=output_dir_html, filename='total_assets_time_series.html')
    print(f"Interactive plot for total_assets saved to {os.path.join(output_dir_html, 'total_assets_time_series.html')}")

    # Create an interactive line chart for log_total_assets over time
    p_log_assets = (
        ggplot(df_agg, aes(x='DT', y='log_total_assets'))
        + geom_line(color='green')
        + labs(title='Log of Total Assets Over Time (Aggregated)', x='Date', y='Log(Total Assets)')
        + ggsize(1000, 500)
    )
    # Save as HTML
    ggsave(p_log_assets, path=output_dir_html, filename='log_total_assets_time_series.html')
    print(f"Interactive plot for log_total_assets saved to {os.path.join(output_dir_html, 'log_total_assets_time_series.html')}")

    print("\n--- Next: Consider plotting individual banks using faceting or specific REGN values. ---")
