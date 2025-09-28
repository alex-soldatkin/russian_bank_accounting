import pandas as pd
import pyarrow.parquet as pq
import os

# Define the path to the Parquet file
data_path = 'data/HOORRAAH_final_banking_indicators_imputed_new.parquet'

# Check if the file exists
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
else:
    print(f"Loading data from {data_path}...")
    # Load the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(data_path)

    print("\n--- DataFrame Info ---")
    df.info()

    print("\n--- DataFrame Head ---")
    print(df.head())

    print("\n--- DataFrame Description ---")
    print(df.describe())

    print("\n--- Number of unique values per column ---")
    print(df.nunique())

    print("\n--- Missing values per column ---")
    print(df.isnull().sum())
