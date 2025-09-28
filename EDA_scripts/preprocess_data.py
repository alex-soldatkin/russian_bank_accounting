import pandas as pd
import numpy as np
import os

# Define the path to the Parquet file
data_path = 'data/HOORRAAH_final_banking_indicators_imputed_new.parquet'

# Define the path for the preprocessed data
preprocessed_data_path = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'

# Check if the file exists
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
else:
    print(f"Loading data from {data_path} for preprocessing...")
    df = pd.read_parquet(data_path)

    # --- Preprocessing Steps ---

    # 1. Handle 'form' column: Fill missing values with 'unknown'
    #    The 'explore_schema.py' script showed some missing values in 'form'.
    #    Since it's an object type and only one unique value was initially observed,
    #    filling with 'unknown' is a safe initial approach.
    df['form'] = df['form'].fillna('unknown')
    print(f"\nMissing values in 'form' after filling: {df['form'].isnull().sum()}")

    # 2. Handle missing numerical values: Impute with median
    #    The 'explore_schema.py' script identified several float64 columns with missing values.
    #    Imputing with the median is a robust strategy for skewed financial data.
    numerical_cols_with_missing = [
        'interest_income', 'operating_income', 'interest_expense',
        'operating_expense', 'net_interest_income', 'net_income_amount',
        'ROA', 'ROE', 'NIM', 'cost_to_income_ratio', 'npl_ratio',
        'llp_to_loans_ratio', 'coverage_ratio', 'loan_to_deposit_ratio',
        'liquid_assets_to_total_assets', 'state_equity_pct'
    ]

    for col in numerical_cols_with_missing:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Missing values in '{col}' after median imputation: {df[col].isnull().sum()}")

    # 3. Basic Normalization Example (Log Transform for highly skewed data)
    #    This is an example for demonstration. Actual normalization will be decided during visualization.
    #    Applying log transform to total_assets as it's likely to be highly skewed.
    #    Add a small constant to avoid log(0) if zeros are present.
    if (df['total_assets'] <= 0).any():
        print("Warning: 'total_assets' contains non-positive values. Adding a small constant before log transform.")
        df['log_total_assets'] = np.log1p(df['total_assets']) # log(1+x)
    else:
        df['log_total_assets'] = np.log(df['total_assets'])

    print("\n--- Preprocessed DataFrame Info (sample of new column) ---")
    print(df[['total_assets', 'log_total_assets']].head())

    # Save the preprocessed DataFrame
    df.to_parquet(preprocessed_data_path, index=False)
    print(f"\nPreprocessed data saved to {preprocessed_data_path}")

    print("\n--- Verification of Preprocessed Data ---")
    df_preprocessed = pd.read_parquet(preprocessed_data_path)
    print(f"Shape of preprocessed data: {df_preprocessed.shape}")
    print(f"Missing values in preprocessed data:\n{df_preprocessed[numerical_cols_with_missing].isnull().sum()}")
