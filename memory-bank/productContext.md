# Product Context: Russian Bank Accounting EDA

## Why this project exists
The project aims to provide a comprehensive understanding of Russian bank accounting time series data. This is crucial for financial analysts, economists, and regulators to monitor the health and stability of the banking sector, identify potential risks, and inform policy decisions.

## Problems it solves
- **Data Complexity:** Raw financial time series data can be complex and difficult to interpret without proper analysis.
- **Hidden Patterns:** Important trends, seasonality, and cyclical patterns might be obscured within large datasets.
- **Outlier Detection:** Anomalous data points (outliers) can indicate significant events, errors, or fraudulent activities, which need to be identified and investigated.
- **Lack of Visual Insight:** Without effective visualizations, it's challenging to convey complex data insights to stakeholders.
- **Data Normalization:** Raw data often has varying scales and distributions, making direct comparisons and modeling difficult. Normalization addresses this.

## How it should work
The EDA process should involve:
1. **Data Loading and Initial Inspection:** Load the parquet file and get a preliminary understanding of its structure, data types, and basic statistics.
2. **Schema Exploration:** Detailed examination of columns, their meanings, and potential relationships.
3. **Data Preprocessing:** Handling missing values, and applying appropriate normalization techniques.
4. **Feature Engineering (if applicable):** Creating new features from existing ones to enhance analytical capabilities.
5. **Visualization Generation:** Producing interactive and static plots to highlight distributions, trends, correlations, and outliers.
6. **Insight Extraction:** Deriving actionable insights and observations from the visualizations.

## User experience goals
- **Intuitive Visualizations:** Dashboards and plots should be easy to understand, even for non-technical users.
- **Interactivity:** Users should be able to explore data points, filter, and drill down into visualizations.
- **Clarity and Precision:** Visualizations should accurately represent the data without misleading interpretations.
- **Comprehensive Coverage:** The EDA should cover various aspects of the banking indicators, providing a holistic view.
- **Reproducibility:** All analysis and visualization scripts should be well-documented and reproducible.
