# Active Context: Russian Bank Accounting EDA

## Current work focus
The current focus is on initiating the schema exploration of the bank accounting time series data and planning for sophisticated visualizations. Initial visualizations have been implemented and their saving mechanism corrected.

## Recent changes
- Initial `projectbrief.md` has been created, outlining the core requirements and goals.
- `productContext.md` has been created, detailing the purpose, problems solved, workflow, and user experience goals.
- `systemPatterns.md` has been created, detailing the system architecture, key technical decisions, design patterns and component relationships.
- `techContext.md` has been created, detailing technologies used, development setup, technical constraints, dependencies and tool usage patterns. The correct `ggsave` syntax for `lets-plot` has been added to `techContext.md`.
- `progress.md` has been created, detailing what works, what's left to build, current status and known issues.
- The `explore_schema.py` script has been created and executed, providing initial insights into the dataset's structure, data types, missing values, and key indicators.
- The `preprocess_data.py` script has been created and executed, handling missing values and providing an example of normalization.
- The `total_assets_time_series.py` script has been created, executed, and corrected to use the proper `ggsave` syntax, generating interactive line charts for total assets over time.
- The `roa_histogram.py` script has been created, executed, and corrected to use the proper `ggsave` syntax, generating interactive histograms and density plots for ROA.

## Next steps
1. Continue implementing the remaining proposed visualizations using `lets-plot`.
2. Refine data preprocessing and normalization strategies as needed for specific visualizations.
3. Implement outlier detection and highlighting within visualizations.

## Active decisions and considerations
- **Environment Management:** Always ensure working within the `.venv` by activating it with `source .venv/bin/activate`. Verify activation by checking for `(.venv)` in the command line prompt.
- **Dependency Management:** Use `uv sync` to install dependencies from `pyproject.toml` and `uv add <package_name>` to install new packages and add them to `pyproject.toml`.
- **Script Execution:** Use `uv run script_name.py` for executing Python scripts within the project's virtual environment.
- Prioritizing `lets-plot` for interactive visualizations.
- Normalization methods (log transform, symlog, minmax scaler) will be evaluated during EDA.
- Visualizations will be saved as separate Python scripts in `./visualisations`.

## Important patterns and preferences
- **Strict Environment Adherence:** All Python-related tasks *must* be performed within the activated `.venv`.
- Modular code structure: Python scripts for EDA and visualizations should be self-contained and focused on a single task.
- Reproducibility: All steps should be clear and easily reproducible.
- Documentation: Extensive comments within code and clear explanations in Memory Bank files.

## Learnings and project insights
- The project requires a strong emphasis on data visualization to convey complex financial insights.
- The time series nature of the data suggests the need for specific time series analysis techniques in addition to general EDA.
- The dataset contains missing values that need to be addressed during preprocessing.
- The `form` column needs further investigation and potential cleaning.
- Correct `ggsave` syntax for `lets-plot` is `ggsave(plot, path=output_dir, filename='plot_name.html')`.

## Proposed Sophisticated Visualizations:

**I. Time Series Analysis (Trends and Seasonality)**

1.  **Interactive Line Charts for Key Financial Indicators over Time (Overall and Per Bank):**
    *   **Indicators:** `total_assets`, `total_equity`, `total_loans`, `total_deposits`, `net_income_amount`.
    *   **Features:** `lets-plot` interactivity for details on hover, faceting by `REGN` for individual bank comparisons, optional rolling averages, and normalization (e.g., log-transform) to highlight relative changes.

**II. Distribution Analysis (Understanding Variable Characteristics)**

2.  **Interactive Histograms/Density Plots with Normalization Overlays:**
    *   **Indicators:** All `float64` columns, with focus on `ROA`, `ROE`, `NIM`, `npl_ratio`, and log-transformed `total_assets`.
    *   **Features:** Dynamic binning, tooltips, overlay of theoretical distribution curves (e.g., normal), and faceting by `form` (after cleaning) or `imputation_flag` to compare distributions.
3.  **Interactive Box Plots / Violin Plots for Ratios and Financial Health Metrics:**
    *   **Indicators:** `ROA`, `ROE`, `NIM`, `npl_ratio`, `coverage_ratio`, `loan_to_deposit_ratio`.
    *   **Features:** Hover-over details for quartiles/median/outliers, grouping by `DT` (annual/quarterly) to show temporal distribution changes, and explicit highlighting of outliers with `REGN` and `DT` details.

**III. Outlier Detection and Relationships**

4.  **Interactive Scatter Plots with Outlier Highlighting:**
    *   **Indicators:** Pairs like `npl_ratio` vs. `coverage_ratio`, `ROA` vs. `ROE`, `total_assets` vs. `total_loans`.
    *   **Features:** `lets-plot` tooltips for `REGN`, `DT`, and values; color-coding by time or `form`; size/transparency for additional variables; integration with simple outlier detection algorithms (e.g., Isolation Forest) to highlight anomalies.
5.  **Interactive Correlation Matrix Heatmap:**
    *   **Indicators:** All `float64` financial indicators.
    *   **Features:** Hover to show exact correlation values, clustering to group correlated variables, and filtering by time period to observe dynamic correlation changes.

**IV. Specific to Imputation Flags**

6.  **Stacked Bar Chart of Imputation Status over Time:**
    *   **Indicators:** Count of 'observed' vs. 'imputed' for key flags (e.g., `imputation_flag__total_assets`).
    *   **Features:** `lets-plot` interactivity for counts and percentages, and faceting for comparison across different imputation flags.
