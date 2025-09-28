# System Patterns: Russian Bank Accounting EDA

## System architecture
The project follows a modular and script-based architecture for EDA and visualization.

- **Data Layer:** The raw data resides in `data/HOORRAAH_final_banking_indicators_imputed_new.parquet`.
- **EDA Layer:** Python scripts in `./EDA_scripts` will handle data loading, initial inspection, schema exploration, and preprocessing steps. These scripts will output insights and potentially cleaned/transformed data for visualization.
- **Visualization Layer:** Python scripts in `./visualisations` will generate interactive and static plots using `lets-plot` and potentially other libraries. These scripts will be independent and focus on specific visualizations.
- **Output Layer:** Visualizations will be saved as HTML (for interactive), PNG, and SVG files in `visualisations/html/`, `visualisations/png/`, and `visualisations/svg/` respectively.

## Key technical decisions
- **Python for Analysis:** Python is chosen for its robust data science ecosystem (Pandas, scikit-learn, lets-plot).
- **Parquet for Data Storage:** Efficient columnar storage format suitable for large datasets and analytical queries.
- **`lets-plot` for Visualization:** Prioritizing interactive and visually appealing dashboards. This aligns with the ggplot philosophy for declarative plotting.
- **`uv` for Environment Management:** Ensures reproducible and isolated development environment. The command `uv run script_name.py` will be used to execute Python scripts.
- **Modular Scripting:** Each EDA step or visualization will be encapsulated in its own Python script for clarity, reusability, and easier management.

## Design patterns in use
- **Modular Design:** Separation of concerns between data loading, EDA, and visualization.
- **Script-based Workflow:** Encourages reproducible research and analysis.
- **Configuration over Code (Implicit):** Visualization scripts will likely take configuration parameters (e.g., column names, normalization methods) rather than hardcoding values directly.

## Component relationships
- `data/HOORRAAH_final_banking_indicators_imputed_new.parquet` is the primary input for all EDA and visualization scripts.
- `./EDA_scripts` produce insights and potentially intermediate data that inform `./visualisations`.
- `./visualisations` consume data (raw or preprocessed by EDA scripts) and generate output files in `visualisations/html/`, `visualisations/png/`, `visualisations/svg/`.

## Critical implementation paths
1. **Data Loading and Schema Exploration:** This is the foundational step. Any issues here will impact all subsequent analysis.
2. **Normalization Strategy:** Selecting and applying appropriate normalization methods is critical for meaningful comparisons and outlier detection.
3. **`lets-plot` Integration:** Ensuring smooth integration and effective utilization of `lets-plot` for interactive outputs.
