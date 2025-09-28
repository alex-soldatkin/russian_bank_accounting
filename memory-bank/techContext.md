# Technical Context: Russian Bank Accounting EDA

## Technologies used
- **Python 3.x:** Primary programming language for data analysis and visualization.
- **`uv`:** Environment and package manager for Python, used for dependency management and script execution.
- **Pandas:** For data manipulation and analysis, especially with DataFrames.
- **NumPy:** For numerical operations.
- **`lets-plot`:** A Python plotting library based on the Grammar of Graphics, used for creating interactive and static visualizations.
- **Scikit-learn:** For data preprocessing, specifically for normalization techniques (e.g., `MinMaxScaler`, `StandardScaler`) and potentially outlier detection algorithms.
- **Parquet:** File format for storing the input data (`HOORRAAH_final_banking_indicators_imputed_new.parquet`).

## Development setup
- **Operating System:** macOS Sonoma (user's environment).
- **IDE:** VSCode (user's environment).
- **Shell:** /bin/bash (user's environment).
- **Virtual Environment:** Managed by `uv`.
- **Project Structure:**
    - `data/`: Contains raw data files.
    - `EDA_scripts/`: Python scripts for schema exploration and initial data analysis.
    - `visualisations/`: Python scripts for generating visualizations.
    - `visualisations/html/`: Output directory for interactive HTML plots.
    - `visualisations/png/`: Output directory for static PNG plots.
    - `visualisations/svg/`: Output directory for static SVG plots.
    - `memory-bank/`: Documentation files for project context.

## Technical constraints
- **Data Source:** Limited to `data/HOORRAAH_final_banking_indicators_imputed_new.parquet`. No other data sources are currently considered.
- **Visualization Library:** Primary focus on `lets-plot`. While other libraries like Matplotlib or Seaborn could be used for specific cases, `lets-plot` is prioritized for interactive dashboards.
- Documentation for `lets-plot`: [Let's Plot Python API](https://lets-plot.org/python/pages/api.html)
- Pay careful attention to the linter warnings regarding lets-plot usage as it might be implemented differently than typical Python plotting libraries or ggplot in R.
- **Environment Management:** `uv` is the mandated tool for running Python scripts.

## Dependencies
The `pyproject.toml` file will manage Python dependencies. Key dependencies will include:
- `pandas`
- `numpy`
- `lets-plot`
- `scikit-learn`
- `pyarrow` (for Parquet file handling)

## Tool usage patterns
- **Environment Activation:** To ensure working within the correct virtual environment, use `source .venv/bin/activate`. Verify activation by checking for `(.venv)` in the command line prompt.
- **`uv run script_name.py`:** Standard command for executing Python scripts within the activated virtual environment.
- **`uv sync`:** To install additional dependencies specified in `pyproject.toml`.
- **`uv add <package_name>`:** As a replacement for `pip install`, this command installs the dependency and adds it to `pyproject.toml`.
- **`read_file`:** To examine script contents and data insights.
- **`write_to_file`:** To create new scripts or update documentation.
- **`list_files`:** To navigate the project structure.
- **`ggsave(plot, path=output_dir, filename='plot_name.html')`:** Correct syntax for saving `lets-plot` visualizations as HTML.
