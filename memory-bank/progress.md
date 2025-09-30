# Progress: Russian Bank Accounting EDA

## What works
- Initial project setup and directory structure are in place.
- Core Memory Bank documentation (`projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`) has been created, providing a foundational understanding of the project.
- The `explore_schema.py` script has been created and executed, providing initial insights into the dataset's structure, data types, missing values, and key indicators.
- Sophisticated visualization proposals have been outlined.
- The `preprocess_data.py` script has been created and executed, handling missing values and providing an example of normalization.
- Initial visualizations have been implemented and executed, specifically:
    - Interactive line charts for total assets (`total_assets_time_series.html`, `log_total_assets_time_series.html`).
    - Interactive histograms and density plots for Return on Assets (ROA) (`roa_histogram.html`, `roa_density_plot.html`).
- The `ggsave` syntax for `lets-plot` has been corrected in both visualization scripts and documented in `memory-bank/techContext.md`.
- **Sankey Dashboard Refactoring and Enhancements**:
    - Data processing logic refactored from pandas to DuckDB.
    - Implemented 'power' transformation mode for better visual distinction of flow thicknesses.
    - Fixed SQL query for `regn_sample` aggregation to resolve `Parser Error`.
    - Corrected passing of `power_exponent` parameter from UI to data processing.
    - Ensured correct link colors, alpha values, and tooltip information (unique bank count, sample REGNs).
    - Correctly positioned 'Exit' bucket node at the bottom.
    - Fixed missing year markers on the Sankey diagram.

## What's left to build
- **Further Refinements**: Based on user feedback, fine-tune the 'power' transform exponent or other visual aspects for optimal clarity.
- **Add Unit Tests**: Consider adding automated unit/integration tests for `compute_sankey_for_variables` and related data processing functions.
- **Code Cleanup**: Address any remaining linting or type warnings.

## Current status
The project has completed the initial schema exploration, visualization planning, data preprocessing, and the implementation of the Sankey dashboard with all requested features and fixes. The next phase involves user verification and potential refinements.

## Known issues
- Pylance errors related to `ModuleType` attribute assignments are noted but do not affect runtime functionality.

## Evolution of project decisions
- The decision to use `uv` for environment management was made to ensure consistency and reproducibility.
- The emphasis on `lets-plot` stems from the requirement for interactive and visually appealing dashboards.
- The modular approach for EDA and visualization scripts is intended to improve maintainability and collaboration.
- The detailed visualization proposals guide the implementation phase.
- The refactoring to DuckDB was a significant architectural decision to improve performance and address data aggregation issues.
- The introduction of the 'power' transform mode and REGN sampling in tooltips addresses user feedback for better visual distinction and data insight.
