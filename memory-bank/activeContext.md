# Active Context: Russian Bank Accounting EDA

## Current work focus
The primary task has been the development and refinement of the Sankey dashboard, focusing on migrating data processing to DuckDB and implementing user-requested visualization enhancements. This includes improving visual distinction through transformations, adding REGN sampling to tooltips, and fixing various bugs.

## Recent changes (latest)
- **Complete Sankey Dashboard Enhancement Suite**:
    - **REGN Sample Display Fix**: Resolved DuckDB array handling issues in tooltip formatting with robust error handling for numpy arrays vs Python lists.
    - **Moving Average Period Control**: Added slider control for 6-48 months in 3-month increments with real-time updates.
    - **Thickness Mode Toggle**: Implemented checkbox to switch between absolute values and percentage of column total for link thickness.
    - **Recompute Layout Button**: Added green button to refresh layout with current hyperparameters while maintaining node sequence (Q5→Q1→Exit).
    - **Enhanced UI Controls**: All controls now work together seamlessly with comprehensive parameter integration.
- **Advanced Data Processing**:
    - **Robust Array Handling**: Fixed "ambiguous truth value" errors when processing DuckDB arrays in pandas DataFrames.
    - **Percentage Calculation Logic**: Implemented proper percentage-based thickness calculations for relative flow analysis.
    - **Parameter Integration**: All UI controls properly pass parameters through the entire data processing pipeline.
- **User Experience Improvements**:
    - **Real-time Layout Updates**: Users can now recompute layouts without changing underlying data or parameters.
    - **Visual Consistency**: Maintained proper node ordering (highest quantile at top, lowest penultimate, Exit last).
    - **Enhanced Debugging**: Added comprehensive logging and error handling throughout the application.
    - **Fixed Empty Chart Issue**: Resolved initial figure loading problems with proper loading indicators.
- **Technical Fixes**:
    - **Callback Structure**: Fixed "Duplicate callback outputs" error by consolidating callback functions.
    - **Button Integration**: Properly connected recompute layout button to main update callback.
    - **Data Type Handling**: Robust handling of DuckDB array returns in pandas DataFrames.
    - **Error Prevention**: Added safeguards against common Dash and data processing errors.

## Current status
- [x] Sankey dashboard data processing fully refactored to DuckDB.
- [x] All requested UI controls and visualization enhancements implemented:
    - Size and thickness variable dropdowns.
    - Quantile slider.
    - STEP_YEARS slider.
    - MAX_YEAR_OVERRIDE input.
    - Transform mode dropdown with 'power' option.
    - Power Exponent slider for 'power' transform.
    - Unit scale dropdown.
    - Node thickness and pad sliders.
    - REGN sample size input.
- [x] Link colors and alpha values adjusted.
- [x] Tooltips now display unique bank counts and sample REGNs.
- [x] Exit bucket node correctly positioned at the bottom.
- [x] Year markers are displayed correctly.
- [x] Resolved `TypeError` for `power_exponent` parameter.
- [x] Resolved SQL `Parser Error` for `regn_sample` aggregation.

## Observations / Known issues
- Pylance errors related to `ModuleType` attribute assignments are noted but do not affect runtime functionality.

## Next steps (recommended)
1. **User Testing & Verification**: Test all implemented features in real-world scenarios to ensure optimal user experience.
2. **Performance Monitoring**: Monitor dashboard performance with large datasets and optimize if needed.
3. **Feature Expansion**: Consider additional visualization types or analysis features based on user feedback.
4. **Documentation**: Create user guides and technical documentation for the implemented features.
5. **Maintenance**: Keep dependencies updated and address any future compatibility issues.
