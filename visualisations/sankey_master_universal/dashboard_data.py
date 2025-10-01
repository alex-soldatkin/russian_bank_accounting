"""
Data preparation and Sankey building utilities for the Sankey dashboard.

This module was extracted from the single-file dashboard to separate data/logic
from the visualization (Dash) code.

This version is refactored to use DuckDB for heavy processing.
"""
from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import duckdb
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# Configuration (edit if needed)
# ------------------------------
DATA_PATH = os.path.join('data', 'HOORRAAH_final_banking_indicators_preprocessed.parquet')

# Sankey visual defaults
UNIT_SCALE = 1e9
UNIT_LABEL = 'bln'
MA_MONTHS = 24
STEP_YEARS = 3
MAX_YEAR_OVERRIDE = 2021
TRANSFORM_MODE = 'signed_log'
NODE_PAD = 22
NODE_THICK = 18
REGN_SAMPLE_SIZE = 5 # Default sample size for REGN in tooltips
POWER_EXPONENT = 0.5 # Default power exponent for 'power' transform mode

# Ownership buckets
OWN_BUCKETS = [
    (0.0, 0.0,  'State 0%'),
    (0.0, 10.0, 'State 0–10%'),
    (10.0,20.0, 'State 10–20%'),
    (20.0,50.0, 'State 20–50%'),
    (50.0,100.1,'State ≥50%'),
]
UNKNOWN_LABEL = 'Unknown'
EXIT_LABEL    = 'Exit'

NODE_COLORS  = {
    'State 0%':        'rgba(33,150,243,0.90)',
    'State 0–10%':     'rgba(63,81,181,0.90)',
    'State 10–20%':    'rgba(0,150,136,0.90)',
    'State 20–50%':    'rgba(255,193,7,0.90)',
    'State ≥50%':      'rgba(244,67,54,0.90)',
    UNKNOWN_LABEL:     'rgba(158,158,158,0.80)',
    EXIT_LABEL:        'rgba(244,67,54,0.90)',  # Red for exit nodes
}

# Diverging color palette for quantiles (Q1 worst -> Q5 best)
QUANTILE_COLORS = {
    'Q1': 'rgba(156,66,33,0.90)',    # Brown - worst performance (not red, reserved for exit)
    'Q2': 'rgba(255,152,0,0.90)',    # Orange
    'Q3': 'rgba(255,235,59,0.90)',   # Yellow
    'Q4': 'rgba(139,195,74,0.90)',   # Light green
    'Q5': 'rgba(76,175,80,0.90)',    # Green - best performance
}

DEFAULT_NUMERIC_COLUMNS = [
    'total_assets', 'total_loans', 'total_deposits', 'total_equity',
    'net_income_amount', 'state_equity_pct',
]

def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Returns a duckdb connection."""
    return duckdb.connect(database=':memory:', read_only=False)

def build_year_grid(min_year: int, max_year: int, step: int) -> List[int]:
    years = list(range(min_year, max_year + 1, step))
    if years and years[-1] != max_year: years.append(max_year)
    return years

def get_sankey_data_from_duckdb(con: duckdb.DuckDBPyConnection, size_var: str, thick_var: str, years: List[int], n_q: int, ma: int, win: float, t_mode: str, regn_sample_size: int) -> pd.DataFrame:
    """
    Executes the main SQL query to generate Sankey link data.
    """
    # Calculate min_periods for moving average (equivalent to max(1, ma//3) in pandas)
    min_periods = max(1, ma // 3)

    # Debug: Print the years being used
    print(f"DEBUG: Years being processed: {years}")
    print(f"DEBUG: STEP_YEARS: {STEP_YEARS}")
    print(f"DEBUG: MAX_YEAR_OVERRIDE: {MAX_YEAR_OVERRIDE}")

    # Construct the CASE statement for bucketing
    if size_var == 'state_equity_pct':
        bucket_case_stmt = f"""
            CASE
                WHEN state_equity_pct IS NULL THEN '{UNKNOWN_LABEL}'
                WHEN state_equity_pct = 0 THEN 'State 0%'
                WHEN state_equity_pct > 0 AND state_equity_pct <= 10 THEN 'State 0–10%'
                WHEN state_equity_pct > 10 AND state_equity_pct <= 20 THEN 'State 10–20%'
                WHEN state_equity_pct > 20 AND state_equity_pct <= 50 THEN 'State 20–50%'
                WHEN state_equity_pct > 50 THEN 'State ≥50%'
                ELSE '{UNKNOWN_LABEL}'
            END
        """
    else:
        # For quantile bucketing, we use ntile
        bucket_case_stmt = f"'Q' || ntile({n_q}) OVER (PARTITION BY year ORDER BY size_ma)"

    sql = f"""
        WITH InputData AS (
            SELECT
                "DT",
                "REGN",
                COALESCE("{size_var}", 0) AS size_raw,
                COALESCE("{thick_var}", 0) AS thick_raw,
                "state_equity_pct"
            FROM '{DATA_PATH}'
        ),
        StandardizedData AS (
            SELECT
                "DT",
                "REGN",
                size_raw,
                thick_raw,
                -- Standardize state_equity_pct similar to v6 logic
                CASE
                    WHEN "state_equity_pct" IS NOT NULL AND ABS("state_equity_pct") < 1e-12 THEN 0.0
                    WHEN "state_equity_pct" IS NOT NULL THEN
                        CASE
                            WHEN (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS("state_equity_pct"))
                                  FROM InputData WHERE "state_equity_pct" IS NOT NULL) <= 1.5
                            THEN "state_equity_pct" * 100.0
                            ELSE "state_equity_pct"
                        END
                    ELSE NULL
                END AS state_equity_pct
            FROM InputData
        ),
        MovingAverages AS (
            SELECT
                "DT",
                "REGN",
                year("DT") as year,
                -- Use conditional aggregation for min_periods equivalent
                CASE
                    WHEN COUNT(*) OVER (PARTITION BY "REGN" ORDER BY "DT" ROWS BETWEEN {ma - 1} PRECEDING AND CURRENT ROW) >= {min_periods}
                    THEN AVG(size_raw) OVER (PARTITION BY "REGN" ORDER BY "DT" ROWS BETWEEN {ma - 1} PRECEDING AND CURRENT ROW)
                    ELSE NULL
                END AS size_ma,
                CASE
                    WHEN COUNT(*) OVER (PARTITION BY "REGN" ORDER BY "DT" ROWS BETWEEN {ma - 1} PRECEDING AND CURRENT ROW) >= {min_periods}
                    THEN AVG(thick_raw) OVER (PARTITION BY "REGN" ORDER BY "DT" ROWS BETWEEN {ma - 1} PRECEDING AND CURRENT ROW)
                    ELSE NULL
                END AS thick_ma,
                state_equity_pct
            FROM StandardizedData
        ),
        ForwardFilledData AS (
            SELECT
                "DT",
                "REGN",
                year,
                -- Forward fill state_equity_pct within each year for each REGN
                LAST_VALUE(state_equity_pct) OVER (
                    PARTITION BY "REGN", year
                    ORDER BY "DT"
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS state_equity_pct_ffill,
                size_ma,
                thick_ma
            FROM MovingAverages
        ),
        YearEndSnapshots AS (
            SELECT * FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY "REGN", year ORDER BY "DT" DESC) as rn
                FROM ForwardFilledData
            ) q
            WHERE rn = 1
        ),
        FinalSnapshots AS (
            SELECT
                "REGN",
                year,
                state_equity_pct_ffill as state_equity_pct,
                -- Use MA if available, otherwise use raw (but raw not available here, so we'll use size_ma)
                COALESCE(size_ma, 0) as size_ma,
                COALESCE(thick_ma, 0) as thick_ma
            FROM YearEndSnapshots
        ),
        BucketedSnapshots AS (
            SELECT
                year,
                "REGN",
                thick_ma,
                {bucket_case_stmt} AS bucket_var
            FROM FinalSnapshots
            WHERE year IN ({','.join(map(str, years))})
        ),
        FlowsWithRegn AS (
            -- Active flows with REGN
            SELECT
                s_from.year AS level_from,
                s_to.year AS level_to,
                s_from.bucket_var AS cat_from,
                s_to.bucket_var AS cat_to,
                s_from."REGN" AS regn_id,
                s_from.thick_ma AS w_raw_from, -- Thick_ma of the bank in the 'from' year
                'active' as flow_type
            FROM BucketedSnapshots s_from
            JOIN BucketedSnapshots s_to ON s_from."REGN" = s_to."REGN" AND s_to.year = s_from.year + {STEP_YEARS}
            WHERE s_from.year IN ({','.join(map(str, years[:-1]))})
              AND s_to.year IN ({','.join(map(str, years))})

            UNION ALL

            -- Exit flows with REGN - target NEXT year Exit node (as requested)
            SELECT
                s_from.year AS level_from,
                s_from.year + {STEP_YEARS} AS level_to,
                s_from.bucket_var AS cat_from,
                '{EXIT_LABEL}' AS cat_to,
                s_from."REGN" AS regn_id,
                s_from.thick_ma AS w_raw_from, -- Thick_ma of the bank in the 'from' year
                'exit' as flow_type
            FROM BucketedSnapshots s_from
            WHERE s_from.year IN ({','.join(map(str, years[:-1]))})
              AND NOT EXISTS (
                SELECT 1 FROM FinalSnapshots s_to
                WHERE s_to."REGN" = s_from."REGN" AND s_to.year = s_from.year + {STEP_YEARS}
              )
        ),
        SampledRegns AS (
            SELECT
                level_from,
                level_to,
                cat_from,
                cat_to,
                flow_type,
                LIST(regn_id::VARCHAR ORDER BY random()) AS regn_sample_list
            FROM FlowsWithRegn
            GROUP BY level_from, level_to, cat_from, cat_to, flow_type
        )
        SELECT
            FWR.level_from,
            FWR.level_to,
            FWR.cat_from,
            FWR.cat_to,
            sum(FWR.w_raw_from) as raw_sum,
            count(DISTINCT FWR.regn_id) as count,
            FWR.flow_type,
            (
                CASE WHEN {regn_sample_size} > 0 THEN
                    array_slice(SR.regn_sample_list, 1, {regn_sample_size})
                ELSE NULL END
            ) AS regn_sample
        FROM FlowsWithRegn FWR
        JOIN SampledRegns SR ON FWR.level_from = SR.level_from
                            AND FWR.level_to = SR.level_to
                            AND FWR.cat_from = SR.cat_from
                            AND FWR.cat_to = SR.cat_to
                            AND FWR.flow_type = SR.flow_type
        GROUP BY 1, 2, 3, 4, 7, SR.regn_sample_list
    """
    return con.execute(sql).fetchdf()


def build_sankey_from_df(links: pd.DataFrame, years: List[int], all_cats: List[str], unit_scale: float, t_mode: str, power_exponent: float, thickness_mode: str = 'absolute') -> Dict:
    if EXIT_LABEL not in all_cats:
        all_cats.append(EXIT_LABEL)

    # Apply transformation to raw_sum to get weight
    s_thick = pd.to_numeric(links['raw_sum'], errors='coerce').fillna(0)
    if t_mode == 'log1p':
        links['weight'] = np.log1p(np.abs(s_thick))
    elif t_mode == 'minmax':
        # Ensure it's a numpy array before reshaping
        s_thick_np = s_thick.to_numpy() if isinstance(s_thick, pd.Series) else s_thick
        links['weight'] = MinMaxScaler().fit_transform(s_thick_np.reshape(-1, 1)).flatten()
    elif t_mode == 'signed_log':
        links['weight'] = np.sign(s_thick) * np.log1p(np.abs(s_thick))
    elif t_mode == 'power':
        links['weight'] = np.sign(s_thick) * np.power(np.abs(s_thick), power_exponent)
    else: # 'raw'
        links['weight'] = s_thick

    xpos = {y: 0.08 + i * (0.88 / max(len(years) - 1, 1)) for i, y in enumerate(years)}
    
    n_id, lbls, n_cust, n_cols, xs, ys = {}, [], [], [], [], []
    for y in years:
        for c in all_cats:
            nid = len(lbls)
            n_id[(y, c)] = nid
            lbls.append(c); n_cust.append([y, c]); xs.append(xpos[y]); ys.append(0.5)
            # Use quantile colors for Q1-Q5, fallback to NODE_COLORS for other categories
            if c.startswith('Q') and c in QUANTILE_COLORS:
                n_cols.append(QUANTILE_COLORS[c])
            else:
                n_cols.append(NODE_COLORS.get(c, 'rgba(158,158,158,0.80)'))  # Gray fallback

    # Custom sort for categories: Qx in reverse, then other categories, then EXIT_LABEL at the very end
    def custom_cat_sort(cat):
        if cat.startswith('Q'):
            return (0, -int(cat[1:])) # Sort Q5 before Q1
        elif cat == EXIT_LABEL:
            return (2, 0) # EXIT_LABEL always last
        else:
            return (1, cat) # Other categories in between

    sorted_cats = sorted(all_cats, key=custom_cat_sort)

    # Ensure better visibility for all quantiles, especially Q1 in last year
    # Use more generous margins and ensure all nodes are within visible bounds
    band_h = (1 - 0.08 - 0.02 * (len(sorted_cats) - 1)) / max(1, len(sorted_cats))
    cat_y = {c: 0.04 + i * (band_h + 0.02) + band_h/2 for i, c in enumerate(sorted_cats)}

    # Ensure Q1 is always visible by setting a minimum Y position
    if 'Q1' in cat_y:
        cat_y['Q1'] = max(cat_y['Q1'], 0.05)
    for y in years:
        for c in sorted_cats: # Iterate over sorted categories
            if (y,c) in n_id:
                ys[n_id[(y, c)]] = cat_y.get(c, 0.5)

    step_tot = links.groupby(['level_from'])['weight'].transform('sum')
    links['value_abs'] = links['weight']
    links['value_share'] = np.where(step_tot > 0, 100 * links['weight'] / step_tot, 0)
    links['raw_abs'] = links['raw_sum'] / unit_scale

    # Apply thickness mode: use percentages of column total if requested
    if thickness_mode == 'percentage':
        # For percentage mode, we want to show the relative contribution within each time period
        # We'll use the existing value_share calculation but scale it appropriately for visualization
        # The value_share already represents the percentage contribution to the total flow from that period
        links['weight'] = links['value_share']
        # Update the absolute values to reflect percentages
        links['value_abs'] = links['value_share']
        # For percentage mode, we don't need to scale by unit_scale since we're showing percentages
        links['raw_abs'] = links['value_share']
    
    # Format REGN sample for tooltip - handle DuckDB array conversion
    def format_regn_sample(regn_sample):
        # Check for None first to avoid array comparison issues
        if regn_sample is None:
            return ''

        # Handle numpy arrays (from DuckDB) - check if empty
        if hasattr(regn_sample, '__len__') and hasattr(regn_sample, 'shape'):
            # This is likely a numpy array
            try:
                if len(regn_sample) == 0:
                    return ''
                return ', '.join(map(str, regn_sample))
            except:
                return str(regn_sample)

        # Handle regular Python lists/tuples
        if isinstance(regn_sample, (list, tuple)):
            if len(regn_sample) == 0:
                return ''
            return ', '.join(map(str, regn_sample))

        # Handle strings
        if isinstance(regn_sample, str):
            if regn_sample == '' or regn_sample.strip() == '':
                return ''
            # If it's a string representation of an array, try to parse it
            try:
                # Remove any brackets and split by comma
                cleaned = str(regn_sample).strip('[]').strip()
                if cleaned:
                    return ', '.join(cleaned.split(','))
                return ''
            except:
                return str(regn_sample)

        # Fallback for any other type
        return str(regn_sample)

    links['regn_sample_formatted'] = links['regn_sample'].apply(format_regn_sample)
    
    return {
        "labels": lbls, "node_customdata": n_cust, "node_x": xs, "node_y": ys, "node_color": n_cols,
        "link_source": [n_id.get((r.level_from, r.cat_from)) for r in links.itertuples() if n_id.get((r.level_from, r.cat_from)) is not None and n_id.get((r.level_to, r.cat_to)) is not None],
        "link_target": [n_id.get((r.level_to, r.cat_to)) for r in links.itertuples() if n_id.get((r.level_from, r.cat_from)) is not None and n_id.get((r.level_to, r.cat_to)) is not None],
        "values_abs": [r.value_abs for r in links.itertuples() if n_id.get((r.level_from, r.cat_from)) is not None and n_id.get((r.level_to, r.cat_to)) is not None],
        "link_color": ['rgba(244,67,54,0.55)' if r.cat_to == EXIT_LABEL else 'rgba(76,175,80,0.50)' for r in links.itertuples() if n_id.get((r.level_from, r.cat_from)) is not None and n_id.get((r.level_to, r.cat_to)) is not None], # Always red when destination is Exit
        "link_customdata": np.stack([links['value_share'], links['raw_abs'].fillna(0), links['level_from'], links['level_to'], links['count'], links['regn_sample_formatted'], links['weight']], axis=1).tolist(), # Added links['count'] and regn_sample_formatted
        "years": years, "annotations": [dict(x=xpos[y], y=0.985, xref='paper', yref='paper', text=str(y), showarrow=False, xanchor='center', yanchor='bottom', font=dict(size=12)) for y in years],
        "hovertemplate": 'From %{source.label}<br>To %{target.label}<br>Raw MA sum: %{customdata[1]:,.1f} ' + UNIT_LABEL + '<br>Thickness value: %{customdata[6]:,.2f}<br>Share of total: %{customdata[0]:.1f}%<br>Unique banks: %{customdata[4]:,.0f}<br>Sample REGNs: %{customdata[5]}<extra></extra>' # Updated hovertemplate to include REGN sample
    }

def build_plotly_sankey(sdata: Dict, title: str = "") -> go.Figure:
    fig = go.Figure(go.Sankey(
        node=dict(label=sdata['labels'], color=sdata['node_color'], x=sdata['node_x'], y=sdata['node_y'], pad=15, thickness=18, customdata=sdata['node_customdata'], hovertemplate="Year %{customdata[0]}<br>Bucket %{customdata[1]}<extra></extra>"),
        link=dict(source=sdata['link_source'], target=sdata['link_target'], value=sdata['values_abs'], color=sdata['link_color'], customdata=sdata['link_customdata'], hovertemplate=sdata['hovertemplate']),
        arrangement='perpendicular'
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        title=title,
        # Set explicit dimensions for consistent SVG export
        width=1200,
        height=720,
        autosize=False
    )
    for ann in sdata.get('annotations', []): fig.add_annotation(ann)
    return fig

def compute_sankey_for_variables(size_var: str, thick_var: str, n_q: int=5, ma: int=MA_MONTHS, step: int=STEP_YEARS, max_y: Optional[int]=MAX_YEAR_OVERRIDE, win: float=99.0, t_mode: str='signed_log', regn_sample_size: int=REGN_SAMPLE_SIZE, power_exponent: float=POWER_EXPONENT, thickness_mode: str='absolute') -> Tuple[go.Figure, Dict, pd.DataFrame]:
    con = get_db_connection()

    # Get min/max year from data to build a proper grid
    min_data_year, max_data_year = con.execute(f"SELECT min(year(DT)), max(year(DT)) FROM '{DATA_PATH}'").fetchone()

    print(f"DEBUG: Raw data years: min={min_data_year}, max={max_data_year}")
    print(f"DEBUG: MAX_YEAR_OVERRIDE parameter: {max_y}")

    max_year = min(max_y, max_data_year) if max_y else max_data_year
    print(f"DEBUG: Computed max_year: {max_year}")

    years = build_year_grid(min_data_year, max_year, step)
    print(f"DEBUG: Generated years grid: {years}")
    
    links_df = get_sankey_data_from_duckdb(con, size_var, thick_var, years, n_q, ma, win, t_mode, regn_sample_size)
    
    if links_df.empty:
        raise RuntimeError("No flow data could be computed. Check variables and parameters.")

    if size_var == 'state_equity_pct':
        all_cats = [b[2] for b in OWN_BUCKETS if b[2] is not None] + [UNKNOWN_LABEL]
        bucket_defs = [b for b in OWN_BUCKETS if b[2] is not None] + [(None, None, UNKNOWN_LABEL)] if OWN_BUCKETS else []
    else:
        all_cats = [f"Q{i+1}" for i in range(n_q)]
        bucket_defs = [(None, None, f"Q{i+1}") for i in range(n_q)]

    sdata = build_sankey_from_df(links_df, years, all_cats, UNIT_SCALE, t_mode, POWER_EXPONENT, 'absolute')
    
    title = (f"Sankey — size={size_var}, thickness={thick_var}, "
             f"buckets={'state' if size_var=='state_equity_pct' else f'{n_q} quantiles'} (MA={ma}M)")
    
    return build_plotly_sankey(sdata, title), {"years": years, "bucket_defs": bucket_defs}, links_df
