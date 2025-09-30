from __future__ import annotations
import os
import json
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# ------------------------------
# Parameters (edit as needed)
# ------------------------------
DATA_PATH = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
OUT_JSON  = 'sankey_data.json'   # written into this directory (visualisations/sankey_master/)
INDICATOR_COL = 'total_assets'   # e.g. 'total_loans', 'total_deposits', ...
MA_MONTHS     = 24               # moving average window (months)
STEP_YEARS    = 3                # step between columns
MAX_YEAR_OVERRIDE = 2021         # cap the plot at this year; set None to auto

# Ownership buckets (percent, 0–100)
OWN_BUCKETS = [
    (0.0, 0.0,  'State 0%'),
    (0.0, 10.0, 'State 0–10%'),
    (10.0,20.0, 'State 10–20%'),
    (20.0,50.0, 'State 20–50%'),
    (50.0,100.1,'State ≥50%'),
]
UNKNOWN_LABEL = 'Unknown'
EXIT_LABEL    = 'Exit'

# Figure + scaling
NODE_PAD     = 22
NODE_THICK   = 18
UNIT_SCALE   = 1e9     # show absolute numbers as billions
UNIT_LABEL   = 'bln'

# Transform of indicator for thickness
TRANSFORM_MODE = 'signed_log'  # 'signed_log' | 'none'

GREEN = 'rgba(76,175,80,0.65)'
RED   = 'rgba(244,67,54,0.70)'
NODE_COLORS  = {
    'State 0%':        'rgba(33,150,243,0.90)',
    'State 0–10%':     'rgba(63,81,181,0.90)',
    'State 10–20%':    'rgba(0,150,136,0.90)',
    'State 20–50%':    'rgba(255,193,7,0.90)',
    'State ≥50%':      'rgba(244,67,54,0.90)',
    UNKNOWN_LABEL:     'rgba(158,158,158,0.80)',
    EXIT_LABEL:        'rgba(120,120,120,0.75)',
}

# ------------------------------
# Utilities (data transforms)
# ------------------------------

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def standardize_percent(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype(float)
    if s.notna().sum() == 0:
        return s
    p95 = float(np.nanpercentile(np.abs(s.dropna()), 95))
    return s * 100.0 if p95 <= 1.5 else s

def year_end_snapshot_ffill(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp['year'] = tmp['DT'].dt.year
    tmp = tmp.sort_values(['REGN','year','DT'])
    for c in cols:
        tmp[c] = tmp.groupby(['REGN','year'])[c].ffill()
    snap = tmp.groupby(['REGN','year']).tail(1)[['REGN','DT','year'] + cols].reset_index(drop=True)
    return snap

def build_year_grid(min_year: int, max_year: int, step: int) -> List[int]:
    years = list(range(min_year, max_year + 1, step))
    if years[-1] != max_year:
        years.append(max_year)
    return years

def decide_kind(name: str) -> str:
    n = (name or '').lower()
    if n.endswith('_pct') or 'ratio' in n or n in {'roa','roe','nim'}:
        return 'percent'
    return 'signed_log' if TRANSFORM_MODE in {'signed_log','auto'} else 'none'

# ------------------------------
# Main generation
# ------------------------------

def generate(out_json: str = OUT_JSON, data_path: str = DATA_PATH) -> str:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    print(f"Loading {data_path} …")
    DF = pd.read_parquet(data_path)
    for c in ['DT','REGN', INDICATOR_COL, 'state_equity_pct']:
        if c not in DF.columns:
            raise ValueError(f"Expected column '{c}' in data")
    DF['DT'] = pd.to_datetime(DF['DT'])
    DF['REGN'] = pd.to_numeric(DF['REGN'], errors='coerce')
    DF[INDICATOR_COL] = pd.to_numeric(DF[INDICATOR_COL], errors='coerce')
    DF['state_equity_pct'] = standardize_percent(DF['state_equity_pct'])

    # --- Indicator MA
    DF = DF.sort_values(['REGN','DT'])
    min_periods = max(1, MA_MONTHS // 3)
    DF['indicator_ma'] = DF.groupby('REGN')[INDICATOR_COL].transform(
        lambda s: s.rolling(MA_MONTHS, min_periods=min_periods).mean()
    )

    # --- Snapshots and final indicator
    SNAP = year_end_snapshot_ffill(DF, ['state_equity_pct', 'indicator_ma'])
    RAW_SNAP = year_end_snapshot_ffill(DF, [INDICATOR_COL])
    SNAP = SNAP.merge(
        RAW_SNAP[['REGN','year', INDICATOR_COL]].rename(columns={INDICATOR_COL: 'indicator_raw'}),
        on=['REGN','year'], how='left'
    )
    SNAP['indicator_ma'] = SNAP['indicator_ma'].where(SNAP['indicator_ma'].notna(), SNAP['indicator_raw'])
    SNAP['indicator_final'] = SNAP['indicator_ma']
    KIND = decide_kind(INDICATOR_COL)
    print(f"[diag] transform kind for {INDICATOR_COL}: {KIND}")

    # --- Year grid & drop empty trailing years
    min_year_data = int(SNAP['year'].min())
    max_year_data = int(SNAP['year'].max())
    max_year = int(min(MAX_YEAR_OVERRIDE, max_year_data)) if MAX_YEAR_OVERRIDE is not None else max_year_data
    YEARS = build_year_grid(min_year_data, max_year, STEP_YEARS)

    def has_year_data(y: int) -> bool:
        sub = SNAP.loc[SNAP['year'] == y]
        if sub.empty:
            return False
        s_ma  = pd.to_numeric(sub['indicator_ma'], errors='coerce')
        s_raw = pd.to_numeric(sub['indicator_raw'], errors='coerce')
        s = np.nan_to_num(s_ma, nan=0.0) + np.nan_to_num(s_raw, nan=0.0)
        return bool(np.isfinite(s).any() and (s.sum() > 0))

    while len(YEARS) > 1 and not has_year_data(YEARS[-1]):
        dropped = YEARS.pop()
        print(f"[diag] Dropped empty trailing year: {dropped}")

    # --- Weights
    s = pd.to_numeric(SNAP['indicator_final'], errors='coerce')
    if KIND == 'percent':
        t = standardize_percent(s); t = np.clip(t, 0, None)
    elif KIND == 'signed_log':
        t = np.log1p(np.abs(s)); t = np.clip(t, 0, None)
    else:
        t = np.clip(s, 0, None)
    SNAP['w_t']   = t
    SNAP['w_raw'] = np.clip(pd.to_numeric(SNAP['indicator_final'], errors='coerce'), 0, None)

    # --- Buckets
    SNAP['bucket'] = SNAP['state_equity_pct'].apply(
        lambda p: (
            'State 0%' if (pd.notna(p) and abs(p) < 1e-12) else
            next((name for lo, hi, name in OWN_BUCKETS if name != 'State 0%' and (p > lo - 1e-12) and (p < hi + 1e-12)), UNKNOWN_LABEL)
        )
    )

    # --- Flows (exits target same-year Exit)
    links_rows = []
    for i in range(len(YEARS) - 1):
        y0, y1 = YEARS[i], YEARS[i+1]
        s0 = SNAP.loc[SNAP['year'] == y0, ['REGN','bucket','w_t','w_raw']].rename(
            columns={'bucket':'cat_from','w_t':'w','w_raw':'raw'})
        s1 = SNAP.loc[SNAP['year'] == y1, ['REGN','bucket']].rename(columns={'bucket':'cat_to'})
        both = s0.merge(s1, on='REGN', how='inner').dropna(subset=['cat_from','cat_to','w'])
        if not both.empty:
            agg = (both.groupby(['cat_from','cat_to'], as_index=False)
                        .agg(weight=('w','sum'), count=('REGN','size')))
            raw_sum = (both.groupby(['cat_from','cat_to'])['raw']
                            .sum(min_count=1)
                            .reset_index(name='raw_sum'))
            agg = agg.merge(raw_sum, on=['cat_from','cat_to'], how='left')
            agg['level_from'] = y0
            agg['level_to']   = y1
            agg['flow_type']  = 'active'
            links_rows.append(agg)
        gone = s0.merge(s1[['REGN']], on='REGN', how='left', indicator=True)
        gone = gone.loc[gone['_merge'] == 'left_only', ['REGN','cat_from','w','raw']]
        if not gone.empty:
            agx = (gone.groupby(['cat_from'], as_index=False)
                        .agg(weight=('w','sum'), count=('REGN','size')))
            raw_sum_x = (gone.groupby(['cat_from'])['raw']
                              .sum(min_count=1)
                              .reset_index(name='raw_sum'))
            agx = agx.merge(raw_sum_x, on='cat_from', how='left')
            agx['cat_to']     = EXIT_LABEL
            agx['level_from'] = y0
            agx['level_to']   = y0
            agx['flow_type']  = 'exit'
            links_rows.append(agx[['cat_from','cat_to','weight','count','raw_sum','level_from','level_to','flow_type']])

    links = pd.concat(links_rows, ignore_index=True) if links_rows else pd.DataFrame(
        columns=['cat_from','cat_to','weight','count','raw_sum','level_from','level_to','flow_type'])
    if links.empty:
        raise RuntimeError('No flows computed. Check availability across step years.')

    # --- Node grid (fixed bands)
    cat_order = ['State ≥50%', 'State 20–50%', 'State 10–20%', 'State 0–10%', 'State 0%', UNKNOWN_LABEL, EXIT_LABEL]
    LEFT_MARGIN, RIGHT_MARGIN = 0.08, 0.04
    ncol = len(YEARS)
    dx = (1 - LEFT_MARGIN - RIGHT_MARGIN) / (max(ncol - 1, 1))
    xpos = {y: (LEFT_MARGIN + i * dx if ncol > 1 else 0.5) for i, y in enumerate(YEARS)}
    EPS_EXIT = (dx * 0.0005) if ncol > 1 else 0.0005

    labels: List[str] = []
    node_customdata: List[list] = []
    node_id: Dict[Tuple[int,str], int] = {}
    node_colors: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    for year in YEARS:
        for cat in cat_order:
            nid = len(labels)
            node_id[(year, cat)] = nid
            labels.append(cat)
            node_customdata.append([year, cat])
            xs.append(xpos[year] + (EPS_EXIT if cat == EXIT_LABEL else 0.0))
            ys.append(0.5)
            node_colors.append(NODE_COLORS.get(cat, 'rgba(150,150,150,0.85)'))

    Y_TOP_MARGIN, Y_BOTTOM_MARGIN = 0.06, 0.06
    BANDS = len(cat_order)
    GAP = 0.02
    usable = max(1.0 - Y_TOP_MARGIN - Y_BOTTOM_MARGIN - GAP * (BANDS - 1), 1e-6)
    band_h = usable / BANDS
    cat_ycenter: Dict[str, float] = {}
    cursor = Y_TOP_MARGIN
    for cat in cat_order:
        cat_ycenter[cat] = cursor + band_h / 2.0
        cursor += band_h + GAP
    for year in YEARS:
        for cat in cat_order:
            ys[node_id[(year, cat)]] = cat_ycenter[cat]

    # --- Link values
    step_tot = (links.groupby(['level_from','level_to'])['weight'].transform('sum'))
    links['value_abs']   = links['weight'] / UNIT_SCALE
    links['value_share'] = np.where(step_tot > 0, 100.0 * links['weight'] / step_tot, 0.0)
    links['raw_abs']     = links['raw_sum'] / UNIT_SCALE
    sources = [node_id[(r.level_from, r.cat_from)] for r in links.itertuples(index=False)]
    targets = [node_id[(r.level_to,   r.cat_to  )] for r in links.itertuples(index=False)]
    values_abs   = links['value_abs'].astype(float).tolist()
    values_share = links['value_share'].astype(float).tolist()
    link_colors  = [GREEN if ft == 'active' else RED for ft in links['flow_type']]
    customdata = np.stack([
        links['value_share'].to_numpy(),
        links['raw_abs'].fillna(0).to_numpy(),
        links['level_from'].to_numpy(),
        links['level_to'].to_numpy()
    ], axis=1)

    hovertemplate = (
        'From %{source.label}<br>'
        'To %{target.label}<br>'
        'Raw MA sum: %{customdata[1]:,.1f} ' + UNIT_LABEL + '<br>'
        'Thickness value (current mode): %{value:,.2f}<br>'
        'Share of step total: %{customdata[0]:.1f}%'
        '<extra></extra>'
    )

    # Build annotations for column years (for plotly layout)
    annotations = [dict(x=xpos[y], y=0.985, xref='paper', yref='paper', text=str(y),
                        showarrow=False, xanchor='center', yanchor='bottom',
                        font=dict(size=12)) for y in YEARS]

    # --- Serialize to a JSON structure suitable for JS
    data = {
        'labels': labels,
        'node_customdata': node_customdata,
        'node_x': [float(x) for x in xs],
        'node_y': [float(y) for y in ys],
        'node_color': node_colors,
        'link_source': [int(s) for s in sources],
        'link_target': [int(t) for t in targets],
        'values_abs': [float(v) for v in values_abs],
        'values_share': [float(v) for v in values_share],
        'link_color': link_colors,
        'link_customdata': customdata.tolist(),
        'years': [int(y) for y in YEARS],
        'link_from': [int(x) for x in links['level_from'].tolist()],
        'link_to': [int(x) for x in links['level_to'].tolist()],
        'annotations': annotations,
        'hovertemplate': hovertemplate,
        'unit_label': UNIT_LABEL,
        'title': (
            f"Ownership buckets (step={STEP_YEARS}y) — Weight = {MA_MONTHS}M MA of {INDICATOR_COL} "
            f"(abs in {UNIT_LABEL}, or % of step)"
        )
    }

    ensure_dirs(os.path.dirname(out_json) or '.')
    with open(out_json, 'w', encoding='utf8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    out_path = os.path.abspath(out_json)
    print("Wrote sankey data to", out_path)
    return out_path

if __name__ == '__main__':
    generate()
