"""
Time‑Sankey of state‑ownership buckets, 4‑year steps (Plotly)
— FULLY DETERMINISTIC GRID (columns = years; rows = fixed order) —

What this version guarantees
- Exactly **one column per year** (2004 → … → 2025) — positions are precomputed.
- **Rows are fixed** top→bottom: State ≥50%, 20–50%, 10–20%, 0–10%, 0%, Unknown, Exit.
- **Exits live in the same‑year column** (tiny epsilon to satisfy left→right).
- **Deterministic vertical layout**: for each year, categories are spaced like a
  stacked bar using that year’s **category totals**. (If a category has zero
  value in a year, it receives a tiny slot to preserve order.)
- A single Sankey trace (no auto layout interference). Thickness toggle: abs vs %.

Output: `visualisations/html/sankey_ownership_4y_ma.html`
"""

import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------
# Parameters
# ------------------------------
DATA_PATH = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
OUT_DIR   = 'visualisations/html'
OUT_FILE  = 'sankey_ownership_4y_ma_FIX.html'  # distinct name to avoid opening stale HTML

INDICATOR_COL = 'total_deposits'   # e.g. 'total_loans', 'total_deposits', ...
MA_MONTHS     = 12                 # moving average window (months)
STEP_YEARS    = 3                  # step between columns

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
FIG_W, FIG_H = 2400, 1400
NODE_PAD     = 22
NODE_THICK   = 18
UNIT_SCALE   = 1e9     # show absolute numbers as billions
UNIT_LABEL   = 'bln'

# Transform of indicator for thickness (mirrors boxplot):
# 'signed_log' | 'auto' | 'none'
TRANSFORM_MODE = 'none'
CLIP_ENABLED   = True
CLIP_LO, CLIP_HI = 0.02, 0.98

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
# Utils
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
    """Within each (REGN, year) forward-fill cols and take the last day."""
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
# Load data & compute weights
# ------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")
print(f"Loading {DATA_PATH} …")
DF = pd.read_parquet(DATA_PATH)

for c in ['DT','REGN', INDICATOR_COL, 'state_equity_pct']:
    if c not in DF.columns:
        raise ValueError(f"Expected column '{c}' in data")

DF['DT'] = pd.to_datetime(DF['DT'])
DF['REGN'] = pd.to_numeric(DF['REGN'], errors='coerce')
DF[INDICATOR_COL] = pd.to_numeric(DF[INDICATOR_COL], errors='coerce')
DF['state_equity_pct'] = standardize_percent(DF['state_equity_pct'])

# Moving average of indicator per bank (monthly rolling by row count)
DF = DF.sort_values(['REGN','DT'])
min_periods = max(1, MA_MONTHS // 3)
DF['indicator_ma'] = DF.groupby('REGN')[INDICATOR_COL].transform(
    lambda s: s.rolling(MA_MONTHS, min_periods=min_periods).mean()
)

# Year-end snapshot + fallback to raw
SNAP = year_end_snapshot_ffill(DF, ['state_equity_pct', 'indicator_ma'])
RAW_SNAP = year_end_snapshot_ffill(DF, [INDICATOR_COL])
SNAP = SNAP.merge(
    RAW_SNAP[['REGN','year', INDICATOR_COL]].rename(columns={INDICATOR_COL: 'indicator_raw'}),
    on=['REGN','year'], how='left'
)
SNAP['indicator_ma'] = SNAP['indicator_ma'].where(SNAP['indicator_ma'].notna(), SNAP['indicator_raw'])

# Transform for flow thickness
KIND = decide_kind(INDICATOR_COL)
print(f"[diag] transform kind for {INDICATOR_COL}: {KIND}")

s = pd.to_numeric(SNAP['indicator_ma'], errors='coerce')
if KIND == 'percent':
    t = standardize_percent(s)
    t = np.clip(t, 0, None)
elif KIND == 'signed_log':
    t = np.log1p(np.abs(s))
    t = np.clip(t, 0, None)
else:
    t = np.clip(s, 0, None)

SNAP['w_t']   = t        # for link thickness (transformed)
SNAP['w_raw'] = np.clip(s, 0, None)  # for node totals / shares (raw, additive)

if CLIP_ENABLED:
    arr = SNAP['w_t'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if arr.size:
        lo = float(np.nanquantile(arr, CLIP_LO))
        hi = float(np.nanquantile(arr, CLIP_HI))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            SNAP['w_t'] = SNAP['w_t'].clip(lower=lo, upper=hi)

# Bucket assignment
SNAP['bucket'] = SNAP['state_equity_pct'].apply(
    lambda p: (
        'State 0%' if (pd.notna(p) and abs(p) < 1e-12) else
        next((name for lo, hi, name in OWN_BUCKETS if name != 'State 0%' and (p > lo - 1e-12) and (p < hi + 1e-12)), UNKNOWN_LABEL)
    )
)

# ------------------------------
# Build 4y steps & flows
# ------------------------------
min_year = int(SNAP['year'].min())
max_year = int(SNAP['year'].max()) 
YEARS    = build_year_grid(min_year, max_year, STEP_YEARS)
if len(YEARS) < 2:
    raise RuntimeError('Not enough annual steps to build a time‑sankey.')
print(f"Levels: {YEARS}")

links_rows = []
for i in range(len(YEARS) - 1):
    y0, y1 = YEARS[i], YEARS[i+1]
    s0 = SNAP.loc[SNAP['year'] == y0, ['REGN','bucket','w_t','w_raw']].rename(
        columns={'bucket':'cat_from','w_t':'w','w_raw':'raw'})
    s1 = SNAP.loc[SNAP['year'] == y1, ['REGN','bucket']].rename(columns={'bucket':'cat_to'})

    # Active flows (banks present at both y0 and y1)
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

    # Exit flows (banks present at y0 but missing at y1) — **target is y0**
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
        agx['level_to']   = y0   # exit sits in the same year column
        agx['flow_type']  = 'exit'
        links_rows.append(agx[['cat_from','cat_to','weight','count','raw_sum','level_from','level_to','flow_type']])

links = pd.concat(links_rows, ignore_index=True) if links_rows else pd.DataFrame(
    columns=['cat_from','cat_to','weight','count','raw_sum','level_from','level_to','flow_type'])
if links.empty:
    raise RuntimeError('No flows computed. Check availability across 4‑year steps.')

# ------------------------------
# Deterministic Node Grid (columns = YEARS, rows = fixed order)
# ------------------------------
cat_order = ['State ≥50%', 'State 20–50%', 'State 10–20%', 'State 0–10%', 'State 0%', UNKNOWN_LABEL, EXIT_LABEL]

# X positions: equal spacing (strictly increasing). A tiny epsilon for Exit.
LEFT_MARGIN, RIGHT_MARGIN = 0.08, 0.04
ncol = len(YEARS)
dx = (1 - LEFT_MARGIN - RIGHT_MARGIN) / (max(ncol - 1, 1))
xpos = {y: (LEFT_MARGIN + i * dx if ncol > 1 else 0.5) for i, y in enumerate(YEARS)}
EPS_EXIT = 1e-6 if ncol > 1 else 0.0

# Node labels & base x/y (we will overwrite y with stacked positions)
nodes_full: List[str] = []   # full labels like "YYYY: Cat" for diagnostics
labels: List[str] = []       # category-only labels rendered on nodes
node_customdata: List[list] = []  # [[year, category], ...]
node_id: Dict[Tuple[int,str], int] = {}
node_colors: List[str] = []
xs: List[float] = []
ys: List[float] = []
for year in YEARS:
    for cat in cat_order:
        nid = len(labels)
        node_id[(year, cat)] = nid
        nodes_full.append(f"{year}: {cat}")
        labels.append(cat)
        node_customdata.append([year, cat])
        xs.append(xpos[year] + (EPS_EXIT if cat == EXIT_LABEL else 0.0))
        ys.append(0.5)  # temp; will be set by stacked layout
        node_colors.append(NODE_COLORS.get(cat, 'rgba(150,150,150,0.85)'))

# --- Compute node 'values' for vertical stacking (deterministic)
# Use **raw** additive totals per year & category from the snapshot.
node_vals_raw = (
    SNAP.groupby(['year','bucket'], as_index=False)['w_raw']
        .sum()
        .rename(columns={'w_raw': 'val'})
)
vals = {(int(r.year), str(r.bucket)): float(r.val) for r in node_vals_raw.itertuples(index=False)}

# For Exit rows, base the value on exit-link totals per year
exit_vals = (
    links[links['flow_type'] == 'exit']
        .groupby(['level_from'], as_index=False)['weight']
        .sum()
        .rename(columns={'level_from': 'year', 'weight': 'val'})
)
for r in exit_vals.itertuples(index=False):
    vals[(int(r.year), EXIT_LABEL)] = float(r.val)

# Stacked layout per year
Y_TOP_MARGIN, Y_BOTTOM_MARGIN = 0.06, 0.06
VERTICAL_GAP = 0.2
MIN_FRAC = 1e-4  # minimal share to preserve order when a category is zero

for year in YEARS:
    series = [max(vals.get((year, cat), 0.0), 0.0) for cat in cat_order]
    total = float(sum(series))
    gap_space = VERTICAL_GAP * (len(cat_order) - 1)
    drawable = max(1.0 - Y_TOP_MARGIN - Y_BOTTOM_MARGIN - gap_space, 1e-6)

    # Convert to fractions, enforce a tiny floor so zero categories keep their slot
    if total > 0:
        fracs = [max(v / total, MIN_FRAC) for v in series]
        # rescale so fractions sum to 1 after flooring
        s = sum(fracs)
        fracs = [f / s for f in fracs]
    else:
        # even split if nothing for that year (degenerate case)
        fracs = [1.0 / len(cat_order)] * len(cat_order)

    y_cursor = Y_TOP_MARGIN
    for ci, cat in enumerate(cat_order):
        h = drawable * fracs[ci]
        y_center = y_cursor + h / 2
        ys[node_id[(year, cat)]] = y_center
        y_cursor += h + VERTICAL_GAP

# Diagnostics
# 1) strict year set minted into nodes
node_years = sorted({int(lbl.split(':')[0]) for lbl in nodes_full})
print('[diag] YEARS (intended):', YEARS)
print('[diag] YEARS (in nodes):', node_years)
assert set(node_years) == set(YEARS), 'Node years diverge from intended YEARS; you may be viewing an old HTML or a stale build.'
print('[diag] X positions by year:')
for y in YEARS:
    uniq = sorted(set(round(v,5) for i,v in enumerate(xs) if nodes_full[i].startswith(f'{y}:')))
    print(f'  {y}: {uniq}')

# ------------------------------
# Link values + modes
# ------------------------------
step_tot = (links.groupby(['level_from','level_to'])['weight'].transform('sum'))
links['value_abs']   = links['weight'] / UNIT_SCALE
links['value_share'] = np.where(step_tot > 0, 100.0 * links['weight'] / step_tot, 0.0)
links['raw_abs']     = links['raw_sum'] / UNIT_SCALE

sources = [node_id[(r.level_from, r.cat_from)] for r in links.itertuples(index=False)]
targets = [node_id[(r.level_to,   r.cat_to  )] for r in links.itertuples(index=False)]

# Validate strict left→right (important for exits)
srcx = np.array([xs[s] for s in sources])
tgtx = np.array([xs[t] for t in targets])
viol = int(np.sum(srcx >= tgtx - 1e-12))
if viol:
    print(
        f"[diag] WARNING: {viol} link(s) violate x[source] < x[target]."
        f"Increase EPS_EXIT slightly if you see nodes pushed to the last stage."
    )

values_abs   = links['value_abs'].astype(float).tolist()
values_share = links['value_share'].astype(float).tolist()
link_colors  = [GREEN if ft == 'active' else RED for ft in links['flow_type']]

# customdata: [share, raw_abs]
customdata = np.stack([
    links['value_share'].to_numpy(),
    links['raw_abs'].fillna(0).to_numpy()
], axis=1)

hovertemplate = (
    'From %{source.label}<br>'
    'To %{target.label}<br>'
    'Raw MA sum: %{customdata[1]:,.1f} ' + UNIT_LABEL + '<br>'
    'Thickness value (current mode): %{value:,.2f}<br>'
    'Share of step total: %{customdata[0]:.1f}%'
)

# --- Single STRICT trace (no auto arrangement) ---
sankey = go.Sankey(
    arrangement='perpendicular',
    domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
    node=dict(
        pad=NODE_PAD,
        thickness=NODE_THICK,
        label=labels,
        customdata=node_customdata,
        hovertemplate='%{customdata[0]}: %{customdata[1]}' + '<extra></extra>',
        color=node_colors,
        x=xs, y=ys,
        line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values_abs,              # default: absolute
        color=link_colors,
        customdata=customdata,
        hovertemplate=hovertemplate,
    ),
)

fig = go.Figure(data=[sankey])
# Year headers above each column (paper coords)
annotations = [dict(x=xpos[y], y=1.02, xref='paper', yref='paper', text=str(y),
                    showarrow=False, xanchor='center', yanchor='bottom',
                    font=dict(size=12)) for y in YEARS]
fig.update_layout(
    title={'text': f"Ownership buckets (step={STEP_YEARS}y) — Weight = {MA_MONTHS}M MA of {INDICATOR_COL} (transform: {decide_kind(INDICATOR_COL)}; abs in {UNIT_LABEL}, or % of step)",
           'x': 0.02, 'xanchor': 'left'},
    width=FIG_W, height=FIG_H,
    margin=dict(l=30, r=30, t=80, b=30),
    annotations=annotations,
    updatemenus=[
        dict(
            type='buttons', x=0.02, y=1.08, xanchor='left',
            buttons=[
                dict(label=f'Thickness: Absolute ({UNIT_LABEL})', method='restyle', args=[{'link.value': [values_abs]}]),
                dict(label='Thickness: Share per step (%)', method='restyle', args=[{'link.value': [values_share]}]),
            ]
        ),
        dict(
            type='buttons', x=0.35, y=1.08, xanchor='left',
            buttons=[
                dict(label='Reset positions', method='restyle', args=[{'node.x': [xs], 'node.y': [ys]}, [0]]),
            ]
        ),
    ]
)

# --- Minimal JS helper: resetPositions() (already wired via button) ---
post_script = (
    """var gd = document.getElementById('{plot_id}');
    // nothing extra for now — deterministic grid already enforced
    """
)

ensure_dirs(OUT_DIR)
fig.write_html(
    os.path.join(OUT_DIR, OUT_FILE),
    include_plotlyjs='cdn', full_html=True,
    post_script=post_script,
)
print('Saved:', os.path.join(OUT_DIR, OUT_FILE))
