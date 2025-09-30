"""
Ownership buckets Time‑Sankey (deterministic grid)
— MA for flows; **RAW for last column** (with safe logic)
— If the **last year has no data**, it is **dropped** from the viz entirely
— Final‑year exits are sent to the last column (only if it exists)
— Layout is fully deterministic and the HTML fills the viewport (no vertical clipping)
— Nodes are positioned with arrangement='fixed' so they don't bunch at the bottom after resize
— Working SVG export via modebar (camera icon) and optional custom button overlay

Output: `visualisations/html/sankey_ownership_4y_ma_LASTYEAR_RAW_fix.html`
"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------
# Parameters (edit as needed)
# ------------------------------
DATA_PATH = 'data/HOORRAAH_final_banking_indicators_preprocessed.parquet'
OUT_DIR   = 'visualisations/html'
OUT_FILE  = 'sankey_ownership_4y_ma_LASTYEAR_RAW_fix.html'

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

# Figure + scaling (actual on‑screen size comes from viewport; these are just defaults)
NODE_PAD     = 10
NODE_THICK   = 18
UNIT_SCALE   = 1e9     # show absolute numbers as billions
UNIT_LABEL   = 'bln'

# Transform of indicator for thickness
# 'signed_log' | 'auto' | 'none'
TRANSFORM_MODE = 'signed_log'
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
    """Within each (REGN, year) forward‑fill cols and take the last row in that year."""
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
# Fill MA with raw when MA is missing (legacy behaviour for non-last years)
SNAP['indicator_ma'] = SNAP['indicator_ma'].where(SNAP['indicator_ma'].notna(), SNAP['indicator_raw'])

# Indicator used for *distribution in the last column* must be RAW year‑end when possible
SNAP['indicator_final'] = SNAP['indicator_ma']
# LAST_COL_YEAR will be decided after building YEARS

# Transform helper
KIND = decide_kind(INDICATOR_COL)
print(f"[diag] transform kind for {INDICATOR_COL}: {KIND}")

# ------------------------------
# Build year grid — then DROP any trailing empty years before building flows
# ------------------------------
min_year_data = int(SNAP['year'].min())
max_year_data = int(SNAP['year'].max())
if MAX_YEAR_OVERRIDE is not None:
    max_year = int(min(MAX_YEAR_OVERRIDE, max_year_data))
else:
    max_year = max_year_data
YEARS = build_year_grid(min_year_data, max_year, STEP_YEARS)

# --- Robustly drop trailing empty years (no rows OR all zeros/NaNs) ---
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

LAST_COL_YEAR = YEARS[-1]

# Now set indicator_final (RAW for the last kept year only)
mask_last = SNAP['year'] == LAST_COL_YEAR
SNAP.loc[mask_last & SNAP['indicator_raw'].notna(), 'indicator_final'] = SNAP.loc[mask_last, 'indicator_raw']

# Final weights (thickness & additive totals)
s = pd.to_numeric(SNAP['indicator_final'], errors='coerce')
if KIND == 'percent':
    t = standardize_percent(s); t = np.clip(t, 0, None)
elif KIND == 'signed_log':
    t = np.log1p(np.abs(s)); t = np.clip(t, 0, None)
else:
    t = np.clip(s, 0, None)
SNAP['w_t']   = t
SNAP['w_raw'] = np.clip(pd.to_numeric(SNAP['indicator_final'], errors='coerce'), 0, None)

# Bucket assignment (by state_equity_pct)
SNAP['bucket'] = SNAP['state_equity_pct'].apply(
    lambda p: (
        'State 0%' if (pd.notna(p) and abs(p) < 1e-12) else
        next((name for lo, hi, name in OWN_BUCKETS if name != 'State 0%' and (p > lo - 1e-12) and (p < hi + 1e-12)), UNKNOWN_LABEL)
    )
)

# ------------------------------
# Build step flows (final‑year exits go to the last column that actually exists)
# ------------------------------
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

    # Exit flows (banks present at y0 but missing at y1)
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
        if y1 == LAST_COL_YEAR:
            agx['level_to']   = LAST_COL_YEAR
            agx['flow_type']  = 'exit_final'
        else:
            agx['level_to']   = y0
            agx['flow_type']  = 'exit'
        links_rows.append(agx[['cat_from','cat_to','weight','count','raw_sum','level_from','level_to','flow_type']])

links = pd.concat(links_rows, ignore_index=True) if links_rows else pd.DataFrame(
    columns=['cat_from','cat_to','weight','count','raw_sum','level_from','level_to','flow_type'])
if links.empty:
    raise RuntimeError('No flows computed. Check availability across step years.')

# ------------------------------
# Deterministic Node Grid (columns = YEARS, rows = fixed order)
# ------------------------------
cat_order = ['State ≥50%', 'State 20–50%', 'State 10–20%', 'State 0–10%', 'State 0%', UNKNOWN_LABEL, EXIT_LABEL]

# X positions: equal spacing (strictly increasing). Tiny epsilon for any Exit in same column.
LEFT_MARGIN, RIGHT_MARGIN = 0.08, 0.04
ncol = len(YEARS)
dx = (1 - LEFT_MARGIN - RIGHT_MARGIN) / (max(ncol - 1, 1))
xpos = {y: (LEFT_MARGIN + i * dx if ncol > 1 else 0.5) for i, y in enumerate(YEARS)}
EPS_EXIT = 1e-6 if ncol > 1 else 0.0

# Node labels & base x/y (we will overwrite y with stacked positions)
labels: List[str] = []
node_customdata: List[list] = []  # [[year, category], ...]
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
        ys.append(0.5)  # temp; will be set by stacked layout
        node_colors.append(NODE_COLORS.get(cat, 'rgba(150,150,150,0.85)'))

# --- Compute node 'values' for vertical stacking (deterministic)
node_vals_raw = (
    SNAP.groupby(['year','bucket'], as_index=False)['w_raw']
        .sum()
        .rename(columns={'w_raw': 'val'})
)
vals = {(int(r.year), str(r.bucket)): float(r.val) for r in node_vals_raw.itertuples(index=False)}

# Exit rows: base the value on exit-link totals per **target year** so final exits inflate the last column
exit_by_target = (
    links[links['cat_to'] == EXIT_LABEL]
        .groupby(['level_to'], as_index=False)['weight']
        .sum()
        .rename(columns={'level_to': 'year', 'weight': 'val'})
)
for r in exit_by_target.itertuples(index=False):
    vals[(int(r.year), EXIT_LABEL)] = float(r.val)

# Stacked layout per year (deterministic)
Y_TOP_MARGIN, Y_BOTTOM_MARGIN = 0.06, 0.06
# Adaptive small gap so everything fits in the viewport
VERTICAL_GAP = 0.02
MIN_FRAC = 1e-4  # minimal share to preserve order when a category is zero

for year in YEARS:
    series = [max(vals.get((year, cat), 0.0), 0.0) for cat in cat_order]
    total = float(sum(series))
    gap_space = VERTICAL_GAP * (len(cat_order) - 1)
    drawable = max(1.0 - Y_TOP_MARGIN - Y_BOTTOM_MARGIN - gap_space, 1e-6)

    # Convert to fractions, enforce a tiny floor so zero categories keep their slot
    if total > 0:
        fracs = [max(v / total, MIN_FRAC) for v in series]
        ssum = sum(fracs)
        fracs = [f / ssum for f in fracs]
    else:
        fracs = [1.0 / len(cat_order)] * len(cat_order)

    y_cursor = Y_TOP_MARGIN
    for ci, cat in enumerate(cat_order):
        h = drawable * fracs[ci]
        y_center = y_cursor + h / 2
        ys[node_id[(year, cat)]] = y_center
        y_cursor += h + VERTICAL_GAP

# Diagnostics
node_years = sorted({y for (y, _) in node_id.keys()})
print('[diag] YEARS (used):', YEARS)
print('[diag] X positions by year:')
for y in YEARS:
    uniq = sorted(set(round(xs[node_id[(y, c)]], 5) for c in cat_order))
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
        f"[diag] WARNING: {viol} link(s) violate x[source] < x[target]. "
        f"Consider increasing EPS_EXIT slightly if nodes get pushed."
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
    arrangement='fixed',  # << key fix so x/y are respected
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
annotations = [dict(x=xpos[y], y=0.985, xref='paper', yref='paper', text=str(y),
                    showarrow=False, xanchor='center', yanchor='bottom',
                    font=dict(size=12)) for y in YEARS]
fig.update_layout(
    title={'text': (
            f"Ownership buckets (step={STEP_YEARS}y) — Weight = {MA_MONTHS}M MA of {INDICATOR_COL} "
            f"(last column uses RAW year‑end; abs in {UNIT_LABEL}, or % of step)"
           ),
           'x': 0.02, 'xanchor': 'left'},
    autosize=True,
    margin=dict(l=16, r=16, t=56, b=12),
    annotations=annotations,
    updatemenus=[
        dict(
            type='buttons', x=0.01, y=0.99, xanchor='left', yanchor='top',
            buttons=[
                dict(label=f'Thickness: Absolute ({UNIT_LABEL})', method='restyle', args=[{'link.value': [values_abs]}]),
                dict(label='Thickness: Share per step (%)', method='restyle', args=[{'link.value': [values_share]}]),
            ]
        ),
        dict(
            type='buttons', x=0.30, y=0.99, xanchor='left', yanchor='top',
            buttons=[
                # More robust reset using relayout path for the 1st sankey trace
                dict(label='Reset positions', method='relayout', args=[{'sankey[0].node.x': xs, 'sankey[0].node.y': ys}]),
            ]
        ),
    ]
)

# --- Full-viewport HTML with working SVG export ---
# Add a small overlay button that calls Plotly.downloadImage, and ensure fixed viewport sizing
post_script = (
    """
    (function(){
      var gd = document.getElementById('sankey_vp');

      function ensureUi(){
        var host = document.getElementById('sankey_ui');
        if(!host){
          host = document.createElement('div');
          host.id = 'sankey_ui';
          host.style.position='fixed';
          host.style.top='8px';
          host.style.right='12px';
          host.style.zIndex='50';
          host.style.fontFamily='sans-serif';
          document.body.appendChild(host);
        }
        if(!document.getElementById('save_svg_btn')){
          var btn = document.createElement('button');
          btn.id = 'save_svg_btn';
          btn.textContent = 'Save SVG';
          btn.style.padding='6px 10px';
          btn.style.border='1px solid #ccc';
          btn.style.borderRadius='6px';
          btn.style.background='#fafafa';
          btn.style.cursor='pointer';
          btn.onclick = function(){
            if(window.Plotly && gd){
              Plotly.downloadImage(gd, {
                format:'svg',
                filename: (document.title || 'sankey').replace(/\\s+/g,'_')
              });
            }
          };
          host.appendChild(btn);
        }
      }

      function resize(){
        document.documentElement.style.height = '100%';
        document.body.style.cssText = 'height:100%;margin:0;overflow:hidden;';
        if (gd){
          gd.style.position = 'fixed';
          gd.style.top = '0';
          gd.style.left = '0';
          gd.style.right = '0';
          gd.style.bottom = '0';
          gd.style.width = '100vw';
          gd.style.height = '100vh';
          if (window.Plotly && gd.data) {
            Plotly.relayout(gd, {autosize:true, margin:{l:16,r:16,t:56,b:12}});
            Plotly.Plots.resize(gd);
          }
        }
        ensureUi();
      }
      window.addEventListener('resize', resize);
      setTimeout(resize, 0);
    })();
    """
)

ensure_dirs(OUT_DIR)
fig.write_html(
    os.path.join(OUT_DIR, OUT_FILE),
    include_plotlyjs='cdn', full_html=True,
    div_id='sankey_vp',
    default_width='100%', default_height='100%',
    config={
        'responsive': True,
        'displaylogo': False,
        # Built-in modebar download will save as SVG by default
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': OUT_FILE.replace('.html',''),
            'height': None, 'width': None, 'scale': 1
        }
    },
    post_script=post_script,
)
print('Saved:', os.path.join(OUT_DIR, OUT_FILE))

# Optional: static SVG snapshot via Kaleido (requires kaleido installed)
# from plotly.io import write_image
# write_image(fig, os.path.join(OUT_DIR, OUT_FILE.replace('.html','.svg')), width=1600, height=900, scale=2)
# print('Saved:', os.path.join(OUT_DIR, OUT_FILE.replace('.html','.svg')))
