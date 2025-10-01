"""
Dash app for the Sankey dashboard (visual layer).

This file is the visualization layer only. It imports the data/prep utilities from
visualisations/sankey_master_universal/dashboard_data.py and exposes UI controls
so the user can adjust key parameters at runtime.

Controls exposed:
 - variable dropdown (numeric variables discovered from the parquet)
 - quantile slider (when variable != state_equity_pct)
 - step years slider (STEP_YEARS)
 - max year override input (MAX_YEAR_OVERRIDE; blank => auto)
 - transform mode (TRANSFORM_MODE)
 - unit scale (UNIT_SCALE) (options: 'bln' -> 1e9, 'none' -> 1)
 - moving average months slider (MA_MONTHS)
 - thickness mode toggle (% of column total vs absolute values)
 - recompute layout button (reorganizes nodes with current hyperparameters)
 - node thickness (NODE_THICK)
 - node pad (NODE_PAD)

Usage (run locally):
    uv run visualisations/sankey_master_universal/dashboard_app.py
    or
    python visualisations/sankey_master_universal/dashboard_app.py

Do not run commands from me; run locally and paste any errors/backtraces here.
"""
from __future__ import annotations
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Dash
try:
    from dash import Dash, dcc, html, Input, Output, State
except Exception as exc:
    raise ImportError("Dash is required. Install with `uv add dash` or `pip install dash`.") from exc

# Import data/prep module
# Try normal package import first; when running the script directly the package
# path may not be available, so fall back to loading the local file via importlib.
try:
    from visualisations.sankey_master_universal import dashboard_data as dd
except Exception:
    import importlib.util
    MOD_PATH = os.path.join(os.path.dirname(__file__), 'dashboard_data.py')
    spec = importlib.util.spec_from_file_location('dashboard_data', MOD_PATH)
    dashboard_data = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dashboard_data)
    dd = dashboard_data

# Host/port
DASH_HOST = "localhost"
DASH_PORT = 2543

# Create app
def create_app():
    app = Dash(__name__)
    numeric_cols = list(dd.DEFAULT_NUMERIC_COLUMNS)

    # UI options
    var_options = [{'label': c, 'value': c} for c in numeric_cols]
    transform_options = [{'label': 'signed_log', 'value': 'signed_log'}, {'label': 'log1p', 'value': 'log1p'}, {'label': 'minmax', 'value': 'minmax'}, {'label': 'none', 'value': 'none'}, {'label': 'power', 'value': 'power'}]
    unit_options = [{'label': 'bln (1e9)', 'value': 'bln'}, {'label': 'none (1)', 'value': 'none'}]

    app.layout = html.Div([
        html.H3("Sankey — variable-driven dashboard (visual layer)"),
        html.Div([
            html.Div([
                html.Label("Size variable (buckets):"),
                dcc.Dropdown(
                    id='size-variable-dropdown',
                    options=var_options,
                    value=('state_equity_pct' if 'state_equity_pct' in numeric_cols else (numeric_cols[0] if numeric_cols else None)),
                    clearable=False,
                    style={'width':'320px'}
                ),
            ], style={'display':'inline-block','marginRight':'16px'}),
            html.Div([
                html.Label("Thickness variable (MA used for link thickness):"),
                dcc.Dropdown(
                    id='thickness-variable-dropdown',
                    options=var_options,
                    value=(numeric_cols[0] if numeric_cols else None),
                    clearable=False,
                    style={'width':'320px'}
                ),
            ], style={'display':'inline-block'})
        ], style={'marginBottom':'8px'}),
        html.Div([
            html.Label("Number of quantiles (used when size variable != state_equity_pct):"),
            dcc.Slider(id='quantile-slider', min=2, max=9, step=1, value=5, marks={i:str(i) for i in range(2,10)}, tooltip={'placement':'bottom'}),
        ], style={'width':'420px','marginBottom':'8px'}),
        html.Div([
            html.Label("STEP_YEARS (step between columns):"),
            dcc.Slider(id='step-years-slider', min=1, max=6, step=1, value=dd.STEP_YEARS, marks={i:str(i) for i in range(1,7)}),
        ], style={'width':'420px','marginBottom':'8px'}),
        html.Div([
            html.Label("MAX_YEAR_OVERRIDE (leave blank for auto):"),
            dcc.Input(id='max-year-input', type='number', placeholder='e.g. 2021', value=dd.MAX_YEAR_OVERRIDE, style={'width':'180px'}),
            html.Span("  (blank = auto)", style={'marginLeft':'8px','color':'#666','fontSize':'12px'})
        ], style={'marginBottom':'8px'}),
        html.Div([
            html.Label("Transform mode:"),
            dcc.Dropdown(id='transform-mode', options=transform_options, value=dd.TRANSFORM_MODE, clearable=False, style={'width':'240px'}),
            html.Label("Power Exponent (for 'power' transform):"),
            html.Div(dcc.Slider(id='power-exponent-slider', min=0.1, max=1.0, step=0.1, value=dd.POWER_EXPONENT, marks={i/10:str(i/10) for i in range(1,11)}, tooltip={'placement':'bottom'}), style={'width':'240px','marginTop':'6px'}),
            html.Label("Unit scale:"),
            dcc.Dropdown(id='unit-scale', options=unit_options, value=('bln' if dd.UNIT_SCALE==1e9 else 'none'), clearable=False, style={'width':'180px','marginTop':'6px'})
        ], style={'marginBottom':'12px'}),
        html.Div([
            html.Label("Moving Average (months):"),
            dcc.Slider(id='ma-months-slider', min=6, max=48, step=3, value=dd.MA_MONTHS, marks={6:'6M', 12:'12M', 18:'18M', 24:'24M', 36:'36M', 48:'48M'}, tooltip={'placement':'bottom'}),
        ], style={'width':'420px','marginBottom':'8px'}),
        html.Div([
            dcc.Checklist(
                id='thickness-mode-toggle',
                options=[{'label': 'Use % of column total for thickness', 'value': 'percentage'}],
                value=[],  # Default to absolute values
                style={'marginTop': '8px'}
            )
        ], style={'marginBottom':'12px'}),
        html.Div([
            html.Label("Winsorize upper bound (%):"),
            dcc.Slider(id='winsor-slider', min=95.0, max=100.0, step=0.1, value=99.0, marks={95:'95', 98:'98', 99:'99', 99.9:'99.9', 100:'100'}, tooltip={'placement':'bottom'}),
        ], style={'width':'420px','marginBottom':'8px'}),
        html.Div([
            html.Label("Node thickness:"),
            dcc.Slider(id='node-thick', min=6, max=48, step=1, value=18, marks={6:'6', 18:'18', 36:'36'}),
            html.Label("Node pad:"),
            dcc.Slider(id='node-pad', min=2, max=48, step=1, value=22, marks={2:'2', 22:'22', 48:'48'}),
        ], style={'width':'520px','marginBottom':'12px'}),
        html.Div([
            html.Label("REGN sample size for tooltip:"),
            dcc.Input(id='regn-sample-size-input', type='number', min=0, max=20, step=1, value=dd.REGN_SAMPLE_SIZE, style={'width':'180px'}),
            html.Span("  (0 = no sample)", style={'marginLeft':'8px','color':'#666','fontSize':'12px'})
        ], style={'marginBottom':'8px'}),
        dcc.Loading(dcc.Graph(id='sankey-graph', figure={'data': [], 'layout': {'title': 'Loading Sankey diagram...'}}, style={'height':'720px'}), type='circle'),
        html.Div(id='variable-warning', style={'color':'#a00','marginTop':'8px'}),
        html.Pre(id='meta-output', style={'fontSize':'12px','whiteSpace':'pre-wrap','marginTop':'8px','color':'#333'}),
        html.Div([
            html.Button(
                'Recompute Layout',
                id='recompute-layout-button',
                style={
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'marginTop': '10px',
                    'marginRight': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                    'transition': 'all 0.2s ease'
                },
                n_clicks=0
            ),
            html.Button(
                'Save to SVG',
                id='save-svg-button',
                style={
                    'backgroundColor': '#007cba',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'marginTop': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                    'transition': 'all 0.2s ease'
                },
                n_clicks=0
            ),
            html.Div(id='export-status', style={'marginTop': '8px', 'color': '#666', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'marginTop': '16px'}),
        # Add JavaScript for debugging button clicks
        html.Script("""
        document.addEventListener('DOMContentLoaded', function() {
            var saveButton = document.getElementById('save-svg-button');
            var layoutButton = document.getElementById('recompute-layout-button');

            if (saveButton) {
                saveButton.addEventListener('click', function() {
                    console.log('Save SVG button clicked! n_clicks should increment');
                });
            } else {
                console.error('Button with id save-svg-button not found!');
            }

            if (layoutButton) {
                layoutButton.addEventListener('click', function() {
                    console.log('Recompute Layout button clicked! n_clicks should increment');
                });
            } else {
                console.error('Button with id recompute-layout-button not found!');
            }
        });
        """)
    ], style={'fontFamily':'Arial, sans-serif', 'padding':'16px'})

    @app.callback(
        Output('sankey-graph', 'figure'),
        Output('variable-warning', 'children'),
        Output('meta-output', 'children'),
        Input('size-variable-dropdown', 'value'),
        Input('thickness-variable-dropdown', 'value'),
        Input('quantile-slider', 'value'),
        Input('step-years-slider', 'value'),
        Input('max-year-input', 'value'),
        Input('transform-mode', 'value'),
        Input('unit-scale', 'value'),
        Input('ma-months-slider', 'value'), # New input for moving average months
        Input('thickness-mode-toggle', 'value'), # New input for thickness mode
        Input('node-thick', 'value'),
        Input('node-pad', 'value'),
        Input('winsor-slider', 'value'),
        Input('regn-sample-size-input', 'value'),
        Input('power-exponent-slider', 'value'), # New input
        Input('recompute-layout-button', 'n_clicks'), # New input for recompute layout button
    )
    def update(size_variable, thickness_variable, n_quantiles, step_years, max_year_input, transform_mode, unit_scale_sel, ma_months, thickness_mode, node_thick, node_pad, winsor_pct, regn_sample_size, power_exponent, n_clicks): # New parameter
        import traceback
        print(f"DEBUG: Update callback triggered with size_var={size_variable}, thick_var={thickness_variable}")
        print(f"DEBUG: All parameters - n_q={n_quantiles}, step={step_years}, ma={ma_months}, transform={transform_mode}")
        if not size_variable:
            return {}, "No size variable selected", ""
        if not thickness_variable:
            return {}, "No thickness variable selected", ""
        # Apply selected parameters into the data module (mutate module-level config)
        try:
            dd.STEP_YEARS = int(step_years) if step_years is not None else dd.STEP_YEARS
            dd.MAX_YEAR_OVERRIDE = int(max_year_input) if (max_year_input is not None and str(max_year_input).strip() != '') else None
            dd.TRANSFORM_MODE = str(transform_mode) if transform_mode is not None else dd.TRANSFORM_MODE
            dd.MA_MONTHS = int(ma_months) if ma_months is not None else dd.MA_MONTHS # New config for moving average months
            dd.NODE_THICK = int(node_thick) if node_thick is not None else dd.NODE_THICK
            dd.NODE_PAD = int(node_pad) if node_pad is not None else dd.NODE_PAD
            dd.UNIT_SCALE = 1e9 if unit_scale_sel == 'bln' else 1.0
            dd.REGN_SAMPLE_SIZE = int(regn_sample_size) if regn_sample_size is not None else dd.REGN_SAMPLE_SIZE
            dd.POWER_EXPONENT = float(power_exponent) if power_exponent is not None else dd.POWER_EXPONENT # New config
        except Exception as e:
            return {}, f"Failed to apply parameters: {e}", ""

        # Compute sankey (size variable -> buckets, thickness variable -> link thickness)
        try:
            fig, meta, links_df = dd.compute_sankey_for_variables(
                size_var=size_variable,
                thick_var=thickness_variable,
                n_q=n_quantiles,
                ma=ma_months, # Use the parameter directly instead of dd.MA_MONTHS
                step=dd.STEP_YEARS,
                max_y=dd.MAX_YEAR_OVERRIDE,
                win=winsor_pct,
                t_mode=transform_mode,
                regn_sample_size=dd.REGN_SAMPLE_SIZE,
                power_exponent=dd.POWER_EXPONENT,
                thickness_mode='percentage' if thickness_mode and 'percentage' in thickness_mode else 'absolute'
            )
            # Convert meta object to string for display
            meta_text = (
                f"Size variable: {size_variable}\n"
                f"Thickness variable: {thickness_variable}\n"
                f"Years: {meta.get('years', [])}\n"
                f"Applied: STEP_YEARS={dd.STEP_YEARS}, MAX_YEAR_OVERRIDE={dd.MAX_YEAR_OVERRIDE}, "
                f"TRANSFORM_MODE={dd.TRANSFORM_MODE}, MA_MONTHS={ma_months}, UNIT_SCALE={dd.UNIT_SCALE}, NODE_THICK={dd.NODE_THICK}, NODE_PAD={dd.NODE_PAD}, POWER_EXPONENT={dd.POWER_EXPONENT}\n"
                f"Debug Links (first 5): {links_df[['level_from', 'level_to']].head(20).to_dict('records') if not links_df.empty else 'No links'}"
            )
            return fig, "", meta_text
        except Exception as exc:
            tb = traceback.format_exc()
            short_tb = "\\n".join(tb.splitlines()[:12])
            return {}, f"Error computing Sankey: {exc}", f"Traceback (first lines):\\n{short_tb}"

    @app.callback(
        Output('export-status', 'children'),
        Input('save-svg-button', 'n_clicks'),
        State('sankey-graph', 'figure'),
        prevent_initial_call=True
    )
    def export_svg(n_clicks, figure):
        print(f"DEBUG: export_svg called with n_clicks={n_clicks}, figure is None: {figure is None}")

        if n_clicks > 0:
            print(f"DEBUG: Button clicked {n_clicks} times")

            if figure:
                data = figure.get('data', [])
                data_length = len(data) if hasattr(data, '__len__') else 0
                print(f"DEBUG: Figure exists, data length: {data_length}")

                try:
                    # Create filename with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"sankey_export_{timestamp}.svg"
                    print(f"DEBUG: Default filename: {default_filename}")

                    # Get the figure's actual layout dimensions for proper aspect ratio
                    layout_width = figure.get('layout', {}).get('width')
                    layout_height = figure.get('layout', {}).get('height')

                    # Fallback to default dimensions if not set
                    if not layout_width:
                        layout_width = 1700  # Default width
                    if not layout_height:
                        layout_height = 720  # Default height

                    print(f"DEBUG: Figure layout dimensions: {layout_width}x{layout_height}")

                    # Convert figure to SVG bytes using to_image with exact dimensions
                    svg_bytes = pio.to_image(
                        figure,
                        format='svg',
                        width=layout_width,
                        height=layout_height,
                        scale=2  # Increase scale for better quality
                    )
                    print(f"DEBUG: SVG generated, size: {len(svg_bytes)} bytes")
                    print(f"DEBUG: Using dimensions: {layout_width}x{layout_height} with scale=2")

                    # Create data URL for download
                    import base64
                    svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')

                    # Create download link that triggers browser download
                    download_link = html.A(
                        "Click here to download SVG",
                        href=f"data:image/svg+xml;base64,{svg_b64}",
                        download=default_filename,
                        style={
                            'color': '#007cba',
                            'textDecoration': 'underline',
                            'fontWeight': 'bold',
                            'display': 'inline-block',
                            'marginLeft': '8px'
                        }
                    )

                    print(f"DEBUG: Download link created successfully")

                    return html.Div([
                        html.Span("✅ SVG ready for download: ", style={'color': '#28a745'}),
                        download_link
                    ])
                except Exception as e:
                    print(f"DEBUG: Export failed with error: {str(e)}")
                    import traceback
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    return html.Div([
                        html.Span(f"❌ Export failed: {str(e)}", style={'color': '#dc3545'})
                    ], style={'marginTop': '8px'})
            else:
                print("DEBUG: Figure is None or empty")
                return html.Div([
                    html.Span("❌ No figure data available", style={'color': '#dc3545'})
                ], style={'marginTop': '8px'})

        print("DEBUG: Initial call or no clicks")
        return ""



    return app

app = create_app()

if __name__ == '__main__':
    print(f"Dashboard ready. Run this script to start the Dash server (host={DASH_HOST}, port={DASH_PORT}).")
    app.run(host=DASH_HOST, port=DASH_PORT, debug=False)
