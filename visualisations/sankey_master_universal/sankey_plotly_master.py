"""
Orchestrator for the refactored Sankey pipeline.

This script runs:
  1) data_prep.generate(...) -> writes sankey_data.json into this directory
  2) plot.build_html(...)     -> writes HTML into visualisations/html (by default)
The HTML produced stores the sankey JSON into localStorage and loads postscript.js
(which reads the JSON from localStorage and renders the Plotly Sankey).

Usage:
  python visualisations/sankey_master/sankey_plotly_master.py

You can re-run the script whenever you update parameters inside data_prep.py.
"""
from __future__ import annotations
import os
import sys

# Make local module imports reliable when executing from project root
HERE = os.path.dirname(__file__) or '.'
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import data_prep
    import plot as plot_mod
except Exception as exc:
    raise ImportError("Failed to import refactored modules. Ensure this file is in "
                      "visualisations/sankey_master/ and that data_prep.py and plot.py exist.") from exc

def main():
    # Paths (all anchored to this folder)
    data_json = os.path.join(HERE, 'sankey_data.json')
    postscript = os.path.join(HERE, 'postscript.js')
    out_dir = os.path.normpath(os.path.join(HERE, '..', 'html'))

    print("Starting Sankey pipeline (data_prep -> build_html)...")
    print(f"Data JSON will be written to: {data_json}")
    print(f"postscript.js expected at: {postscript}")
    print(f"HTML output directory: {out_dir}")

    # 1) Generate data JSON
    # Let data_prep use its own DATA_PATH default; avoid passing None which can trigger typing issues.
    data_out = data_prep.generate(out_json=data_json)
    print("Data prep completed. JSON:", data_out)

    # 2) Build HTML which places JSON into localStorage and includes postscript.js
    html_out = plot_mod.build_html(
        data_json_path=data_out,
        out_dir=out_dir,
        out_file=getattr(plot_mod, 'DEFAULT_OUT_FILE', 'sankey_ownership_4y_ma_LASTYEAR_RAW_fix.html'),
        postscript_js_path=postscript
    )
    print("HTML generated:", html_out)
    print("Pipeline finished. Open the HTML file in a browser (file://) to view the visualization.")
    print("Note: the renderer reads sankey JSON from localStorage (key: 'sankey_data') and renders via postscript.js.")

if __name__ == '__main__':
    main()
