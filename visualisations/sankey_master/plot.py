from __future__ import annotations
import os
import json
from typing import Optional

# Defaults (edit if needed)
DEFAULT_DATA_JSON = 'sankey_data.json'          # created by data_prep.py (in same dir)
DEFAULT_OUT_DIR   = os.path.join('..', 'html')  # relative to this script dir -> visualisations/html
DEFAULT_OUT_FILE  = 'sankey_ownership_4y_ma_LASTYEAR_RAW_fix.html'
POSTSCRIPT_JS     = 'postscript.js'             # path (relative to sankey_master/) written earlier
PLOTLY_CDN        = 'https://cdn.plot.ly/plotly-latest.min.js'


def build_html(
    data_json_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_file: Optional[str] = None,
    postscript_js_path: Optional[str] = None,
) -> str:
    """
    Build an HTML file that stores sankey JSON into localStorage and then
    calls the external postscript.js which reads it and renders the plot.

    data_json_path: path to sankey_data.json (defaults to sankey_master/sankey_data.json)
    out_dir: directory to write the HTML (defaults to visualisations/html)
    out_file: filename for HTML output
    postscript_js_path: path to postscript.js (defaults to sankey_master/postscript.js)
    """
    base_dir = os.path.dirname(__file__)  # visualisations/sankey_master
    data_json_path = data_json_path or os.path.join(base_dir, DEFAULT_DATA_JSON)
    out_dir = out_dir or os.path.normpath(os.path.join(base_dir, DEFAULT_OUT_DIR))
    out_file = out_file or DEFAULT_OUT_FILE
    postscript_js_path = postscript_js_path or os.path.join(base_dir, POSTSCRIPT_JS)

    if not os.path.exists(data_json_path):
        raise FileNotFoundError(f"data json not found: {data_json_path}")
    if not os.path.exists(postscript_js_path):
        raise FileNotFoundError(f"postscript.js not found: {postscript_js_path}")

    # read data (so we can inline it safely)
    with open(data_json_path, 'r', encoding='utf8') as fh:
        data = json.load(fh)

    # ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_file)

    # compute relative path from HTML file to postscript.js
    rel_postscript = os.path.relpath(postscript_js_path, start=out_dir)

    # Build HTML:
    # - include Plotly CDN
    # - include a script that sets localStorage['sankey_data'] = <JSON object>
    # - include postscript.js via a script tag (relative path)
    # - call renderSankeyFromLocalStorage() on load
    html_parts = []
    html_parts.append('<!doctype html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head>')
    html_parts.append('  <meta charset="utf-8">')
    html_parts.append('  <meta name="viewport" content="width=device-width, initial-scale=1">')
    html_parts.append(f'  <title>{data.get("title","Sankey")}</title>')
    html_parts.append('  <style>html,body{height:100%;margin:0}#sankey_vp{width:100vw;height:100vh;}</style>')
    html_parts.append(f'  <script src="{PLOTLY_CDN}"></script>')
    html_parts.append('</head>')
    html_parts.append('<body>')
    # The plot div will be created by the postscript if missing; keep an empty container to match previous behaviour
    html_parts.append('  <div id="sankey_vp"></div>')
    # Inline data: we inject as a JS literal to avoid issues with quoting; JSON produced by python is valid JS
    json_literal = json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':'))
    html_parts.append('  <script>')
    html_parts.append('    // Store sankey data into localStorage for the renderer (postscript.js) to consume')
    html_parts.append('    (function(){')
    html_parts.append(f'      var SANKEY_DATA = {json_literal};')
    html_parts.append("      try { localStorage.setItem('sankey_data', JSON.stringify(SANKEY_DATA)); } catch(e) { console.error('Failed to write sankey_data to localStorage', e); }")
    html_parts.append('    })();')
    html_parts.append('  </script>')
    # include postscript.js
    html_parts.append(f'  <script src="{rel_postscript}"></script>')
    # call renderer
    html_parts.append('  <script>')
    html_parts.append('    // Kick off render once scripts have loaded')
    html_parts.append("    (function(){")
    html_parts.append("      function start(){ if(window.renderSankeyFromLocalStorage){ renderSankeyFromLocalStorage('sankey_vp'); } else { setTimeout(start, 50); } }")
    html_parts.append("      start();")
    html_parts.append("    })();")
    html_parts.append('  </script>')
    html_parts.append('</body>')
    html_parts.append('</html>')

    with open(out_path, 'w', encoding='utf8') as fh:
        fh.write('\n'.join(html_parts))

    print('Wrote HTML:', os.path.abspath(out_path))
    return os.path.abspath(out_path)


if __name__ == '__main__':
    # convenience runner when executed directly from this directory
    build_html()
