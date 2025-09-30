/*
postscript.js
Responsible for reading sankey data from localStorage and rendering the Plotly Sankey.
Exposes a single entrypoint: renderSankeyFromLocalStorage(divId)
*/
(function (global) {
  function eqArr(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function ensureUi(gd, YEARS, setYears) {
    var host = document.getElementById('sankey_ui');
    if (!host) {
      host = document.createElement('div');
      host.id = 'sankey_ui';
      host.style.position = 'fixed';
      host.style.top = '8px';
      host.style.right = '12px';
      host.style.zIndex = '50';
      host.style.fontFamily = 'sans-serif';
      host.style.background = 'rgba(255,255,255,0.85)';
      host.style.backdropFilter = 'blur(2px)';
      host.style.padding = '6px 8px';
      host.style.border = '1px solid #ddd';
      host.style.borderRadius = '8px';
      document.body.appendChild(host);
    }
    if (!document.getElementById('save_svg_btn')) {
      var row = document.createElement('div');
      row.style.display = 'flex';
      row.style.gap = '6px';
      row.style.alignItems = 'center';

      var btn = document.createElement('button');
      btn.id = 'save_svg_btn';
      btn.textContent = 'Save SVG';
      btn.style.padding = '6px 10px';
      btn.style.border = '1px solid #ccc';
      btn.style.borderRadius = '6px';
      btn.style.background = '#fafafa';
      btn.style.cursor = 'pointer';
      btn.onclick = function () {
        var prev1 = host ? host.style.display : null;
        var modebar = document.querySelector('.modebar');
        var prev2 = modebar ? modebar.style.display : null;
        if (host) host.style.display = 'none';
        if (modebar) modebar.style.display = 'none';
        if (window.Plotly && gd) {
          Plotly.downloadImage(gd, { format: 'svg', filename: (document.title || 'sankey').replace(/\s+/g,'_') })
            .finally(function () {
              if (host) host.style.display = prev1;
              if (modebar) modebar.style.display = prev2;
            });
        }
      };
      row.appendChild(btn);

      var form = document.createElement('div');
      form.id = 'year_toggles';
      form.style.display = 'flex';
      form.style.gap = '8px';
      form.style.flexWrap = 'wrap';
      form.style.maxWidth = '32vw';
      YEARS.forEach(function (y) {
        var label = document.createElement('label');
        label.style.display = 'inline-flex';
        label.style.alignItems = 'center';
        label.style.gap = '4px';
        var cb = document.createElement('input');
        cb.type = 'checkbox'; cb.value = String(y); cb.checked = true;
        label.appendChild(cb);
        label.appendChild(document.createTextNode(String(y)));
        form.appendChild(label);
      });

      var apply = document.createElement('button');
      apply.textContent = 'Apply columns';
      apply.style.padding = '4px 8px';
      apply.style.border = '1px solid #ccc';
      apply.style.borderRadius = '6px';
      apply.style.background = '#fff';
      apply.style.cursor = 'pointer';
      apply.onclick = function () {
        var boxes = form.querySelectorAll('input[type=checkbox]');
        var keep = [];
        boxes.forEach(function (b) { if (b.checked) keep.push(parseInt(b.value)); });
        setYears(keep);
      };

      var all = document.createElement('button'); all.textContent = 'All'; all.onclick = function () { var bs=form.querySelectorAll('input'); bs.forEach(function(b){b.checked=true;}); };
      var none = document.createElement('button'); none.textContent = 'None'; none.onclick = function () { var bs=form.querySelectorAll('input'); bs.forEach(function(b){b.checked=false;}); };
      [all, none].forEach(function(b){ b.style.padding='4px 6px'; b.style.border='1px solid #ccc'; b.style.borderRadius='6px'; b.style.background='#fff'; b.style.cursor='pointer'; });

      host.appendChild(row);
      host.appendChild(document.createElement('div')).style.height='6px';
      host.appendChild(form);
      var ctl = document.createElement('div'); ctl.style.marginTop='6px'; ctl.style.display='flex'; ctl.style.gap='6px'; ctl.appendChild(apply); ctl.appendChild(all); ctl.appendChild(none);
      host.appendChild(ctl);
    }
  }

  function resize(gd) {
    document.documentElement.style.height = '100%';
    document.body.style.cssText = 'height:100%;margin:0;overflow:hidden;';
    if (gd) {
      gd.style.position = 'fixed';
      gd.style.top = '0';
      gd.style.left = '0';
      gd.style.right = '0';
      gd.style.bottom = '0';
      gd.style.width = '100vw';
      gd.style.height = '100vh';
      if (window.Plotly && gd.data) {
        Plotly.relayout(gd, {autosize:true, margin:{l:16,r:16,t:56,b:12}});
        Plotly.relayout(gd, {'sankey[0].node.x': gd.data[0].node.x, 'sankey[0].node.y': gd.data[0].node.y});
        Plotly.Plots.resize(gd);
      }
    }
    ensureUi(gd, window.__SANKEY_YEARS || [], window.__setYears || function(){});
  }

  function renderSankeyFromLocalStorage(divId) {
    divId = divId || 'sankey_vp';
    var raw = localStorage.getItem('sankey_data');
    if (!raw) {
      console.error('sankey_data not found in localStorage');
      return;
    }
    var d;
    try { d = JSON.parse(raw); } catch (err) { console.error('Failed parsing sankey_data', err); return; }

    var VALUES_ABS = d.values_abs || [];
    var VALUES_SHARE = d.values_share || [];
    var YEARS = d.years || [];
    var LINK_FROM = d.link_from || [];
    var LINK_TO = d.link_to || [];

    var gd = document.getElementById(divId);
    if (!gd) {
      gd = document.createElement('div');
      gd.id = divId;
      document.body.appendChild(gd);
    }
    // Build trace
    var trace = {
      type: 'sankey',
      arrangement: 'perpendicular',
      node: {
        pad: d.node_pad || 22,
        thickness: d.node_thick || 18,
        label: d.labels,
        customdata: d.node_customdata,
        hovertemplate: d.hovertemplate || '%{customdata[0]}: %{customdata[1]}<extra></extra>',
        color: d.node_color,
        x: d.node_x,
        y: d.node_y,
        line: { width: 0.5, color: 'rgba(0,0,0,0.3)' }
      },
      link: {
        source: d.link_source,
        target: d.link_target,
        value: VALUES_ABS.slice(), // default to abs
        color: d.link_color,
        customdata: d.link_customdata,
        hovertemplate: d.hovertemplate || ''
      }
    };

    var layout = {
      title: { text: d.title || '', x: 0.02, xanchor: 'left' },
      autosize: true,
      margin: { l: 16, r: 16, t: 56, b: 12 },
      annotations: d.annotations || []
    };

    var config = {
      responsive: true,
      displaylogo: false,
      toImageButtonOptions: {
        format: 'svg',
        filename: (d.title || 'sankey').replace(/\s+/g,'_'),
        height: null, width: null, scale: 1
      }
    };

    Plotly.newPlot(gd, [trace], layout, config).then(function () {
      // store state on gd
      gd.__VALUES_ABS = VALUES_ABS;
      gd.__VALUES_SHARE = VALUES_SHARE;
      gd.__YEARS = YEARS;
      gd.__LINK_FROM = LINK_FROM;
      gd.__LINK_TO = LINK_TO;
      gd.__useShare = false;

      // hook restyle to detect which mode is active
      if (gd && gd.on) {
        gd.on('plotly_restyle', function (e) {
          try {
            if (e && e[0] && e[0]['link.value']) {
              var arr = e[0]['link.value'][0];
              gd.__useShare = eqArr(arr, gd.__VALUES_SHARE) ? true : eqArr(arr, gd.__VALUES_ABS) ? false : gd.__useShare;
            }
          } catch (err) { /* ignore */ }
        });
      }

      // expose setYears
      function setYears(keepYears) {
        if (!gd || !gd.data || !gd.data[0]) return;
        var tr = gd.data[0];
        var keepSet = (keepYears && keepYears.length ? new Set(keepYears) : new Set(YEARS));
        window.__yearsKeep = keepSet;
        var labels = tr.node.label.slice();
        var colors = tr.node.color.slice();
        for (var i = 0; i < tr.node.customdata.length; i++) {
          var yr = tr.node.customdata[i][0];
          if (!keepSet.has(yr)) {
            labels[i] = '';
            colors[i] = 'rgba(0,0,0,0)';
          }
        }
        var useShare = (gd.__useShare === true) ? true : (gd.__useShare === false ? false : false);
        var baseVals = useShare ? gd.__VALUES_SHARE : gd.__VALUES_ABS;
        var masked = baseVals.map(function (v, idx) {
          var lf = gd.__LINK_FROM[idx];
          var lt = gd.__LINK_TO[idx];
          return (keepSet.has(lf) && keepSet.has(lt)) ? v : 0;
        });
        Plotly.restyle(gd, { 'node.label': [labels], 'node.color': [colors], 'link.value': [masked] }, [0]);
      }

      window.__setYears = setYears;
      window.__SANKEY_YEARS = YEARS;

      ensureUi(gd, YEARS, setYears);
      window.addEventListener('resize', function () { resize(gd); });
      setTimeout(function () { resize(gd); }, 0);
    });
  }

  // expose
  global.renderSankeyFromLocalStorage = renderSankeyFromLocalStorage;
})(window);
