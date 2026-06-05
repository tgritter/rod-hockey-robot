# tools/annotate_zones.py
"""Draw player-zone polygons on a saved dynamic-crop frame, in your browser.

    .venv/bin/python tools/annotate_zones.py --image <frame.jpeg> --out robot/zones.json

Starts a tiny local web server (stdlib only — no matplotlib / Qt / tkinter) and
opens a page with the frame on a canvas. The current robot/zones.json is loaded
for editing, and the legacy boxes are shown faintly for reference.

Interaction:
  - Click to drop polygon vertices on the image.
  - Pick player + side, click "Finish zone" to commit the polygon.
  - "Undo point" / "Remove last zone" to correct mistakes.
  - "Save" writes robot/zones.json (normalized to [0,1]). Vertices are normalized
    server-side via build_zone, the same function used by the seeded zones.
"""

import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Allow running directly as `python tools/annotate_zones.py` (not only `-m tools...`):
# put the repo root on sys.path so the `tools` package resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from tools.seed_zones_from_legacy import LEGACY_ZONES, REF_W, REF_H

PLAYERS = ["center", "right_wing", "left_wing", "right_d", "left_d"]
SIDES = ["left", "right", "bottom_left", "bottom_right"]


def build_zone(player, side, verts_px, width, height):
    """Return a zone dict with the vertices normalized to [0, 1] (rounded 4dp)."""
    polygon = [[round(x / width, 4), round(y / height, 4)] for (x, y) in verts_px]
    return {"player": player, "side": side, "polygon": polygon}


def _legacy_boxes_px(width, height):
    """Legacy zone boxes scaled from the reference crop to this image's pixels."""
    sx, sy = width / REF_W, height / REF_H
    boxes = []
    for z in LEGACY_ZONES:
        boxes.append({
            "player": z["player"], "side": z["side"],
            "x": z["x_min"] * sx, "y": z["y_min"] * sy,
            "w": (z["x_max"] - z["x_min"]) * sx, "h": (z["y_max"] - z["y_min"]) * sy,
        })
    return boxes


def _existing_zones_px(out_path, width, height):
    """Existing zones.json polygons converted to image-pixel vertices for editing."""
    if not os.path.exists(out_path):
        return []
    with open(out_path) as f:
        raw = json.load(f)
    out = []
    for z in raw:
        out.append({
            "player": z["player"], "side": z["side"],
            "points": [[u * width, v * height] for u, v in z["polygon"]],
        })
    return out


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>annotate zones</title>
<style>
  body { font-family: sans-serif; margin: 12px; }
  #wrap { display: flex; gap: 16px; align-items: flex-start; }
  canvas { border: 1px solid #888; cursor: crosshair; }
  #side-panel { min-width: 240px; }
  button { margin: 2px 0; }
  #zonelist { font-size: 12px; max-height: 320px; overflow:auto; border:1px solid #ccc; padding:4px; }
  .row { margin: 6px 0; }
  #status { color: #060; font-weight: bold; min-height: 18px; }
</style></head>
<body>
<h3>Zone annotator</h3>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="side-panel">
    <div class="row">
      player <select id="player"></select>
      side <select id="side"></select>
    </div>
    <div class="row">
      <button onclick="finishZone()">Finish zone</button>
      <button onclick="undoPoint()">Undo point</button>
    </div>
    <div class="row">
      <button onclick="removeLast()">Remove last zone</button>
      <button onclick="saveZones()">Save</button>
    </div>
    <div class="row"><label><input type="checkbox" id="showLegacy" checked onchange="redraw()"> show legacy boxes</label></div>
    <div id="status"></div>
    <div class="row">zones (<span id="count">0</span>):</div>
    <div id="zonelist"></div>
  </div>
</div>
<script>
const SCALE = 2;
let IMG = new Image();
let W = 0, H = 0;
let legacy = [];
let zones = [];        // committed: {player, side, points:[[x,y],...]} in image px
let current = [];      // in-progress vertices, image px
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

function fillSelect(id, opts){ const s=document.getElementById(id); opts.forEach(o=>{const e=document.createElement('option');e.value=o;e.textContent=o;s.appendChild(e);}); }
fillSelect('player', %PLAYERS%);
fillSelect('side', %SIDES%);

async function init(){
  const d = await (await fetch('/data')).json();
  W = d.width; H = d.height; legacy = d.legacy; zones = d.zones;
  canvas.width = W*SCALE; canvas.height = H*SCALE;
  IMG.onload = redraw;
  IMG.src = '/image?ts=' + Date.now();
  refreshList();
}

function centroid(pts){ let x=0,y=0; pts.forEach(p=>{x+=p[0];y+=p[1];}); return [x/pts.length, y/pts.length]; }

function redraw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (IMG.complete) ctx.drawImage(IMG, 0, 0, W*SCALE, H*SCALE);
  if (document.getElementById('showLegacy').checked){
    ctx.strokeStyle = 'rgba(0,200,255,0.5)'; ctx.lineWidth = 1;
    legacy.forEach(b => ctx.strokeRect(b.x*SCALE, b.y*SCALE, b.w*SCALE, b.h*SCALE));
  }
  zones.forEach(z => {
    drawPoly(z.points, 'rgba(255,215,0,0.95)', 'rgba(255,215,0,0.12)');
    const c = centroid(z.points);
    ctx.fillStyle = '#000'; ctx.font = '12px sans-serif';
    ctx.fillText(z.player+'/'+z.side, c[0]*SCALE+2, c[1]*SCALE);
  });
  if (current.length){
    drawPoly(current, 'rgba(255,0,0,0.95)', 'rgba(255,0,0,0.10)', true);
    current.forEach(p => { ctx.fillStyle='red'; ctx.beginPath(); ctx.arc(p[0]*SCALE,p[1]*SCALE,3,0,7); ctx.fill(); });
  }
}

function drawPoly(pts, stroke, fill, open){
  if (!pts.length) return;
  ctx.beginPath(); ctx.moveTo(pts[0][0]*SCALE, pts[0][1]*SCALE);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0]*SCALE, pts[i][1]*SCALE);
  if (!open) ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke();
}

canvas.addEventListener('click', e => {
  const r = canvas.getBoundingClientRect();
  current.push([ (e.clientX-r.left)/SCALE, (e.clientY-r.top)/SCALE ]);
  redraw();
});

function undoPoint(){ current.pop(); redraw(); }

function finishZone(){
  if (current.length < 3){ setStatus('need at least 3 points', true); return; }
  zones.push({ player: document.getElementById('player').value,
               side: document.getElementById('side').value,
               points: current });
  current = []; redraw(); refreshList(); setStatus('zone added');
}

function removeLast(){ zones.pop(); redraw(); refreshList(); }

function refreshList(){
  document.getElementById('count').textContent = zones.length;
  document.getElementById('zonelist').innerHTML =
    zones.map((z,i)=>`${i}: ${z.player}/${z.side} (${z.points.length} pts)`).join('<br>');
}

async function saveZones(){
  const r = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({zones})});
  const j = await r.json();
  setStatus(j.ok ? ('saved '+j.count+' zones to '+j.path) : ('ERROR: '+j.error), !j.ok);
}

function setStatus(msg, err){ const s=document.getElementById('status'); s.textContent=msg; s.style.color = err?'#c00':'#060'; }

init();
</script>
</body></html>
"""


def _make_handler(image_path, out_path, width, height):
    legacy = _legacy_boxes_px(width, height)
    existing = _existing_zones_px(out_path, width, height)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                page = (PAGE.replace("%PLAYERS%", json.dumps(PLAYERS))
                            .replace("%SIDES%", json.dumps(SIDES)))
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/image":
                with open(image_path, "rb") as f:
                    data = f.read()
                ctype = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                self._send(200, data, ctype)
            elif path == "/data":
                self._send(200, json.dumps({
                    "width": width, "height": height,
                    "legacy": legacy, "zones": existing,
                }))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path != "/save":
                self._send(404, json.dumps({"error": "not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                out = [build_zone(z["player"], z["side"], z["points"], width, height)
                       for z in payload["zones"]]
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                self._send(200, json.dumps({"ok": True, "count": len(out), "path": out_path}))
            except Exception as e:  # surface the error to the page
                self._send(200, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="/home/nick/Downloads/dynamic-crop-2026-06-04_16_56_57.jpeg")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "robot", "zones.json"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    out_path = os.path.abspath(args.out)

    width, height = Image.open(args.image).size

    handler = _make_handler(args.image, out_path, width, height)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Annotating {args.image} ({width}x{height}) -> {out_path}")
    print(f"Open {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
