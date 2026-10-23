# -*- coding: utf-8 -*-
"""Render a .drawio file to PNG with headless Edge via the diagrams.net viewer embed (the diagram XML stays inside the
local HTML page; only the viewer script and MathJax are fetched).
Usage: render_drawio.py in.drawio out.png zoom [x0,y0,x1,y1 crop in diagram units]"""
import io, os, sys, json, html, math, subprocess
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
src, out, zoom = sys.argv[1], sys.argv[2], float(sys.argv[3])
crop = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
EDGE = "C:/Program Files/Google/Chrome/Application/chrome.exe"
xml = io.open(src, encoding="utf-8").read()
root = ET.fromstring(xml); xs, ys = [], []
for c in root.iter("mxCell"):
    g = c.find("mxGeometry")
    if g is None: continue
    if c.get("vertex") == "1" and g.get("x") is not None:
        x, y, w, h = (float(g.get(k, 0)) for k in ("x", "y", "width", "height")); xs += [x, x + w]; ys += [y, y + h]
    for p in g.iter("mxPoint"):
        if p.get("x"): xs.append(float(p.get("x"))); ys.append(float(p.get("y")))
bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
BORDER = 10
cfg = {"highlight": "#0000ff", "nav": False, "resize": True, "toolbar": "", "zoom": zoom, "border": BORDER, "xml": xml}
page = ('<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:0;background:#ffffff}</style></head><body>'
        '<div class="mxgraph" style="border:0;" data-mxgraph="' + html.escape(json.dumps(cfg), quote=True) + '"></div>'
        '<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
hp = os.path.splitext(out)[0] + ".html"; io.open(hp, "w", encoding="utf-8").write(page)
W = int(math.ceil((bx1 - bx0) * zoom + 2 * BORDER + 40)); H = int(math.ceil((by1 - by0) * zoom + 2 * BORDER + 40))
prof = os.path.join(os.path.dirname(out), "edge_profile")
cmd = [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
       "--user-data-dir=" + prof, "--virtual-time-budget=30000", "--window-size=%d,%d" % (W, H), "--screenshot=" + out,
       "file:///" + hp.replace(os.sep, "/")]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
tail = (r.stderr or "").strip().splitlines()
print("edge exit", r.returncode, "|", tail[-1][:160] if tail else "")
import time; [time.sleep(1) for _ in range(60) if not os.path.exists(out)]; assert os.path.exists(out), "no screenshot"
from PIL import Image
im = Image.open(out)
print("rendered", out, im.size, "diagram bounds x %.0f..%.0f y %.0f..%.0f zoom %.1f" % (bx0, bx1, by0, by1, zoom))
if crop:
    x0, y0, x1, y1 = crop
    px = lambda x: int((x - bx0) * zoom + BORDER); py = lambda y: int((y - by0) * zoom + BORDER)
    box = (max(0, px(x0)), max(0, py(y0)), min(im.size[0], px(x1)), min(im.size[1], py(y1)))
    cp = os.path.splitext(out)[0] + "_crop.png"; im.crop(box).save(cp); print("crop", box, "->", cp)
