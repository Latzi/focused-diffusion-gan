# -*- coding: utf-8 -*-
"""Export a .drawio file to a cropped, single-page vector PDF with headless Chrome: the diagrams.net viewer embed renders
the diagram as SVG (MathJax included) in a local HTML page whose @page size equals the diagram bounds, and Chrome prints
it. Usage: drawio_pdf.py in.drawio out.pdf"""
import io, os, sys, json, html, math, subprocess, time
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
src, out = sys.argv[1], sys.argv[2]
CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
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
BORDER = 8
W = int(math.ceil(bx1 - bx0 + 2 * BORDER)); H = int(math.ceil(by1 - by0 + 2 * BORDER))
cfg = {"highlight": "#0000ff", "nav": False, "resize": True, "toolbar": "", "zoom": 1, "border": BORDER, "xml": xml}
page = ('<!DOCTYPE html><html><head><meta charset="utf-8"><style>@page{size:%dpx %dpx;margin:0}'
        'html,body{margin:0;padding:0;background:#ffffff;width:%dpx;height:%dpx;overflow:hidden}</style></head><body>'
        '<div class="mxgraph" style="border:0;" data-mxgraph="' % (W, H, W, H) + html.escape(json.dumps(cfg), quote=True) + '"></div>'
        '<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
hp = os.path.splitext(out)[0] + "_print.html"; io.open(hp, "w", encoding="utf-8").write(page)
prof = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chrome_profile")
cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--user-data-dir=" + prof,
       "--virtual-time-budget=30000", "--no-pdf-header-footer", "--print-to-pdf=" + out, "file:///" + hp.replace(os.sep, "/")]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
tail = [l for l in (r.stderr or "").splitlines() if "external_registry" not in l]
print("chrome exit", r.returncode, "|", tail[-1][:160] if tail else "")
[time.sleep(1) for _ in range(60) if not os.path.exists(out)]
assert os.path.exists(out), "no pdf"
from pypdf import PdfReader
rd = PdfReader(out); pg = rd.pages[0]; mb = pg.mediabox
print("pdf pages", len(rd.pages), "| page size %.1f x %.1f pt = %.0f x %.0f px (expected %d x %d px) | images on page: %d | file %d bytes"
      % (float(mb.width), float(mb.height), float(mb.width) / 0.75, float(mb.height) / 0.75, W, H, len(pg.images), os.path.getsize(out)))
txt = pg.extract_text() or ""
print("text sample:", " ".join(txt.split())[:200])
