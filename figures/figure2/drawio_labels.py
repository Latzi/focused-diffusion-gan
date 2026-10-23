# -*- coding: utf-8 -*-
"""List every text label of a .drawio file (handles the compressed <diagram> format)."""
import sys, io, re, base64, zlib, html
from urllib.parse import unquote
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
p = sys.argv[1]
raw = io.open(p, encoding="utf-8").read()
root = ET.fromstring(raw)
diagrams = root.findall(".//diagram")
print(f"{p}\n  {len(raw)} chars, {len(diagrams)} diagram(s)")
for d in diagrams:
    if len(list(d)) == 0 and (d.text or "").strip():   # compressed
        data = base64.b64decode(d.text.strip())
        xml_txt = unquote(zlib.decompress(data, -15).decode("utf-8"))
        model = ET.fromstring(xml_txt); fmt = "compressed"
    else:
        model = d.find("mxGraphModel"); fmt = "plain"
    cells = model.findall(".//mxCell")
    print(f"  diagram {d.get('name')!r}: {fmt}, {len(cells)} cells")
    for c in cells:
        v = c.get("value")
        if v:
            txt = re.sub(r"<[^>]+>", " ", html.unescape(v)); txt = re.sub(r"\s+", " ", txt).strip()
            if txt: print(f"    id={c.get('id'):>6s} | {txt[:140]}")
