# -*- coding: utf-8 -*-
"""Rev15 -> Rev16: redraw the shared BB-mask channel as an explicit fan-out.
Before: a dotted line started loose on the ANPM block border, ran under the 'BB Mask (shared)' text, up into the
diffusion-pass concat and (via an invisible junction plus a degenerate stub edge) down into the GAN-pass concat, so it
read as one double-headed arrow. After: dotted trunk from the ANPM block border to a visible junction dot, two dotted
arrows from the dot (up into the diffusion-pass concat, down into the GAN-pass concat), arrowheads only at the concats,
label 'BB Mask (shared)' below the trunk, clear of every line. Two stray zero-length edges are removed."""
import io, os, sys, re, html, shutil
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
D = "C:/Dropbox/Vasile Marian Dropbox/Vasile Marian/iMatrix_Business/Phd/FDGAN_Paper_IVC"
SRC = D + "/fdgan_editable_Rev15.drawio"; DST = D + "/fdgan_editable_Rev16.drawio"
COPY = ("C:/Dropbox/Vasile Marian Dropbox/Vasile Marian/iMatrix_Business/Phd/Progress_Review_3/Papers/Latest_Current/"
        "FDGAN_IVC/10_Revision_Workplan/Step48_figure2/fdgan_editable_Rev16.drawio")
assert not os.path.exists(DST), "Rev16 exists"
root = ET.fromstring(io.open(SRC, encoding="utf-8").read())
cells = {c.get("id"): c for c in root.iter("mxCell")}; n0 = len(cells)
parent_of = {c: p for p in root.iter() for c in p}
strip = lambda v: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(v or ""))).strip()
def by_suffix(s):
    m = [k for k in cells if k.endswith(s)]; assert len(m) == 1, (s, m); return m[0]
JUNC, UP, DOWN = "AVFs8ge52PxbxqRaAwgc-34", "qG9IAKI1T2MbTa6LQLkF-5", "AVFs8ge52PxbxqRaAwgc-27"
LABEL, TOP, BOT = "AVFs8ge52PxbxqRaAwgc-12", "qG9IAKI1T2MbTa6LQLkF-2", "AVFs8ge52PxbxqRaAwgc-4"
ANPM = by_suffix("IMbT2-22")
assert strip(cells[LABEL].get("value")) == "BB Mask (shared)" and "swimlane" in cells[ANPM].get("style")
for cid in (TOP, BOT): assert "summing_function" in cells[cid].get("style")
JY = 202.0                       # y of the trunk / junction; concat centres are x = 623.5 (both at x 616, w 15)
# 1. remove the degenerate stub and two stray zero-length edges
for cid in ("AVFs8ge52PxbxqRaAwgc-35", "AVFs8ge52PxbxqRaAwgc-20", "AVFs8ge52PxbxqRaAwgc-21"):
    c = cells[cid]; assert c.get("edge") == "1" and not strip(c.get("value"))
    parent_of[c].remove(c); print("removed edge", cid)
# 2. junction: visible 6 px dot centred at (623.5, JY)
j = cells[JUNC]; g = j.find("mxGeometry")
g.set("x", "613.5"); g.set("y", str(JY - 10)); g.set("width", "20"); g.set("height", "20")
j.set("style", "shape=waypoint;sketch=0;fillStyle=solid;size=6;pointerEvents=1;points=[];fillColor=#330000;"
               "strokeColor=#330000;strokeWidth=1;resizable=0;rotatable=0;perimeter=centerPerimeter;snapToPoint=1;")
print("junction -> visible dot at (623.5, %.1f)" % JY)
DOT = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;strokeColor=#330000;dashed=1;dashPattern=1 1;"
def reset_geometry(e):
    g = e.find("mxGeometry")
    for ch in list(g): g.remove(ch)
    for k in list(g.attrib):
        if k not in ("relative", "as"): del g.attrib[k]
    g.set("relative", "1")
# 3. arrow up into the diffusion-pass concat
e = cells[UP]; e.set("source", JUNC); e.set("target", TOP); reset_geometry(e)
e.set("style", DOT + "entryX=0.5;entryY=1;entryDx=0;entryDy=0;entryPerimeter=0;startArrow=none;endArrow=classic;")
print("edge", UP, "-> junction to top concat, straight, arrowhead at the concat")
# 4. arrow down into the GAN-pass concat
e = cells[DOWN]; e.set("source", JUNC); e.set("target", BOT); reset_geometry(e)
e.set("style", DOT + "entryX=0.5;entryY=0;entryDx=0;entryDy=0;entryPerimeter=0;startArrow=none;endArrow=classic;")
print("edge", DOWN, "-> junction to bottom concat, straight, arrowhead at the concat")
# 5. trunk: ANPM block right border -> junction, no arrowheads
ag = cells[ANPM].find("mxGeometry"); ay, ah = float(ag.get("y")), float(ag.get("height")); ex = (JY - ay) / ah
layer = parent_of[cells[DOWN]]
t = ET.SubElement(layer, "mxCell", {"id": "Rev16-mask-trunk", "edge": "1", "parent": cells[DOWN].get("parent"),
                                    "source": ANPM, "target": JUNC,
                                    "style": DOT + "exitX=1;exitY=%.4f;exitDx=0;exitDy=0;startArrow=none;endArrow=none;" % ex})
ET.SubElement(t, "mxGeometry", {"relative": "1", "as": "geometry"})
print("new trunk edge from ANPM block border (x=%.0f, y=%.0f) to the junction" % (float(ag.get("x")) + float(ag.get("width")), JY))
# 6. label below the trunk, between the block border (561) and the vertical line (623.5)
l = cells[LABEL]; g = l.find("mxGeometry"); before = {k: g.get(k) for k in ("x", "y", "width", "height")}
g.set("x", "562"); g.set("y", str(JY + 5)); g.set("width", "58"); g.set("height", "30")
l.set("style", "text;whiteSpace=wrap;html=1;fontSize=11;")
print("label 'BB Mask (shared)':", before, "-> x=562 y=%.0f w=58 h=30, fontSize 11" % (JY + 5))
out = ET.tostring(root, encoding="unicode")
io.open(DST, "w", encoding="utf-8", newline="\n").write(out)
os.makedirs(os.path.dirname(COPY), exist_ok=True); shutil.copy2(DST, COPY)
r2 = ET.fromstring(io.open(DST, encoding="utf-8").read()); n2 = len(list(r2.iter("mxCell")))
assert n2 == n0 - 3 + 1 and 'math="1"' in out, (n0, n2)
print("written", DST, "and copy;", n0, "->", n2, "cells; math flag kept; Rev15 kept on disk")
