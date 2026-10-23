# -*- coding: utf-8 -*-
"""Traffic Signs box audit: JSON conditioning boxes vs YOLO label boxes vs sampler meta boxes, with contact sheets.
Usage: traffic_box_check.py <img_dir> <bb_json_dir> <yolo_label_dir> <sample_zip> <out_dir>"""
import sys, os, json, glob, zipfile, random
import numpy as np
from PIL import Image, ImageDraw

img_dir, bb_dir, lab_dir, zip_path, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
random.seed(3)

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ---- 1. JSON convention -------------------------------------------------------------------------------------------
js = sorted(glob.glob(os.path.join(bb_dir, "*.json")))
print("json files:", len(js))
keys = set(); as_xyxy_ok = 0; as_xywh_small = 0; vals = []
boxes_json = {}
for p in js:
    d = json.load(open(p))
    if isinstance(d, list):
        d = d[0]
    keys |= set(d.keys())
    x, y, w, h = [float(d[k]) for k in ("x", "y", "w", "h")]
    vals.append((x, y, w, h))
    if w > x and h > y:
        as_xyxy_ok += 1
    if w < 128 and h < 128:
        as_xywh_small += 1
    boxes_json[os.path.splitext(os.path.basename(p))[0]] = (x, y, w, h)
v = np.array(vals)
print("keys:", sorted(keys))
print("x,y,w,h min:", v.min(0).round(1), " max:", v.max(0).round(1), " mean:", v.mean(0).round(1))
print(f"files where w>x and h>y (consistent with x1,y1,x2,y2): {as_xyxy_ok}/{len(js)}")
print(f"files where w<128 and h<128 (would be typical if w,h were widths): {as_xywh_small}/{len(js)}")

# ---- 2. YOLO labels for the same stems ------------------------------------------------------------------------------
labs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(lab_dir, "*.txt"))}
common = [s for s in boxes_json if s in labs]
print("label files:", len(labs), "| stems in common with the json boxes:", len(common))
W = H = 256
n_multi = 0; ious_xyxy = []; ious_xywh = []; ious_best_xyxy = []
yolo_boxes = {}
for s in common:
    rows = [l.split() for l in open(labs[s]).read().strip().splitlines() if l.strip()]
    bxs = []
    for r in rows:
        cx, cy, bw, bh = [float(t) for t in r[1:5]]
        bxs.append((W * (cx - bw / 2), H * (cy - bh / 2), W * (cx + bw / 2), H * (cy + bh / 2)))
    yolo_boxes[s] = bxs
    if len(bxs) > 1:
        n_multi += 1
    x, y, w, h = boxes_json[s]
    j_xyxy = (x, y, w, h)
    j_xywh = (x, y, x + w, y + h)
    ious_best_xyxy.append(max(iou(j_xyxy, b) for b in bxs))
    ious_xyxy.append(iou(j_xyxy, bxs[0]))
    ious_xywh.append(max(iou(j_xywh, b) for b in bxs))
a = np.array(ious_best_xyxy); b = np.array(ious_xywh)
print(f"label files with more than one sign: {n_multi}/{len(common)}")
print(f"JSON read as x1,y1,x2,y2 vs best YOLO label box: mean IoU {a.mean():.3f}, share IoU>=0.5 {100 * (a >= 0.5).mean():.1f} %")
print(f"JSON read as x,y,w,h           vs best YOLO label box: mean IoU {b.mean():.3f}, share IoU>=0.5 {100 * (b >= 0.5).mean():.1f} %")
# image size check
im0 = Image.open(os.path.join(img_dir, os.listdir(img_dir)[0])); print("real image size:", im0.size)

# ---- 3. sampler meta boxes --------------------------------------------------------------------------------------------
with zipfile.ZipFile(zip_path) as z:
    mname = [n for n in z.namelist() if n.endswith("meta.json")][0]
    meta = json.loads(z.read(mname))
    pngs = {os.path.basename(n): n for n in z.namelist() if n.endswith(".png") and os.path.basename(n).startswith("sample_")}
    bb = np.array([s["bb_xyxy"] for s in meta["samples"]], dtype=float)
    print("meta boxes:", len(bb), "| x2>x1 and y2>y1 for all:", bool(((bb[:, 2] > bb[:, 0]) & (bb[:, 3] > bb[:, 1])).all()),
          "| width mean %.1f (min %.0f max %.0f) height mean %.1f" % ((bb[:, 2] - bb[:, 0]).mean(), (bb[:, 2] - bb[:, 0]).min(), (bb[:, 2] - bb[:, 0]).max(), (bb[:, 3] - bb[:, 1]).mean()))
    # do the meta boxes equal the JSON boxes of the named real image?
    eq = sum(1 for s in meta["samples"] if s["bb_stem"] in boxes_json and np.allclose(boxes_json[s["bb_stem"]], s["bb_xyxy"], atol=1.0))
    print(f"meta box == JSON box (read as x1,y1,x2,y2) for the named real image: {eq}/{len(meta['samples'])}")
    # contact sheet of generated images with their boxes
    picks = random.sample(meta["samples"], 8)
    sheet = Image.new("RGB", (8 * 256, 256), (255, 255, 255))
    for k, s in enumerate(picks):
        im = Image.open(z.open(pngs[s["file"]])).convert("RGB")
        dr = ImageDraw.Draw(im); x1, y1, x2, y2 = s["bb_xyxy"]; dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        sheet.paste(im, (k * 256, 0))
    sheet.save(os.path.join(out, "generated_with_conditioning_boxes.png"))

# ---- 4. real images: JSON box (red) and YOLO label boxes (green) -------------------------------------------------------
picks = random.sample(common, 8)
sheet = Image.new("RGB", (8 * 256, 256), (255, 255, 255))
for k, s in enumerate(picks):
    cands = glob.glob(os.path.join(img_dir, s + ".*"))
    im = Image.open(cands[0]).convert("RGB").resize((256, 256))
    dr = ImageDraw.Draw(im)
    x, y, w, h = boxes_json[s]
    dr.rectangle([min(x, w), min(y, h), max(x, w), max(y, h)], outline=(255, 0, 0), width=2)          # JSON read as x1,y1,x2,y2 (red)
    dr.rectangle([x, y, min(255, x + w), min(255, y + h)], outline=(0, 0, 255), width=2)              # JSON read as x,y,width,height (blue)
    for bx in yolo_boxes[s]:
        dr.rectangle([max(0, bx[0]), max(0, bx[1]), min(255, bx[2]), min(255, bx[3])], outline=(0, 255, 0), width=2)   # YOLO label (green)
    dr.text((4, 4), s[-18:], fill=(255, 255, 0))
    sheet.paste(im, (k * 256, 0))
sheet.save(os.path.join(out, "real_json_red_xyxy_blue_xywh_vs_yolo_label_green.png"))
# YOLO label box sizes in pixels (if the labels refer to these 256 px images)
sz = np.array([(b[2] - b[0], b[3] - b[1]) for s in common for b in yolo_boxes[s]])
print("YOLO label box size (px, if normalised to 256): width mean %.1f median %.1f | height mean %.1f median %.1f" % (sz[:, 0].mean(), np.median(sz[:, 0]), sz[:, 1].mean(), np.median(sz[:, 1])))
jw = np.array([(abs(w - x), abs(h - y)) for (x, y, w, h) in boxes_json.values()])
print("JSON box size read as corners (px): width mean %.1f median %.1f | height mean %.1f median %.1f" % (jw[:, 0].mean(), np.median(jw[:, 0]), jw[:, 1].mean(), np.median(jw[:, 1])))
print("sheets written to", out)
