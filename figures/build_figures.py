# -*- coding: utf-8 -*-
"""Revision figures from figure_samples_23Oct.zip (first K samples by index of each set; sample i = box i, identical across sets).
Outputs in 10_Revision_Workplan/Step45_figures/:
  Fig3_matched_3x8.png        main Figure 3, one row per dataset, two box-matched quartets (Real+box | ADM | FDGAN* | FDGAN) per row
  Fig3_matched_6x4.png        same content, two rows per dataset, four columns
  S7_<dataset>_<gan|diff>.jpg the six old supplement grids with the Original / FDGAN (/ ADM) columns replaced by the new matched samples
  S7_ablation_grid.png        Cityscapes ablation models, box-matched rows
  S7_seeds_guidance_grid.png  Cityscapes seeds and guidance, box-matched rows
Rule: rows use sample indices 0, 1, 2, ... of every set (deterministic); boxes drawn in red on every panel of the matched grids."""
import os, io, json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.stdout.reconfigure(encoding="utf-8")
SP = "C:/Users/latzi/AppData/Local/Temp/claude/C--Users-latzi-Downloads-New-folder--5-/2874049a-ada9-431d-826d-b7ea81ba130f/scratchpad"
FS = os.path.join(SP, "figsamples")   # figure_samples_23Oct.zip = boxes 0-11 of every set (lot A); the second extraction (figure_samples_lotA.zip) carried wrong files for the four FDGAN sets and is NOT used
IM = "C:/Dropbox/Vasile Marian Dropbox/Vasile Marian/iMatrix_Business/Phd/Progress_Review_3/Papers/Latest_Current/FDGAN_IVC/11_Overleaf_Source/Overleaf_current_22Oct/images"
OUT = "C:/Dropbox/Vasile Marian Dropbox/Vasile Marian/iMatrix_Business/Phd/Progress_Review_3/Papers/Latest_Current/FDGAN_IVC/10_Revision_Workplan/Step45_figures"
os.makedirs(OUT, exist_ok=True)
FONT = "C:/Windows/Fonts/timesbd.ttf"
idx = json.load(open(os.path.join(FS, "index.json")))
REAL = {ds: {os.path.splitext(f)[0]: os.path.join(FS, ds, "real", f) for f in os.listdir(os.path.join(FS, ds, "real"))} for ds in ("Cityscapes", "Traffic", "COCO")}
SETS = {
    "Cityscapes": {"ADM": "cityscapes_adm_ema200000_n3000_r250_colsample_anc_lv", "FDGAN*": "cityscapes_adm_mask_ema200000_n3000_r500_colsample_anc500_cfg20",
                   "FDGAN": "cityscapes_fdgan_mask_ema200000_n3000_r500_colsample_anc500_cfg20",
                   "nonoise": "cityscapes_fdgan_nonoise_ema200000_n3000_r500_colsample_anc500_cfg20", "global": "cityscapes_fdgan_globalnoise_ema200000_n3000_r500_colsample_anc500_cfg20",
                   "smallD": "cityscapes_fdgan_mask_smallD_ema200000_n3000_r500_colsample_anc500_cfg20", "nomask": "cityscapes_fdgan_nomask_ema200000_n3000_r500_colsample_anc500",
                   "FDGAN*_s2": "cityscapes_adm_mask_rep2_ema200000_n3000_r500_colsample_anc500_cfg20", "FDGAN_s2": "cityscapes_fdgan_mask_rep2_ema200000_n3000_r500_colsample_anc500_cfg20",
                   "FDGAN*_nocfg": "cityscapes_adm_mask_ema200000_n3000_r500_colsample_anc500_nocfg", "FDGAN_nocfg": "cityscapes_fdgan_mask_ema200000_n3000_r500_colsample_anc500_nocfg"},
    "Traffic": {"ADM": "traffic_adm_ema200000_n3000_r250_colsample_anc_lv", "FDGAN*": "traffic2_adm_mask_ema200000_n3000_r500_colsample_anc500_cfg20", "FDGAN": "traffic2_fdgan_mask_ema200000_n3000_r500_colsample_anc500_cfg20"},
    "COCO": {"ADM": "coco_adm_lsun_ema200000_n3000_r250_colsample_anc_lv", "FDGAN*": "coco_adm_mask_lsun_ema200000_n3000_r500_colsample_anc500_cfg20", "FDGAN": "coco_fdgan_mask_lsun_ema200000_n3000_r500_colsample_anc500_cfg20"},
}
DSNAME = {"Cityscapes": "Cityscapes--Pedestrian", "Traffic": "Traffic Signs", "COCO": "COCO potted plant"}
RED = (230, 30, 30)

def rec(ds, key, i):
    s = SETS[ds][key]
    return next(r for r in idx if r["dataset"] == ds and r["set"] == s and r.get("row", r["index"]) == i)
def panel(ds, key, i, size, box=True, width=3):
    """key = 'real' or a SETS key; returns a PIL image of the given size with the conditioning box drawn."""
    r0 = rec(ds, "FDGAN", i); x1, y1, x2, y2 = r0["bb_xyxy"]
    if key == "real": im = Image.open(REAL[ds][r0["bb_stem"]]).convert("RGB")
    else:
        rr = rec(ds, key, i)
        if key != "ADM":   # ADM is unconditional: its i-th sample is shown with the box of the matched quartet drawn for reference
            assert rr["bb_stem"] == r0["bb_stem"] and rr["bb_xyxy"] == r0["bb_xyxy"], (ds, key, i)
        im = Image.open(os.path.join(FS, rr["file"])).convert("RGB")
    W0 = im.size[0]; im = im.resize(size, Image.LANCZOS)
    if box:
        s = size[0] / W0; d = ImageDraw.Draw(im)
        d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=RED, width=width)
    return im
def text_w(font, t): b = font.getbbox(t); return b[2] - b[0]

def grid(blocks, cols, nrows, P=256, G=6, header=44, left=0, rowlabels=None, font_size=28, sep_after=None, box=True):
    """blocks: list of (ds, key) per column (per row index i); cols: header labels; rowlabels: optional text per row (rotated)."""
    font = ImageFont.truetype(FONT, font_size)
    ncol = len(cols); extra = (len(sep_after) * 10) if sep_after else 0
    Wd = left + G + ncol * (P + G) + extra; Hd = header + nrows * (P + G)
    sheet = Image.new("RGB", (Wd, Hd), "white"); d = ImageDraw.Draw(sheet)
    xs = []; x = left + G
    for c in range(ncol):
        xs.append(x); x += P + G
        if sep_after and c in sep_after: x += 10
    for c, name in enumerate(cols):
        d.text((xs[c] + (P - text_w(font, name)) / 2, (header - font_size) / 2 - 2), name, fill="black", font=font)
    for i in range(nrows):
        y = header + i * (P + G)
        for c, b in enumerate(blocks[i]):
            ds, key = b[0], b[1]; ii = b[2] if len(b) > 2 else i
            sheet.paste(panel(ds, key, ii, (P, P), box=box), (xs[c], y))
    if rowlabels:
        for i, lab in enumerate(rowlabels):
            if not lab: continue
            tw = text_w(font, lab) + 10
            t = Image.new("RGB", (tw, font_size + 12), "white"); ImageDraw.Draw(t).text((5, 2), lab, fill="black", font=font)
            t = t.rotate(90, expand=True); y = header + i * (P + G) + (P - t.size[1]) // 2
            sheet.paste(t, ((left - t.size[0]) // 2, y))
    return sheet

# ---------------------------------------------------------------- Figure 3, layout A: 3 rows x 8 columns (two quartets per dataset)
quart = ["Real (box)", "ADM", "FDGAN*", "FDGAN"]
blocksA = [[(ds, "real", 0), (ds, "ADM", 0), (ds, "FDGAN*", 0), (ds, "FDGAN", 0), (ds, "real", 1), (ds, "ADM", 1), (ds, "FDGAN*", 1), (ds, "FDGAN", 1)] for ds in ("Cityscapes", "Traffic", "COCO")]
figA = grid(blocksA, quart + quart, 3, P=256, G=6, header=44, left=56, rowlabels=["Cityscapes", "Traffic Signs", "COCO potted plant"], font_size=26, sep_after=[3])
figA.save(os.path.join(OUT, "Fig3_matched_3x8.png")); print("Fig3_matched_3x8.png", figA.size)
figA.save(os.path.join(OUT, "Fig3_matched_grid.png")); print("Fig3_matched_grid.png (final name for Overleaf)", figA.size)
figA.save(os.path.join(OUT, "Fig3_matched_grid.png")); print("Fig3_matched_grid.png (final name for Overleaf)", figA.size)
# ---------------------------------------------------------------- Figure 3, layout B: 6 rows x 4 columns
blocksB = []; labs = []
for ds in ("Cityscapes", "Traffic", "COCO"):
    for i in (0, 1):
        blocksB.append([(ds, "real", i), (ds, "ADM", i), (ds, "FDGAN*", i), (ds, "FDGAN", i)]); labs.append({"Cityscapes": "Cityscapes", "Traffic": "Traffic Signs", "COCO": "COCO potted plant"}[ds] if i == 0 else "")
figB = grid(blocksB, quart, 6, P=256, G=6, header=44, left=56, rowlabels=labs, font_size=26)
figB.save(os.path.join(OUT, "Fig3_matched_6x4.png")); print("Fig3_matched_6x4.png", figB.size)

# ---------------------------------------------------------------- ablation grid (Cityscapes, guidance 2.0; no-mask row has no guidance)
abl_cols = ["Real (box)", "ADM", "FDGAN*", "w/o localized noise", "global noise", "+ small ROI D", "w/o box mask", "FDGAN"]
abl_keys = ["real", "ADM", "FDGAN*", "nonoise", "global", "smallD", "nomask", "FDGAN"]
NR = 5
blocksC = [[("Cityscapes", k, i) for k in abl_keys] for i in range(NR)]
figC = grid(blocksC, abl_cols, NR, P=256, G=6, header=44, font_size=22)
figC.save(os.path.join(OUT, "S7_ablation_grid.png")); print("S7_ablation_grid.png", figC.size)
# ---------------------------------------------------------------- seeds and guidance grid
sg_cols = ["Real (box)", "FDGAN* seed 1", "FDGAN seed 1", "FDGAN* seed 2", "FDGAN seed 2", "FDGAN* no guid.", "FDGAN no guid."]
sg_keys = ["real", "FDGAN*", "FDGAN", "FDGAN*_s2", "FDGAN_s2", "FDGAN*_nocfg", "FDGAN_nocfg"]
blocksD = [[("Cityscapes", k, i) for k in sg_keys] for i in range(NR)]
figD = grid(blocksD, sg_cols, NR, P=256, G=6, header=44, font_size=22)
figD.save(os.path.join(OUT, "S7_seeds_guidance_grid.png")); print("S7_seeds_guidance_grid.png", figD.size)

# ---------------------------------------------------------------- S7 grids: splice new columns into the old images
def runs(mask):
    """maximal runs of True in a 1-D boolean array -> list of (start, end_exclusive)"""
    out, start = [], None
    for k, v in enumerate(mask):
        if v and start is None: start = k
        if not v and start is not None: out.append((start, k)); start = None
    if start is not None: out.append((start, len(mask)))
    return out
def geometry(im, min_len=100):
    a = np.asarray(im.convert("L")); white = a > 235
    row_white = white.mean(axis=1) > 0.97; col_white = white.mean(axis=0) > 0.97
    rows = [r for r in runs(~row_white) if r[1] - r[0] >= min_len]
    body = slice(rows[0][0], rows[-1][1])
    col_white_b = white[body].mean(axis=0) > 0.97
    cols = [c for c in runs(~col_white_b) if c[1] - c[0] >= min_len]
    return rows, cols
OLD = {("Cityscapes", "gan"): ("Fig7_grid_gan_x6_1.jpg", {"Original": 0, "FDGAN": 1}),
       ("Cityscapes", "diff"): ("Fig7_grid_diff.jpg", {"Original": 0, "FDGAN": 1, "ADM": 2}),
       ("Traffic", "gan"): ("Fig9_grid_GAN.jpg", {"Original": 0, "FDGAN": 1}),
       ("Traffic", "diff"): ("Fig9_grid_Diff.jpg", {"Original": 0, "FDGAN": 1, "ADM": 2}),
       ("COCO", "gan"): ("COCO_Grid_GAN.jpg", {"Original": 0, "FDGAN": 1}),
       ("COCO", "diff"): ("COCO_Grid_Diff.jpg", {"Original": 0, "FDGAN": 1, "ADM": 2})}
KEYMAP = {"Original": "real", "FDGAN": "FDGAN", "ADM": "ADM"}
for (ds, kind), (fname, colmap) in OLD.items():
    im = Image.open(os.path.join(IM, fname)).convert("RGB")
    rows, cols = geometry(im)
    print(f"{fname}: {len(rows)} rows {len(cols)} cols; row heights {[r[1]-r[0] for r in rows]}; col widths {[c[1]-c[0] for c in cols]}; header {rows[0][0]} px")
    assert len(rows) == 6 and len(cols) in (6, 7), (fname, len(rows), len(cols))
    for label, c in colmap.items():
        x0, x1 = cols[c]
        for i, (y0, y1) in enumerate(rows):
            im.paste(panel(ds, KEYMAP[label], i, (x1 - x0, y1 - y0), box=True, width=3), (x0, y0))
    outp = os.path.join(OUT, f"S7_{ds}_{kind}.jpg"); im.save(outp, quality=93); print("  ->", os.path.basename(outp), im.size)
print("done")
