# -*- coding: utf-8 -*-
"""Turn any folder of images with YOLO labels into what FDGAN trains and samples from: 256 px object-centric crops and
one box file per crop.

Input: --images (searched recursively; jpg, jpeg, png) and --labels (YOLO text files of the same stem: one line per
box, "class x_center y_center width height", normalised to the image). Images of any size are accepted.

What happens to each image:
  1. one box is chosen (--rule largest | centre | first; --class_id restricts the choice to one class);
  2. a square window of side min(width, height) is centred on that box and clamped to the image (--window object);
     with --window whole the window is the centre square of the image instead;
  3. the window is resized to --size (256) and saved as PNG under --out_images;
  4. the chosen box is mapped into the crop and written to --out_boxes/<stem>.json as {"x": x1, "y": y1, "w": width,
     "h": height} in pixels, the x-y-width-height form that the FDGAN flags --bb_is_xywh True expect.

Images without a label file or without a usable box are skipped and counted.

Usage:
  python tools/prepare_dataset.py --images <dir> --labels <dir> --out_images <dir> --out_boxes <dir>
"""
import argparse, glob, json, os, sys
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def pick(boxes, rule):
    if rule == "first":
        return boxes[0]
    if rule == "largest":
        return max(boxes, key=lambda b: b[2] * b[3])
    if rule == "centre":
        return min(boxes, key=lambda b: (b[0] - 0.5) ** 2 + (b[1] - 0.5) ** 2)
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_images", required=True)
    ap.add_argument("--out_boxes", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--rule", default="largest", choices=("largest", "centre", "first"))
    ap.add_argument("--window", default="object", choices=("object", "whole"))
    ap.add_argument("--class_id", type=int, default=-1, help="use only boxes of this class (-1 = any class)")
    a = ap.parse_args()

    labels = {}
    for lp in glob.glob(os.path.join(a.labels, "**", "*.txt"), recursive=True):
        labels[os.path.splitext(os.path.basename(lp))[0]] = lp
    images = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True) if p.endswith(EXTS))
    if not images:
        print("no images found under", a.images, file=sys.stderr); sys.exit(1)
    os.makedirs(a.out_images, exist_ok=True); os.makedirs(a.out_boxes, exist_ok=True)
    n_ok = n_nolabel = n_nobox = n_multi = 0
    S = a.size
    for ip in images:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = labels.get(stem)
        if lp is None:
            n_nolabel += 1; continue
        boxes = []
        with open(lp) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5 and (a.class_id < 0 or int(float(parts[0])) == a.class_id):
                    boxes.append(tuple(float(v) for v in parts[1:5]))
        if not boxes:
            n_nobox += 1; continue
        if len(boxes) > 1:
            n_multi += 1
        cx, cy, bw, bh = pick(boxes, a.rule)
        im = Image.open(ip).convert("RGB"); W, H = im.size
        side = min(W, H)
        if a.window == "object":
            wx0 = int(round(cx * W - side / 2)); wy0 = int(round(cy * H - side / 2))
        else:
            wx0 = (W - side) // 2; wy0 = (H - side) // 2
        wx0 = max(0, min(W - side, wx0)); wy0 = max(0, min(H - side, wy0))
        crop = im.crop((wx0, wy0, wx0 + side, wy0 + side)).resize((S, S), Image.BICUBIC)
        sc = S / float(side)
        x1 = int(round(((cx - bw / 2) * W - wx0) * sc)); y1 = int(round(((cy - bh / 2) * H - wy0) * sc))
        x2 = int(round(((cx + bw / 2) * W - wx0) * sc)); y2 = int(round(((cy + bh / 2) * H - wy0) * sc))
        x1 = max(0, min(S - 1, x1)); y1 = max(0, min(S - 1, y1)); x2 = max(x1 + 1, min(S, x2)); y2 = max(y1 + 1, min(S, y2))
        crop.save(os.path.join(a.out_images, stem + ".png"))
        with open(os.path.join(a.out_boxes, stem + ".json"), "w") as f:
            json.dump({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "source_image": os.path.basename(ip),
                       "source_size": [W, H], "window": [wx0, wy0, side], "rule": a.rule, "n_boxes": len(boxes)}, f)
        n_ok += 1
    print(f"{n_ok} crops written to {a.out_images} and {n_ok} box files to {a.out_boxes} ({S} px, window {a.window}, rule {a.rule}); "
          f"skipped: {n_nolabel} images without a label file, {n_nobox} without a usable box; {n_multi} images had more than one box")


if __name__ == "__main__":
    main()
