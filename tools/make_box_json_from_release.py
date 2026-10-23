# -*- coding: utf-8 -*-
"""Write the per-image box files that the FDGAN loader and sampler read, from the TrafficSigns-OC crops of our dataset
release (https://github.com/Latzi/object-centric-low-data-datasets).

The release stores YOLO labels (class, x_center, y_center, width, height, all normalised) with up to five boxes per
256 px crop. FDGAN conditions on ONE box per image, so this script picks one box per crop and writes
<out_boxes>/<stem>.json = {"x": x1, "y": y1, "w": width, "h": height} in pixels, the x-y-width-height format that the
Traffic Signs runs use (--bb_is_xywh True). The rule for the pick is --rule: largest (default), centre (closest to the
crop centre) or first (first line of the label file).

Usage:
  python tools/make_box_json_from_release.py --traffic_signs <release>/traffic_signs --out_boxes <folder>
                                             [--out_images <folder>] [--rule largest|centre|first] [--splits train,val,test]
--out_images copies every crop into one flat folder, which is convenient for --data_dir in training; for sampling only
the box folder is needed.
"""
import argparse, glob, json, os, shutil, sys


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
    ap.add_argument("--traffic_signs", required=True, help="the traffic_signs folder of the release (holds Train_Data/)")
    ap.add_argument("--out_boxes", required=True, help="folder for the <stem>.json files")
    ap.add_argument("--out_images", default="", help="optional flat folder that receives a copy of every crop")
    ap.add_argument("--rule", default="largest", choices=("largest", "centre", "first"))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--size", type=int, default=256, help="crop side in pixels (the release crops are 256)")
    a = ap.parse_args()

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    pairs = []
    for s in splits:
        if s == "test":
            lab_dir, img_dir = os.path.join(a.traffic_signs, "Train_Data", "test", "labels"), os.path.join(a.traffic_signs, "Train_Data", "test", "images")
        else:
            lab_dir, img_dir = os.path.join(a.traffic_signs, "Train_Data", "labels", s), os.path.join(a.traffic_signs, "Train_Data", "images", s)
        labels = sorted(glob.glob(os.path.join(lab_dir, "*.txt")))
        if not labels:
            print(f"no label files in {lab_dir}", file=sys.stderr); sys.exit(1)
        for lp in labels:
            stem = os.path.splitext(os.path.basename(lp))[0]
            ip = None
            for ext in (".jpg", ".png", ".jpeg"):
                if os.path.exists(os.path.join(img_dir, stem + ext)):
                    ip = os.path.join(img_dir, stem + ext); break
            pairs.append((s, stem, lp, ip))

    os.makedirs(a.out_boxes, exist_ok=True)
    if a.out_images:
        os.makedirs(a.out_images, exist_ok=True)
    n_written = n_multi = n_noimg = 0
    for s, stem, lp, ip in pairs:
        boxes = []
        with open(lp) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append(tuple(float(v) for v in parts[1:5]))
        if not boxes:
            continue
        if len(boxes) > 1:
            n_multi += 1
        cx, cy, w, h = pick(boxes, a.rule)
        S = a.size
        x1 = int(round((cx - w / 2) * S)); y1 = int(round((cy - h / 2) * S))
        x2 = int(round((cx + w / 2) * S)); y2 = int(round((cy + h / 2) * S))
        x1 = max(0, min(S - 1, x1)); y1 = max(0, min(S - 1, y1)); x2 = max(x1 + 1, min(S, x2)); y2 = max(y1 + 1, min(S, y2))
        with open(os.path.join(a.out_boxes, stem + ".json"), "w") as f:
            json.dump({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "split": s, "n_boxes_in_crop": len(boxes), "rule": a.rule}, f)
        n_written += 1
        if ip is None:
            n_noimg += 1
        elif a.out_images:
            shutil.copy2(ip, os.path.join(a.out_images, os.path.basename(ip)))
    print(f"{n_written} box files written to {a.out_boxes} (rule: {a.rule}; {n_multi} crops had more than one box; "
          f"{n_noimg} label files without an image)" + (f"; crops copied to {a.out_images}" if a.out_images else ""))


if __name__ == "__main__":
    main()
