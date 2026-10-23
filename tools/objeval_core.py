# -*- coding: utf-8 -*-
"""Object-level evaluation core: box crops of generated and matching real images, and person detection
inside the conditioned box with a COCO-pretrained YOLO. Shared by the local test and the Colab notebook."""
import os, json, glob
import numpy as np
from PIL import Image

PERSON_CLASS = 0          # COCO class index of "person" in the Ultralytics models
CROP_SIZE = 128           # crops are saved at this size; dgm-eval's encoders resize to their input anyway


def square_crop(im, box, pad_ratio=0.10):
    """Square crop around the box (padded by pad_ratio of the longer side), clipped to the image, resized to CROP_SIZE."""
    W, H = im.size
    x1, y1, x2, y2 = box
    side = int(round(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * pad_ratio))); side = max(8, min(side, min(W, H)))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sx1 = min(max(0, cx - side // 2), W - side); sy1 = min(max(0, cy - side // 2), H - side)
    return im.crop((sx1, sy1, sx1 + side, sy1 + side)).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)


def build_crop_sets(gen_dir, meta, real_index, out_gen, out_real):
    """gen_dir: flat folder of sample_*.png; meta: the set's meta.json dict; real_index: {stem: real image path}.
    Writes paired crops (same box) for the generated image and the real image the box came from."""
    os.makedirs(out_gen, exist_ok=True); os.makedirs(out_real, exist_ok=True); n = 0; missing = 0
    for s in meta["samples"]:
        gp = os.path.join(gen_dir, s["file"]); rp = real_index.get(s["bb_stem"])
        if not (os.path.exists(gp) and rp):
            missing += 1; continue
        box = [int(v) for v in s["bb_xyxy"]]
        square_crop(Image.open(gp).convert("RGB"), box).save(os.path.join(out_gen, s["file"]))
        square_crop(Image.open(rp).convert("RGB"), box).save(os.path.join(out_real, s["file"]))
        n += 1
    return n, missing


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detect_persons(model, image_paths, boxes, conf=0.25, batch=32):
    """Runs the detector on full images. For each image: whether any person was found anywhere, whether one overlaps the
    conditioned box with IoU >= 0.5, and the best IoU. Returns per-image records."""
    recs = []
    for i in range(0, len(image_paths), batch):
        paths = image_paths[i:i + batch]
        results = model.predict(paths, conf=conf, classes=[PERSON_CLASS], verbose=False, imgsz=256)
        for p, box, r in zip(paths, boxes[i:i + batch], results):
            dets = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
            best = max((iou(box, d) for d in dets), default=0.0)
            recs.append(dict(file=os.path.basename(p), n_persons=len(dets), any_person=int(len(dets) > 0), hit_iou50=int(best >= 0.5), best_iou=best))
    return recs


def summarize(recs):
    a = np.array([[r["any_person"], r["hit_iou50"], r["best_iou"]] for r in recs], dtype=float)
    return dict(n=len(recs), person_detected_pct=100 * a[:, 0].mean(), person_in_box_iou50_pct=100 * a[:, 1].mean(), mean_best_iou=float(a[:, 2].mean()))
