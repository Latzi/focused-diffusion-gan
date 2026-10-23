#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the FDGAN2026 runs of one dataset and initialisation on the held-out real test split (the 'val:' entry of the
data yaml, as in the test jobs) and append the numbers to FDGAN2026_results.csv.
    python fdgan2026_val.py --dataset Pedestrian --init Pretrained [--conf 0.001]
conf: detection threshold passed to yolo val; 0.001 is the Ultralytics default. Use the same value as for the companion
paper's Table 3 (set it here or with --conf)."""
import argparse, csv, io, os, sys
ROOT = "/fred/oz097/latzi/YOLOv11/ultralytics"
GEN = "FDGAN2026"
RATIOS = [10, 25, 50, 75, 100, 125, 150]
PROJECTS = {"Pedestrian": {"Pretrained": f"{ROOT}/runs_768px_Pedestrian_Pretrained_600", "FromScratch": f"{ROOT}/runs_768_Pedestrian_FromScratch"},
            "Traffic": {"Pretrained": f"{ROOT}/runs_768px_Traffic_Pretrained_600", "FromScratch": f"{ROOT}/runs_768px_Traffic_FromScratch_600"},
            "PottedPlant": {"Pretrained": f"{ROOT}/runs_512_PottedPlant_Pretrained", "FromScratch": f"{ROOT}/runs_512_PottedPlant_FromScratch"}}
IMGSZ = {"Pedestrian": 768, "Traffic": 768, "PottedPlant": 512}
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(PROJECTS))
ap.add_argument("--init", required=True, choices=["Pretrained", "FromScratch"])
ap.add_argument("--conf", type=float, default=0.001)
a = ap.parse_args()
from ultralytics import YOLO
out = os.path.join(ROOT, f"{GEN}_results.csv")
new_file = not os.path.exists(out)
with io.open(out, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new_file: w.writerow(["dataset", "init", "ratio", "run", "best_epoch_val_mAP50-95", "test_mAP50-95", "test_mAP50", "test_precision", "test_recall", "conf", "n_train_images"])
    for rt in RATIOS:
        run = os.path.join(PROJECTS[a.dataset][a.init], f"{GEN}_{rt}")
        best = os.path.join(run, "weights", "best.pt")
        if not os.path.exists(best):
            print(f"[skip] no best.pt for ratio {rt}: {best}"); continue
        # best validation mAP during training, from results.csv
        best_val = ""
        rc = os.path.join(run, "results.csv")
        if os.path.exists(rc):
            rows = list(csv.DictReader(io.open(rc, encoding="utf-8")))
            key = next((k for k in rows[0] if k.strip() == "metrics/mAP50-95(B)"), None)
            if key: best_val = f"{max(float(r[key]) for r in rows):.5f}"
        data = os.path.join(ROOT, f"{a.dataset}{GEN}_{rt}.yaml")
        m = YOLO(best).val(data=data, imgsz=IMGSZ[a.dataset], conf=a.conf, split="val", plots=False, verbose=False,
                           project=run, name=f"Val_{GEN}_{rt}_conf{a.conf}", exist_ok=True)
        n_tr = ""
        try:   # number of training images = files in the 'train:' folder of the yaml
            tr = next(l.split(":", 1)[1].split("#")[0].strip() for l in io.open(data, encoding="utf-8") if l.startswith("train:"))
            n_tr = len([x for x in os.listdir(tr) if x.lower().endswith((".jpg", ".jpeg", ".png"))])
        except Exception:
            pass
        row = [a.dataset, a.init, rt, run, best_val, f"{m.box.map:.5f}", f"{m.box.map50:.5f}", f"{m.box.mp:.5f}", f"{m.box.mr:.5f}", a.conf, n_tr]
        w.writerow(row); f.flush()
        print(f"{a.dataset} {a.init} ratio {rt:3d}: test mAP50-95 {m.box.map:.4f}  mAP50 {m.box.map50:.4f}  (best val during training {best_val})")
print("appended to", out)
