#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate EVERY trained YOLO run of the six projects (all generators, all ratios, the real-only baselines) on the
same held-out test split and with the same threshold as the FDGAN2026 rows, so that the numbers are comparable.

Scans the six project folders for sub-folders that hold weights/best.pt, reads <generator>_<ratio> from the folder
name (a name without a trailing number is a baseline, ratio 0), validates best.pt with the FDGAN2026_10 yaml of the
dataset (its 'val:' entry is the shared test split; identical for every generator), conf 0.001, and appends one row
per run to FDGAN2026_all_generators_results.csv. Runs already in the csv are skipped, so the job can be re-run.

    python eval_all_generators.py --check          # list what would be evaluated (no GPU needed)
    python eval_all_generators.py                  # evaluate (GPU job: sbatch eval_all_generators.sh)
    python eval_all_generators.py --only Pedestrian --init Pretrained
"""
import argparse, csv, io, os, re, sys, time
ROOT = os.environ.get("FDGAN2026_ROOT", "/fred/oz097/latzi/YOLOv11/ultralytics")
PROJECTS = {"Pedestrian": {"Pretrained": f"{ROOT}/runs_768px_Pedestrian_Pretrained_600", "FromScratch": f"{ROOT}/runs_768_Pedestrian_FromScratch"},
            "Traffic": {"Pretrained": f"{ROOT}/runs_768px_Traffic_Pretrained_600", "FromScratch": f"{ROOT}/runs_768px_Traffic_FromScratch_600"},
            "PottedPlant": {"Pretrained": f"{ROOT}/runs_512_PottedPlant_Pretrained", "FromScratch": f"{ROOT}/runs_512_PottedPlant_FromScratch"}}
IMGSZ = {"Pedestrian": 768, "Traffic": 768, "PottedPlant": 512}
OUT = os.path.join(ROOT, "FDGAN2026_all_generators_results.csv")
COLS = ["dataset", "init", "generator", "ratio", "run", "epochs_logged", "best_epoch_val_mAP50-95", "test_mAP50-95", "test_mAP50",
        "test_precision", "test_recall", "conf", "n_train_images", "test_yaml"]

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--only", default="", help="dataset: Pedestrian, Traffic or PottedPlant")
ap.add_argument("--init", default="", help="Pretrained or FromScratch")
ap.add_argument("--conf", type=float, default=0.001)
a = ap.parse_args()


def parse_name(name):
    m = re.match(r"^(.+?)_(\d+)$", name)
    if m: return m.group(1), int(m.group(2))
    return name, 0


def train_count(run):
    """number of training images of a run: the 'train:' folder of the data yaml named in its args.yaml"""
    try:
        args = os.path.join(run, "args.yaml")
        data = next(l.split(":", 1)[1].strip() for l in io.open(args, encoding="utf-8") if l.startswith("data:"))
        tr = next(l.split(":", 1)[1].split("#")[0].strip() for l in io.open(data, encoding="utf-8") if l.startswith("train:"))
        if not os.path.isabs(tr):
            base = next((l.split(":", 1)[1].strip() for l in io.open(data, encoding="utf-8") if l.startswith("path:")), os.path.dirname(data))
            tr = os.path.join(base, tr)
        return len([x for x in os.listdir(tr) if x.lower().endswith((".jpg", ".jpeg", ".png"))])
    except Exception:
        return ""


done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(io.open(OUT, encoding="utf-8")):
        done.add(r["run"])
todo = []
for ds, inits in PROJECTS.items():
    if a.only and ds != a.only: continue
    yaml = os.path.join(ROOT, f"{ds}FDGAN2026_10.yaml")
    for init, proj in inits.items():
        if a.init and init != a.init: continue
        if not os.path.isdir(proj):
            print(f"[missing project folder] {proj}"); continue
        names = sorted(os.listdir(proj))
        found = []
        for n in names:
            run = os.path.join(proj, n)
            if n.startswith("Val_") or not os.path.exists(os.path.join(run, "weights", "best.pt")): continue
            gen, ratio = parse_name(n)
            found.append((gen, ratio, run))
        gens = {}
        for gen, ratio, run in found: gens.setdefault(gen, []).append(ratio)
        print(f"\n{ds} / {init}: {len(found)} runs with best.pt in {os.path.basename(proj)}  (test yaml {'ok' if os.path.exists(yaml) else 'MISSING'}: {os.path.basename(yaml)})")
        for gen in sorted(gens): print(f"   {gen:24s} ratios {sorted(gens[gen])}")
        for gen, ratio, run in found:
            if run in done: continue
            todo.append((ds, init, gen, ratio, run, yaml))
print(f"\n{len(todo)} runs to evaluate ({len(done)} already in {os.path.basename(OUT)})")
if a.check:
    sys.exit(0)

from ultralytics import YOLO
new_file = not os.path.exists(OUT)
with io.open(OUT, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new_file: w.writerow(COLS)
    for i, (ds, init, gen, ratio, run, yaml) in enumerate(todo, 1):
        if not os.path.exists(yaml):
            print(f"[skip] no test yaml for {ds}: {yaml}"); continue
        rc = os.path.join(run, "results.csv"); n_rows = 0; best_val = ""
        if os.path.exists(rc):
            rows = list(csv.DictReader(io.open(rc, encoding="utf-8"))); n_rows = len(rows)
            key = next((k for k in (rows[0] if rows else {}) if k.strip() == "metrics/mAP50-95(B)"), None)
            if key: best_val = f"{max(float(r[key]) for r in rows):.5f}"
        t0 = time.time()
        m = YOLO(os.path.join(run, "weights", "best.pt")).val(data=yaml, imgsz=IMGSZ[ds], conf=a.conf, split="val", plots=False, verbose=False,
                                                              project=run, name=f"Val_all_conf{a.conf}", exist_ok=True)
        row = [ds, init, gen, ratio, run, n_rows, best_val, f"{m.box.map:.5f}", f"{m.box.map50:.5f}", f"{m.box.mp:.5f}", f"{m.box.mr:.5f}",
               a.conf, train_count(run), os.path.basename(yaml)]
        w.writerow(row); f.flush()
        print(f"[{i}/{len(todo)}] {ds} {init} {gen} {ratio:3d}: test mAP50-95 {m.box.map:.4f} mAP50 {m.box.map50:.4f} ({time.time()-t0:.0f} s)")
print("done; rows appended to", OUT)
