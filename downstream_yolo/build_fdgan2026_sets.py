#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the YOLO11 augmentation datasets for the revised FDGAN ("FDGAN2026") in the exact layout of the existing
generator folders, label the synthetic images with the dataset's real-only pretrained detector (the teacher step of the
companion study), write the 21 data yamls and the six training job files.

    python build_fdgan2026_sets.py --check      # login node: verify every path and count, write nothing
    sbatch build_fdgan2026_sets.sh              # GPU job: unzip, teacher labels, folders, yamls, job files

Nothing existing is modified. New things only:
    datasets/<Dataset>/FDGAN2026/src/...                       unzipped 256-px samples + meta.json + teacher labels
    datasets/<Dataset>/FDGAN2026/Train_Data_<ratio>/...        real train/val (hard links to the Baseline folder) + synthetic
    <Prefix>FDGAN2026_<ratio>.yaml                             copies of <Prefix>FDGAN_<ratio>.yaml with the new folder
    train_FDGAN2026_<Prefix>_<Init>.sh                         six job files (7 chained trainings + evaluation)
Synthetic subset per ratio: the first round(ratio x N_real) kept images in sorted order (nested across ratios)."""
import argparse, io, json, os, re, shutil, subprocess, sys, zipfile

ROOT = "/fred/oz097/latzi/YOLOv11/ultralytics"
SAMPLES = "/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/samples_Step11"
GEN = "FDGAN2026"
RATIOS = [10, 25, 50, 75, 100, 125, 150]
TEACHER_CONF = 0.25          # detection threshold of the teacher (Ultralytics predict default)
DATASETS = {
    "Pedestrian": dict(ds_dir=f"{ROOT}/datasets/Pedestrian768px", imgsz=768,
                       zip=f"{SAMPLES}/cityscapes_fdgan_mask_ema200000_n3000_r500_colsample_anc500_cfg20.zip",
                       projects={"Pretrained": f"{ROOT}/runs_768px_Pedestrian_Pretrained_600",
                                 "FromScratch": f"{ROOT}/runs_768_Pedestrian_FromScratch"},
                       teacher=[f"{ROOT}/runs_768_Pedestrian_Pretrained_600/Pedestrian_Baseline_check/weights/best.pt",
                                f"{ROOT}/runs_768px_Pedestrian_Pretrained/Pedestrian_Baseline/weights/best.pt"]),
    "Traffic": dict(ds_dir=f"{ROOT}/datasets/TrafficSigns768px", imgsz=768,
                    zip=f"{SAMPLES}/traffic2_fdgan_mask_ema200000_n3000_r500_colsample_anc500_cfg20.zip",
                    projects={"Pretrained": f"{ROOT}/runs_768px_Traffic_Pretrained_600",
                              "FromScratch": f"{ROOT}/runs_768px_Traffic_FromScratch_600"},
                    teacher=[f"{ROOT}/runs_768px_Traffic_Pretrained_600/BASELINE_Pretrained/weights/best.pt"]),
    "PottedPlant": dict(ds_dir=f"{ROOT}/datasets/PottedPlant512px", imgsz=512,
                        zip=f"{SAMPLES}/coco_fdgan_mask_lsun_ema200000_n3000_r500_colsample_anc500_cfg20.zip",
                        projects={"Pretrained": f"{ROOT}/runs_512_PottedPlant_Pretrained",
                                  "FromScratch": f"{ROOT}/runs_512_PottedPlant_FromScratch"},
                        teacher=[f"{ROOT}/runs_512_PottedPlant_Pretrained/PottedPlant_Baseline2/weights/best.pt",
                                 f"{ROOT}/runs_512_PottedPlant_Pretrained/PottedPlant_Baseline_check/weights/best.pt"]),
}
# training arguments copied from the old FDGAN_25 run of each project (args.yaml); only these keys are carried over
ARG_KEYS = ["model", "pretrained", "epochs", "imgsz", "patience", "batch", "optimizer", "lr0", "lrf", "momentum",
            "weight_decay", "dropout", "cos_lr", "single_cls", "augment", "workers", "seed", "deterministic",
            "warmup_epochs", "close_mosaic", "device", "amp", "rect", "multi_scale", "freeze", "nbs"]
FALLBACK_ARGS = {"Pretrained": "model=/fred/oz097/latzi/YOLOv11/ultralytics/yolo11m.pt epochs=600 patience=50 augment=True single_cls=True optimizer=SGD lr0=0.001 lrf=0.01 momentum=0.67 dropout=0.2 batch=32 weight_decay=0.001 workers=4 cos_lr=True",
                 "FromScratch": None}   # from scratch: must come from args.yaml (model=yolo11m.yaml / pretrained=False)
SBATCH_HEADER = """#!/bin/bash
#
#SBATCH --job-name={job}
#SBATCH --output=YOLOv11_log_%j.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:a100:2

module load gcc/12.2.0
module load openmpi/4.1.4
module load python-scientific/3.10.8-foss-2022b
export PSM2_CUDA=1
source /fred/oz097/latzi/YOLOv11/yolov11/bin/activate
cd {root}
"""

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="verify paths and counts only")
ap.add_argument("--only", default="", help="comma-separated subset of Pedestrian,Traffic,PottedPlant")
a = ap.parse_args()
problems = []
def note(msg): print("   " + msg)
def problem(msg): problems.append(msg); print("   PROBLEM: " + msg)
def count_files(d, exts=(".jpg", ".jpeg", ".png", ".txt")):
    return len([f for f in os.listdir(d) if f.lower().endswith(exts)]) if os.path.isdir(d) else -1
def link_or_copy(src, dst):
    if os.path.exists(dst): return
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

def inspect(name, cfg):
    """Return a dict with the resolved paths and counts, recording problems."""
    print(f"\n===== {name}")
    r = {"name": name}
    ds = cfg["ds_dir"]
    base = os.path.join(ds, "Baseline", "Train_Data")
    r["base_train_img"] = os.path.join(base, "images", "train"); r["base_train_lab"] = os.path.join(base, "labels", "train")
    r["base_val_img"] = os.path.join(base, "images", "val"); r["base_val_lab"] = os.path.join(base, "labels", "val")
    for k in ("base_train_img", "base_train_lab", "base_val_img", "base_val_lab"):
        n = count_files(r[k]); note(f"{k}: {r[k]}  ({n} files)")
        if n <= 0: problem(f"{name}: missing or empty {r[k]}")
    r["n_real"] = count_files(r["base_train_img"], (".jpg", ".jpeg", ".png"))
    note(f"N_real (training images) = {r['n_real']}; synthetic images needed at 150 % = {round(1.5 * r['n_real'])}")
    r["old_yaml"] = {rt: os.path.join(ROOT, f"{name}FDGAN_{rt}.yaml") for rt in RATIOS}
    for rt, p in r["old_yaml"].items():
        if not os.path.exists(p): problem(f"{name}: old yaml missing {p}")
    y25 = io.open(r["old_yaml"][25], encoding="utf-8").read() if os.path.exists(r["old_yaml"][25]) else ""
    if "/FDGAN/Train_Data_25" not in y25: problem(f"{name}: {r['old_yaml'][25]} does not reference /FDGAN/Train_Data_25 as expected")
    note("old yaml (25):\n      " + y25.strip().replace("\n", "\n      "))
    # held-out test set: either a shared datasets/<ds>/test folder, or a test/ folder inside each Train_Data_<ratio> (PottedPlant)
    val_line = next((l for l in y25.split("\n") if l.startswith("val:")), "")
    r["test_in_train_data"] = "/FDGAN/Train_Data_25/" in val_line
    if r["test_in_train_data"]:
        cands = [os.path.join(base, "test"), os.path.join(ds, "FDGAN", "Train_Data_25", "test")]
        r["test_src"] = next((c for c in cands if os.path.isdir(os.path.join(c, "images"))), None)
        note(f"held-out test set is inside Train_Data (val: {val_line.split('#')[0].split(':', 1)[1].strip()}); will be linked into every new folder from {r['test_src']}")
        if r["test_src"] is None: problem(f"{name}: no test folder with images/ found among {cands}")
        else:
            note(f"  test images: {count_files(os.path.join(r['test_src'], 'images'), ('.jpg', '.jpeg', '.png'))}, test labels: {count_files(os.path.join(r['test_src'], 'labels'))}")
    else:
        r["test_src"] = None; shared = os.path.join(ds, "test")
        note(f"held-out test folder: {shared} (exists: {os.path.isdir(shared)})")
        if not os.path.isdir(shared): problem(f"{name}: no test folder {shared}")
    r["zip"] = cfg["zip"]; note(f"FDGAN sample zip: {r['zip']} (exists: {os.path.exists(r['zip'])})")
    if not os.path.exists(r["zip"]): problem(f"{name}: sample zip missing")
    r["teacher"] = next((t for t in cfg["teacher"] if os.path.exists(t)), None)
    note(f"teacher detector: {r['teacher']}")
    if r["teacher"] is None: problem(f"{name}: none of the teacher weights exist: {cfg['teacher']}")
    r["projects"] = cfg["projects"]; r["args"] = {}
    for init, proj in cfg["projects"].items():
        ok = os.path.isdir(proj); note(f"project {init}: {proj} (exists: {ok})")
        if not ok: problem(f"{name}: project folder missing {proj}")
        ay = os.path.join(proj, "FDGAN_25", "args.yaml")
        if os.path.exists(ay):
            try:
                import yaml
                d = yaml.safe_load(io.open(ay, encoding="utf-8"))
                r["args"][init] = {k: d[k] for k in ARG_KEYS if k in d and d[k] is not None}
                note(f"  training args from {ay}: " + " ".join(f"{k}={v}" for k, v in r['args'][init].items()))
            except Exception as e:
                problem(f"{name}: cannot read {ay}: {e!r}")
        else:
            note(f"  no args.yaml at {ay}")
            if FALLBACK_ARGS[init] is None: problem(f"{name}: {init} training arguments unknown (no args.yaml and no fallback)")
            else: note(f"  using fallback arguments: {FALLBACK_ARGS[init]}")
    r["imgsz"] = cfg["imgsz"]; r["gen_dir"] = os.path.join(ds, GEN)
    note(f"output folder: {r['gen_dir']} (exists already: {os.path.isdir(r['gen_dir'])})")
    return r

def yolo_line(r, init, ratio):
    data = os.path.join(ROOT, f"{r['name']}{GEN}_{ratio}.yaml"); proj = r["projects"][init]
    if init in r["args"]:
        args = " ".join(f"{k}={v}" for k, v in r["args"][init].items())
    else:
        args = FALLBACK_ARGS[init] + f" imgsz={r['imgsz']}"
    return f"srun yolo train data={data} project={proj} name={GEN}_{ratio} {args}"

def build(r):
    name = r["name"]; gen_dir = r["gen_dir"]; src = os.path.join(gen_dir, "src")
    os.makedirs(src, exist_ok=True)
    # 1. unzip the 256-px samples (flat) + meta.json
    img_dir = os.path.join(src, "images"); os.makedirs(img_dir, exist_ok=True)
    meta = None
    if not os.path.exists(os.path.join(src, "meta.json")):
        with zipfile.ZipFile(r["zip"]) as z:
            for m in z.namelist():
                b = os.path.basename(m)
                if b == "meta.json":
                    io.open(os.path.join(src, "meta.json"), "wb").write(z.read(m))
                elif b.lower().endswith(".png"):
                    io.open(os.path.join(img_dir, b), "wb").write(z.read(m))
    meta = json.load(io.open(os.path.join(src, "meta.json"), encoding="utf-8"))
    recs = {s["file"]: s for s in meta["samples"]}
    imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    print(f"   {len(imgs)} synthetic images unpacked, {len(recs)} meta records")
    # 2. labels = the conditioning box of each image (exact, from meta.json); the teacher (the dataset's real-only
    #    pretrained detector) only confirms that the object is present: kept = a detection overlapping the conditioning
    #    box at IoU >= 0.5 (the in-box criterion of the paper). Images without that confirmation are dropped.
    lab_dir = os.path.join(src, "labels_condbox"); os.makedirs(lab_dir, exist_ok=True)
    summ_path = os.path.join(src, "teacher_summary.json")
    if not os.path.exists(summ_path):
        from ultralytics import YOLO
        model = YOLO(r["teacher"])
        kept, n_nodet, n_lowiou, n_nometa = [], 0, 0, 0
        for res in model.predict(source=img_dir, imgsz=r["imgsz"], conf=TEACHER_CONF, stream=True, verbose=False):
            fname = os.path.basename(res.path); stem = os.path.splitext(fname)[0]
            rec = recs.get(fname)
            if rec is None:
                n_nometa += 1; continue
            H, W = res.orig_shape[:2]; x1, y1, x2, y2 = [float(v) for v in rec["bb_xyxy"]]
            xc, yc, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
            io.open(os.path.join(lab_dir, stem + ".txt"), "w").write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                n_nodet += 1; continue
            best = 0.0
            for bx in boxes.xyxy.cpu().numpy().tolist():
                ix1, iy1, ix2, iy2 = max(x1, bx[0]), max(y1, bx[1]), min(x2, bx[2]), min(y2, bx[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = (x2 - x1) * (y2 - y1) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - inter
                best = max(best, inter / union if union > 0 else 0.0)
            if best >= 0.5: kept.append(fname)
            else: n_lowiou += 1
        summ = {"teacher": r["teacher"], "conf": TEACHER_CONF, "imgsz": r["imgsz"], "label_source": "conditioning box (meta.json bb_xyxy)",
                "keep_rule": "teacher detection with IoU >= 0.5 to the conditioning box", "n_images": len(imgs), "n_kept": len(kept),
                "n_dropped_no_detection": n_nodet, "n_dropped_iou_below_0.5": n_lowiou, "n_without_meta": n_nometa, "kept": sorted(kept)}
        json.dump(summ, io.open(summ_path, "w"), indent=1)
    summ = json.load(io.open(summ_path))
    kept = summ["kept"]
    print(f"   labels = conditioning boxes; teacher {os.path.relpath(summ['teacher'], ROOT)} confirms the object: kept {summ['n_kept']} of {summ['n_images']} images "
          f"(dropped {summ['n_dropped_no_detection']} with no detection, {summ['n_dropped_iou_below_0.5']} with IoU < 0.5)")
    # 3. dataset folders per ratio
    for rt in RATIOS:
        n_syn = round(rt / 100 * r["n_real"])
        if n_syn > len(kept):
            print(f"   WARNING ratio {rt}: needs {n_syn} synthetic images, only {len(kept)} kept -> using {len(kept)}"); n_syn = len(kept)
        td = os.path.join(gen_dir, f"Train_Data_{rt}")
        for sub in ("images/train", "images/val", "labels/train", "labels/val"): os.makedirs(os.path.join(td, sub), exist_ok=True)
        for src_dir, dst_sub in ((r["base_train_img"], "images/train"), (r["base_train_lab"], "labels/train"),
                                 (r["base_val_img"], "images/val"), (r["base_val_lab"], "labels/val")):
            for f in os.listdir(src_dir):
                p = os.path.join(src_dir, f)
                if os.path.isfile(p): link_or_copy(p, os.path.join(td, dst_sub, f))
        for f in kept[:n_syn]:
            stem = os.path.splitext(f)[0]
            link_or_copy(os.path.join(img_dir, f), os.path.join(td, "images/train", f"fdgan2026_{f}"))
            shutil.copyfile(os.path.join(lab_dir, stem + ".txt"), os.path.join(td, "labels/train", f"fdgan2026_{stem}.txt"))
        if r.get("test_in_train_data"):   # PottedPlant: the held-out test set sits inside each Train_Data folder
            for sub in ("images", "labels"):
                s = os.path.join(r["test_src"], sub); d = os.path.join(td, "test", sub); os.makedirs(d, exist_ok=True)
                for f in os.listdir(s):
                    p = os.path.join(s, f)
                    if os.path.isfile(p): link_or_copy(p, os.path.join(d, f))
        n_tr = count_files(os.path.join(td, "images/train"), (".jpg", ".jpeg", ".png")); n_va = count_files(os.path.join(td, "images/val"), (".jpg", ".jpeg", ".png"))
        # yaml: copy of the old FDGAN yaml with the folder substituted
        old = io.open(r["old_yaml"][rt], encoding="utf-8").read()
        new = old.replace(f"/FDGAN/Train_Data_{rt}/", f"/{GEN}/Train_Data_{rt}/")
        assert new.count(f"/{GEN}/Train_Data_{rt}/") >= 2, f"yaml substitution failed for ratio {rt}"
        yp = os.path.join(ROOT, f"{name}{GEN}_{rt}.yaml")
        io.open(yp, "w", encoding="utf-8", newline="\n").write(new)
        print(f"   ratio {rt:3d}: {n_tr} train images ({r['n_real']} real + {n_syn} synthetic), {n_va} val -> {os.path.basename(yp)}")
    # 4. job files
    for init in r["projects"]:
        job = f"train_{GEN}_{name}_{init}.sh"
        lines = [SBATCH_HEADER.format(job=f"{GEN}_{name[:3]}_{init[:3]}", root=ROOT)]
        for rt in RATIOS: lines.append(yolo_line(r, init, rt) + "\n")
        lines.append(f"srun python {ROOT}/fdgan2026_val.py --dataset {name} --init {init}\n")
        io.open(os.path.join(ROOT, job), "w", encoding="utf-8", newline="\n").write("\n".join(lines))
        print(f"   job file written: {job}")
    return True

only = [s for s in a.only.split(",") if s] or list(DATASETS)
resolved = {n: inspect(n, DATASETS[n]) for n in only}
if problems:
    print(f"\n{len(problems)} problem(s) found; fix them (or edit the paths at the top of this script) and run --check again."); sys.exit(1)
print("\nall checks passed")
if a.check:
    print("\nTraining lines that would be generated (review them):")
    for n, r in resolved.items():
        for init in r["projects"]: print("  " + yolo_line(r, init, 25))
    sys.exit(0)
for n, r in resolved.items():
    print(f"\n===== building {n}"); build(r)
print("\nDone. Submit the six training jobs with:")
for n in only:
    for init in DATASETS[n]["projects"]: print(f"  sbatch {ROOT}/train_{GEN}_{n}_{init}.sh")
