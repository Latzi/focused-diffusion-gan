#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write resume versions of the FDGAN2026 YOLO job files after a TIMEOUT.

For every train_FDGAN2026_<Dataset>_<Init>.sh in this folder a train_FDGAN2026_<Dataset>_<Init>_resume.sh is written
with a 2-day limit, in which each of the seven ratio runs is
  - skipped when it finished (results.png exists, or results.csv holds every epoch),
  - continued from its last epoch when it was interrupted (weights/last.pt exists: 'yolo train resume model=...'),
  - started as in the original job file otherwise,
followed by the same evaluation line. The original job files are never changed.

    python3 make_FDGAN2026_resume_jobs.py --check     # only show the state of every ratio run
    python3 make_FDGAN2026_resume_jobs.py             # show the state and write the _resume.sh files
"""
import argparse, csv, glob, io, os, re, sys
ROOT = os.environ.get("FDGAN2026_ROOT", "/fred/oz097/latzi/YOLOv11/ultralytics")
GEN = "FDGAN2026"

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--time", default="2-00:00:00", help="walltime of the resume jobs")
a = ap.parse_args()

jobs = sorted(p for p in glob.glob(os.path.join(ROOT, f"train_{GEN}_*.sh")) if not p.endswith("_resume.sh"))
if not jobs:
    print("no train_FDGAN2026_*.sh job files in", ROOT); sys.exit(1)
written = 0
for job in jobs:
    text = io.open(job, encoding="utf-8").read().splitlines()
    train_lines = [l for l in text if l.startswith("srun yolo train ")]
    val_lines = [l for l in text if "fdgan2026_val.py" in l]
    header = [l for l in text if not l.startswith("srun ")]
    print(f"\n{os.path.basename(job)}: {len(train_lines)} training lines")
    out_lines = []
    for l in train_lines:
        m_proj = re.search(r"project=(\S+)", l); m_name = re.search(r"name=(\S+)", l); m_ep = re.search(r"epochs=(\d+)", l)
        if not (m_proj and m_name):
            print("   cannot parse:", l[:100]); out_lines.append(l); continue
        run = os.path.join(m_proj.group(1), m_name.group(1)); epochs = int(m_ep.group(1)) if m_ep else None
        rc = os.path.join(run, "results.csv"); best = os.path.join(run, "weights", "best.pt"); last = os.path.join(run, "weights", "last.pt")
        n_rows = 0
        if os.path.exists(rc):
            try: n_rows = sum(1 for _ in csv.DictReader(io.open(rc, encoding="utf-8")))
            except Exception: n_rows = 0
        finished = os.path.exists(os.path.join(run, "results.png")) or (epochs is not None and n_rows >= epochs and os.path.exists(best))
        if finished:
            state = f"finished ({n_rows} epochs logged)"; out_lines.append(f"# {m_name.group(1)}: finished, skipped")
        elif os.path.exists(last):
            state = f"INTERRUPTED after {n_rows} of {epochs} epochs -> resume"; out_lines.append(f"srun yolo train resume model={last}")
        elif os.path.isdir(run):
            state = f"folder exists but no last.pt ({n_rows} epochs logged) -> start again"; out_lines.append(l)
        else:
            state = "not started -> start"; out_lines.append(l)
        print(f"   {m_name.group(1):16s} {state}")
    if a.check:
        continue
    # keep the header part of the original (everything before its first srun line), with a new time limit and job name
    first_srun = next(i for i, l in enumerate(text) if l.startswith("srun "))
    hdr = [re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={a.time}", h) for h in text[:first_srun]]
    hdr = [re.sub(r"^(#SBATCH --job-name=\S+)$", r"\1_rs", h) for h in hdr]
    out = os.path.join(ROOT, os.path.basename(job)[:-3] + "_resume.sh")
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(hdr + out_lines + val_lines) + "\n")
    written += 1
    print(f"   -> written {os.path.basename(out)} (time {a.time}, {sum(1 for l in out_lines if l.startswith('srun'))} training lines)")
if not a.check:
    print(f"\n{written} resume job files written. Submit with: sbatch train_{GEN}_<Dataset>_<Init>_resume.sh")
