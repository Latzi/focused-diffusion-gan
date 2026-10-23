# -*- coding: utf-8 -*-
"""File the 21 Oct sheets under the canonical Step14/Step16 names; the previous canonical file is renamed with its own date, never deleted."""
import os, shutil, sys, time
sys.stdout.reconfigure(encoding="utf-8")
DL = r"C:\Users\latzi\Downloads"
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
PAIRS = [("Cityscapes_200k_metrics (6).xlsx", "Step14_Cityscapes_200k_metrics.xlsx"),      # CT of the new rows still blank; re-filed when the C_T cell has run
         ("Traffic_200k_metrics (3).xlsx", "Step14_Traffic_200k_metrics.xlsx"),
         ("COCO_200k_metrics (2).xlsx", "Step14_COCO_200k_metrics.xlsx"),
         ("Cityscapes_bootstrap_FD (2).xlsx", "Step14_Cityscapes_bootstrap_FD.xlsx"),
         ("Cityscapes_bootstrap_replicates (1).csv", "Step14_Cityscapes_bootstrap_replicates.csv"),
         ("Cityscapes_object_metrics (3).xlsx", "Step16_Cityscapes_object_metrics.xlsx"),
         ("COCO_object_metrics (2).xlsx", "Step16_COCO_object_metrics.xlsx")]
# Traffic_object_metrics (3).xlsx is stale (no new row) and is NOT filed.
for src, dst in PAIRS:
    s, d = os.path.join(DL, src), os.path.join(W, dst)
    assert os.path.exists(s), s
    if os.path.exists(d):
        stem, ext = os.path.splitext(dst)
        old_date = time.strftime("%d%b", time.localtime(os.path.getmtime(d)))
        keep = os.path.join(W, f"{stem}_superseded_{old_date}{ext}")
        n = 1
        while os.path.exists(keep):
            n += 1; keep = os.path.join(W, f"{stem}_superseded_{old_date}_{n}{ext}")
        os.rename(d, keep); print(f"kept old: {os.path.basename(keep)}")
    shutil.copy2(s, d); print(f"filed:    {dst}  <- {src}")
print("done")
