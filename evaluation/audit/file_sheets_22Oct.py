# -*- coding: utf-8 -*-
"""File the 22 Oct 11:22 Cityscapes sheets (bundle C: two no-guidance seed-2 rows) under the canonical names.
   The 21 Oct canonical files are renamed *_superseded_21Oct_bundleB*, never deleted."""
import os, shutil, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
DL = r"C:\Users\latzi\Downloads"
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
PAIRS = [("Cityscapes_200k_metrics (7).xlsx", "Step14_Cityscapes_200k_metrics.xlsx"),
         ("Cityscapes_bootstrap_FD (3).xlsx", "Step14_Cityscapes_bootstrap_FD.xlsx"),
         ("Cityscapes_bootstrap_replicates (2).csv", "Step14_Cityscapes_bootstrap_replicates.csv"),
         ("Cityscapes_object_metrics (4).xlsx", "Step16_Cityscapes_object_metrics.xlsx")]
for src, dst in PAIRS:
    s, d = os.path.join(DL, src), os.path.join(W, dst)
    assert os.path.exists(s), s
    if os.path.exists(d):
        stem, ext = os.path.splitext(dst)
        keep = os.path.join(W, f"{stem}_superseded_21Oct_bundleB{ext}")
        assert not os.path.exists(keep), keep
        os.rename(d, keep); print(f"kept old: {os.path.basename(keep)}")
    shutil.copy2(s, d); print(f"filed:    {dst}  <- {src}")

# pairs sheet re-check with the sheet's own convention (diff = FD(A) - FD(B), A = FDGAN cfg15; P(A better) = share of replicates with FD_A < FD_B)
bn = pd.read_excel(os.path.join(W, "Step14_Cityscapes_bootstrap_FD.xlsx"), sheet_name=None)
pm = bn["per_model"].rename(columns={"Unnamed: 0": "model"}).set_index("model"); pr = bn["pairs"]
rep = pd.read_csv(os.path.join(W, "Step14_Cityscapes_bootstrap_replicates.csv"))
e = 0.0; ok = True
for _, r in pr.iterrows():
    A, B = r["A"], r["B"]; d = rep[A].values - rep[B].values
    if abs((float(pm.at[A, "fd_point"]) - float(pm.at[B, "fd_point"])) - float(r["diff_point"])) > 1e-6: ok = False
    e = max(e, abs(d.mean() - r["diff_boot_mean"]), abs(np.percentile(d, 2.5) - r["diff_ci95_low"]), abs(np.percentile(d, 97.5) - r["diff_ci95_high"]), abs((d < 0).mean() - r["P(A better)"]))
print(f"pairs sheet: diff_point == FD(A)-FD(B) for all 22 pairs: {ok}; max err of mean/ci/P vs replicates: {e:.6f}")
# sanity: per-model intervals sit above the point FD (bootstrap FD bias), as in every earlier file
above = int((pm["ci95_low"] > pm["fd_point"]).sum()); print(f"per-model: ci95_low > fd_point in {above} of {len(pm)} rows (expected all; differences cancel the bias)")
print("done")
