# -*- coding: utf-8 -*-
"""Bundle C check: the four Cityscapes files of 22 Oct 11:22 against the filed canonical sheets.
   - metrics / object sheets: every common row identical cell by cell; the two new nocfg rep2 rows complete
   - bootstrap per_model: fd_point of every row equals the metrics sheet dinov2_fd; intervals reproduce from the replicates CSV
   - pairs: diff = FD(B) - FD(A) with A = FDGAN cfg15; intervals and P(A better) reproduce from the replicates"""
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
N = r"C:\Users\latzi\AppData\Local\Temp\claude\C--Users-latzi-Downloads-New-folder--5-\2874049a-ada9-431d-826d-b7ea81ba130f\scratchpad\bundleC_in"
NEW_ROWS = ["cityscapes_adm_mask_rep2_ema200000_n3000_r500_colsample_anc500_nocfg",
            "cityscapes_fdgan_mask_rep2_ema200000_n3000_r500_colsample_anc500_nocfg"]

def same(x, y):
    if pd.isna(x) and pd.isna(y): return True
    try: return abs(float(x) - float(y)) <= 1e-9
    except Exception: return str(x) == str(y)

def compare(old, new, key, show):
    o = pd.read_excel(os.path.join(W, old)); n = pd.read_excel(os.path.join(N, new))
    print(f"\n=== {old} {o.shape} -> {new} {n.shape}")
    so, sn = set(o[key]), set(n[key])
    added = [s for s in n[key] if s not in so]; removed = [s for s in o[key] if s not in sn]
    print(" added rows:", added); print(" removed rows:", removed)
    print(" columns only in old:", [c for c in o.columns if c not in n.columns]); print(" columns only in new:", [c for c in n.columns if c not in o.columns])
    common = [s for s in o[key] if s in sn]
    oi, ni = o.set_index(key), n.set_index(key)
    cols = [c for c in o.columns if c != key and c in n.columns]
    diffs = 0
    for s in common:
        for c in cols:
            if not same(oi.at[s, c], ni.at[s, c]):
                diffs += 1
                if diffs <= 10: print("   DIFF", s, c, oi.at[s, c], ni.at[s, c])
    print(f" common rows {len(common)} x {len(cols)} cols: {diffs} cells differ")
    for s in added:
        row = ni.loc[s]
        nans = [c for c in n.columns if c != key and pd.isna(row[c])]
        print(f" new row {s}: NaN columns = {nans}")
        for c in show:
            if c in row.index: print(f"    {c} = {row[c]}")
    return n

m = compare("Step14_Cityscapes_200k_metrics.xlsx", "Cityscapes_200k_metrics (7).xlsx", "set",
            ["dinov2_fd", "dinov2_fd_infinity_value", "dinov2_kd_value", "dinov2_fls", "dinov2_ct", "dinov2_ct_modified", "dinov2_fls_overfit", "inception_fd", "inception_precision", "inception_recall"])
ob = compare("Step16_Cityscapes_object_metrics.xlsx", "Cityscapes_object_metrics (4).xlsx", "set",
             ["crop_dinov2_fd", "crop_dinov2_kd_value", "crop_inception_fd", "det_person_detected_pct", "det_person_in_box_iou50_pct", "det_mean_best_iou"])

# ---------------- bootstrap
bo = pd.read_excel(os.path.join(W, "Step14_Cityscapes_bootstrap_FD.xlsx"), sheet_name=None)
bn = pd.read_excel(os.path.join(N, "Cityscapes_bootstrap_FD (3).xlsx"), sheet_name=None)
pmo = bo["per_model"].rename(columns={"Unnamed: 0": "model"}).set_index("model")
pmn = bn["per_model"].rename(columns={"Unnamed: 0": "model"}).set_index("model")
print(f"\n=== bootstrap per_model: old {pmo.shape} -> new {pmn.shape}")
print(" added:", [s for s in pmn.index if s not in pmo.index]); print(" removed:", [s for s in pmo.index if s not in pmn.index])
print(" n_gen/n_real/B unique:", pmn["n_gen"].unique(), pmn["n_real"].unique(), pmn["B"].unique())
ms = m.set_index("set")
mx = 0.0
for s in pmn.index:
    d = abs(float(pmn.at[s, "fd_point"]) - float(ms.at[s, "dinov2_fd"])); mx = max(mx, d)
print(f" max |fd_point - sheet dinov2_fd| over {len(pmn)} rows: {mx:.6f}")
mxo = max(abs(float(pmn.at[s, "fd_point"]) - float(pmo.at[s, "fd_point"])) for s in pmo.index if s in pmn.index)
print(f" max |fd_point new - old| over common rows: {mxo:.6f}")
shift = max(abs(float(pmn.at[s, "ci95_low"]) - float(pmo.at[s, "ci95_low"])) for s in pmo.index if s in pmn.index)
print(f" max |ci95_low new - old| over common rows (expected > 0, shared RNG stream): {shift:.3f}")

rep = pd.read_csv(os.path.join(N, "Cityscapes_bootstrap_replicates (2).csv"))
print(f"\n=== replicates: {rep.shape}; models = {rep.shape[1]-1}; models match per_model: {set(rep.columns[1:]) == set(pmn.index)}")
e_mean = e_lo = e_hi = 0.0
for s in pmn.index:
    v = rep[s].values
    e_mean = max(e_mean, abs(v.mean() - float(pmn.at[s, "boot_mean"])))
    e_lo = max(e_lo, abs(np.percentile(v, 2.5) - float(pmn.at[s, "ci95_low"])))
    e_hi = max(e_hi, abs(np.percentile(v, 97.5) - float(pmn.at[s, "ci95_high"])))
print(f" per_model vs replicates: max err mean {e_mean:.6f}, ci_low {e_lo:.6f}, ci_high {e_hi:.6f}")
pr = bn["pairs"]
print(f"\n=== pairs {pr.shape}; A unique: {pr['A'].unique()}")
e = 0.0; sign_ok = True
for _, r in pr.iterrows():
    A, B = r["A"], r["B"]
    d = rep[B].values - rep[A].values
    pt = float(pmn.at[B, "fd_point"]) - float(pmn.at[A, "fd_point"])
    if abs(pt - float(r["diff_point"])) > 1e-6: sign_ok = False
    e = max(e, abs(d.mean() - r["diff_boot_mean"]), abs(np.percentile(d, 2.5) - r["diff_ci95_low"]), abs(np.percentile(d, 97.5) - r["diff_ci95_high"]), abs((rep[A].values < rep[B].values).mean() - r["P(A better)"]))
print(f" diff_point == FD(B)-FD(A) for all pairs: {sign_ok}; max err of mean/ci/P vs replicates: {e:.6f}")
print("\n=== new-row numbers (for the paper) ===")
for s in NEW_ROWS:
    print(f" {s}: FD {pmn.at[s,'fd_point']:.2f} [{pmn.at[s,'ci95_low']:.1f}, {pmn.at[s,'ci95_high']:.1f}]  KD {ms.at[s,'dinov2_kd_value']:.3f}  FID {ms.at[s,'inception_fd']:.2f}  objFD {ob.set_index('set').at[s,'crop_dinov2_fd']:.1f}  inbox {ob.set_index('set').at[s,'det_person_in_box_iou50_pct']:.1f}")
    row = pr[pr["B"] == s].iloc[0]
    print(f"    vs FDGAN cfg15: diff_point {row['diff_point']:+.1f} [{row['diff_ci95_low']:+.1f}, {row['diff_ci95_high']:+.1f}]")
a, b = NEW_ROWS[1], NEW_ROWS[0]  # FDGAN rep2 nocfg - FDGAN* rep2 nocfg
d = rep[a].values - rep[b].values
print(f" seed-2 no guidance FDGAN - FDGAN*: point {pmn.at[a,'fd_point']-pmn.at[b,'fd_point']:+.1f}  boot mean {d.mean():+.1f}  CI [{np.percentile(d,2.5):+.1f}, {np.percentile(d,97.5):+.1f}]  P(FDGAN better) {(d<0).mean():.3f}")
a1, b1 = "cityscapes_fdgan_mask_ema200000_n3000_r500_colsample_anc500_nocfg", "cityscapes_adm_mask_ema200000_n3000_r500_colsample_anc500_nocfg"
d1 = rep[a1].values - rep[b1].values
print(f" seed-1 no guidance FDGAN - FDGAN* (regenerated): point {pmn.at[a1,'fd_point']-pmn.at[b1,'fd_point']:+.1f}  CI [{np.percentile(d1,2.5):+.1f}, {np.percentile(d1,97.5):+.1f}]  P {(d1<0).mean():.3f}")
