# -*- coding: utf-8 -*-
"""Bundle B (replicate training seed): check the four Cityscapes sheets against the filed ones, compute the seed comparisons, file them."""
import io, os, shutil, sys, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
DL = r"C:\Users\latzi\Downloads"
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
FILES = {"metrics": ("Cityscapes_200k_metrics (6).xlsx", "Step14_Cityscapes_200k_metrics.xlsx"),
         "object": ("Cityscapes_object_metrics (3).xlsx", "Step16_Cityscapes_object_metrics.xlsx"),
         "boot": ("Cityscapes_bootstrap_FD (2).xlsx", "Step14_Cityscapes_bootstrap_FD.xlsx"),
         "reps": ("Cityscapes_bootstrap_replicates (1).csv", "Step14_Cityscapes_bootstrap_replicates.csv")}
REP2 = ["cityscapes_adm_mask_rep2_ema200000_n3000_r500_colsample_anc500_cfg15", "cityscapes_adm_mask_rep2_ema200000_n3000_r500_colsample_anc500_cfg20",
        "cityscapes_fdgan_mask_rep2_ema200000_n3000_r500_colsample_anc500_cfg15", "cityscapes_fdgan_mask_rep2_ema200000_n3000_r500_colsample_anc500_cfg20"]
def load(p): d = pd.read_excel(p, index_col=0); d.index = d.index.astype(str); return d
def compare(old, new, label):
    added = [r for r in new.index if r not in old.index]; missing = [r for r in old.index if r not in new.index]
    diffs = []
    for r in old.index:
        if r not in new.index: continue
        for c in old.columns:
            if c not in new.columns: continue
            a, b = old.loc[r, c], new.loc[r, c]
            if pd.isna(a) and pd.isna(b): continue
            try:
                if abs(float(a) - float(b)) < 1e-6: continue
            except Exception:
                if a == b: continue
            diffs.append((r[:55], c, a, b))
    print(f"[{label}] old {len(old)} rows | new {len(new)} | added {added if len(added) < 6 else len(added)} | missing {missing or 'none'} | changed cells in old rows {len(diffs)}")
    for d in diffs[:10]: print("    CHANGED", d)
    return added, diffs
ok = True
m_new, m_old = load(os.path.join(DL, FILES["metrics"][0])), load(os.path.join(W, FILES["metrics"][1]))
added, diffs = compare(m_old, m_new, "whole-image"); ok &= (sorted(added) == sorted(REP2) and not diffs)
cols = ["dinov2_fd", "dinov2_fd_infinity_value", "dinov2_kd_value", "inception_fd", "dinov2_precision", "dinov2_recall", "dinov2_ct", "dinov2_ct_modified", "ct_seed",
        "dinov2_fls", "dinov2_fls_overfit", "failure_rate_pct", "tint", "saturation", "seconds_per_image", "mask_cfg_scale", "roi_bump_mode"]
print(m_new.loc[REP2, cols].round(3).to_string())
blank = m_new.loc[REP2, "dinov2_ct_modified"].isna().sum(); print("   replicate rows with blank CT-mod:", blank); ok &= (blank == 0)
o_new, o_old = load(os.path.join(DL, FILES["object"][0])), load(os.path.join(W, FILES["object"][1]))
added_o, diffs_o = compare(o_old, o_new, "object"); ok &= (sorted(added_o) == sorted(REP2) and not diffs_o)
print(o_new.loc[REP2, ["crop_dinov2_fd", "crop_dinov2_kd_value", "crop_inception_fd", "det_person_in_box_iou50_pct", "real_det_person_in_box_iou50_pct"]].round(3).to_string())
per = pd.read_excel(os.path.join(DL, FILES["boot"][0]), sheet_name="per_model", index_col=0); pairs = pd.read_excel(os.path.join(DL, FILES["boot"][0]), sheet_name="pairs")
rep = pd.read_csv(os.path.join(DL, FILES["reps"][0]), index_col=0)
print(f"[bootstrap] per_model rows {len(per)} | csv {rep.shape} | B {per['B'].iloc[0]} | n_real {per['n_real'].iloc[0]} | A = {pairs['A'].iloc[0][:70]}")
ok &= (len(per) == 21 and rep.shape == (200, 21) and "rep2" not in pairs["A"].iloc[0])
bad = 0
for t in per.index:
    v = rep[t].values
    for col, val in (("boot_mean", v.mean()), ("boot_std", v.std(ddof=1)), ("ci95_low", np.percentile(v, 2.5)), ("ci95_high", np.percentile(v, 97.5))):
        if abs(per.loc[t, col] - val) > 1e-6: bad += 1
    if abs(per.loc[t, "fd_point"] - m_new.loc[t, "dinov2_fd"]) > 1e-6: bad += 1
print("   per_model statistics and point FDs recomputed from csv/sheet: mismatches =", bad); ok &= (bad == 0)
# the 17 old rows: did their replicate values change vs the filed 17-row csv? (they will, draws shift with added rows)
rep_old = pd.read_csv(os.path.join(W, FILES["reps"][1]), index_col=0)
same = all(np.allclose(rep_old[c].values, rep[c].values) for c in rep_old.columns if c in rep.columns)
print("   replicate values of the 17 previous rows identical to the filed csv:", same)

def P(a, b, src=rep):
    d = src[a].values - src[b].values
    return m_new.loc[a, "dinov2_fd"] - m_new.loc[b, "dinov2_fd"], np.percentile(d, 2.5), np.percentile(d, 97.5), float((d < 0).mean())
R = {"F15": "cityscapes_fdgan_mask_ema200000_n3000_r500_colsample_anc500_cfg15", "F20": "cityscapes_fdgan_mask_ema200000_n3000_r500_colsample_anc500_cfg20",
     "S15": "cityscapes_adm_mask_ema200000_n3000_r500_colsample_anc500_cfg15", "S20": "cityscapes_adm_mask_ema200000_n3000_r500_colsample_anc500_cfg20",
     "F15r": REP2[2], "F20r": REP2[3], "S15r": REP2[0], "S20r": REP2[1]}
print("\nSEED COMPARISONS (point difference, 95% paired interval, P(first better)):")
for name, a, b in [("FDGAN rep2 - FDGAN* rep2 @1.5", "F15r", "S15r"), ("FDGAN rep2 - FDGAN* rep2 @2.0", "F20r", "S20r"),
                   ("FDGAN rep1 - FDGAN* rep1 @1.5", "F15", "S15"), ("FDGAN rep1 - FDGAN* rep1 @2.0", "F20", "S20"),
                   ("FDGAN rep2 - FDGAN rep1 @1.5", "F15r", "F15"), ("FDGAN rep2 - FDGAN rep1 @2.0", "F20r", "F20"),
                   ("FDGAN* rep2 - FDGAN* rep1 @1.5", "S15r", "S15"), ("FDGAN* rep2 - FDGAN* rep1 @2.0", "S20r", "S20"),
                   ("FDGAN rep1 - FDGAN* rep2 @2.0", "F20", "S20r"), ("FDGAN rep2 - FDGAN* rep1 @2.0", "F20r", "S20"),
                   ("FDGAN rep1 - FDGAN* rep2 @1.5", "F15", "S15r"), ("FDGAN rep2 - FDGAN* rep1 @1.5", "F15r", "S15")]:
    p, lo, hi, pb = P(R[a], R[b]); print(f"   {name:32s} {p:+8.1f}  [{lo:+6.1f}, {hi:+6.1f}]  P={pb:.3f}")
print("\nPOINT VALUES per seed (FD / KD / FID / objFD / inbox):")
for k in ["F15", "F15r", "S15", "S15r", "F20", "F20r", "S20", "S20r"]:
    t = R[k]; print(f"   {k:5s} {m_new.loc[t,'dinov2_fd']:7.1f} {m_new.loc[t,'dinov2_kd_value']:6.3f} {m_new.loc[t,'inception_fd']:6.2f} {o_new.loc[t,'crop_dinov2_fd']:7.1f} {o_new.loc[t,'det_person_in_box_iou50_pct']:5.1f}")
# mean over the two seeds and the two-seed difference
for cfg, F, S in [("1.5", ("F15", "F15r"), ("S15", "S15r")), ("2.0", ("F20", "F20r"), ("S20", "S20r"))]:
    fF = [m_new.loc[R[k], "dinov2_fd"] for k in F]; fS = [m_new.loc[R[k], "dinov2_fd"] for k in S]
    print(f"   guidance {cfg}: FDGAN mean of 2 seeds {np.mean(fF):.1f} (seeds {fF[0]:.1f}, {fF[1]:.1f}) | FDGAN* mean {np.mean(fS):.1f} (seeds {fS[0]:.1f}, {fS[1]:.1f}) | mean difference {np.mean(fF)-np.mean(fS):+.1f}")
print("\nALL CHECKS PASSED:", ok)
if ok and "--file" in sys.argv:
    for key, (src, dst) in FILES.items():
        d = os.path.join(W, dst)
        if os.path.exists(d):
            stem, ext = os.path.splitext(dst); keep = os.path.join(W, f"{stem}_superseded_21Oct_bundleA{ext}")
            n = 1
            while os.path.exists(keep): n += 1; keep = os.path.join(W, f"{stem}_superseded_21Oct_bundleA_{n}{ext}")
            os.rename(d, keep); print("kept:", os.path.basename(keep))
        shutil.copy2(os.path.join(DL, src), d); print("filed:", dst)
