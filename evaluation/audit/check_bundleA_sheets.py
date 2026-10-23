# -*- coding: utf-8 -*-
"""Check the 21 Oct sheets (bundle A results) against the filed 19/20 Oct sheets, and compute the new paired comparisons."""
import io, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
DL = r"C:\Users\latzi\Downloads"
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
NEW = {"Cityscapes": "Cityscapes_200k_metrics (6).xlsx", "Traffic": "Traffic_200k_metrics (3).xlsx", "COCO": "COCO_200k_metrics (2).xlsx"}
NEWOBJ = {"Cityscapes": "Cityscapes_object_metrics (3).xlsx", "Traffic": "Traffic_object_metrics (3).xlsx", "COCO": "COCO_object_metrics (2).xlsx"}
EXPECTED_NEW = {"Cityscapes": ["cityscapes_adm_ema200000_n3000_r500_colsample_anc500_lv", "cityscapes_adm_mask_ema200000_n3000_r500_colsample_anc500_nocfg",
                               "cityscapes_fdgan_nonoise_ema200000_n3000_r500_colsample_anc500_cfg20", "cityscapes_fdgan_mask_smallD_ema200000_n3000_r500_colsample_anc500_cfg20",
                               "cityscapes_fdgan_globalnoise_ema200000_n3000_r500_colsample_anc500_cfg20"],
                "Traffic": ["traffic_adm_ema200000_n3000_r500_colsample_anc500_lv"], "COCO": ["coco_adm_lsun_ema200000_n3000_r500_colsample_anc500_lv"]}

def load(p):
    d = pd.read_excel(p, index_col=0); d.index = d.index.astype(str); return d

def compare(old, new, label):
    common_rows = [r for r in old.index if r in new.index]
    missing = [r for r in old.index if r not in new.index]
    added = [r for r in new.index if r not in old.index]
    num_cols = [c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c])]
    diffs = []
    for r in common_rows:
        for c in num_cols:
            a, b = old.loc[r, c], new.loc[r, c]
            if pd.isna(a) and pd.isna(b): continue
            if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > 1e-6:
                diffs.append((r, c, a, b))
    print(f"[{label}] old rows {len(old)} | new rows {len(new)} | added {len(added)} | missing from new {missing or 'none'} | changed cells in old rows: {len(diffs)}")
    for d in diffs[:12]: print("    CHANGED:", d)
    return added

print("================ WHOLE-IMAGE SHEETS ================")
new_metrics = {}
for ds, f in NEW.items():
    new = load(os.path.join(DL, f)); old = load(os.path.join(W, f"Step14_{ds}_200k_metrics.xlsx"))
    added = compare(old, new, f"{ds} whole-image")
    assert sorted(added) == sorted(EXPECTED_NEW[ds]), f"unexpected new rows in {ds}: {added}"
    cols = ["dinov2_fd", "dinov2_fd_infinity_value", "dinov2_kd_value", "inception_fd", "dinov2_precision", "dinov2_recall", "dinov2_ct", "dinov2_ct_modified",
            "dinov2_fls", "dinov2_fls_overfit", "failure_rate_pct", "grain_median", "tint", "saturation", "unet_evals_per_image", "seconds_per_image", "timestep_respacing", "mask_cfg_scale", "roi_bump_mode"]
    print(new.loc[added, [c for c in cols if c in new.columns]].round(3).to_string())
    nan_ct = new["dinov2_ct_modified"].isna().sum()
    print(f"   rows with blank CT-mod: {nan_ct}" + ("" if nan_ct == 0 else "   <-- C_T fill cell not run?"))
    new_metrics[ds] = new

print("\n================ OBJECT SHEETS ================")
for ds, f in NEWOBJ.items():
    new = load(os.path.join(DL, f)); old = load(os.path.join(W, f"Step16_{ds}_object_metrics.xlsx"))
    added = compare(old, new, f"{ds} object")
    if sorted(added) != sorted(EXPECTED_NEW[ds]):
        print(f"   WARNING {ds} object sheet: expected new rows {EXPECTED_NEW[ds]}, found {added} -> STALE SHEET, re-send after the loop/detector cells")
    detcols = [c for c in new.columns if c.startswith("det_") or c.startswith("real_det_") or c == "detector"]
    print("   columns:", list(new.columns))
    show = [c for c in ["crop_dinov2_fd", "crop_dinov2_kd_value", "crop_inception_fd"] + detcols if c in new.columns]
    print(new.loc[added, show].round(3).to_string())
    if "detector" in new.columns:
        print("   detector column values over all rows:", new["detector"].astype(str).unique().tolist())
    for c in [c for c in new.columns if c.startswith("real_det_")]:
        print(f"   {c}: unique over rows = {np.round(new[c].dropna().unique(), 3).tolist()}")

print("\n================ BOOTSTRAP ================")
bx = os.path.join(DL, "Cityscapes_bootstrap_FD (2).xlsx"); bc = os.path.join(DL, "Cityscapes_bootstrap_replicates (1).csv")
per = pd.read_excel(bx, sheet_name="per_model", index_col=0); pairs = pd.read_excel(bx, sheet_name="pairs")
rep = pd.read_csv(bc, index_col=0)
print("per_model rows:", len(per), "| replicates csv shape:", rep.shape, "| B =", per["B"].iloc[0], "| n_real =", per["n_real"].iloc[0])
print("rows:", list(rep.columns))
# recompute per-model stats from the csv
ok = True
for t in per.index:
    v = rep[t].values
    for col, val in (("boot_mean", v.mean()), ("boot_std", v.std(ddof=1)), ("ci95_low", np.percentile(v, 2.5)), ("ci95_high", np.percentile(v, 97.5))):
        if abs(per.loc[t, col] - val) > 1e-6: ok = False; print("   MISMATCH", t, col, per.loc[t, col], val)
print("per_model statistics recomputed from the csv:", "all match" if ok else "MISMATCHES ABOVE")
# fd_point vs the new sheet
cs = new_metrics["Cityscapes"]
for t in per.index:
    if abs(per.loc[t, "fd_point"] - cs.loc[t, "dinov2_fd"]) > 1e-6: print("   fd_point differs from sheet:", t)
print("\npairs sheet (A = reference row):"); print(pairs.round(2).to_string(index=False))

def short(t):
    return (t.replace("cityscapes_", "").replace("_ema200000_n3000", "").replace("_colsample", ""))
def pair(a, b):
    d = rep[a].values - rep[b].values
    return (cs.loc[a, "dinov2_fd"] - cs.loc[b, "dinov2_fd"], np.percentile(d, 2.5), np.percentile(d, 97.5), float((d < 0).mean()))
R = {k: next(t for t in rep.columns if all(s in t for s in v)) for k, v in {
    "FDGAN15": ["fdgan_mask_ema", "anc500_cfg15"], "FDGAN20": ["fdgan_mask_ema", "anc500_cfg20"], "FDGAN10": ["fdgan_mask_ema", "anc500_nocfg"],
    "STAR15": ["adm_mask_ema", "anc500_cfg15"], "STAR20": ["adm_mask_ema", "anc500_cfg20"], "STAR10": ["adm_mask_ema", "anc500_nocfg"],
    "ADM250": ["cityscapes_adm_ema", "r250", "anc_lv"], "ADM500": ["cityscapes_adm_ema", "r500", "anc500_lv"],
    "NONOISE15": ["nonoise", "anc500_cfg15"], "NONOISE20": ["nonoise", "anc500_cfg20"], "SMALLD15": ["smallD", "anc500_cfg15"], "SMALLD20": ["smallD", "anc500_cfg20"],
    "GLOBAL15": ["globalnoise", "anc500_cfg15"], "GLOBAL20": ["globalnoise", "anc500_cfg20"], "NOMASK": ["nomask", "anc500"]}.items()}
print("\nNEW paired comparisons from the replicates (point difference, 95% interval, P(first better)):")
for a, b in [("FDGAN20", "STAR20"), ("FDGAN15", "STAR15"), ("FDGAN10", "STAR10"), ("FDGAN10", "ADM500"), ("STAR10", "ADM500"), ("STAR10", "ADM250"),
             ("ADM500", "ADM250"), ("FDGAN20", "NONOISE20"), ("FDGAN20", "SMALLD20"), ("FDGAN20", "GLOBAL20"), ("FDGAN15", "NONOISE15"), ("FDGAN15", "SMALLD15"),
             ("FDGAN15", "GLOBAL15"), ("FDGAN15", "NOMASK"), ("FDGAN15", "ADM250"), ("FDGAN20", "ADM250"), ("FDGAN20", "ADM500"), ("NONOISE20", "STAR20"), ("GLOBAL20", "NONOISE20")]:
    p, lo, hi, pb = pair(R[a], R[b])
    print(f"   {a:10s} - {b:10s}: {p:+8.2f}  [{lo:+7.1f}, {hi:+7.1f}]   P={pb:.3f}")

print("\nOLD (19 Oct, 12-row) bootstrap pairs for the same comparisons, to see how far the intervals moved:")
old_rep = pd.read_csv(os.path.join(W, "Step14_Cityscapes_bootstrap_replicates.csv"), index_col=0)
old_cs = load(os.path.join(W, "Step14_Cityscapes_200k_metrics.xlsx"))
def opair(a, b):
    d = old_rep[a].values - old_rep[b].values
    return (old_cs.loc[a, "dinov2_fd"] - old_cs.loc[b, "dinov2_fd"], np.percentile(d, 2.5), np.percentile(d, 97.5), float((d < 0).mean()))
for a, b in [("FDGAN20", "STAR20"), ("FDGAN15", "STAR15"), ("FDGAN15", "NONOISE15"), ("FDGAN15", "SMALLD15"), ("FDGAN15", "GLOBAL15"), ("FDGAN15", "NOMASK"), ("FDGAN15", "ADM250"), ("FDGAN10", "FDGAN15")]:
    if R[a] in old_rep.columns and R[b] in old_rep.columns:
        p, lo, hi, pb = opair(R[a], R[b]); print(f"   {a:10s} - {b:10s}: {p:+8.2f}  [{lo:+7.1f}, {hi:+7.1f}]   P={pb:.3f}")
