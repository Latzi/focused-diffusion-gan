# -*- coding: utf-8 -*-
"""Step 23 tables: Tables 1-3 + the 'Percent improvements' paragraph from the filed sheets (22 Oct, bundle C included).
   Differences to build_step28.py: FDGAN* labels (never 'ADM + box mask'), two seed-2 no-guidance rows in Table 2,
   captions updated, no letter docx. Output: Step38_Tables_1_2_3_LaTeX.tex + a JSON of every number the text update needs."""
import sys, os, io, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
rd = lambda f: pd.read_excel(os.path.join(W, f), index_col=0)
whole = {ds: rd(f"Step14_{ds}_200k_metrics.xlsx") for ds in ("Cityscapes", "COCO", "Traffic")}
obj = {ds: rd(f"Step16_{ds}_object_metrics.xlsx") for ds in ("Cityscapes", "COCO", "Traffic")}
reps = {ds: pd.read_csv(os.path.join(W, f"Step14_{ds}_bootstrap_replicates.csv"), index_col=0) for ds in ("Cityscapes", "COCO", "Traffic")}
for d in list(whole.values()) + list(obj.values()):
    d.index = d.index.astype(str)
assert whole["Cityscapes"].shape[0] == 39 and reps["Cityscapes"].shape == (200, 23), (whole["Cityscapes"].shape, reps["Cityscapes"].shape)
whole["Cityscapes2"], obj["Cityscapes2"], reps["Cityscapes2"] = whole["Cityscapes"], obj["Cityscapes"], reps["Cityscapes"]

def row(ds, key):
    toks = key if isinstance(key, (list, tuple)) else [key]
    hits = [i for i in whole[ds].index if all(t in i for t in toks) and i.endswith(toks[-1])]
    assert len(hits) == 1, (ds, key, hits)
    return hits[0]

TAGS = {
    "Cityscapes": {
        "ADM":    {None: row("Cityscapes", ["cityscapes_adm_ema", "anc_lv"])},
        "ADM500": {None: row("Cityscapes", ["cityscapes_adm_ema", "anc500_lv"])},
        "FDGAN*": {1.5: row("Cityscapes", ["cityscapes_adm_mask_ema", "anc500_cfg15"]), 2.0: row("Cityscapes", ["cityscapes_adm_mask_ema", "anc500_cfg20"]), 1.0: row("Cityscapes", ["cityscapes_adm_mask_ema", "anc500_nocfg"])},
        "FDGAN":  {1.5: row("Cityscapes", ["cityscapes_fdgan_mask_ema", "anc500_cfg15"]), 2.0: row("Cityscapes", ["cityscapes_fdgan_mask_ema", "anc500_cfg20"]), 1.0: row("Cityscapes", ["cityscapes_fdgan_mask_ema", "anc500_nocfg"])},
        "nonoise": {1.5: row("Cityscapes", ["fdgan_nonoise", "anc500_cfg15"]), 2.0: row("Cityscapes", ["fdgan_nonoise", "anc500_cfg20"])},
        "globalnoise": {1.5: row("Cityscapes", ["globalnoise", "anc500_cfg15"]), 2.0: row("Cityscapes", ["globalnoise", "anc500_cfg20"])},
        "smallD": {1.5: row("Cityscapes", ["smallD", "anc500_cfg15"]), 2.0: row("Cityscapes", ["smallD", "anc500_cfg20"])},
        "nomask": {None: row("Cityscapes", ["fdgan_nomask", "anc500"])},
        "FDGAN_rep2":  {1.5: row("Cityscapes", ["fdgan_mask_rep2", "anc500_cfg15"]), 2.0: row("Cityscapes", ["fdgan_mask_rep2", "anc500_cfg20"]), 1.0: row("Cityscapes", ["fdgan_mask_rep2", "anc500_nocfg"])},
        "FDGAN*_rep2": {1.5: row("Cityscapes", ["adm_mask_rep2", "anc500_cfg15"]), 2.0: row("Cityscapes", ["adm_mask_rep2", "anc500_cfg20"]), 1.0: row("Cityscapes", ["adm_mask_rep2", "anc500_nocfg"])},
    },
    "Cityscapes2": {
        "FDGAN*": {1.5: row("Cityscapes", ["adm_mask_rep2", "anc500_cfg15"]), 2.0: row("Cityscapes", ["adm_mask_rep2", "anc500_cfg20"]), 1.0: row("Cityscapes", ["adm_mask_rep2", "anc500_nocfg"])},
        "FDGAN":  {1.5: row("Cityscapes", ["fdgan_mask_rep2", "anc500_cfg15"]), 2.0: row("Cityscapes", ["fdgan_mask_rep2", "anc500_cfg20"]), 1.0: row("Cityscapes", ["fdgan_mask_rep2", "anc500_nocfg"])},
    },
    "COCO": {
        "ADM":    {None: row("COCO", ["coco_adm_lsun", "anc_lv"])},
        "FDGAN*": {1.5: row("COCO", ["adm_mask_lsun", "anc500_cfg15"]), 2.0: row("COCO", ["adm_mask_lsun", "anc500_cfg20"])},
        "FDGAN":  {1.5: row("COCO", ["fdgan_mask_lsun", "anc500_cfg15"]), 2.0: row("COCO", ["fdgan_mask_lsun", "anc500_cfg20"])},
        "nonoise": {1.5: row("COCO", ["nonoise_lsun", "anc500_cfg15"]), 2.0: row("COCO", ["nonoise_lsun", "anc500_cfg20"])},
    },
    "Traffic": {
        "ADM":    {None: row("Traffic", ["traffic_adm_ema", "anc_lv"])},
        "FDGAN*": {1.5: row("Traffic", ["traffic2_adm_mask", "anc500_cfg15"]), 2.0: row("Traffic", ["traffic2_adm_mask", "anc500_cfg20"])},
        "FDGAN":  {1.5: row("Traffic", ["traffic2_fdgan_mask_ema", "anc500_cfg15"]), 2.0: row("Traffic", ["traffic2_fdgan_mask_ema", "anc500_cfg20"])},
        "nonoise": {1.5: row("Traffic", ["traffic2_fdgan_nonoise", "anc500_cfg15"]), 2.0: row("Traffic", ["traffic2_fdgan_nonoise", "anc500_cfg20"])},
        "nomask": {None: row("Traffic", ["traffic2_fdgan_nomask", "anc500"])},
    },
}
def M(ds, model, cfg, col): return float(whole[ds].loc[TAGS[ds][model][cfg], col])
def O(ds, model, cfg, col): return float(obj[ds].loc[TAGS[ds][model][cfg], col])
INBOX = {"Cityscapes": "det_person_in_box_iou50_pct", "Cityscapes2": "det_person_in_box_iou50_pct", "COCO": "det_target_in_box_iou50_pct", "Traffic": "det_target_in_box_iou50_pct"}
REALREF = {ds: float(obj[ds].iloc[0]["real_" + INBOX[ds]]) for ds in obj}
DS_T3 = ["Cityscapes", "Cityscapes2", "Traffic", "COCO"]
DS_NAME = {"Cityscapes": "Cityscapes, seed 1", "Cityscapes2": "Cityscapes, seed 2", "Traffic": "Traffic", "COCO": "COCO"}

def paired(ds, a, b):
    r = reps[ds]; d = r[a] - r[b]
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d < 0).mean())

STAR = r"FDGAN$^{\ast}$"
BASE = {
    "Projected GAN":   [(828.32, 3.98, 15.81), (586.91, 2.24, 18.18), (1094.78, 1.78, 44.15)],
    "StyleGAN2-ADA":   [(1948.78, 8.77, 71.36), (1487.25, 5.86, 96.84), (1450.62, 3.04, 76.57)],
    "DiT-XL/2-G":      [(2254.54, 11.64, 92.65), (1349.91, 4.63, 140.43), (926.54, 1.53, 56.10)],
    "Diffusion-GAN":   [(920.67, 4.33, 14.80), (616.85, 2.40, 16.88), (1010.54, 1.61, 30.71)],
    "PSGAN":           [(774.13, 5.01, 78.56), None, None],
    "OC-ProjectedGAN": [(1076.03, 5.12, 22.26), (693.43, 2.93, 33.66), (1316.79, 2.52, 61.75)],
    "Pix2Pix":         [(958.23, 4.27, 88.78), (828.10, 3.16, 116.92), (1623.55, 3.98, 62.79)],
    "LayoutDiffusion": [(1313.52, 5.27, 75.00), (680.03, 2.74, 57.65), (1047.62, 1.62, 62.95)],
}
DS_ORDER = ["Cityscapes", "Traffic", "COCO"]
CFG_T1 = 2.0
new = {}
for ds in DS_ORDER:
    new[("ADM", ds)] = (M(ds, "ADM", None, "dinov2_fd"), M(ds, "ADM", None, "dinov2_kd_value"), M(ds, "ADM", None, "inception_fd"))
    new[("FDGAN*", ds)] = (M(ds, "FDGAN*", CFG_T1, "dinov2_fd"), M(ds, "FDGAN*", CFG_T1, "dinov2_kd_value"), M(ds, "FDGAN*", CFG_T1, "inception_fd"))
    new[("FDGAN", ds)] = (M(ds, "FDGAN", CFG_T1, "dinov2_fd"), M(ds, "FDGAN", CFG_T1, "dinov2_kd_value"), M(ds, "FDGAN", CFG_T1, "inception_fd"))

ROWS_T1 = [
    ("GAN", "Projected GAN", BASE["Projected GAN"]), ("GAN", "StyleGAN2-ADA", BASE["StyleGAN2-ADA"]),
    ("DM", "ADM", [new[("ADM", ds)] for ds in DS_ORDER]),
    ("DM", STAR + " (no adversarial stage)", [new[("FDGAN*", ds)] for ds in DS_ORDER]),
    ("DM", "DiT-XL/2-G", BASE["DiT-XL/2-G"]),
    ("Hyb", r"\textbf{FDGAN} (ours)", [new[("FDGAN", ds)] for ds in DS_ORDER]), ("Hyb", "Diffusion-GAN", BASE["Diffusion-GAN"]),
    ("OC", r"PSGAN$^{\dagger}$", BASE["PSGAN"]), ("OC", r"OC-ProjectedGAN$^{\dagger}$", BASE["OC-ProjectedGAN"]),
    ("OC", r"Pix2Pix$^{\dagger}$", BASE["Pix2Pix"]), ("OC", r"LayoutDiffusion$^{\dagger}$", BASE["LayoutDiffusion"]),
]
best = {}
for di, ds in enumerate(DS_ORDER):
    for mi in range(3):
        vals = [r[2][di][mi] for r in ROWS_T1 if r[2][di] is not None]
        best[(di, mi)] = min(vals)
from decimal import Decimal, ROUND_HALF_UP
def rhu(v, nd):
    q = Decimal(1).scaleb(-nd)
    return f"{Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP):.{nd}f}"
def fmt(v, di, mi):
    if v is None: return "--"
    s = rhu(v, 2) if mi != 0 else rhu(v, 1)
    return r"\textbf{" + s + "}" if abs(v - best[(di, mi)]) < 1e-9 else s

GROUPS = {"GAN": "GAN", "DM": "DM", "Hyb": "Hyb", "OC": "OC"}
lines = []
lines.append(r"""% ---------------- COPY FROM HERE: TABLE 1 (replaces the block from \begin{center} ... \label{tab:all-datasets} ... \end{center}) ----------------
\begin{center}
\begin{minipage}{\textwidth}
\captionof{table}{Whole-image fidelity on the three low-data object-centric sets (3{,}000 samples per model, lower is better). DINOv2 FD and KD, Inception-V3 FID. ADM, FDGAN$^{\ast}$ and FDGAN are trained with one implementation and budget (128-channel U-Net, 200k iterations) and sampled with one sampler (ancestral, learned variance, colour guidance): the two box-conditioned models with 500 steps and mask guidance scale 2.0, selected on Cityscapes--Pedestrian by FD and applied unchanged to the other two sets, ADM with 250 steps and no mask guidance (ADM at 500 steps is a row of Table~\ref{tab:ablation_cityscapes}). FDGAN$^{\ast}$ is the FDGAN model trained with the adversarial stage disabled; it keeps the box-mask input channel and is the matched box-mask-conditioned control (ADM takes no mask input). On COCO these three rows start from the LSUN-bedroom checkpoint. $^{\dagger}$object-centric baselines. LayoutDiffusion is trained from scratch on Cityscapes--Pedestrian and Traffic Signs (the initialisation of its COCO row is not recorded); its COCO-Stuff-pretrained variant reaches FD 1124.0 on Cityscapes--Pedestrian and 353.9 on Traffic Signs. The other baseline rows are unchanged from the original submission. Best value per column in bold.}
\label{tab:all-datasets}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrrr}
\toprule
& & \multicolumn{3}{c}{Cityscapes--Pedestrian} & \multicolumn{3}{c}{Traffic Signs} & \multicolumn{3}{c}{COCO potted plant} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
& Model & FD$\downarrow$ & KD$\downarrow$ & FID$\downarrow$ & FD$\downarrow$ & KD$\downarrow$ & FID$\downarrow$ & FD$\downarrow$ & KD$\downarrow$ & FID$\downarrow$ \\
\midrule""")
prev = None
for g, label, vals in ROWS_T1:
    cells = []
    for di in range(3):
        v = vals[di]
        cells += [fmt(v[mi] if v is not None else None, di, mi) for mi in range(3)]
    glabel = ""
    if g != prev:
        n = sum(1 for r in ROWS_T1 if r[0] == g)
        glabel = r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{\scriptsize\textbf{%s}}}" % (n, GROUPS[g])
        if prev is not None: lines.append(r"\midrule")
    lines.append(glabel + " & " + label + " & " + " & ".join(cells) + r" \\")
    prev = g
lines.append(r"""\bottomrule
\end{tabular}}
\end{minipage}
\end{center}
""")

# ---------------------------------------------------------------- Table 2
A15 = TAGS["Cityscapes"]["FDGAN"][1.5]
abl = [
    ("ADM (no box mask, no adversarial stage), 250 steps", "ADM", None),
    ("ADM, 500 steps", "ADM500", None),
    ("FDGAN without box mask", "nomask", None),
    (STAR + " without mask guidance", "FDGAN*", 1.0),
    ("FDGAN without mask guidance", "FDGAN", 1.0),
    (STAR + ": no adversarial stage", "FDGAN*", 1.5),
    ("FDGAN without localized noise", "nonoise", 1.5),
    ("FDGAN with global noise (matched energy)", "globalnoise", 1.5),
    ("FDGAN + small ROI discriminator", "smallD", 1.5),
    (r"\textbf{FDGAN} (full)", "FDGAN", 1.5),
    (STAR + ", second training seed", "FDGAN*_rep2", 1.5),
    ("FDGAN (full), second training seed", "FDGAN_rep2", 1.5),
    (STAR + ", second training seed, without mask guidance", "FDGAN*_rep2", 1.0),
    ("FDGAN (full), second training seed, without mask guidance", "FDGAN_rep2", 1.0),
]
T2 = []
for label, key, cfg in abl:
    t = TAGS["Cityscapes"][key][cfg]
    fd, kd, fid = M("Cityscapes", key, cfg, "dinov2_fd"), M("Cityscapes", key, cfg, "dinov2_kd_value"), M("Cityscapes", key, cfg, "inception_fd")
    cfd = O("Cityscapes", key, cfg, "crop_dinov2_fd"); inbox = O("Cityscapes", key, cfg, INBOX["Cityscapes"])
    if t == A15:
        dstr = "--"
    else:
        mean, lo, hi, p = paired("Cityscapes", t, A15)
        pt = fd - M("Cityscapes", "FDGAN", 1.5, "dinov2_fd")
        dstr = f"${pt:+.0f}$ [{lo:+.0f}, {hi:+.0f}]"
    fd20 = "--"
    if cfg == 1.5 and 2.0 in TAGS["Cityscapes"][key]:
        fd20 = f"{M('Cityscapes', key, 2.0, 'dinov2_fd'):.1f}"
    T2.append((label, fd, kd, fid, cfd, inbox, dstr, fd20))
lines.append(r"""% ---------------- COPY TO HERE: END OF TABLE 1 ----------------

% ---------------- COPY FROM HERE: TABLE 2 (replaces the block from \begin{center} ... \label{tab:ablation_cityscapes} ... \end{center}) ----------------
\begin{center}
\begin{minipage}{\textwidth}
\captionof{table}{Component ablation on Cityscapes--Pedestrian. Every row is trained with the same code, data and budget and differs from the full model by the flags named; all rows are sampled with the final sampler (3{,}000 samples), the box-conditioned rows at mask guidance 1.5 unless the row says otherwise. The last four rows are FDGAN$^{\ast}$ and the full model trained again with a second training seed, sampled with mask guidance (1.5, and 2.0 in the last column) and without it. $\Delta$FD is the FD of the row minus the FD of the full model (first seed) at guidance 1.5, with the 95\,\% paired bootstrap interval of that difference in brackets (200 replicates, shared real resample; positive = worse than FDGAN). Object FD is the DINOv2 FD of square crops at the conditioned box (1.2 times the longer box side, resized to 128 px); ``in box'' is the share of samples in which a COCO-pretrained YOLO11n detects a person inside the conditioned box (IoU $\geq 0.5$; real images: @@REF@@\,\%). The last column gives FD at guidance 2.0 for the rows sampled at both scales.}
\label{tab:ablation_cityscapes}
\setlength{\tabcolsep}{5pt}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{lrrrrrcr}
\toprule
Configuration & FD$\downarrow$ & KD$\downarrow$ & FID$\downarrow$ & Object FD$\downarrow$ & In box (\%) & $\Delta$FD vs.\ FDGAN & FD @ 2.0 \\
\midrule""".replace("@@REF@@", f"{REALREF['Cityscapes']:.1f}"))
for label, fd, kd, fid, cfd, inbox, dstr, fd20 in T2:
    lines.append(f"{label} & {fd:.1f} & {kd:.3f} & {fid:.2f} & {cfd:.1f} & {inbox:.1f} & {dstr} & {fd20} \\\\")
lines.append(r"""\bottomrule
\end{tabular}}
\end{minipage}
\end{center}
""")

# ---------------------------------------------------------------- Table 3
T3 = []
for ds in DS_T3:
    for cfg in (1.5, 2.0):
        a, b = TAGS[ds]["FDGAN"][cfg], TAGS[ds]["FDGAN*"][cfg]
        fdA, fdB = M(ds, "FDGAN", cfg, "dinov2_fd"), M(ds, "FDGAN*", cfg, "dinov2_fd")
        mean, lo, hi, p = paired(ds, a, b)
        rel = lambda col, src=M: 100 * (src(ds, "FDGAN", cfg, col) / src(ds, "FDGAN*", cfg, col) - 1)
        T3.append((ds, cfg, fdA, fdB, fdA - fdB, lo, hi, p, rel("dinov2_kd_value"), rel("inception_fd"),
                   100 * (O(ds, "FDGAN", cfg, "crop_dinov2_fd") / O(ds, "FDGAN*", cfg, "crop_dinov2_fd") - 1),
                   O(ds, "FDGAN", cfg, INBOX[ds]), O(ds, "FDGAN*", cfg, INBOX[ds])))
lines.append(r"""% ---------------- COPY TO HERE: END OF TABLE 2 ----------------

% ---------------- COPY FROM HERE: TABLE 3 ----------------
\begin{center}
\begin{minipage}{\textwidth}
\captionof{table}{Effect of the adversarial stage: FDGAN against the identically trained and sampled model with the adversarial stage disabled (FDGAN$^{\ast}$), at both guidance scales, 3{,}000 samples each. On Cityscapes--Pedestrian both models were trained twice with different training seeds (seed 1 = the rows of Table~\ref{tab:all-datasets}); the comparison is made within each seed. $\Delta$FD is the paired bootstrap difference with its 95\,\% interval and the share of replicates in which FDGAN is better; the relative columns are FDGAN relative to FDGAN$^{\ast}$ (negative = FDGAN better). ``In box'' uses a COCO-pretrained YOLO11n (person) on Cityscapes and detectors trained on the real training images of each set for Traffic Signs and COCO; the real-image references are @@R1@@, @@R2@@ and @@R3@@\,\%.}
\label{tab:decomposition}
\setlength{\tabcolsep}{5pt}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrcrrrr}
\toprule
Dataset & Guidance & FD FDGAN & FD FDGAN$^{\ast}$ & $\Delta$FD [95\,\%] & $P$(FDGAN better) & $\Delta$KD & $\Delta$FID & $\Delta$Object FD & In box FDGAN / FDGAN$^{\ast}$ (\%) \\
\midrule""".replace("@@R1@@", f"{REALREF['Cityscapes']:.1f}").replace("@@R2@@", f"{REALREF['Traffic']:.1f}").replace("@@R3@@", f"{REALREF['COCO']:.1f}"))
for ds, cfg, fdA, fdB, d, lo, hi, p, dkd, dfid, dcrop, ibA, ibB in T3:
    star = r"$^{\ast}$" if hi < 0 else ""
    lines.append(f"{DS_NAME[ds]} & {cfg:.1f} & {fdA:.1f} & {fdB:.1f} & ${d:+.1f}$ [{lo:+.0f}, {hi:+.0f}]{star} & {p:.2f} & {dkd:+.1f}\\,\\% & {dfid:+.1f}\\,\\% & {dcrop:+.1f}\\,\\% & {ibA:.1f} / {ibB:.1f} \\\\")
lines.append(r"""\bottomrule
\end{tabular}}
\end{minipage}
\end{center}
% $^{\ast}$ interval entirely below zero.
""")

# ---------------------------------------------------------------- 'Percent improvements' paragraph
c15 = [r for r in T3 if r[0] == "Cityscapes" and r[1] == 1.5][0]; c20 = [r for r in T3 if r[0] == "Cityscapes" and r[1] == 2.0][0]
s15 = [r for r in T3 if r[0] == "Cityscapes2" and r[1] == 1.5][0]; s20 = [r for r in T3 if r[0] == "Cityscapes2" and r[1] == 2.0][0]
co20 = [r for r in T3 if r[0] == "COCO" and r[1] == 2.0][0]; tr20 = [r for r in T3 if r[0] == "Traffic" and r[1] == 2.0][0]
pc = lambda a, b: 100 * (1 - a / b)
fd_c, fd_t, fd_o = new[("FDGAN", "Cityscapes")][0], new[("FDGAN", "Traffic")][0], new[("FDGAN", "COCO")][0]
adm_c, adm_t, adm_o = new[("ADM", "Cityscapes")][0], new[("ADM", "Traffic")][0], new[("ADM", "COCO")][0]
st_c, st_t, st_o = new[("FDGAN*", "Cityscapes")][0], new[("FDGAN*", "Traffic")][0], new[("FDGAN*", "COCO")][0]
C = TAGS["Cityscapes"]; C2 = TAGS["Cityscapes2"]
ng1 = paired("Cityscapes", C["FDGAN"][1.0], C["FDGAN*"][1.0]); ng1_pt = M("Cityscapes", "FDGAN", 1.0, "dinov2_fd") - M("Cityscapes", "FDGAN*", 1.0, "dinov2_fd")
ng2 = paired("Cityscapes", C2["FDGAN"][1.0], C2["FDGAN*"][1.0]); ng2_pt = M("Cityscapes2", "FDGAN", 1.0, "dinov2_fd") - M("Cityscapes2", "FDGAN*", 1.0, "dinov2_fd")
para1 = (r"\noindent\emph{Percent improvements.} For lower-is-better metrics we report relative reductions as $100\times(\text{baseline}-\text{FDGAN})/\text{baseline}$ (DINOv2 FD unless stated). "
         f"On Cityscapes--Pedestrian, FDGAN reaches a DINOv2 FD of {fd_c:.1f}, {pc(fd_c, 774.13):.1f}\\,\\% below PSGAN (774.1) and {pc(fd_c, 828.32):.1f}\\,\\% below Projected GAN (828.3); against the plain ADM trained with the same code and budget ({adm_c:.1f}) the reduction is {pc(fd_c, adm_c):.1f}\\,\\%, of which the box-mask conditioning with guidance accounts for most: FDGAN$^{{\\ast}}$ ({st_c:.1f}), the same model without the adversarial stage, is {pc(st_c, adm_c):.1f}\\,\\% below ADM, and the adversarial stage lowers FD by a further {abs(c20[4]):.1f}, i.e.\\ {pc(fd_c, st_c):.1f}\\,\\% of the FDGAN$^{{\\ast}}$ value (paired bootstrap difference ${c20[4]:+.1f}$, 95\\,\\% interval [{c20[5]:+.0f}, {c20[6]:+.0f}]; at guidance 1.5: ${c15[4]:+.1f}$ [{c15[5]:+.0f}, {c15[6]:+.0f}]; without mask guidance: ${ng1_pt:+.1f}$ [{ng1[1]:+.0f}, {ng1[2]:+.0f}]); with a second training seed of both models the same comparison gives ${s20[4]:+.1f}$ [{s20[5]:+.0f}, {s20[6]:+.0f}] at 2.0, ${s15[4]:+.1f}$ [{s15[5]:+.0f}, {s15[6]:+.0f}] at 1.5 and ${ng2_pt:+.1f}$ [{ng2[1]:+.0f}, {ng2[2]:+.0f}] without guidance, so under guidance the adversarial gain in FD is within training-seed variation, while without guidance it is resolved for both seeds (Section~\\ref{{sec:ablation}}). "
         f"On Traffic Signs, FDGAN ({fd_t:.1f}) and FDGAN$^{{\\ast}}$ ({st_t:.1f}) are statistically indistinguishable (${tr20[4]:+.1f}$ [{tr20[5]:+.0f}, {tr20[6]:+.0f}]); both are 45 to 46\\,\\% below Projected GAN (586.9) and below ADM ({adm_t:.1f}), which differ by 0.3 FD. "
         f"On the COCO potted-plant subset, with the three rows trained with our implementation (ADM, FDGAN$^{{\\ast}}$, FDGAN) initialised from the LSUN-bedroom checkpoint, FDGAN reaches {fd_o:.1f}, {pc(fd_o, 926.54):.1f}\\,\\% below DiT-XL/2-G (926.5) and {pc(fd_o, 1094.78):.1f}\\,\\% below Projected GAN; the difference to FDGAN$^{{\\ast}}$ ({st_o:.1f}) is ${co20[4]:+.1f}$ [{co20[5]:+.0f}, {co20[6]:+.0f}], within sampling uncertainty, while KD and the object-level FD favour FDGAN by {abs(co20[8]):.1f}\\,\\% and {abs(co20[10]):.1f}\\,\\%. "
         r"We therefore attribute most of the gain over the original ADM baseline to the box-mask conditioning with guidance; adversarial training with localized perturbation adds a smaller gain that is resolved against sampling uncertainty for one training seed on Cityscapes--Pedestrian (and for both seeds when no mask guidance is applied), within training-seed variation under guidance when a second seed is trained, and within sampling uncertainty on the other two sets (Table~\ref{tab:decomposition}).")
lines.append("% ---------------- COPY TO HERE: END OF TABLE 3 ----------------")
lines.append("")
lines.append("% ===================== SECTION 4 TEXT =====================")
lines.append(r"% REPLACE the paragraph that begins with:  \noindent\emph{Percent improvements.} ... by:")
lines.append(para1 + "\n")
tex = "\n".join(lines)
out_tex = os.path.join(W, "Step38_Tables_1_2_3_LaTeX.tex")
io.open(out_tex, "w", encoding="utf-8", newline="\n").write(tex)
print("wrote", out_tex, len(tex), "chars")

# ---------------------------------------------------------------- every number the text update needs (JSON for patch_step23.py)
def P(ds, a, b):
    m, lo, hi, p = paired(ds, a, b); return {"mean": m, "lo": lo, "hi": hi, "lo_s": f"{lo:+.0f}", "hi_s": f"{hi:+.0f}", "p": p}
def PT(ds, a, b): return float(whole[ds].loc[a, "dinov2_fd"] - whole[ds].loc[b, "dinov2_fd"])
Q = {}
def q(name, ds, a, b):
    d = P(ds, a, b); d["point"] = PT(ds, a, b); d["abs_lo"] = f"{abs(d['lo']):.0f}"; d["abs_hi"] = f"{abs(d['hi']):.0f}"; Q[name] = d
q("nomask-FDGAN15", "Cityscapes", C["nomask"][None], C["FDGAN"][1.5])
q("FDGAN10-FDGAN15", "Cityscapes", C["FDGAN"][1.0], C["FDGAN"][1.5])
q("FDGAN*10-FDGAN10", "Cityscapes", C["FDGAN*"][1.0], C["FDGAN"][1.0])          # seed-1 no-guidance adversarial effect, positive = FDGAN better
q("FDGAN*15-FDGAN15", "Cityscapes", C["FDGAN*"][1.5], C["FDGAN"][1.5])
q("FDGAN*20-FDGAN20", "Cityscapes", C["FDGAN*"][2.0], C["FDGAN"][2.0])
q("nonoise15-FDGAN15", "Cityscapes", C["nonoise"][1.5], C["FDGAN"][1.5])
q("nonoise20-FDGAN20", "Cityscapes", C["nonoise"][2.0], C["FDGAN"][2.0])
q("global15-FDGAN15", "Cityscapes", C["globalnoise"][1.5], C["FDGAN"][1.5])
q("global20-FDGAN20", "Cityscapes", C["globalnoise"][2.0], C["FDGAN"][2.0])
q("nonoise15-FDGAN*15", "Cityscapes", C["nonoise"][1.5], C["FDGAN*"][1.5])
q("nonoise20-FDGAN*20", "Cityscapes", C["nonoise"][2.0], C["FDGAN*"][2.0])
q("smallD15-FDGAN15", "Cityscapes", C["smallD"][1.5], C["FDGAN"][1.5])
q("smallD20-FDGAN20", "Cityscapes", C["smallD"][2.0], C["FDGAN"][2.0])
q("FDGANrep2-rep1_15", "Cityscapes", C2["FDGAN"][1.5], C["FDGAN"][1.5])
q("FDGANrep2-rep1_20", "Cityscapes", C2["FDGAN"][2.0], C["FDGAN"][2.0])
q("FDGAN*rep2-rep1_15", "Cityscapes", C2["FDGAN*"][1.5], C["FDGAN*"][1.5])
q("FDGAN*rep2-rep1_20", "Cityscapes", C2["FDGAN*"][2.0], C["FDGAN*"][2.0])
q("rep2_FDGAN-FDGAN*_15", "Cityscapes", C2["FDGAN"][1.5], C2["FDGAN*"][1.5])
q("rep2_FDGAN-FDGAN*_20", "Cityscapes", C2["FDGAN"][2.0], C2["FDGAN*"][2.0])
q("rep2_FDGAN-FDGAN*_10", "Cityscapes", C2["FDGAN"][1.0], C2["FDGAN*"][1.0])    # seed-2 no-guidance adversarial effect, negative = FDGAN better
q("rep1_FDGAN-FDGAN*_10", "Cityscapes", C["FDGAN"][1.0], C["FDGAN*"][1.0])
q("ADM500-ADM250", "Cityscapes", C["ADM500"][None], C["ADM"][None])
Q["values"] = {"FDGAN*_rep2_nocfg_fd": M("Cityscapes2", "FDGAN*", 1.0, "dinov2_fd"), "FDGAN_rep2_nocfg_fd": M("Cityscapes2", "FDGAN", 1.0, "dinov2_fd"),
               "FDGAN*_rep2_nocfg_inbox": O("Cityscapes2", "FDGAN*", 1.0, INBOX["Cityscapes"]), "FDGAN_rep2_nocfg_inbox": O("Cityscapes2", "FDGAN", 1.0, INBOX["Cityscapes"]),
               "FDGAN*_rep2_nocfg_objfd": O("Cityscapes2", "FDGAN*", 1.0, "crop_dinov2_fd"), "FDGAN_rep2_nocfg_objfd": O("Cityscapes2", "FDGAN", 1.0, "crop_dinov2_fd"),
               "FDGAN*_nocfg_fd": M("Cityscapes", "FDGAN*", 1.0, "dinov2_fd"), "FDGAN_nocfg_fd": M("Cityscapes", "FDGAN", 1.0, "dinov2_fd")}
json.dump(Q, io.open(os.path.join(W, "Step38_numbers_for_text.json"), "w", encoding="utf-8"), indent=1)
print("\n[quantities for the text update]  name: point | mean [lo, hi] P(a better)")
for k, d in Q.items():
    if k == "values": continue
    print(f"  {k:24s} point {d['point']:+7.1f} | {d['mean']:+7.1f} [{d['lo_s']}, {d['hi_s']}]  P={d['p']:.3f}")
print("  values:", {k: round(v, 2) for k, v in Q["values"].items()})
print("\nTable 2 rows:")
for t in T2: print("  ", t[0][:58].ljust(58), f"{t[1]:.1f} {t[2]:.3f} {t[3]:.2f} | {t[4]:.1f} {t[5]:.1f} | {t[6]} | {t[7]}")
print("\nTable 3 rows:")
for r in T3: print("  ", r[0], r[1], f"FD {r[2]:.1f} vs {r[3]:.1f} | d {r[4]:+.1f} [{r[5]:+.0f},{r[6]:+.0f}] P {r[7]:.2f} | KD {r[8]:+.1f}% FID {r[9]:+.1f}% crop {r[10]:+.1f}% | inbox {r[11]:.1f}/{r[12]:.1f}")
