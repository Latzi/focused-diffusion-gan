# -*- coding: utf-8 -*-
"""Guidance-scale figure for the supplement, from the three Step24 sheets. Palette: reference categorical slots 1-3 (validated)."""
import os, sys
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Dropbox\Vasile Marian Dropbox\Vasile Marian\iMatrix_Business\Phd\Progress_Review_3\Papers\Latest_Current\FDGAN_IVC\10_Revision_Workplan"
OUT = os.path.join(W, "Step32_figure_guidance_curves")
os.makedirs(OUT, exist_ok=True)
DS = [("Cityscapes", "Cityscapes--Pedestrian"), ("Traffic", "Traffic Signs"), ("COCO", "COCO potted plant")]
SERIES = [("FDGAN", "FDGAN", "#2a78d6"), ("ADM* (mask-conditioned ADM)", "FDGAN$^{\\ast}$ (no adversarial stage)", "#eb6834"),
          ("FDGAN without box noise", "FDGAN without localized noise", "#1baf7a")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXT, "pdf.fonttype": 42})
tabs = {ds: pd.read_excel(os.path.join(W, f"Step24_{ds}_guidance_curve_n1000.xlsx"), index_col=0) for ds, _ in DS}
rows = [("dinov2_fd", "DINOv2 FD (1,000 samples)"), ("dinov2_kd_value", "DINOv2 KD"), ("failure_rate_pct", "degenerate samples (%)")]
fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2), sharex=True)
for j, (ds, title) in enumerate(DS):
    t = tabs[ds]
    for i, (col, ylab) in enumerate(rows):
        ax = axes[i, j]
        ax.set_facecolor("#fcfcfb"); ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for spine in ("top", "right"): ax.spines[spine].set_visible(False)
        if col == "failure_rate_pct" and ds == "COCO":
            ax.text(0.5, 0.5, "threshold not applicable on COCO:\n25.6 % of the real images exceed it", ha="center", va="center", color=MUTED, fontsize=8.5, transform=ax.transAxes)
            ax.set_yticks([]); ax.set_xlabel("mask guidance scale"); continue
        for key, label, colr in SERIES:
            d = t[t["model"] == key].sort_values("cfg")
            ax.plot(d["cfg"], d[col], color=colr, linewidth=2, marker="o", markersize=5.5, markeredgecolor="#fcfcfb", markeredgewidth=1.0, label=label, zorder=3)
        if i == 0: ax.set_title(title, fontsize=10, pad=6)
        if j == 0: ax.set_ylabel(ylab)
        if i == 2: ax.set_xlabel("mask guidance scale")
        ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    # direct labels at the right end of the FD panel (top row)
    ax = axes[0, j]; t0 = tabs[ds]
    ends = sorted([(float(t0[t0["model"] == k].sort_values("cfg")["dinov2_fd"].iloc[-1]), lab, c) for k, lab, c in SERIES])
for ds_i, (ds, _) in enumerate(DS):
    pass
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
fig.suptitle("Tolerance to the mask guidance scale, final sampler (ancestral 500 steps, learned variance, colour guidance)", fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0.03, 1, 0.98))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"guidance_curves.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
print("saved", OUT)
# a compact numeric table alongside (for the supplement caption / checks)
summary = []
for ds, _ in DS:
    t = tabs[ds]
    for key, label, _c in SERIES:
        d = t[t["model"] == key].sort_values("cfg")
        summary.append([ds, label.replace("$^{\\ast}$", "*")] + [f"{v:.0f}" for v in d["dinov2_fd"]] )
print(pd.DataFrame(summary, columns=["dataset", "model", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5"]).to_string(index=False))
