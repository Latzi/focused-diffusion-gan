"""Per-image global colour statistics of a folder of images, saved as JSON for the sampler's colour guidance.
Usage: real_color_stats.py <image_root> <out_json>"""
import sys, os, json, numpy as np
from PIL import Image

root, out = sys.argv[1], sys.argv[2]
ps = []
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            ps.append(os.path.join(dp, f))
ps.sort()
means, stds, sats, bad = [], [], [], 0
for i, p in enumerate(ps):
    try:
        with Image.open(p) as im0:
            im = im0.convert("RGB")
            a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0          # model units [-1, 1]
            means.append(a.reshape(-1, 3).mean(0)); stds.append(a.reshape(-1, 3).std(0))
            small = im.resize((32, 32), Image.BOX).convert("HSV")
            sats.append(float(np.asarray(small, dtype=np.float32)[..., 1].mean()))
    except Exception as e:
        bad += 1
        print("skip", p, e)
    if (i + 1) % 500 == 0:
        print("done", i + 1, flush=True)
means = np.array(means); stds = np.array(stds)
tint = means.std(1) * 127.5
res = dict(n=int(len(means)), note="per-image RGB means/stds of the training images, model units [-1,1]; tint = std across channel means in grey levels",
           mean_rgb=means.mean(0).tolist(), std_of_means_rgb=means.std(0).tolist(), mean_std_rgb=stds.mean(0).tolist(),
           tint_mean=float(tint.mean()), tint_p95=float(np.percentile(tint, 95)), saturation_32px=float(np.mean(sats)),
           per_image_means=np.round(means, 4).tolist())
json.dump(res, open(out, "w"))
print("images", len(means), "bad", bad)
print("dataset-average image colour R,G,B in [-1,1]:", np.round(means.mean(0), 3), "= grey levels", np.round((means.mean(0) + 1) * 127.5, 1))
print("spread of per-image mean colour across images (std R,G,B, grey levels):", np.round(means.std(0) * 127.5, 1))
print("tint score: mean %.1f  p95 %.1f  max %.1f" % (tint.mean(), np.percentile(tint, 95), tint.max()))
print("within-image contrast (std R,G,B, grey levels):", np.round(stds.mean(0) * 127.5, 1))
print("saturation (32 px, HSV S):", round(float(np.mean(sats)), 1))
