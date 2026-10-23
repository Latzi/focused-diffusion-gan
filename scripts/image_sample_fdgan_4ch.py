# -*- coding: utf-8 -*-
"""
image_sample_fdgan_4ch.py  -  sample from a checkpoint trained with scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py
(the 2026 FDGAN code), with or without the four-channel mask input.

Default behaviour = the paper's stated sampling procedure:
  * BB mask rasterised from the dataset's per-image JSON boxes ({x,y,w,h} = x1,y1,x2,y2), cycled in sorted order;
  * with --use_bb_mask_cond True the mask (times a gain) is fed to the U-Net as the 4th channel at every reverse step
    (--mask_mode always | off | band | ramp);
  * ANPM at sampling (--roi_bump_mode): 'predict' (default) perturbs the box of a COPY of the state at every in-band step
    and lets the network predict from it, as in the training perturbed pass, while the chain continues unperturbed;
    'once' / 'band' add noise to the chain itself; 'none' disables it. Window: --bump_tmin_rev/--bump_tmax_rev (reverse-t of
    the training horizon), mapped onto the respaced chain;
  * discriminator NOT used; mask classifier-free guidance OFF (--mask_cfg_scale 1.0); fixed posterior variance (--sigma_mode small).

Optional quality knobs (all off by default so the default run is unchanged):
  --mask_cfg_scale S      classifier-free guidance on the mask channel, x0 space: x0 = x0_box + (S-1)*(x0_box - x0_nobox), where
                          x0_nobox is the prediction with the mask channel zeroed. S=1 off; 1.5..3 sharpens the box content.
                          --mask_cfg_region all|roi (roi = only inside the padded box), --cfg_tmin_rev/--cfg_tmax_rev window.
                          Costs a second U-Net pass per step inside the window.
  --sigma_mode learned    ancestral chain uses the U-Net's learned variance (ADM's own sampler) instead of the fixed posterior one.

Optional (off by default, --guide_mode main | small | both): plug-and-play guidance from the discriminators saved with
every checkpoint: D_main<step>.pt (MainDWithMaskAdapter around SpectralNLayerDiscriminator, whole image + mask) and, for
runs trained with --enable_small_roi_d, D_small<step>.pt (SmallROIPatchD on the 128 px crop around the box). This is a
port of the September 2025 image_sample_pnpd_jit.py guidance step (d_space x0, mean_roi normalisation, pnp_target) onto
the 2026 models. Using it means the discriminator IS used at inference, which the paper must then say.
Strength: with --pnp_norm mean_roi and --pnp_target T the mean push per guided step inside the box is pnp_eta * T in
image units (the image range is 2), so eta 0.006 * 0.45 = 0.003 per step is invisible; eta 0.05 .. 0.3 is the useful
range. --max_step_scale caps the normalised scale and silently weakens the push when the raw gradient is tiny; the log
prints the realised mean push and how often the cap bit.
Global colour guidance (--color_guide_mode mean | sample, off by default): the samples of the 128-channel models carry
strong global colour casts (tint score ~26 against 4.9 for real images). At every step inside --color_guide_tmin_rev ..
--color_guide_tmax_rev the per-channel means of the clean estimate are pulled (strength 1.0 = set) towards a target
colour: the dataset-average colour ('mean') or, per sample, the mean colour of a randomly drawn real training image
('sample', which reproduces the real spread of global colour). Targets come from --color_stats_json, written by
real_color_stats.py from the training images. This is an inference-time statistical correction and must be applied to
every compared model alike and stated in the paper.

Outputs <out_dir>/sample_000000.png ... (flat, dgm-eval ready) and <out_dir>/meta.json. Smoke-test with --num_samples 16.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# ----------------------------------------------------------------------------- boxes and masks
def load_bbs(bb_dir):
    """Read every box JSON in bb_dir in sorted file order; returns a list of (box dict, file stem). Accepts the same key
    variants as the training loader.
    """
    out = []
    for name in sorted(os.listdir(bb_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(bb_dir, name), "r") as f:
            bb = json.load(f)
        if isinstance(bb, list):
            if len(bb) == 0:
                continue
            bb = bb[0]
        if not isinstance(bb, dict):
            continue
        def pick(*keys):
            for k in keys:
                if k in bb:
                    return bb[k]
            return None
        x, y, w, h = pick("x", "x1", "xmin"), pick("y", "y1", "ymin"), pick("w", "x2", "xmax", "width"), pick("h", "y2", "ymax", "height")
        if None in (x, y, w, h):
            continue
        out.append(({"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, os.path.splitext(name)[0]))
    return out


def bb_to_xyxy(bb, H, W, bb_is_xywh, end_inclusive):
    """Box dict to clamped exclusive corner coordinates, for both stored formats."""
    x0, y0 = int(bb["x"]), int(bb["y"])
    if bb_is_xywh:
        x1, y1 = x0 + int(bb["w"]), y0 + int(bb["h"])
    else:
        x1, y1 = int(bb["w"]), int(bb["h"])
        if end_inclusive:
            x1, y1 = x1 + 1, y1 + 1
    x0 = max(0, min(W - 1, x0)); y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1)); y1 = max(y0 + 1, min(H, y1))
    return x0, y0, x1, y1


def xyxy_to_mask(x0, y0, x1, y1, H, W, device):
    """Binary (1, H, W) mask of a box."""
    m = th.zeros((1, H, W), device=device, dtype=th.float32)
    m[:, y0:y1, x0:x1] = 1.0
    return m


def pad_xyxy(x0, y0, x1, y1, H, W, pad_ratio):
    """Grow a box by pad_ratio of its width and height on each side, clamped to the image."""
    bw, bh = x1 - x0, y1 - y0
    px, py = int(round(bw * pad_ratio)), int(round(bh * pad_ratio))
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


# ----------------------------------------------------------------------------- four-channel generator wrapper
def inflate_first_conv_to_4ch(unet):
    """Widen the first U-Net convolution to four input channels so that the trained 4-channel weights can be loaded; the
    extra channel starts at zero and is overwritten by the checkpoint.
    """
    conv = unet.input_blocks[0][0]
    if conv.in_channels == 4:
        return
    new_conv = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
    with th.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :3] = conv.weight
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    unet.input_blocks[0][0] = new_conv
    if hasattr(unet, "in_channels"):
        unet.in_channels = 4


class BB4ChWrapper(nn.Module):
    """model(x3, t, bb_mask=...) -> cat([x, mask * gain]) -> U-Net. gain is 1.0 here; the per-step gain is in the mask."""
    def __init__(self, unet, mask_gain=1.0):
        super().__init__()
        self.unet = unet
        self.mask_gain = float(mask_gain)

    def forward(self, x, timesteps, **kwargs):
        bb_mask = kwargs.pop("bb_mask", None)
        if bb_mask is None:
            bb_mask = th.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        if bb_mask.dim() == 3:
            bb_mask = bb_mask.unsqueeze(1)
        x4 = th.cat([x, bb_mask.to(dtype=x.dtype) * self.mask_gain], dim=1)
        if "y" in kwargs:
            return self.unet(x4, timesteps, y=kwargs["y"])
        return self.unet(x4, timesteps)


def load_weights(model, path):
    """Load an EMA or model checkpoint into the model, adding or removing the 'unet.' prefix of the wrapper as needed.
    """
    sd = dist_util.load_state_dict(path, map_location="cpu")
    tgt_prefixed = next(iter(model.state_dict().keys())).startswith("unet.")
    src_prefixed = next(iter(sd.keys())).startswith("unet.")
    if tgt_prefixed and not src_prefixed:
        sd = {"unet." + k: v for k, v in sd.items()}
    elif src_prefixed and not tgt_prefixed:
        sd = {k[len("unet."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        logger.log(f"[load] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
        for k in list(missing)[:6]:
            logger.log("   missing: " + k)
        if len(missing) > 0:
            raise RuntimeError("checkpoint does not match the model: wrong --use_bb_mask_cond or wrong model flags?")


# ----------------------------------------------------------------------------- the saved main discriminator
class SpectralNLayerDiscriminator(nn.Module):
    """Identical to the class in scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py (1 Mar 2026)."""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_sigmoid=False):
        super().__init__()
        layers = []
        ch = input_nc
        for i in range(n_layers):
            out_ch = int(ndf * min(2 ** i, 4))
            stride = 2 if i < (n_layers - 1) else 1
            layers += [spectral_norm(nn.Conv2d(ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False)),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch = out_ch
        layers += [spectral_norm(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1, bias=False))]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MainDWithMaskAdapter(nn.Module):
    """Identical to the class in guided_diffusion/train_util_coco_fdgan_late_band.py: RGB+mask -> 1x1 adapter -> PatchGAN."""
    def __init__(self, patch_d, use_mask=True):
        super().__init__()
        self.use_mask = use_mask
        self.adapter = nn.Conv2d(4, 3, kernel_size=1, bias=True) if use_mask else None
        self.patch_d = patch_d

    def forward(self, x, mask=None):
        if self.use_mask:
            x = self.adapter(th.cat([x, mask], dim=1))
        return self.patch_d(x)


def load_main_d(path, ndf, n_layers, use_mask_adapter, device):
    """Load a saved main discriminator for the optional guidance, frozen and in eval mode."""
    d = MainDWithMaskAdapter(SpectralNLayerDiscriminator(3, ndf, n_layers, False), use_mask=use_mask_adapter)
    sd = th.load(path, map_location="cpu")
    missing, unexpected = d.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"D_main does not match the discriminator definition: missing {list(missing)[:4]}")
    d.to(device).eval()
    for p in d.parameters():
        p.requires_grad_(False)
    return d


class SmallROIPatchD(nn.Module):
    """Identical to the class in guided_diffusion/train_util_coco_fdgan_late_band.py (3-layer conv D on the ROI crop)."""
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def crop_resize_square_roi(x, bbs, roi_size, pad_ratio):
    """Identical to train_util: square crop around each box (padded), resized to roi_size. Differentiable w.r.t. x."""
    B, C, H, W = x.shape
    rois = []
    for i in range(B):
        x1, y1, x2, y2 = bbs[i]
        bw = max(1, x2 - x1); bh = max(1, y2 - y1)
        side = max(2, int(max(bw, bh) * (1.0 + 2.0 * pad_ratio)))
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        sx1 = max(0, cx - side // 2); sy1 = max(0, cy - side // 2)
        sx2 = min(W, sx1 + side); sy2 = min(H, sy1 + side)
        if sx2 - sx1 < side:
            sx1 = max(0, sx2 - side)
        if sy2 - sy1 < side:
            sy1 = max(0, sy2 - side)
        crop = x[i:i + 1, :, sy1:sy2, sx1:sx2]
        rois.append(F.interpolate(crop, size=(roi_size, roi_size), mode="bilinear", align_corners=False))
    return th.cat(rois, dim=0)


def load_small_d(path, device):
    """Load a saved small discriminator for the optional guidance, frozen and in eval mode."""
    d = SmallROIPatchD(3, 64)
    sd = th.load(path, map_location="cpu")
    d.load_state_dict(sd, strict=True)
    d.to(device).eval()
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def guidance_step(x0_pred, roi, roi_grad, D_main, D_small, boxes, *, w_main, w_small, roi_size, roi_pad_ratio,
                  eta, norm, target, gain, max_scale):
    """September 2025 pnpd_guidance_step in x0 space: LSGAN generator loss of the main D (whole image + mask) and/or the
    small ROI D (crop around the box), gradient restricted to the padded box, normalised, stepped. Returns (x0_guided, dbg)."""
    d_main_mean = d_small_mean = float("nan")
    with th.enable_grad():
        x0 = x0_pred.detach().requires_grad_(True)
        loss = 0.0
        if D_main is not None and w_main > 0:
            out = D_main(x0, roi)                                           # (B,1,h,w) patch map
            d_main_mean = float(out.detach().mean())
            loss = loss + w_main * ((out - 1.0) ** 2).mean(dim=(1, 2, 3)).sum()     # LSGAN generator loss, per image (real = 1)
        if D_small is not None and w_small > 0:
            out_s = D_small(crop_resize_square_roi(x0, boxes, roi_size, roi_pad_ratio))
            d_small_mean = float(out_s.detach().mean())
            loss = loss + w_small * ((out_s - 1.0) ** 2).mean(dim=(1, 2, 3)).sum()
        g = th.autograd.grad(loss, x0)[0]
    g = g * roi_grad                                                    # only inside the (padded) box
    B = g.shape[0]
    n_roi = roi_grad.flatten(1).sum(1) * g.shape[1] + 1e-9             # box elements per image (channels included)
    if norm == "mean_roi":
        denom = g.abs().flatten(1).sum(1) / n_roi
    elif norm == "max_roi":
        denom = g.abs().flatten(1).amax(1)
    else:
        denom = th.ones(B, device=g.device)
    denom = denom + 1e-9
    scale = (target / denom) if target > 0 else (gain / denom)
    capped = bool((scale > max_scale).any())
    scale = scale.clamp(max=max_scale).view(B, 1, 1, 1)
    step = eta * scale * g
    x0_guided = (x0.detach() - step).clamp(-1, 1)
    push = float((step.abs().flatten(1).sum(1) / n_roi).mean())        # realised mean |push| per box element, image units
    dbg = dict(d_main=d_main_mean, d_small=d_small_mean, scale=float(scale.mean()), push=push, capped=capped)
    return x0_guided, dbg


def mask_gain_at(i, T, mode, gain, gain_start, gain_end, power, band_lo, band_hi):
    """Gain applied to the mask channel at step i: constant ('always'), zero ('off'), only inside the band ('band') or a
    power ramp from gain_start to gain_end ('ramp'). Our sets use 'always'.
    """
    if mode == "off":
        return 0.0
    if mode == "always":
        return gain
    if mode == "band":
        return gain if band_lo <= i <= band_hi else 0.0
    if mode == "ramp":
        frac = i / float(max(T - 1, 1))
        return gain_end + (gain_start - gain_end) * (frac ** power)
    raise ValueError(mode)


# ----------------------------------------------------------------------------- the reverse chain
@th.no_grad()
def sample_batch(diffusion, model, shape, device, a, *, roi, roi_grad, use_ddim, bump_lo, bump_hi, guide_lo, guide_hi, cfg_lo, cfg_hi,
                 D, D_small, boxes, color_lo=-1, color_hi=-1, color_target=None):
    """The reverse chain for one batch: ancestral sampling with the fixed posterior variance or ADM's learned variance,
    or DDIM. At in-band steps with roi_bump_mode 'predict' the network predicts from a box-perturbed copy of the state
    and the chain continues from the unperturbed one, as in training. Optional in the same loop: classifier-free
    guidance on the mask channel in clean-image space (with the rescaling against over-saturation), colour guidance
    towards a target colour, and gradient guidance from the saved discriminators. Returns the batch in [-1, 1] and the
    guidance statistics.
    """
    B = shape[0]
    x = th.randn(shape, device=device)
    T = diffusion.num_timesteps
    sqrt_1m_acp = th.from_numpy((1.0 - diffusion.alphas_cumprod) ** 0.5).to(device=device, dtype=x.dtype)
    acp = th.from_numpy(diffusion.alphas_cumprod).to(device=device, dtype=x.dtype)
    acp_prev = th.from_numpy(diffusion.alphas_cumprod_prev).to(device=device, dtype=x.dtype)
    cfg_on = bool(a.use_bb_mask_cond) and abs(float(a.mask_cfg_scale) - 1.0) > 1e-6 and cfg_lo <= cfg_hi
    # region of the training-consistent perturbation: the box (FDGAN) or the whole image at matched energy (global-noise control)
    bump_m = roi if a.roi_bump_region == "box" else roi.float().mean(dim=(1, 2, 3), keepdim=True).sqrt().expand_as(roi)
    cfg_region = roi_grad if a.mask_cfg_region == "roi" else None
    do_bump = a.roi_bump_mode in ("once", "band") and a.k_focus > 0 and bump_lo <= bump_hi
    t_bump = th.randint(low=bump_lo, high=bump_hi + 1, size=(B,), device=device) if (do_bump and a.roi_bump_mode == "once") else None
    gstat = dict(n=0, push=0.0, capped=0, d_main=0.0, d_small=0.0)
    for i in reversed(range(T)):
        t = th.full((B,), i, device=device, dtype=th.long)
        if do_bump and bump_lo <= i <= bump_hi:
            if a.roi_bump_mode == "once":
                sel = (t_bump == i).view(B, 1, 1, 1).to(x.dtype)
                if sel.any():
                    x = x + sel * sqrt_1m_acp[t_bump].view(B, 1, 1, 1).to(x.dtype) * a.k_focus * roi * th.randn_like(x)
            else:
                x = x + sqrt_1m_acp[i] * a.k_focus * roi * th.randn_like(x)
        kw = {}
        if a.use_bb_mask_cond:
            kw["bb_mask"] = roi * mask_gain_at(i, T, a.mask_mode, a.mask_gain, a.mask_gain_start, a.mask_gain_end, a.mask_gain_power, bump_lo, bump_hi)
        guided_now = (D is not None or D_small is not None) and (guide_lo <= i <= guide_hi)
        # "predict" mode = the training-time use of ANPM: the network sees a box-perturbed COPY of the state at in-band steps
        # (exactly the perturbed pass of training), and the chain continues from the UNPERTURBED state with that prediction.
        predict_now = (a.roi_bump_mode == "predict") and (a.k_focus > 0) and (bump_lo <= i <= bump_hi)
        cfg_now = cfg_on and (cfg_lo <= i <= cfg_hi)
        color_now = (color_target is not None) and (color_lo <= i <= color_hi)
        plain = not (guided_now or predict_now or cfg_now or color_now)
        if use_ddim and plain and a.noise_temp == 1.0:
            x = diffusion.ddim_sample(model=model, x=x, t=t, clip_denoised=a.clip_denoised, model_kwargs=kw)["sample"]
            continue
        x_in = x + sqrt_1m_acp[i] * a.k_focus * bump_m * th.randn_like(x) if predict_now else x
        pmv = diffusion.p_mean_variance(model=model, x=x_in, t=t, clip_denoised=(a.clip_denoised and not cfg_now), model_kwargs=kw)
        x0 = pmv["pred_xstart"]
        if cfg_now:
            # classifier-free guidance on the mask channel, in x0 space: move away from the prediction made WITHOUT the box.
            # The no-box pass always sees the unperturbed state, so in predict mode the extrapolation is box-vs-no-box only.
            kw_u = dict(kw); kw_u["bb_mask"] = th.zeros_like(roi)
            x0_u = diffusion.p_mean_variance(model=model, x=x, t=t, clip_denoised=False, model_kwargs=kw_u)["pred_xstart"]
            d = (x0 - x0_u) * (float(a.mask_cfg_scale) - 1.0)
            if cfg_region is not None:
                d = d * cfg_region
            x0_g = x0 + d
            if float(a.cfg_rescale) > 0:
                # CFG rescale (Lin et al. 2024): bring the guided estimate's spread back towards the conditional one's,
                # which removes the over-saturation and over-contrast that strong guidance produces
                # mean-preserving: only the spread around each image's own mean is rescaled, so brightness and colour are untouched
                mu_g = x0_g.mean(dim=(1, 2, 3), keepdim=True)
                std_c = x0.flatten(1).std(dim=1).view(-1, 1, 1, 1) + 1e-6
                std_g = x0_g.flatten(1).std(dim=1).view(-1, 1, 1, 1) + 1e-6
                x0_resc = mu_g + (x0_g - mu_g) * (std_c / std_g)
                x0_g = float(a.cfg_rescale) * x0_resc + (1.0 - float(a.cfg_rescale)) * x0_g
            x0 = x0_g
            if a.clip_denoised:
                x0 = x0.clamp(-1, 1)
        if color_now:
            # global colour guidance: pull the per-channel means of the clean estimate towards the target colour (a DC shift)
            cur = x0.mean(dim=(2, 3), keepdim=True)
            x0 = x0 + float(a.color_guide_strength) * (color_target - cur)
            if a.clip_denoised:
                x0 = x0.clamp(-1, 1)
        if guided_now:
            x0, dbg = guidance_step(x0, roi, roi_grad, D, D_small, boxes, w_main=a.guide_main_w, w_small=a.guide_small_w,
                                    roi_size=a.roi_size, roi_pad_ratio=a.roi_pad_ratio, eta=a.pnp_eta, norm=a.pnp_norm,
                                    target=a.pnp_target, gain=a.pnp_gain, max_scale=a.max_step_scale)
            gstat["n"] += 1; gstat["push"] += dbg["push"]; gstat["capped"] += int(dbg["capped"])
            gstat["d_main"] += dbg["d_main"] if dbg["d_main"] == dbg["d_main"] else 0.0
            gstat["d_small"] += dbg["d_small"] if dbg["d_small"] == dbg["d_small"] else 0.0
        if use_ddim:
            # DDIM update from the (guided) clean estimate, eta = 0
            eps = (x - acp[i].sqrt() * x0) / (1 - acp[i]).sqrt()
            x = acp_prev[i].sqrt() * x0 + (1 - acp_prev[i]).sqrt() * eps
        else:
            mean, _, post_log_var = diffusion.q_posterior_mean_variance(x_start=x0, x_t=x, t=t)
            log_var = pmv["log_variance"] if a.sigma_mode == "learned" else post_log_var
            x = mean + (th.exp(0.5 * log_var) * a.noise_temp * th.randn_like(x) if i > 0 else 0.0)
    return x, (gstat if gstat["n"] > 0 else None)


def rev_to_idx(tmin_rev, tmax_rev, T_train, T):
    """Map a window given in reverse-time units of the training horizon onto step indices of the respaced chain."""
    fwd_min = (T_train - 1) - int(tmax_rev)
    fwd_max = (T_train - 1) - int(tmin_rev)
    scale = T / float(T_train)
    lo = max(0, min(T - 1, int(round(fwd_min * scale))))
    hi = max(0, min(T - 1, int(round(fwd_max * scale))))
    return lo, hi


# ----------------------------------------------------------------------------- main
def main():
    """Build the model, load the weights, read the boxes, and write num_samples images (sample_000000.png onwards, boxes
    cycled in sorted order) plus meta.json with every box, the arguments and the measured cost.
    """
    a = create_argparser().parse_args()
    th.manual_seed(a.seed); np.random.seed(a.seed)
    dist_util.setup_dist()
    logger.configure(dir=a.out_dir)
    device = dist_util.dev()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
    if a.use_bb_mask_cond:
        inflate_first_conv_to_4ch(model)
        model = BB4ChWrapper(model, 1.0)
        if a.mask_mode == "ramp":
            logger.log(f"[info] four-channel input ON, mask_mode=ramp gain {a.mask_gain_start} -> {a.mask_gain_end} (power {a.mask_gain_power})")
        else:
            logger.log(f"[info] four-channel input ON, mask_mode={a.mask_mode}, mask_gain={a.mask_gain}")
    else:
        logger.log("[info] four-channel input OFF (plain 3-channel U-Net)")
    load_weights(model, a.model_path)
    model.to(device).eval()

    use_ddim = str(a.timestep_respacing).startswith("ddim")
    T, T_train = diffusion.num_timesteps, int(a.diffusion_steps)
    bump_lo, bump_hi = rev_to_idx(a.bump_tmin_rev, a.bump_tmax_rev, T_train, T)
    logger.log(f"[info] sampler={'ddim' if use_ddim else 'ancestral'} T={T} noise_temp={a.noise_temp} sigma_mode={a.sigma_mode} | ANPM bump mode={a.roi_bump_mode} "
               f"k_focus={a.k_focus} band rev[{a.bump_tmin_rev},{a.bump_tmax_rev}] -> respaced idx [{bump_lo},{bump_hi}]")
    cfg_lo, cfg_hi = rev_to_idx(a.cfg_tmin_rev, a.cfg_tmax_rev, T_train, T)
    if a.use_bb_mask_cond and abs(float(a.mask_cfg_scale) - 1.0) > 1e-6:
        logger.log(f"[info] MASK CFG ON: scale={a.mask_cfg_scale} region={a.mask_cfg_region} rescale={a.cfg_rescale} window rev[{a.cfg_tmin_rev},{a.cfg_tmax_rev}] "
                   f"-> idx [{cfg_lo},{cfg_hi}] (second U-Net pass per step in the window)")
    else:
        logger.log("[info] mask CFG off (scale 1.0)")

    color_lo = color_hi = -1; color_mean = None; color_per = None; color_rng = None
    if a.color_guide_mode != "none":
        if a.color_guide_mode not in ("mean", "sample"):
            raise ValueError(f"--color_guide_mode must be none | mean | sample, got {a.color_guide_mode}")
        if not a.color_stats_json or not os.path.exists(a.color_stats_json):
            raise RuntimeError(f"--color_stats_json not found: {a.color_stats_json!r} (write it with real_color_stats.py from the training images)")
        with open(a.color_stats_json, "r") as f:
            st = json.load(f)
        color_mean = np.asarray(st["mean_rgb"], dtype=np.float32)
        color_per = np.asarray(st.get("per_image_means", []), dtype=np.float32).reshape(-1, 3)
        if a.color_guide_mode == "sample" and len(color_per) == 0:
            raise RuntimeError("colour stats json has no per_image_means; use --color_guide_mode mean")
        color_lo, color_hi = rev_to_idx(a.color_guide_tmin_rev, a.color_guide_tmax_rev, T_train, T)
        color_rng = np.random.RandomState(int(a.seed) + 1234)
        logger.log(f"[info] COLOUR GUIDANCE ON: mode={a.color_guide_mode} strength={a.color_guide_strength} | dataset-average colour R,G,B = "
                   f"{np.round(color_mean, 3).tolist()} from {st.get('n', '?')} images, {len(color_per)} per-image means | window rev[{a.color_guide_tmin_rev},{a.color_guide_tmax_rev}] -> idx [{color_lo},{color_hi}]")
    else:
        logger.log("[info] colour guidance off")

    D = None; D_small = None; guide_lo = guide_hi = -1
    if a.guide_mode not in ("none", "main", "small", "both"):
        raise ValueError(f"--guide_mode must be none | main | small | both, got {a.guide_mode}")
    if a.guide_mode != "none":
        m = re.search(r"(\d{6})\.pt$", os.path.basename(a.model_path))

        def auto_path(given, prefix):
            if given:
                return given
            if not m:
                raise RuntimeError(f"cannot infer the {prefix} file from --model_path; pass --d_main_path / --d_small_path")
            return os.path.join(os.path.dirname(a.model_path), f"{prefix}{m.group(1)}.pt")

        names = []
        if a.guide_mode in ("main", "both"):
            p = auto_path(a.d_main_path, "D_main")
            if not os.path.exists(p):
                raise RuntimeError(f"D_main not found: {p}")
            D = load_main_d(p, a.ndf, a.n_layers_D, a.use_mask_adapter, device); names.append(os.path.basename(p))
        if a.guide_mode in ("small", "both"):
            p = auto_path(a.d_small_path, "D_small")
            if not os.path.exists(p):
                raise RuntimeError(f"D_small not found: {p} (only runs trained with --enable_small_roi_d True save it)")
            D_small = load_small_d(p, device); names.append(os.path.basename(p))
        guide_lo, guide_hi = rev_to_idx(a.guide_tmin_rev, a.guide_tmax_rev, T_train, T)
        logger.log(f"[info] DISCRIMINATOR GUIDANCE ON ({a.guide_mode}): {' + '.join(names)} | window rev[{a.guide_tmin_rev},{a.guide_tmax_rev}] -> idx [{guide_lo},{guide_hi}] "
                   f"| eta={a.pnp_eta} norm={a.pnp_norm} target={a.pnp_target} => nominal mean push per guided step {a.pnp_eta * a.pnp_target:.4f} image units (range 2) "
                   f"| w_main={a.guide_main_w} w_small={a.guide_small_w} roi_size={a.roi_size} roi_pad={a.roi_pad_ratio} max_scale={a.max_step_scale} grad_pad={a.grad_pad_ratio}")
    else:
        logger.log("[info] discriminator guidance off (the paper's stated procedure)")

    bbs = load_bbs(a.bb_dir)
    if not bbs:
        raise RuntimeError(f"no box json files in {a.bb_dir}")
    logger.log(f"[info] {len(bbs)} boxes from {a.bb_dir}; writing {a.num_samples} samples to {a.out_dir}")
    os.makedirs(a.out_dir, exist_ok=True)

    H = W = a.image_size
    meta = {"args": vars(a), "samples": []}
    idx = int(a.start_index); n_done = 0
    # cost accounting for the paper: network function evaluations (NFE) per image = chain length + one extra U-Net pass per
    # step inside the CFG window; predict and colour guidance add no passes; discriminator guidance adds one D forward+backward
    # per guided step (counted separately)
    cfg_steps = (cfg_hi - cfg_lo + 1) if (a.use_bb_mask_cond and abs(float(a.mask_cfg_scale) - 1.0) > 1e-6 and cfg_lo <= cfg_hi) else 0
    d_steps = (guide_hi - guide_lo + 1) if (D is not None or D_small is not None) else 0
    logger.log(f"[cost] U-Net evaluations per image: {T + cfg_steps} = chain {T} + CFG {cfg_steps}; discriminator evaluations per image: {d_steps}")
    if device.type == "cuda":
        th.cuda.synchronize()
    t0 = time.time()
    while n_done < a.num_samples:
        B = min(a.batch_size, a.num_samples - n_done)
        batch = [bbs[(idx + j) % len(bbs)] for j in range(B)]
        boxes = [bb_to_xyxy(bb, H, W, a.bb_is_xywh, a.bb_xyxy_end_inclusive) for bb, _ in batch]
        roi = th.stack([xyxy_to_mask(*b, H, W, device) for b in boxes], 0)                                   # (B,1,H,W)
        roi_grad = th.stack([xyxy_to_mask(*pad_xyxy(*b, H, W, a.grad_pad_ratio), H, W, device) for b in boxes], 0)
        color_target = None; tgt = None
        if color_mean is not None:
            if a.color_guide_mode == "mean":
                tgt = np.tile(color_mean[None, :], (B, 1))
            else:
                tgt = color_per[color_rng.randint(0, len(color_per), size=B)]
            color_target = th.from_numpy(np.ascontiguousarray(tgt, dtype=np.float32)).to(device).view(B, 3, 1, 1)
        x, dbg = sample_batch(diffusion, model, (B, 3, H, W), device, a, roi=roi, roi_grad=roi_grad, use_ddim=use_ddim,
                              bump_lo=bump_lo, bump_hi=bump_hi, guide_lo=guide_lo, guide_hi=guide_hi, cfg_lo=cfg_lo, cfg_hi=cfg_hi,
                              D=D, D_small=D_small, boxes=boxes, color_lo=color_lo, color_hi=color_hi, color_target=color_target)
        x = (x.clamp(-1, 1) + 1) / 2
        for j in range(B):
            name = f"sample_{idx + j:06d}.png"
            save_image(x[j], os.path.join(a.out_dir, name))
            rec = {"file": name, "bb_stem": batch[j][1], "bb_xyxy": list(boxes[j])}
            if tgt is not None:
                rec["color_target_rgb"] = [round(float(v), 4) for v in tgt[j]]
            meta["samples"].append(rec)
        idx += B; n_done += B
        msg = f"created {n_done}/{a.num_samples}"
        if dbg is not None:
            n = dbg["n"]
            msg += (f"  [guidance: {n} steps, realised mean push/step {dbg['push'] / n:.4f} image units, cap hit {dbg['capped']}/{n}, "
                    f"D_main mean {dbg['d_main'] / n:.3f}, D_small mean {dbg['d_small'] / n:.3f}]")
        logger.log(msg)
    if device.type == "cuda":
        th.cuda.synchronize()
    dt = time.time() - t0
    gpu = th.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    meta["cost"] = {"images": n_done, "seconds": round(dt, 2), "seconds_per_image": round(dt / max(n_done, 1), 3), "batch_size": a.batch_size,
                    "unet_evals_per_image": T + cfg_steps, "chain_steps": T, "cfg_steps": cfg_steps, "discriminator_evals_per_image": d_steps, "device": gpu}
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    logger.log(f"[cost] {n_done} images in {dt:.1f} s = {dt / max(n_done, 1):.2f} s/image (batch {a.batch_size}, {gpu}); "
               f"U-Net evals/image {T + cfg_steps}, D evals/image {d_steps}")
    logger.log("sampling complete")


def create_argparser():
    """Every sampler setting with its default; the comments next to the defaults say what each option does."""
    defaults = dict(
        model_path="", bb_dir="", out_dir="",
        num_samples=3000, batch_size=16, start_index=0, seed=17,
        clip_denoised=True,
        noise_temp=1.0,              # <1 sharpens ancestral sampling at the cost of diversity (1.0 = standard)
        sigma_mode="small",          # ancestral chain variance: small = fixed posterior (September sampler) | learned = ADM's p_sample
        # classifier-free guidance on the mask channel (x0 space). 1.0 = off. region all | roi (padded box only)
        mask_cfg_scale=1.0, mask_cfg_region="all", cfg_tmin_rev=0, cfg_tmax_rev=999,
        cfg_rescale=0.0,             # 0 = off; 0.5 .. 0.7 = CFG rescale (Lin et al. 2024) against over-saturation at high scales
        # global colour guidance: per-channel means of the clean estimate pulled towards real-image colour statistics
        color_guide_mode="none",     # none | mean (dataset-average colour) | sample (a real image's mean colour drawn per sample)
        color_stats_json="",         # written by real_color_stats.py: mean_rgb + per_image_means, model units [-1,1]
        color_guide_strength=1.0, color_guide_tmin_rev=0, color_guide_tmax_rev=400,   # window = the first 40% of the chain
        # four-channel input (must match how the checkpoint was trained)
        use_bb_mask_cond=True,
        mask_gain=50.0,
        mask_mode="always",          # always | off | band | ramp
        mask_gain_start=80.0, mask_gain_end=50.0, mask_gain_power=1.5,   # ramp mode
        bb_is_xywh=False, bb_xyxy_end_inclusive=False,
        # ANPM bump (training-like): once inside a window given in reverse-t units of the training horizon
        k_focus=1.75, roi_bump_mode="predict", bump_tmin_rev=700, bump_tmax_rev=820,
        roi_bump_region="box",       # box (FDGAN) | global = same total energy spread over the image (the global-noise control)
        # roi_bump_mode: predict = perturb only the network INPUT at in-band steps (training-consistent, nothing accumulates)
        #                once    = add the noise to the chain once in the window;  band = add it at every window step;  none
        # OPTIONAL discriminator guidance (September 2025 PNPD step, main D, x0 space). Off by default.
        guide_mode="none",           # none | main | small | both   (small = the ROI discriminator of --enable_small_roi_d runs)
        d_main_path="", d_small_path="",   # default: D_main<step>.pt / D_small<step>.pt next to the EMA checkpoint
        ndf=128, n_layers_D=3, use_mask_adapter=True,
        guide_main_w=1.0, guide_small_w=1.0,   # loss weights (September 'both' used main 0.25, small 1.0)
        roi_size=128, roi_pad_ratio=0.03,      # crop fed to the small D, as in training
        guide_tmin_rev=225, guide_tmax_rev=300,   # = steps 280..310 of the September 400-step chain
        pnp_eta=0.006, pnp_norm="mean_roi", pnp_target=0.45, pnp_gain=1.0, max_step_scale=60000.0,
        grad_pad_ratio=0.5,          # gradient allowed in the box padded by this fraction of its size
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
