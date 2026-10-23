# -*- coding: utf-8 -*-
"""
FDGAN training loop.

Replaces the TrainLoop of guided-diffusion. The diffusion loss is ADM's and runs on every sample; inside the band of
reverse-time steps the in-band samples also go through the adversarial stage (box-localised perturbation of the
input, clean-image prediction, PatchGAN discriminator with the box mask as an extra input, least-squares loss weighted
per sample by the timestep) and the reconstruction terms (L1, background L1, box MSE, colour statistics, saturation).
The discriminator and its optimiser are saved and resumed together with the generator; if a run resumes without
them, the generator's adversarial term is ramped in over a few thousand steps. The learning-rate anneal is applied to
the discriminator optimisers as well.

Used by scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py.
"""

import os
import copy
import contextlib
import math
import json
import re
from typing import Dict, Any, Optional, Tuple, List

import blobfile as bf

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.optim as optim
import torchvision.utils as vutils

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


# ---------------------------------------------------------------------------
# Device helpers (safe for multi-GPU + LOCAL_RANK)
# ---------------------------------------------------------------------------
def _get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        lr = _get_local_rank()
        torch.cuda.set_device(lr)
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")


def _maybe_patch_dist_util_dev(dev: torch.device) -> None:
    """
    guided-diffusion's dist_util.dev() sometimes returns cuda:0 for all ranks.
    Patching it here prevents accidental cross-rank device mismatch.
    """
    try:
        dist_util.dev = lambda: dev  # type: ignore[assignment]
    except Exception:
        pass


_LOCAL_RANK = _get_local_rank()
_DEV = _get_device()
_maybe_patch_dist_util_dev(_DEV)

# perf knobs
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _dist_is_ready() else 1


def _rank() -> int:
    return dist.get_rank() if _dist_is_ready() else 0


# ---------------------------------------------------------------------------
# Helpers for safe "freeze D params during G update"
# ---------------------------------------------------------------------------
def _unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


@contextlib.contextmanager
def _freeze_params(m: nn.Module):
    """
    Temporarily set requires_grad=False on ALL params in m.
    This prevents computing/storing grads for D params, while still allowing
    grads to flow wrt the INPUT (x_t_fake) because params are treated as constants.
    """
    req = []
    params = list(m.parameters())
    for p in params:
        req.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(params, req):
            p.requires_grad_(r)


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------
def _get_bb_xyxy(
    bb: Dict[str, Any],
    idx: int,
    H: int,
    W: int,
    *,
    bb_is_xywh: bool = False,
    x_key: str = "x",
    y_key: str = "y",
    w_key: str = "w",
    h_key: str = "h",
) -> Tuple[int, int, int, int]:
    """
    Returns bbox as (x1, y1, x2, y2) clamped to image.
    Supports either:
      - bb_is_xywh=False: stored as x1,y1,x2,y2 under keys x,y,w,h (xyxy)
      - bb_is_xywh=True:  stored as x1,y1,width,height under keys x,y,w,h (xywh)
    """
    def _get_field(k: str) -> int:
        v = bb[k]
        if isinstance(v, torch.Tensor):
            return int(v[idx].item())
        return int(v[idx])

    x1 = _get_field(x_key)
    y1 = _get_field(y_key)
    a = _get_field(w_key)
    b = _get_field(h_key)

    if bb_is_xywh:
        x2 = x1 + a
        y2 = y1 + b
    else:
        x2 = a
        y2 = b

    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 == x1:
        x2 = min(W, x1 + 1)
    if y2 == y1:
        y2 = min(H, y1 + 1)

    return x1, y1, x2, y2


def _build_bb_mask_xyxy(
    bb: Dict[str, Any],
    B: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    bb_is_xywh: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
    mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    bbs: List[Tuple[int, int, int, int]] = []
    for j in range(B):
        x1, y1, x2, y2 = _get_bb_xyxy(bb, j, H, W, bb_is_xywh=bb_is_xywh)
        bbs.append((x1, y1, x2, y2))
        mask[j, :, y1:y2, x1:x2] = 1.0
    return mask, bbs


def _bbox_area_frac(bb_xyxy: Tuple[int, int, int, int], H: int, W: int) -> float:
    x1, y1, x2, y2 = bb_xyxy
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return float(area) / float(H * W + 1e-8)


# ---------------------------------------------------------------------------
# GAN loss (weighted LSGAN)
# ---------------------------------------------------------------------------
class WeightedLSGANLoss(nn.Module):
    """
    Least-Squares GAN loss with optional per-sample weights.
    Works for PatchGAN outputs with shape (B, ...).
    """
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label

    def forward(self, pred: torch.Tensor, target_is_real: bool, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_val = self.real_label if target_is_real else self.fake_label
        target = torch.full_like(pred, fill_value=target_val)
        loss = (pred - target) ** 2

        if weight is not None:
            assert weight.dim() == 1 and weight.shape[0] == pred.shape[0], "weight must be (B,)"
            w = weight
            while w.dim() < loss.dim():
                w = w.view(w.shape[0], *([1] * (loss.dim() - 1)))
            loss = loss * w

        return loss.mean()


# ---------------------------------------------------------------------------
# DiffAug: integer translate (reflect pad)
# ---------------------------------------------------------------------------
def diffaug_translate(x: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return x
    B, C, H, W = x.shape
    pad = px
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = torch.empty_like(x)
    for i in range(B):
        dx = int(torch.randint(-px, px + 1, (1,), device=x.device).item())
        dy = int(torch.randint(-px, px + 1, (1,), device=x.device).item())
        xs = pad + dx
        ys = pad + dy
        out[i] = x_pad[i, :, ys:ys + H, xs:xs + W]
    return out


def diffaug_translate_pair(x: torch.Tensor, mask: torch.Tensor, px: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Translate x and mask with the SAME random integer shift per sample."""
    if px <= 0:
        return x, mask
    B, C, H, W = x.shape
    pad = px
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    m_pad = F.pad(mask, (pad, pad, pad, pad), mode="reflect")
    out_x = torch.empty_like(x)
    out_m = torch.empty_like(mask)
    for i in range(B):
        dx = int(torch.randint(-px, px + 1, (1,), device=x.device).item())
        dy = int(torch.randint(-px, px + 1, (1,), device=x.device).item())
        xs = pad + dx
        ys = pad + dy
        out_x[i] = x_pad[i, :, ys:ys + H, xs:xs + W]
        out_m[i] = m_pad[i, :, ys:ys + H, xs:xs + W]
    return out_x, out_m


# ---------------------------------------------------------------------------
# ROI crop util
# ---------------------------------------------------------------------------
def crop_resize_square_roi(
    x: torch.Tensor,
    bbs: List[Tuple[int, int, int, int]],
    roi_size: int,
    pad_ratio: float,
) -> torch.Tensor:
    """
    Crop a square ROI around bbox (with padding), then resize to (roi_size, roi_size).
    x: (B,C,H,W)
    """
    B, C, H, W = x.shape
    rois = []
    for i in range(B):
        x1, y1, x2, y2 = bbs[i]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = int(max(bw, bh) * (1.0 + 2.0 * pad_ratio))
        side = max(2, side)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        sx1 = max(0, cx - side // 2)
        sy1 = max(0, cy - side // 2)
        sx2 = min(W, sx1 + side)
        sy2 = min(H, sy1 + side)

        if sx2 - sx1 < side:
            sx1 = max(0, sx2 - side)
        if sy2 - sy1 < side:
            sy1 = max(0, sy2 - side)

        crop = x[i:i+1, :, sy1:sy2, sx1:sx2]
        crop = F.interpolate(crop, size=(roi_size, roi_size), mode="bilinear", align_corners=False)
        rois.append(crop)
    return torch.cat(rois, dim=0)


# ---------------------------------------------------------------------------
# Small ROI discriminator (3 conv layers)
# ---------------------------------------------------------------------------
class SmallROIPatchD(nn.Module):
    """
    3-layer conv discriminator for cropped ROIs.
    Output is patch map (B,1,h,w).
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main D wrapper (RGB+mask -> 3ch adapter -> PatchGAN)
# ---------------------------------------------------------------------------
class MainDWithMaskAdapter(nn.Module):
    def __init__(self, patch_d: nn.Module, use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        if use_mask:
            self.adapter = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        else:
            self.adapter = None
        self.patch_d = patch_d

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_mask:
            assert mask is not None, "mask required when use_mask=True"
            x = torch.cat([x, mask], dim=1)
            x = self.adapter(x)
        return self.patch_d(x)


# ---------------------------------------------------------------------------
# Checkpoint loading helpers (generator)
# ---------------------------------------------------------------------------
def _maybe_prefix_state_dict(state_dict: dict, prefix: str) -> dict:
    """Return a new dict with prefix added to keys that don't already have it."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k] = v
        else:
            out[prefix + k] = v
    return out


def _inflate_first_conv_weight_if_needed(state_dict: dict, model_state: dict) -> dict:
    """Inflate 3ch -> 4ch first conv weights (RGB -> RGB+mask) when needed."""
    cand_keys = [
        "unet.input_blocks.0.0.weight",
        "input_blocks.0.0.weight",
    ]
    for ck in cand_keys:
        if ck in state_dict:
            w = state_dict[ck]
            if ck in model_state:
                w_target = model_state[ck]
            else:
                continue

            if (
                isinstance(w, torch.Tensor)
                and isinstance(w_target, torch.Tensor)
                and w.ndim == 4
                and w_target.ndim == 4
                and w.shape[1] == 3
                and w_target.shape[1] == 4
                and w.shape[0] == w_target.shape[0]
                and w.shape[2:] == w_target.shape[2:]
            ):
                # 15 Oct 2026: when a 3-channel (pretrained) checkpoint is loaded into the 4-channel model, give the mask channel
                # the copy_mean initialisation (mean of the loaded RGB filters x 0.10) instead of zeros, as the paper states.
                w_new = torch.zeros_like(w_target)
                w_new[:, :3] = w
                w_new[:, 3:4] = w.mean(dim=1, keepdim=True) * 0.10
                state_dict = dict(state_dict)
                state_dict[ck] = w_new
    return state_dict


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a checkpoint into `model`, handling:
    - plain guided-diffusion ADM checkpoints (no 'unet.' prefix)
    - our wrapper checkpoints (with 'unet.' prefix)
    - 3-channel -> 4-channel first conv inflation for BB4Ch
    """
    state = dist_util.load_state_dict(checkpoint_path, map_location=dist_util.dev())
    model_state = model.state_dict()

    if not any(k in model_state for k in state.keys()):
        state = _maybe_prefix_state_dict(state, "unet.")

    state = _inflate_first_conv_weight_if_needed(state, model_state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if _rank() == 0:
        if missing:
            logger.log(f"[ckpt] missing keys (showing up to 20): {missing[:20]}")
        if unexpected:
            logger.log(f"[ckpt] unexpected keys (showing up to 20): {unexpected[:20]}")


# ---------------------------------------------------------------------------
# Train Loop (COCO late-band variant)
# ---------------------------------------------------------------------------
class TrainLoop:
    """
    The FDGAN training loop:
      - band of reverse-time steps that slides from its starting position to its final one and then stays
      - diffusion loss on every sample, drawn over the full timestep range by the schedule sampler
      - adversarial and reconstruction terms only for the samples inside the band
      - logging aggregated per optimisation step across the micro-batches
      - discriminator and optimiser states saved and resumed with the generator
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        checkpoint_dir: str = "",
        # --- Stabilizers (main D)
        r1_gamma_main: float = 5.0,
        r1_every_main: int = 16,
        diffaug_translate_px_main: int = 0,
        # --- GAN strength modifiers
        gan_scale_cap: float = 0.0,
        gan_ramp_steps: int = 0,
        use_fp16: bool = False,
        fp16_scale_growth: float = 1e-3,
        schedule_sampler=None,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        # --- Discriminator(s)
        discriminator: nn.Module = None,
        use_mask_adapter: bool = True,
        lr_D: float = 1e-5,
        betas_D: Tuple[float, float] = (0.5, 0.999),
        # --- COCO late-band schedule (reverse t)
        tmin_start: int = 700,
        tmin_end: int = 540,
        band_width: int = 120,
        tmax_cap: int = 820,
        band_ramp_steps: int = 50000,
        diffusion_in_band_weight: float = 0.7,
        # --- GAN scaling schedule
        gan_initial_scale: float = 1e-4,
        gan_final_scale: float = 2e-2,
        # --- global warm-up: adversarial stage and extra terms off for the first N steps
        gan_warmup_steps: int = 40000,
        # --- Recon scaling (global L1 schedule)
        l1_initial_scale: float = 1e-2,
        l1_final_scale: float = 1e-2,
        # Background L1 anchor weight as a fraction of main L1 (10–25% is typical)
        l1_bg_frac: float = 0.2,
        lambda_enhanced: float = 5e-2,
        # --- ROI noise injection (GAN path)
        noise_option: str = "extra_gan_only",
        k_focus: float = 1.75,
        bb_is_xywh: bool = False,
        # --- GAN sampling control
        max_gan_per_microbatch: int = 4,
        # --- Small ROI D (optional)
        enable_small_roi_d: bool = False,
        roi_size: int = 128,
        roi_pad_ratio: float = 0.03,
        train_roi_frac_cap: float = 0.15,
        diffaug_translate_px: int = 4,
        r1_gamma_small: float = 5.0,
        r1_every: int = 16,
        lr_small_D: float = 1e-5,
        # --- Debug image saving
        image_save_every: int = 1000,
        image_save_dir: str = "",
        # --- box mask as the fourth input channel of the U-Net
        use_bb_mask_cond: bool = False,
        bb_mask_key: str = "bb_mask",
        # --- Color stabilizers
        color_stats_scale: float = 0.0,
        sat_penalty_scale: float = 0.0,
        sat_threshold: float = 0.95,
        # --- saving / resuming the discriminator(s)
        save_discriminators: bool = True,
        resume_discriminators: bool = True,
        # --- if resuming without discriminator checkpoints, ramp in the generator's adversarial term
        # (number of *local* steps after resume; i.e., counts from 0 after restart)
        g_adv_warmup_steps_if_no_d_ckpt: int = 2000,
    ):
        assert discriminator is not None, "You must pass a discriminator module."

        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.device = dist_util.dev()

        self.save_discriminators = bool(save_discriminators)
        self.resume_discriminators = bool(resume_discriminators)
        self.g_adv_warmup_steps_if_no_d_ckpt = int(g_adv_warmup_steps_if_no_d_ckpt)

        # Sanity warning if T != 1000 (band values assume 1000).
        if getattr(self.diffusion, "num_timesteps", None) is not None:
            T = int(self.diffusion.num_timesteps)
            if T != 1000 and _rank() == 0:
                logger.warn(
                    f"COCO late-band defaults assume T=1000, but diffusion.num_timesteps={T}. "
                    f"Band numbers (700..820) may be wrong unless you remap."
                )

        # global-noise control: "extra_gan_only_global" spreads the SAME total extra-noise energy over the whole
        # image instead of the box: per-pixel multiplier sqrt(box_area / image_area), so sum over pixels of (k * m)^2 is equal.
        assert noise_option in ("extra_gan_only", "extra_gan_only_global", "renorm_shared")
        self.noise_option = noise_option
        self.k_focus = float(k_focus)
        self.bb_is_xywh = bool(bb_is_xywh)

        # Box mask: passed to the U-Net through model_kwargs.
        self.use_bb_mask_cond = bool(use_bb_mask_cond)
        self.bb_mask_key = str(bb_mask_key)

        # Optional color stabilizers.
        self.color_stats_scale = float(color_stats_scale)
        self.sat_penalty_scale = float(sat_penalty_scale)
        self.sat_threshold = float(sat_threshold)

        # Band schedule
        self.tmin_start = int(tmin_start)
        self.tmin_end = int(tmin_end)
        self.band_width = int(band_width)
        self.tmax_cap = int(tmax_cap)
        self.band_ramp_steps = max(1, int(band_ramp_steps))
        self.diffusion_in_band_weight = float(diffusion_in_band_weight)

        # Schedules
        assert gan_initial_scale > 0 and gan_final_scale > 0
        self.gan_initial_scale = float(gan_initial_scale)
        self.gan_final_scale = float(gan_final_scale)

        # Global warmup: GAN + extras fully OFF for the first N steps.
        self.gan_warmup_steps = int(gan_warmup_steps)

        assert l1_initial_scale > 0 and l1_final_scale > 0
        self.l1_initial_scale = float(l1_initial_scale)
        self.l1_final_scale = float(l1_final_scale)
        self.l1_bg_frac = float(l1_bg_frac)

        self.lambda_enhanced = float(lambda_enhanced)

        # GAN sampling control
        self.max_gan_per_microbatch = int(max_gan_per_microbatch)

        # Small ROI D settings
        self.enable_small_roi_d = bool(enable_small_roi_d)
        self.roi_size = int(roi_size)
        self.roi_pad_ratio = float(roi_pad_ratio)
        self.train_roi_frac_cap = float(train_roi_frac_cap)
        self.diffaug_translate_px = int(diffaug_translate_px)
        self.r1_gamma_small = float(r1_gamma_small)
        self.r1_every = int(r1_every)
        self.lr_small_D = float(lr_small_D)

        # Basics
        self.batch_size = int(batch_size)
        self.microbatch = int(microbatch) if microbatch > 0 else int(batch_size)
        self.lr = float(lr)

        # ema_rate parsing
        if isinstance(ema_rate, float):
            self.ema_rate = [ema_rate]
        elif isinstance(ema_rate, str):
            self.ema_rate = [float(x) for x in ema_rate.split(",")]
        else:
            self.ema_rate = [float(x) for x in ema_rate]

        self.log_interval = int(log_interval)
        self.save_interval = int(save_interval)
        self.resume_checkpoint = resume_checkpoint

        # Where to save checkpoints
        self.checkpoint_dir = checkpoint_dir or logger.get_dir()
        if _rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Stabilizers / caps
        self.r1_gamma_main = float(r1_gamma_main)
        self.r1_every_main = int(r1_every_main)
        self.diffaug_translate_px_main = int(diffaug_translate_px_main)
        self.gan_scale_cap = float(gan_scale_cap)
        self.gan_ramp_steps = int(gan_ramp_steps)
        self.use_fp16 = bool(use_fp16)
        self.fp16_scale_growth = float(fp16_scale_growth)
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = float(weight_decay)
        self.lr_anneal_steps = int(lr_anneal_steps)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * _world_size()

        # Debug image saving
        self.image_save_every = int(image_save_every)
        self.image_save_dir = image_save_dir
        if self.image_save_dir:
            os.makedirs(self.image_save_dir, exist_ok=True)

        # Load model weights
        self._load_and_sync_parameters()

        # MP trainer + optimizer for generator
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )
        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)

        # EMA
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.mp_trainer.master_params) for _ in range(len(self.ema_rate))]

        # DDP for generator
        if torch.cuda.is_available():
            self.use_ddp = _dist_is_ready() and _world_size() > 1
            self.ddp_model = DDP(
                self.model.to(dist_util.dev()),
                device_ids=[_LOCAL_RANK],
                output_device=_LOCAL_RANK,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            ) if self.use_ddp else self.model.to(dist_util.dev())
        else:
            self.use_ddp = False
            self.ddp_model = self.model

        # Build discriminators (DDP-wrapped)
        if isinstance(discriminator, nn.DataParallel):
            discriminator = discriminator.module
        discriminator = discriminator.to(dist_util.dev()).train()
        self.main_D = MainDWithMaskAdapter(discriminator, use_mask=use_mask_adapter).to(dist_util.dev()).train()

        self.use_ddp_D = _dist_is_ready() and _world_size() > 1 and torch.cuda.is_available()
        if self.use_ddp_D:
            self.main_D = DDP(
                self.main_D,
                device_ids=[_LOCAL_RANK],
                output_device=_LOCAL_RANK,
                broadcast_buffers=False,
                bucket_cap_mb=64,
                find_unused_parameters=False,
            )

        self.opt_D = optim.Adam(self.main_D.parameters(), lr=lr_D, betas=betas_D)

        # Record base LR so lr_anneal_steps scales D LR correctly (no compounding decay).
        for pg in self.opt_D.param_groups:
            pg.setdefault("base_lr", float(pg.get("lr", lr_D)))

        self.small_D = None
        self.opt_small_D = None
        if self.enable_small_roi_d:
            small = SmallROIPatchD(in_ch=3, base_ch=64).to(dist_util.dev()).train()
            if self.use_ddp_D:
                small = DDP(
                    small,
                    device_ids=[_LOCAL_RANK],
                    output_device=_LOCAL_RANK,
                    broadcast_buffers=False,
                    bucket_cap_mb=32,
                    find_unused_parameters=False,
                )
            self.small_D = small
            self.opt_small_D = optim.Adam(self.small_D.parameters(), lr=self.lr_small_D, betas=betas_D)

            for pg in self.opt_small_D.param_groups:
                pg.setdefault("base_lr", float(pg.get("lr", self.lr_small_D)))

        self.gan_loss = WeightedLSGANLoss().to(dist_util.dev())

        # --- load discriminator checkpoints if present ---
        self._loaded_d_ckpt = False
        if self.resume_step and self.resume_discriminators:
            self._loaded_d_ckpt = self._load_discriminators_and_opts_if_present()

        # If resuming but D ckpts are missing -> protect generator with warmup ramp
        self._g_adv_warmup_steps_effective = 0
        if self.resume_step and (not self._loaded_d_ckpt) and (self.g_adv_warmup_steps_if_no_d_ckpt > 0):
            self._g_adv_warmup_steps_effective = int(self.g_adv_warmup_steps_if_no_d_ckpt)
            if _rank() == 0:
                logger.log(
                    f"[FDGAN] Resumed G at step={self.resume_step} but D ckpts were not found. "
                    f"Applying generator-adv warmup ramp for {self._g_adv_warmup_steps_effective} local steps."
                )

    # ---------------------------
    # Schedule helpers
    # ---------------------------
    def _max_t(self) -> int:
        return int(getattr(self.diffusion, "num_timesteps", 1000)) - 1

    def _to_reverse_t(self, t_fwd: torch.Tensor) -> torch.Tensor:
        return self._max_t() - t_fwd

    def _current_band_edges(self, global_step: int) -> Tuple[int, int]:
        p = min(1.0, float(global_step) / float(self.band_ramp_steps))
        tmin = int(round(self.tmin_start + p * (self.tmin_end - self.tmin_start)))
        tmax = min(self.tmax_cap, tmin + self.band_width)
        return tmin, tmax

    def _gan_scale(self, t_rev: torch.Tensor, global_step: Optional[int] = None) -> torch.Tensor:
        max_t = float(self._max_t())
        decay = math.log(self.gan_final_scale / self.gan_initial_scale) / max_t
        scale = self.gan_initial_scale * torch.exp(decay * t_rev.float())

        if self.gan_ramp_steps and global_step is not None and self.gan_ramp_steps > 0:
            # If a global warmup is used, start the ramp *after* the warmup.
            # (Keeps old runs identical when gan_warmup_steps=0)
            s = float(global_step)
            if getattr(self, "gan_warmup_steps", 0) > 0:
                s = max(0.0, s - float(self.gan_warmup_steps))
            p = min(1.0, s / float(self.gan_ramp_steps))
            scale = scale * p

        if self.gan_scale_cap and self.gan_scale_cap > 0:
            scale = torch.clamp(scale, max=self.gan_scale_cap)

        return scale

    def _l1_scale(self, t_rev: torch.Tensor) -> torch.Tensor:
        max_t = float(self._max_t())
        decay = math.log(self.l1_final_scale / self.l1_initial_scale) / max_t
        return self.l1_initial_scale * torch.exp(decay * t_rev.float())

    def _g_adv_warmup_factor(self) -> float:
        """
        Local-step warmup factor for generator adversarial losses.
        Only used when resuming without discriminator checkpoints.
        """
        steps = int(getattr(self, "_g_adv_warmup_steps_effective", 0))
        if steps <= 0:
            return 1.0
        # self.step starts at 0 after resume. Ramp linearly to 1.0.
        return float(min(1.0, max(0.0, float(self.step) / float(steps))))

    # ---------------------------
    # Checkpoint / EMA helpers (generator)
    # ---------------------------
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if _rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            load_state_dict_flexible(self.model, resume_checkpoint)
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if _rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06d}.pt")
        if bf.exists(opt_checkpoint):
            if _rank() == 0:
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    # ---------------------------
    # discriminator checkpoint helpers
    # ---------------------------
    def _d_ckpt_dir_candidates(self) -> List[str]:
        """
        Where to look for D checkpoints.
        - 1) directory of resume_checkpoint (most common)
        - 2) checkpoint_dir used for this run
        """
        cands: List[str] = []
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint:
            cands.append(bf.dirname(main_checkpoint))
        if self.checkpoint_dir:
            cands.append(self.checkpoint_dir)
        # unique while preserving order
        out: List[str] = []
        for p in cands:
            if p and (p not in out):
                out.append(p)
        return out

    def _d_paths_for_step(self, step: int, base_dir: str) -> Dict[str, str]:
        return {
            "D_main": bf.join(base_dir, f"D_main{step:06d}.pt"),
            "optD_main": bf.join(base_dir, f"optD_main{step:06d}.pt"),
            "D_small": bf.join(base_dir, f"D_small{step:06d}.pt"),
            "optD_small": bf.join(base_dir, f"optD_small{step:06d}.pt"),
        }

    def _load_discriminators_and_opts_if_present(self) -> bool:
        """
        Returns True if it loaded the main discriminator checkpoint.
        (small_D is optional)
        """
        step = int(self.resume_step)
        if step <= 0:
            return False

        loaded_any = False
        for base_dir in self._d_ckpt_dir_candidates():
            paths = self._d_paths_for_step(step, base_dir)

            # ---- main D
            if bf.exists(paths["D_main"]):
                if _rank() == 0:
                    logger.log(f"[ckpt] loading main D: {paths['D_main']}")
                sd = dist_util.load_state_dict(paths["D_main"], map_location=dist_util.dev())
                try:
                    _unwrap_ddp(self.main_D).load_state_dict(sd, strict=True)
                except Exception as e:
                    if _rank() == 0:
                        logger.log(f"[warn] Failed to load main D checkpoint (shape mismatch?). Starting fresh D. Error: {e}")
                    continue
                dist_util.sync_params(self.main_D.parameters())
                loaded_any = True

                # optimizer (load on all ranks; avoids manual broadcast of optimizer state)
                if bf.exists(paths["optD_main"]):
                    if _rank() == 0:
                        logger.log(f"[ckpt] loading main D optimizer: {paths['optD_main']}")
                    opt_sd = dist_util.load_state_dict(paths["optD_main"], map_location=dist_util.dev())
                    try:
                        self.opt_D.load_state_dict(opt_sd)
                    except Exception as e:
                        if _rank() == 0:
                            logger.log(f"[warn] Failed to load optD_main state; continuing with fresh optD. Error: {e}")

                # ---- small D (optional)
                if self.small_D is not None and bf.exists(paths["D_small"]):
                    if _rank() == 0:
                        logger.log(f"[ckpt] loading small D: {paths['D_small']}")
                    sd_s = dist_util.load_state_dict(paths["D_small"], map_location=dist_util.dev())
                    try:
                        _unwrap_ddp(self.small_D).load_state_dict(sd_s, strict=True)
                    except Exception as e:
                        if _rank() == 0:
                            logger.log(f"[warn] Failed to load small D checkpoint. Starting fresh small D. Error: {e}")
                    else:
                        dist_util.sync_params(self.small_D.parameters())

                    if self.opt_small_D is not None and bf.exists(paths["optD_small"]):
                        if _rank() == 0:
                            logger.log(f"[ckpt] loading small D optimizer: {paths['optD_small']}")
                        opt_s_sd = dist_util.load_state_dict(paths["optD_small"], map_location=dist_util.dev())
                        try:
                            self.opt_small_D.load_state_dict(opt_s_sd)
                        except Exception as e:
                            if _rank() == 0:
                                logger.log(f"[warn] Failed to load optD_small state; continuing fresh. Error: {e}")

                return loaded_any  # stop at first dir that has D_main

        if _rank() == 0:
            logger.log(f"[ckpt] No discriminator checkpoints found for step {step:06d} in: {self._d_ckpt_dir_candidates()}")
        return False

    # ---------------------------
    # Main loop
    # ---------------------------
    def run_loop(self):
        while (not self.lr_anneal_steps) or (self.step + self.resume_step < self.lr_anneal_steps):
            for batch_data in self.data:
                # 11 Oct 2026 stop guard: the data generator is infinite, so without this check the loop never returns to the
                # while condition, training runs past lr_anneal_steps and _anneal_lr turns the learning rate NEGATIVE
                # (the OzSTAR runs diverged after 200k; the 270k EMA gave black images). No effect before the horizon.
                if self.lr_anneal_steps and (self.step + self.resume_step) >= self.lr_anneal_steps:
                    break
                images, additional_info = batch_data
                batch = images.to(dist_util.dev(), non_blocking=True)
                cond = additional_info
                self.run_step(batch, cond)

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()

                if self.step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.opt.zero_grad(set_to_none=True)
        self.opt_D.zero_grad(set_to_none=True)
        if self.opt_small_D is not None:
            self.opt_small_D.zero_grad(set_to_none=True)

        self.forward_backward(batch, cond)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()

        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        """
        Microbatch accumulation:
          - backward(diffusion_loss)
          - if GAN-active: update D(s), then backward(extra_g_loss)
        WITH per-step aggregation across microbatches for logging.
        """
        self.mp_trainer.zero_grad()

        global_step = int(self.step + self.resume_step)

        # --------------------------------------------------------
        # Global warmup: GAN + extras OFF for first gan_warmup_steps
        # --------------------------------------------------------
        gan_enabled = (global_step >= int(getattr(self, "gan_warmup_steps", 0)))
        if _rank() == 0:
            logger.logkv("gan_warmup_steps", float(getattr(self, "gan_warmup_steps", 0)))
            logger.logkv("gan_enabled", float(gan_enabled))
            if getattr(self, "gan_warmup_steps", 0) > 0:
                logger.logkv(
                    "gan_warmup_remaining",
                    float(max(0, int(getattr(self, "gan_warmup_steps", 0)) - global_step)),
                )
        tmin_rev, tmax_rev = self._current_band_edges(global_step)
        if _rank() == 0:
            logger.logkv("band_tmin_rev", float(tmin_rev))
            logger.logkv("band_tmax_rev", float(tmax_rev))

        # -----------------------
        # Per-step accumulators
        # -----------------------
        diff_loss_sum = 0.0
        diff_loss_den = 0.0
        gan_frac_sum = 0.0
        gan_frac_den = 0.0

        gan_sel_sum = 0.0
        gan_updates = 0

        d_loss_sum = 0.0
        d_loss_raw_sum = 0.0
        g_adv_sum = 0.0
        g_l1_sum = 0.0
        g_l1_bg_sum = 0.0
        g_roi_sum = 0.0
        g_color_sum = 0.0
        g_sat_sum = 0.0

        gan_scale_mean_sum = 0.0
        gan_scale_max_max = 0.0

        D_grad_norm_sum = 0.0
        D_grad_norm_max = 0.0

        r1_sum = 0.0
        r1_n = 0

        d_small_sum = 0.0
        d_small_n = 0

        # warm-up factor for the generator's adversarial term
        g_adv_factor = self._g_adv_warmup_factor()
        if _rank() == 0:
            logger.logkv("g_adv_warmup_factor", float(g_adv_factor))

        for mb_start in range(0, batch.shape[0], self.microbatch):
            micro = batch[mb_start: mb_start + self.microbatch].to(dist_util.dev(), non_blocking=True)
            B_mb, C, H, W = micro.shape
            last_mb = (mb_start + self.microbatch) >= batch.shape[0]

            # Slice cond for microbatch
            micro_cond: Dict[str, Any] = {}
            for k, v in cond.items():
                if isinstance(v, torch.Tensor):
                    micro_cond[k] = v[mb_start: mb_start + B_mb].to(dist_util.dev(), non_blocking=True)
                elif isinstance(v, dict):
                    micro_cond[k] = {}
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            micro_cond[k][kk] = vv[mb_start: mb_start + self.microbatch].to(dist_util.dev(), non_blocking=True)
                        else:
                            micro_cond[k][kk] = vv[mb_start: mb_start + self.microbatch]
                elif isinstance(v, (list, tuple)):
                    micro_cond[k] = v[mb_start: mb_start + B_mb]
                else:
                    micro_cond[k] = v

            t, weights = self.schedule_sampler.sample(B_mb, dist_util.dev())
            t_rev = self._to_reverse_t(t)

            in_band = (t_rev >= tmin_rev) & (t_rev <= tmax_rev)
            if not gan_enabled:
                # Hard disable GAN/extras during warmup.
                in_band = torch.zeros_like(in_band, dtype=torch.bool)
            gan_active = bool(in_band.any().item())
            gan_active_frac = float(in_band.float().mean().item())

            sync_ctx = self.ddp_model.no_sync() if (self.use_ddp and not last_mb) else contextlib.nullcontext()

            # -----------------------
            # Diffusion loss (FULL t range)
            # -----------------------
            model_kwargs_mb = {k: v for k, v in micro_cond.items() if k not in ("flip", "bb", "image_filenames")}

            # Build bb_mask once per microbatch
            bb_dict = micro_cond.get("bb", None)
            if bb_dict is None:
                bb_mask_mb = torch.zeros((B_mb, 1, H, W), device=self.device, dtype=micro.dtype)
                bbs_mb = [(0, 0, W, H)] * B_mb
            else:
                bb_mask_mb, bbs_mb = _build_bb_mask_xyxy(
                    bb_dict,
                    B_mb,
                    H,
                    W,
                    device=self.device,
                    dtype=micro.dtype,
                    bb_is_xywh=self.bb_is_xywh,
                )

            if self.use_bb_mask_cond:
                model_kwargs_mb[self.bb_mask_key] = bb_mask_mb

            losses = self.diffusion.training_losses(self.ddp_model, micro, t, model_kwargs=model_kwargs_mb)
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            diff_weight = torch.where(
                in_band,
                torch.full_like(weights, self.diffusion_in_band_weight),
                torch.ones_like(weights),
            )
            diffusion_loss = (losses["loss"] * weights * diff_weight).mean()

            # Accumulate per-step diffusion stats
            diff_loss_sum += float(diffusion_loss.detach().item()) * float(B_mb)
            diff_loss_den += float(B_mb)
            gan_frac_sum += float(gan_active_frac) * float(B_mb)
            gan_frac_den += float(B_mb)

            with sync_ctx:
                self.mp_trainer.backward(diffusion_loss)

                # -----------------------
                # GAN + recon extras (ONLY for in-band samples)
                # -----------------------
                if gan_active:
                    idxs = in_band.nonzero(as_tuple=False).squeeze(1)
                    if idxs.numel() > self.max_gan_per_microbatch:
                        perm = torch.randperm(idxs.numel(), device=micro.device)
                        idxs = idxs[perm[: self.max_gan_per_microbatch]]

                    micro_sel = micro[idxs]
                    t_sel = t[idxs]
                    t_rev_sel = t_rev[idxs]
                    B_sel = micro_sel.shape[0]

                    # subselect model_kwargs
                    model_kwargs_sel: Dict[str, Any] = {}
                    for k, v in model_kwargs_mb.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] == B_mb:
                            model_kwargs_sel[k] = v[idxs]
                        else:
                            model_kwargs_sel[k] = v

                    bb_mask_sel = bb_mask_mb[idxs]
                    bbs_sel = [bbs_mb[int(j.item())] for j in idxs]

                    roi_keep = torch.ones((B_sel,), device=micro.device, dtype=torch.bool)
                    if self.train_roi_frac_cap > 0:
                        keep_list = []
                        for bb in bbs_sel:
                            keep_list.append(_bbox_area_frac(bb, H, W) <= self.train_roi_frac_cap)
                        roi_keep = torch.tensor(keep_list, device=micro.device, dtype=torch.bool)

                    # ROI-focused noise eps'
                    eps_base = torch.randn_like(micro_sel)
                    eps_focus = torch.randn_like(micro_sel)
                    if self.noise_option == "extra_gan_only":
                        eps_prime = eps_base + (self.k_focus * bb_mask_sel) * eps_focus
                    elif self.noise_option == "extra_gan_only_global":
                        m_global = bb_mask_sel.float().mean(dim=(1, 2, 3), keepdim=True).sqrt()   # (B,1,1,1): sqrt(area fraction)
                        eps_prime = eps_base + (self.k_focus * m_global) * eps_focus              # same energy, everywhere
                    else:
                        denom = math.sqrt(1.0 + self.k_focus ** 2)
                        mixed = (eps_base + self.k_focus * eps_focus) / denom
                        eps_prime = (1.0 - bb_mask_sel) * eps_base + bb_mask_sel * mixed

                    # x_t real and pred_x0
                    x_t_real = self.diffusion.q_sample(x_start=micro_sel, t=t_sel, noise=eps_prime)
                    pmv = self.diffusion.p_mean_variance(
                        self.ddp_model,
                        x_t_real,
                        t_sel,
                        clip_denoised=True,
                        model_kwargs=model_kwargs_sel,
                    )
                    x0_pred = pmv["pred_xstart"].clamp(-1, 1)

                    # NOTE: We train D on clean x0 (real=micro_sel, fake=x0_pred),
                    # so we do NOT need to build x_t_fake here.

                    gan_scale = self._gan_scale(t_rev_sel, global_step=global_step).detach()
                    l1_scale = self._l1_scale(t_rev_sel).detach()

                    # Log-scale debug values
                    gan_scale_mean_val = float(gan_scale.float().mean().item())
                    gan_scale_max_val = float(gan_scale.float().max().item())

                    # -----------------------
                    # Main D update (backward + PatchGAN-safe R1)
                    # -----------------------
                    self.opt_D.zero_grad(set_to_none=True)

                    x_real_D = micro_sel.detach().float()
                    x_fake_D = x0_pred.detach().float()
                    m_D = bb_mask_sel.detach().float()

                    if self.diffaug_translate_px_main > 0:
                        x_real_D, m_real_D = diffaug_translate_pair(x_real_D, m_D, self.diffaug_translate_px_main)
                        x_fake_D, m_fake_D = diffaug_translate_pair(x_fake_D, m_D, self.diffaug_translate_px_main)
                    else:
                        m_real_D = m_D
                        m_fake_D = m_D

                    do_r1 = (self.r1_every_main > 0) and (global_step % self.r1_every_main == 0) and (self.r1_gamma_main > 0)
                    if do_r1:
                        x_real_D.requires_grad_(True)

                    real_pred = self.main_D(x_real_D, m_real_D)
                    fake_pred = self.main_D(x_fake_D, m_fake_D)

                    # d_loss_raw: unweighted LSGAN (no gan_scale weighting, no R1)
                    with torch.no_grad():
                        d_loss_raw = 0.5 * (
                            ((real_pred - 1.0) ** 2).mean() +
                            ((fake_pred - 0.0) ** 2).mean()
                        )

                    d_loss = 0.5 * (
                        self.gan_loss(real_pred, True, weight=gan_scale.float()) +
                        self.gan_loss(fake_pred, False, weight=gan_scale.float())
                    )

                    if do_r1:
                        # PatchGAN-safe reduction to per-sample scalar before grad
                        real_score = real_pred
                        if real_score.dim() > 1:
                            real_score = real_score.view(real_score.shape[0], -1).mean(1)

                        grad = torch.autograd.grad(
                            outputs=real_score.sum(),
                            inputs=x_real_D,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        r1_pen = 0.5 * self.r1_gamma_main * (grad.pow(2).reshape(grad.shape[0], -1).sum(1)).mean()
                        d_loss = d_loss + r1_pen
                        r1_sum += float(r1_pen.detach().item())
                        r1_n += 1

                    # Backprop + grad norm logging + step
                    d_loss.backward()

                    # D_grad_norm (per-update)
                    with torch.no_grad():
                        s = 0.0
                        for p in self.main_D.parameters():
                            if p.grad is not None:
                                s += p.grad.detach().float().pow(2).sum().item()
                        D_grad_norm_val = math.sqrt(s) if s > 0 else 0.0

                    self.opt_D.step()
                    self.opt_D.zero_grad(set_to_none=True)

                    # -----------------------
                    # Generator adversarial (freeze D params during G update)
                    # (AND apply resume-warmup factor)
                    # -----------------------
                    mainD_fwd = _unwrap_ddp(self.main_D)
                    with _freeze_params(mainD_fwd):
                        fake_pred_g = mainD_fwd(x0_pred.float(), bb_mask_sel.detach().float())
                        g_adv = self.gan_loss(fake_pred_g, True, weight=gan_scale.float())

                    g_adv = g_adv * float(g_adv_factor)

                    # -----------------------
                    # Recon losses
                    # -----------------------
                    l1_map = (x0_pred - micro_sel).abs().mean(dim=(1, 2, 3))
                    g_l1 = (l1_map * l1_scale).mean()

                    # Background L1 anchor (helps prevent global hue/contrast drift)
                    bg_mask_c = (1.0 - bb_mask_sel).expand(-1, C, -1, -1)
                    l1_bg_map = ((x0_pred - micro_sel).abs() * bg_mask_c).mean(dim=(1, 2, 3))
                    g_l1_bg = (l1_bg_map * l1_scale * float(self.l1_bg_frac)).mean()

                    if roi_keep.any():
                        bb_mask_c = bb_mask_sel.expand(-1, C, -1, -1)
                        mse_map = ((x0_pred - micro_sel) ** 2) * bb_mask_c
                        denom = (bb_mask_c.sum(dim=(1, 2, 3)) + 1e-8)
                        mse_roi = (mse_map.sum(dim=(1, 2, 3)) / denom)
                        mse_roi = mse_roi[roi_keep].mean()
                        g_roi = self.lambda_enhanced * mse_roi
                    else:
                        g_roi = torch.zeros((), device=micro.device, dtype=micro.dtype)

                    # -----------------------
                    # Small ROI D (optional)
                    # -----------------------
                    g_adv_small = torch.zeros((), device=micro.device, dtype=micro.dtype)
                    d_small_loss = torch.zeros((), device=micro.device, dtype=micro.dtype)

                    if self.enable_small_roi_d and (self.small_D is not None) and (self.opt_small_D is not None) and roi_keep.any():
                        kept_idx = roi_keep.nonzero(as_tuple=False).squeeze(1)
                        micro_kept = micro_sel[kept_idx]
                        pred_kept = x0_pred[kept_idx]
                        bbs_kept = [bbs_sel[int(k.item())] for k in kept_idx]

                        real_roi = crop_resize_square_roi(micro_kept, bbs_kept, self.roi_size, self.roi_pad_ratio)
                        fake_roi = crop_resize_square_roi(pred_kept, bbs_kept, self.roi_size, self.roi_pad_ratio)

                        real_roi = diffaug_translate(real_roi, self.diffaug_translate_px)
                        fake_roi = diffaug_translate(fake_roi, self.diffaug_translate_px)

                        # small D update
                        self.opt_small_D.zero_grad(set_to_none=True)

                        real_pred_s = self.small_D(real_roi.detach().float())
                        fake_pred_s = self.small_D(fake_roi.detach().float())
                        gan_scale_kept = gan_scale[kept_idx].float()

                        d_small = 0.5 * (
                            self.gan_loss(real_pred_s, True, weight=gan_scale_kept) +
                            self.gan_loss(fake_pred_s, False, weight=gan_scale_kept)
                        )

                        # optional R1 (PatchGAN-safe reduction)
                        if self.r1_every > 0 and (global_step % self.r1_every == 0) and (self.r1_gamma_small > 0):
                            real_roi_r1 = real_roi.detach().float().requires_grad_(True)
                            real_pred_r1 = self.small_D(real_roi_r1)
                            real_score_r1 = real_pred_r1
                            if real_score_r1.dim() > 1:
                                real_score_r1 = real_score_r1.view(real_score_r1.shape[0], -1).mean(1)

                            grad = torch.autograd.grad(
                                outputs=real_score_r1.sum(),
                                inputs=real_roi_r1,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                            )[0]
                            r1_pen = 0.5 * self.r1_gamma_small * (grad.pow(2).reshape(grad.shape[0], -1).sum(1)).mean()
                            d_small = d_small + r1_pen

                        d_small.backward()
                        self.opt_small_D.step()
                        self.opt_small_D.zero_grad(set_to_none=True)
                        d_small_loss = d_small.detach()

                        # small G adv (freeze small D params during G update)
                        smallD_fwd = _unwrap_ddp(self.small_D)
                        with _freeze_params(smallD_fwd):
                            fake_pred_s_g = smallD_fwd(fake_roi.float())
                            g_adv_small = self.gan_loss(fake_pred_s_g, True, weight=gan_scale_kept)

                        g_adv_small = g_adv_small * float(g_adv_factor)

                        d_small_sum += float(d_small_loss.detach().item())
                        d_small_n += 1

                    # -----------------------
                    # Optional color stabilizers
                    # -----------------------
                    g_color_stats = torch.tensor(0.0, device=self.device)
                    if self.color_stats_scale > 0:
                        x0_f = x0_pred.float()
                        xr_f = micro_sel.float()
                        mean_pred = x0_f.mean(dim=(2, 3))
                        mean_real = xr_f.mean(dim=(2, 3))
                        std_pred = x0_f.std(dim=(2, 3))
                        std_real = xr_f.std(dim=(2, 3))
                        color_stats_loss = (mean_pred - mean_real).abs().mean() + (std_pred - std_real).abs().mean()
                        g_color_stats = float(self.color_stats_scale) * color_stats_loss

                    g_sat = torch.tensor(0.0, device=self.device)
                    if self.sat_penalty_scale > 0:
                        thr = float(self.sat_threshold)
                        sat_excess = (x0_pred.float().abs() - thr).clamp(min=0.0)
                        sat_loss = (sat_excess ** 2).mean()
                        g_sat = float(self.sat_penalty_scale) * sat_loss

                    extras_mean = g_adv + g_adv_small + g_l1 + g_l1_bg + g_roi + g_color_stats + g_sat
                    extra_g_loss = extras_mean * (float(B_sel) / float(B_mb))

                    # no_sync DDP discriminators during G backward (even though params frozen on forward)
                    d_nosync = self.main_D.no_sync() if self.use_ddp_D else contextlib.nullcontext()
                    sd_nosync = (
                        self.small_D.no_sync()
                        if (self.use_ddp_D and (self.small_D is not None) and hasattr(self.small_D, "no_sync"))
                        else contextlib.nullcontext()
                    )
                    with d_nosync, sd_nosync:
                        self.mp_trainer.backward(extra_g_loss)

                    # Debug image save
                    if self.image_save_dir and (_rank() == 0) and (global_step % self.image_save_every == 0):
                        filenames_mb = micro_cond.get("image_filenames", None)
                        filenames_sel = None
                        if isinstance(filenames_mb, (list, tuple)):
                            filenames_sel = [filenames_mb[int(j.item())] for j in idxs]

                        self._save_debug_pairs(
                            x_t_real=x_t_real.detach(),
                            x0_pred=x0_pred.detach(),
                            bbs=bbs_sel,
                            step=global_step,
                            outdir=self.image_save_dir,
                            max_n=min(8, B_sel),
                            filenames=filenames_sel,
                            t=t_sel.detach().cpu().tolist(),
                            t_rev=t_rev[idxs].detach().cpu().tolist(),
                        )

                    # -----------------------
                    # Accumulate per-step GAN stats (weight by B_sel)
                    # -----------------------
                    gan_updates += 1
                    gan_sel_sum += float(B_sel)

                    d_loss_sum += float(d_loss.detach().item()) * float(B_sel)
                    d_loss_raw_sum += float(d_loss_raw.detach().item()) * float(B_sel)

                    g_adv_sum += float(g_adv.detach().item()) * float(B_sel)
                    g_l1_sum += float(g_l1.detach().item()) * float(B_sel)
                    g_l1_bg_sum += float(g_l1_bg.detach().item()) * float(B_sel)
                    g_roi_sum += float(g_roi.detach().item()) * float(B_sel)

                    if self.color_stats_scale > 0:
                        g_color_sum += float(g_color_stats.detach().item()) * float(B_sel)
                    if self.sat_penalty_scale > 0:
                        g_sat_sum += float(g_sat.detach().item()) * float(B_sel)

                    gan_scale_mean_sum += gan_scale_mean_val * float(B_sel)
                    gan_scale_max_max = max(gan_scale_max_max, gan_scale_max_val)

                    D_grad_norm_sum += float(D_grad_norm_val)
                    D_grad_norm_max = max(D_grad_norm_max, float(D_grad_norm_val))

        # -----------------------
        # Emit per-step aggregated logs (ONE place only)
        # -----------------------
        if diff_loss_den > 0:
            logger.logkv("diffusion_loss", diff_loss_sum / diff_loss_den)
        else:
            logger.logkv("diffusion_loss", 0.0)

        if gan_frac_den > 0:
            logger.logkv("gan_active_frac", gan_frac_sum / gan_frac_den)
        else:
            logger.logkv("gan_active_frac", 0.0)

        logger.logkv("gan_updates", float(gan_updates))
        logger.logkv("gan_sel", float(gan_sel_sum))

        if gan_sel_sum > 0:
            logger.logkv("d_loss", d_loss_sum / gan_sel_sum)
            logger.logkv("d_loss_raw", d_loss_raw_sum / gan_sel_sum)

            logger.logkv("g_adv", g_adv_sum / gan_sel_sum)
            logger.logkv("g_l1", g_l1_sum / gan_sel_sum)
            logger.logkv("g_l1_bg", g_l1_bg_sum / gan_sel_sum)
            logger.logkv("g_roi", g_roi_sum / gan_sel_sum)

            if self.color_stats_scale > 0:
                logger.logkv("g_color_stats", g_color_sum / gan_sel_sum)
            if self.sat_penalty_scale > 0:
                logger.logkv("g_sat", g_sat_sum / gan_sel_sum)

            logger.logkv("gan_scale_mean", gan_scale_mean_sum / gan_sel_sum)
            logger.logkv("gan_scale_max", gan_scale_max_max)

            logger.logkv("D_grad_norm", D_grad_norm_sum / max(1.0, float(gan_updates)))
            logger.logkv("D_grad_norm_max", D_grad_norm_max)

            if r1_n > 0:
                logger.logkv("r1_main", r1_sum / float(r1_n))

            if self.enable_small_roi_d and d_small_n > 0:
                logger.logkv("d_small_loss", d_small_sum / float(d_small_n))
        else:
            # keep table stable when GAN not active this step
            logger.logkv("d_loss", 0.0)
            logger.logkv("d_loss_raw", 0.0)
            logger.logkv("g_adv", 0.0)
            logger.logkv("g_l1", 0.0)
            logger.logkv("g_l1_bg", 0.0)
            logger.logkv("g_roi", 0.0)
            if self.color_stats_scale > 0:
                logger.logkv("g_color_stats", 0.0)
            if self.sat_penalty_scale > 0:
                logger.logkv("g_sat", 0.0)
            logger.logkv("gan_scale_mean", 0.0)
            logger.logkv("gan_scale_max", 0.0)
            logger.logkv("D_grad_norm", 0.0)
            logger.logkv("D_grad_norm_max", 0.0)

    # ---------------------------
    # Debug image saver
    # ---------------------------
    def _save_debug_pairs(self, x_t_real, x0_pred, bbs, step, outdir, max_n=8, filenames=None, t=None, t_rev=None):
        """
        Saves a grid with [x_t_real | x0_pred] rows, both converted from [-1,1] to [0,1],
        with bbox rectangles drawn.
        """
        os.makedirs(outdir, exist_ok=True)
        B = min(max_n, x_t_real.shape[0])
        rows = []
        for i in range(B):
            x1, y1, x2, y2 = bbs[i]
            xt = x_t_real[i].clone()
            xp = x0_pred[i].clamp(-1, 1).clone()

            for img in (xt, xp):
                img[:, y1:y1+1, x1:x2] = 1
                img[:, y2-1:y2, x1:x2] = 1
                img[:, y1:y2, x1:x1+1] = 1
                img[:, y1:y2, x2-1:x2] = 1

            xt = (xt + 1) / 2
            xp = (xp + 1) / 2
            xt = xt.clamp(0, 1)
            xp = xp.clamp(0, 1)

            pair = torch.cat([xt, xp], dim=2)
            rows.append(pair.unsqueeze(0))

        grid = torch.cat(rows, dim=0)
        save_path = os.path.join(outdir, f"PAIR_xt_left_x0pred_right_step_{int(step):06d}.png")
        vutils.save_image(grid, save_path)

        try:
            meta = {"step": int(step)}
            if filenames is not None:
                meta["filenames"] = [str(x) for x in filenames]
            if bbs is not None:
                meta["bbs_xyxy"] = [[int(a), int(b), int(c), int(d)] for (a, b, c, d) in bbs]
            if t is not None:
                meta["t"] = [int(x) for x in t]
            if t_rev is not None:
                meta["t_rev"] = [int(x) for x in t_rev]

            meta_path = os.path.join(outdir, f"PAIR_meta_step_{int(step):06d}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.log(f"[warn] Failed to write debug pair meta JSON: {e}")

    # ---------------------------
    # EMA / LR / Saving
    # ---------------------------
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        new_lr = self.lr * (1 - frac_done)
        for pg in self.opt.param_groups:
            pg["lr"] = new_lr

        # anneal the discriminator learning rates too
        ratio = new_lr / max(self.lr, 1e-12)
        for pg in self.opt_D.param_groups:
            base = float(pg.get("base_lr", pg.get("lr", 0.0)))
            pg["lr"] = base * ratio

        if self.opt_small_D is not None:
            for pg in self.opt_small_D.param_groups:
                base = float(pg.get("base_lr", pg["lr"]))
                pg["lr"] = base * (new_lr / max(self.lr, 1e-12))

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)

        try:
            if self.use_bb_mask_cond:
                if hasattr(self.model, "mask_weight_stats"):
                    s = self.model.mask_weight_stats()
                    logger.logkv("bb_mask_w_absmean", float(s.get("absmean", 0.0)))
                    logger.logkv("bb_mask_w_absmax", float(s.get("absmax", 0.0)))
                    logger.logkv("bb_mask_w_mean", float(s.get("mean", 0.0)))
                elif hasattr(self.model, "adapter"):
                    w = self.model.adapter.weight.detach().float()[:, 3, 0, 0]
                    logger.logkv("bb_mask_w_absmean", float(w.abs().mean().item()))
                    if w.numel() >= 3:
                        logger.logkv("bb_mask_w_r", float(w[0].item()))
                        logger.logkv("bb_mask_w_g", float(w[1].item()))
                        logger.logkv("bb_mask_w_b", float(w[2].item()))
                if hasattr(self.model, "mask_gain"):
                    logger.logkv("mask_gain", float(getattr(self.model, "mask_gain")))
        except Exception:
            pass

        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """
        Saves:
          - generator model + EMA
          - generator optimizer state
          - discriminator(s) + discriminator optimizer(s)
        """
        step = int(self.step + self.resume_step)

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if _rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{step:06d}.pt"
                with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if _rank() == 0:
            with bf.BlobFile(bf.join(self.checkpoint_dir, f"opt{step:06d}.pt"), "wb") as f:
                th.save(self.opt.state_dict(), f)

        # save discriminators + optimisers (rank 0 only)
        if self.save_discriminators and _rank() == 0:
            try:
                # main D weights
                D_sd = _unwrap_ddp(self.main_D).state_dict()
                with bf.BlobFile(bf.join(self.checkpoint_dir, f"D_main{step:06d}.pt"), "wb") as f:
                    th.save(D_sd, f)

                # main D optimizer
                with bf.BlobFile(bf.join(self.checkpoint_dir, f"optD_main{step:06d}.pt"), "wb") as f:
                    th.save(self.opt_D.state_dict(), f)

                # small D (optional)
                if self.small_D is not None and self.opt_small_D is not None:
                    D2_sd = _unwrap_ddp(self.small_D).state_dict()
                    with bf.BlobFile(bf.join(self.checkpoint_dir, f"D_small{step:06d}.pt"), "wb") as f:
                        th.save(D2_sd, f)
                    with bf.BlobFile(bf.join(self.checkpoint_dir, f"optD_small{step:06d}.pt"), "wb") as f:
                        th.save(self.opt_small_D.state_dict(), f)

                logger.log(f"[ckpt] saved discriminator checkpoints at step {step:06d}")
            except Exception as e:
                logger.log(f"[warn] Failed to save discriminator checkpoints: {e}")

        if _dist_is_ready():
            dist.barrier()


# ---------------------------------------------------------------------------
# Checkpoint helpers (generator)
# ---------------------------------------------------------------------------
def parse_resume_step_from_filename(filename: str) -> int:
    """
    Robustly parse step from:
      - .../model000123.pt
      - .../ema_0.9999_000123.pt
      - .../<anything>000123.pt
    """
    b = os.path.basename(filename)
    m = re.search(r"model(\d+)\.pt$", b)
    if m:
        return int(m.group(1))
    m = re.search(r"ema_[^_]+_(\d+)\.pt$", b)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\.pt$", b)
    if m:
        return int(m.group(1))
    return 0


def find_resume_checkpoint():
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
