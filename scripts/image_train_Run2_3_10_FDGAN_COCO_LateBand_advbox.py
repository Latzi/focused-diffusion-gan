# -*- coding: utf-8 -*-
"""
FDGAN training entry point.

Builds the ADM model and diffusion (guided_diffusion.script_util), widens the first U-Net convolution to four input
channels when the box mask is used, builds the spectral-norm PatchGAN discriminator, the box-aware data loader
(guided_diffusion.image_datasets_Run2_3_10_COCO) and the FDGAN training loop
(guided_diffusion.train_util_coco_fdgan_late_band.TrainLoop). Every run of the paper calls this script through one
of the job files in jobs/.
"""

import os
import argparse
import inspect

# -----------------------------------------------------------------------------
# IMPORTANT: ensure LOCAL_RANK is set BEFORE importing train_util module
# (that module sets torch.cuda.set_device() at import time using LOCAL_RANK).
# -----------------------------------------------------------------------------
if "LOCAL_RANK" not in os.environ:
    for k in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        if k in os.environ:
            os.environ["LOCAL_RANK"] = os.environ[k]
            break
    else:
        os.environ["LOCAL_RANK"] = "0"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import spectral_norm

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets_Run2_3_10_COCO import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.train_util_coco_fdgan_late_band_advbox import TrainLoop


# -----------------------------------------------------------------------------
# Spectral-Norm PatchGAN discriminator (NO BatchNorm)
# -----------------------------------------------------------------------------
class SpectralNLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral-norm convolutions and no batch norm, which is more stable at our
    micro-batch size.
    """

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, use_sigmoid: bool = False):
        super().__init__()
        assert n_layers >= 1
        layers = []
        ch = input_nc
        # Stacked convs; last block uses stride=1 (more PatchGAN-like)
        for i in range(n_layers):
            out_ch = int(ndf * min(2 ** i, 4))
            stride = 2 if i < (n_layers - 1) else 1
            layers += [
                spectral_norm(nn.Conv2d(ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch
        # Final 1-channel conv (patch map)
        layers += [spectral_norm(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1, bias=False))]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _filter_kwargs_for_call(fn, kwargs):
    """Keep only kwargs that exist in fn(...) signature (prevents loader mismatches)."""
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def create_argparser():
    """Every setting of a run with its default: data, optimiser, output, discriminator, band, adversarial and
    reconstruction weights, ANPM, small discriminator, colour terms, checkpointing and the mask channel, plus the ADM
    model and diffusion flags from script_util. The job files override what differs per row.
    """
    defaults = dict(
        # data
        data_dir="",
        bb_dir="",
        bb_is_xywh=False,
        bb_xyxy_end_inclusive=True,

        # schedule sampler
        schedule_sampler="uniform",

        # optimizer/training
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

        # output
        log_dir="",
        checkpoint_dir="",
        image_save_dir="",
        image_save_every=1000,

        # discriminator flags (GAN_modules define_image_D)
        use_mask_adapter=True,
        ndf=128,
        n_layers_D=3,
        use_sigmoid=False,
        lr_D=1e-5,
        beta1_D=0.5,
        beta2_D=0.999,

        # main D stabilizers
        r1_gamma_main=5.0,
        r1_every_main=16,
        diffaug_translate_px_main=0,

        # band defaults (the job files override them where needed)
        tmin_start=820,
        tmin_end=700,
        band_width=120,
        tmax_cap=820,
        band_ramp_steps=50000,
        diffusion_in_band_weight=0.7,

        # GAN scaling schedule
        gan_initial_scale=1e-4,
        gan_final_scale=1e-2,
        # Global warmup: disable GAN + extras for first N steps
        gan_warmup_steps=40000,
        gan_scale_cap=0.0,
        gan_ramp_steps=0,

        # Recon schedules
        l1_initial_scale=5e-2,
        l1_final_scale=5e-2,
        # Background L1 anchor (fraction of main L1 weight, 0.10–0.25 typical)
        l1_bg_frac=0.20,
        lambda_enhanced=5e-2,

        # ROI noise
        noise_option="extra_gan_only",
        k_focus=1.75,
        max_gan_per_microbatch=4,

        # small ROI D (optional)
        enable_small_roi_d=False,
        adv_grad_in_box_only=False,
        roi_size=128,
        roi_pad_ratio=0.03,
        train_roi_frac_cap=0.15,
        diffaug_translate_px=4,
        r1_gamma_small=5.0,
        r1_every=16,
        lr_small_D=1e-5,

        # color stabilizers
        color_stats_scale=0.0,
        sat_penalty_scale=0.0,
        sat_threshold=0.95,

        # save/resume discriminator states
        save_discriminators=True,
        resume_discriminators=True,
        g_adv_warmup_steps_if_no_d_ckpt=2000,

        # --- BB mask conditioning into the UNet (4ch: RGB+mask)
        # If True, TrainLoop will pass a bb mask tensor via model_kwargs (key=bb_mask_key)
        # and this script will wrap the UNet so it consumes RGB+mask.
        use_bb_mask_cond=False,
        bb_mask_key="bb_mask",
        # Scale applied to the mask channel before concatenation.
        mask_gain=50.0,
        # How to initialize the extra (mask) channel weights in the UNet first conv.
        #   - copy_mean: copy mean(R,G,B) weights into mask channel, scaled by mask_init_scale
        #   - zero:      initialize mask channel weights to 0
        mask_init_mode="copy_mean",
        mask_init_scale=0.10,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    """Build the ADM model and diffusion, widen the U-Net input to four channels when the box mask is used, build the
    discriminator, the loader and the training loop, and train.
    """
    args = create_argparser().parse_args()

    if not args.data_dir:
        raise ValueError("--data_dir is required")

    # Distributed init
    dist_util.setup_dist()

    # Logging dir (separate from checkpoint dir)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        logger.configure(dir=args.log_dir)
    else:
        logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # ------------------------------------------------------------------
    # BB mask conditioning support
    # ------------------------------------------------------------------
    # With --use_bb_mask_cond True the TrainLoop provides a bb mask
    # in model_kwargs (key=args.bb_mask_key). Here we ensure the UNet consumes
    # RGB+mask (4ch) while the outer interface remains compatible with
    # guided-diffusion (still called with 3ch x_t).
    try:
        from guided_diffusion.unet import UNetModel  # type: ignore
    except Exception:
        UNetModel = None  # type: ignore

    def _inflate_unet_first_conv_to_4ch(unet: nn.Module, init_mode: str, init_scale: float):
        """Replace the first UNet conv (3ch) with 4ch and init mask channel."""
        conv = None
        try:
            conv = unet.input_blocks[0][0]
        except Exception as e:
            raise RuntimeError(f"Could not access unet.input_blocks[0][0] for inflation: {e}")

        if not isinstance(conv, nn.Conv2d):
            raise RuntimeError(f"Expected first layer to be nn.Conv2d, got {type(conv)}")

        if conv.in_channels == 4:
            return
        if conv.in_channels != 3:
            raise RuntimeError(f"Unexpected first conv in_channels={conv.in_channels}; expected 3 or 4")

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
        )

        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :3] = conv.weight

            init_mode = str(init_mode)
            if init_mode == "zero":
                pass
            elif init_mode == "copy_mean":
                mean = conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:4] = mean * float(init_scale)
            else:
                raise ValueError(f"Unknown mask_init_mode: {init_mode}")

            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)

        unet.input_blocks[0][0] = new_conv
        if hasattr(unet, "in_channels"):
            try:
                unet.in_channels = 4
            except Exception:
                pass

    class _BB4ChWrapper(nn.Module):
        """Wrap a 4ch UNet so it can be called with (x3, t, bb_mask=...)."""

        def __init__(self, unet: nn.Module, bb_mask_key: str, mask_gain: float):
            super().__init__()
            self.unet = unet
            self.bb_mask_key = str(bb_mask_key)
            self.mask_gain = float(mask_gain)

        def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs):
            bb_mask = kwargs.pop(self.bb_mask_key, None)
            if bb_mask is None:
                bb_mask = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
            if bb_mask.dim() == 3:
                bb_mask = bb_mask.unsqueeze(1)
            x4 = torch.cat([x, bb_mask.to(dtype=x.dtype) * self.mask_gain], dim=1)

            # UNetModel signature: forward(x, timesteps, y=None)
            if "y" in kwargs:
                return self.unet(x4, timesteps, y=kwargs["y"])
            return self.unet(x4, timesteps)
        def convert_to_fp16(self):
            # guided-diffusion MixedPrecisionTrainer expects this on the model.
            if hasattr(self.unet, "convert_to_fp16"):
                self.unet.convert_to_fp16()
            return self

        def convert_to_fp32(self):
            if hasattr(self.unet, "convert_to_fp32"):
                self.unet.convert_to_fp32()
            return self


    if bool(getattr(args, "use_bb_mask_cond", False)):
        if UNetModel is not None and isinstance(model, UNetModel):
            _inflate_unet_first_conv_to_4ch(
                model,
                init_mode=getattr(args, "mask_init_mode", "copy_mean"),
                init_scale=float(getattr(args, "mask_init_scale", 0.10)),
            )
            model = _BB4ChWrapper(
                model,
                bb_mask_key=getattr(args, "bb_mask_key", "bb_mask"),
                mask_gain=float(getattr(args, "mask_gain", 50.0)),
            )
        else:
            # A model that is already wrapped: expose mask_gain for logging.
            if not hasattr(model, "mask_gain"):
                try:
                    setattr(model, "mask_gain", float(getattr(args, "mask_gain", 50.0)))
                except Exception:
                    pass
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # -----------------------------
    # Discriminator (SpectralNorm, no BatchNorm)
    # -----------------------------
    input_nc = 3
    discriminator = SpectralNLayerDiscriminator(
        input_nc=input_nc,
        ndf=args.ndf,
        n_layers=args.n_layers_D,
        use_sigmoid=args.use_sigmoid,
    )

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        print("Discriminator Architecture:")
        print(discriminator)
        total_params = sum(p.numel() for p in discriminator.parameters())
        print(f"Total parameters in discriminator: {total_params}")

    # -----------------------------
    # Data loader (box-aware)
    # -----------------------------
    logger.log("creating data loader...")
    loader_kwargs = dict(
        data_dir=args.data_dir,
        bb_dir=args.bb_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        bb_is_xywh=args.bb_is_xywh,
        bb_xyxy_end_inclusive=args.bb_xyxy_end_inclusive,
    )
    loader_kwargs = _filter_kwargs_for_call(load_data, loader_kwargs)
    data = load_data(**loader_kwargs)

    # -----------------------------
    # Checkpoint dir
    # -----------------------------
    ckpt_dir = args.checkpoint_dir
    if not ckpt_dir:
        # If not given, default to the resume dir, else the logger dir.
        if args.resume_checkpoint:
            ckpt_dir = os.path.dirname(args.resume_checkpoint)
        else:
            ckpt_dir = logger.get_dir()
    os.makedirs(ckpt_dir, exist_ok=True)

    # Debug image dir
    img_dir = args.image_save_dir or ""
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)

    logger.log("training (FDGAN COCO late-band)...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_dir=ckpt_dir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,

        discriminator=discriminator,
        use_mask_adapter=args.use_mask_adapter,
        lr_D=args.lr_D,
        betas_D=(args.beta1_D, args.beta2_D),

        # main D stabilizers
        r1_gamma_main=args.r1_gamma_main,
        r1_every_main=args.r1_every_main,
        diffaug_translate_px_main=args.diffaug_translate_px_main,

        # COCO late-band
        tmin_start=args.tmin_start,
        tmin_end=args.tmin_end,
        band_width=args.band_width,
        tmax_cap=args.tmax_cap,
        band_ramp_steps=args.band_ramp_steps,
        diffusion_in_band_weight=args.diffusion_in_band_weight,

        # GAN schedule
        gan_initial_scale=args.gan_initial_scale,
        gan_final_scale=args.gan_final_scale,
        gan_warmup_steps=args.gan_warmup_steps,
        gan_scale_cap=args.gan_scale_cap,
        gan_ramp_steps=args.gan_ramp_steps,

        # recon
        l1_initial_scale=args.l1_initial_scale,
        l1_final_scale=args.l1_final_scale,
        l1_bg_frac=args.l1_bg_frac,
        lambda_enhanced=args.lambda_enhanced,

        # ROI noise + bb format
        noise_option=args.noise_option,
        k_focus=args.k_focus,
        bb_is_xywh=args.bb_is_xywh,
        max_gan_per_microbatch=args.max_gan_per_microbatch,

        # small ROI D
        enable_small_roi_d=args.enable_small_roi_d,
        adv_grad_in_box_only=args.adv_grad_in_box_only,
        roi_size=args.roi_size,
        roi_pad_ratio=args.roi_pad_ratio,
        train_roi_frac_cap=args.train_roi_frac_cap,
        diffaug_translate_px=args.diffaug_translate_px,
        r1_gamma_small=args.r1_gamma_small,
        r1_every=args.r1_every,
        lr_small_D=args.lr_small_D,

        # debug images
        image_save_every=args.image_save_every,
        image_save_dir=img_dir,

        # BB conditioning into the UNet
        use_bb_mask_cond=args.use_bb_mask_cond,
        bb_mask_key=args.bb_mask_key,

        # color
        color_stats_scale=args.color_stats_scale,
        sat_penalty_scale=args.sat_penalty_scale,
        sat_threshold=args.sat_threshold,

        # D save/resume
        save_discriminators=args.save_discriminators,
        resume_discriminators=args.resume_discriminators,
        g_adv_warmup_steps_if_no_d_ckpt=args.g_adv_warmup_steps_if_no_d_ckpt,
    ).run_loop()


if __name__ == "__main__":
    main()
