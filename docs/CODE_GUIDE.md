# Code guide

This is a walk through the files we wrote, function by function, with the ADM origin stated where there is one.
The code is the code that produced the paper's runs. Before publishing we rewrote the module docstrings and a few
comments that referred to our own working setup, and we added a docstring to every function and class; no executable
line changed.

## The ADM modules

`guided_diffusion/` holds the guided-diffusion package as it was on the cluster. Against OpenAI's repository,
`unet.py`, `respace.py`, `nn.py`, `losses.py`, `logger.py`, `fp16_util.py` and `__init__.py` are byte-identical, and
so is `setup.py`. Four modules carry small leftovers from our earlier experiments:

- `dist_util.py`: `GPUS_PER_NODE` is 2 instead of 8.
- `script_util.py`: a channel multiplier for 32 px images (`image_size == 32`).
- `resample.py`: an `EarlyTimestepSampler` class, and `np.int` replaced by `int` in the loss-aware sampler for newer
  numpy.
- `gaussian_diffusion.py`: a helper `get_noise_scale_for_timestep`.

None of these is called by the FDGAN scripts; the runs use the uniform timestep sampler and the 256 px model. We left
them in so that the package is the one that ran.

## guided_diffusion/train_util_coco_fdgan_late_band.py

Ours. It replaces ADM's `TrainLoop` (guided-diffusion `train_util.py`, which is not shipped because nothing here
imports it). The ADM parts we still call are `diffusion.training_losses`, `diffusion.q_sample`,
`diffusion.p_mean_variance`, the `MixedPrecisionTrainer`, the EMA update and the schedule sampler.

Helpers at the top of the file:

- `_get_bb_xyxy`, `_build_bb_mask_xyxy`: turn the box dictionary from the loader into clamped corner coordinates and a
  binary mask of shape (B, 1, H, W). Both box formats are handled (`bb_is_xywh`).
- `_bbox_area_frac`: box area as a fraction of the image; the box MSE and the small discriminator only use boxes below
  `train_roi_frac_cap` (0.15).
- `WeightedLSGANLoss`: least-squares GAN loss with one weight per sample. It works on patch maps of any shape.
- `diffaug_translate`, `diffaug_translate_pair`: integer translation with reflect padding, used on the crops fed to the
  small discriminator (4 px); off for the main discriminator in our runs.
- `crop_resize_square_roi`: square crop around the box, padded by 3 % of its side, resized to 128 px. Used by the small
  discriminator and by the sampler's optional guidance.
- `SmallROIPatchD`: the three-convolution discriminator on the box crop (the `+ small ROI D` ablation row).
- `MainDWithMaskAdapter`: wraps the main PatchGAN; concatenates image and mask, maps 4 to 3 channels with a 1 x 1
  convolution, then runs the PatchGAN.
- `_inflate_first_conv_weight_if_needed`, `load_state_dict_flexible`: let a 3-channel ADM checkpoint load into the
  4-channel model and tolerate the wrapper prefix.

`TrainLoop`:

- `__init__`: all FDGAN settings are keyword arguments; the job files pass them as flags. Band: `tmin_start`,
  `tmin_end`, `band_width`, `tmax_cap`, `band_ramp_steps`. Adversarial weight: `gan_initial_scale`, `gan_final_scale`,
  `gan_warmup_steps`. Reconstruction: `l1_initial_scale`, `l1_final_scale`, `l1_bg_frac`, `lambda_enhanced`,
  `color_stats_scale`, `sat_penalty_scale`, `sat_threshold`. ANPM: `noise_option`, `k_focus`. Discriminator: `lr_D`,
  `betas_D`, `r1_gamma_main`, `r1_every_main`, `use_mask_adapter`. Small discriminator: `enable_small_roi_d` and the
  `roi_*` arguments. Mask channel: `use_bb_mask_cond`, `bb_mask_key`.
- `_current_band_edges(step)`: the band's lower edge moves linearly from `tmin_start` to `tmin_end` over
  `band_ramp_steps`; the upper edge is lower edge plus `band_width`, capped at `tmax_cap`. Reverse-time units.
- `_gan_scale(t_rev)`: lambda_GAN(t) = initial * exp(log(final / initial) * t_rev / 999). Optional ramp and cap, both
  unused in our runs.
- `_l1_scale(t_rev)`: same form for the L1 weight; initial and final are equal in our runs, so it is a constant.
- `forward_backward(batch, cond)`: one optimisation step over the micro-batches.
  1. `gan_enabled` is true once `global_step >= gan_warmup_steps`.
  2. For every micro-batch: sample timesteps, convert to reverse time, mark `in_band`.
  3. Build the box mask; if `use_bb_mask_cond`, put it into `model_kwargs` so the U-Net wrapper concatenates it.
  4. Diffusion loss on every sample through `diffusion.training_losses` (ADM), in-band samples scaled by
     `diffusion_in_band_weight`. Backward.
  5. If any sample is in band (at most `max_gan_per_microbatch` of them, 4): draw `eps_base` and `eps_focus`, form
     `eps_prime` according to `noise_option`, make `x_t_real` with `q_sample`, predict `x0_pred` with
     `p_mean_variance` (clipped to [-1, 1]).
  6. Discriminator update on (real image, mask) against (`x0_pred`, mask), each sample weighted by lambda_GAN(t);
     R1 penalty on the real images every `r1_every_main` steps; gradient norm logged.
  7. Generator adversarial loss through the frozen discriminator, then L1, background L1, box MSE, the optional small
     discriminator pair, colour statistics and saturation penalty. Their sum, scaled by the in-band fraction of the
     micro-batch, goes through a second backward pass.
  8. Per-step logging keys: `diffusion_loss`, `gan_active_frac`, `d_loss`, `d_loss_raw`, `g_adv`, `g_l1`, `g_l1_bg`,
     `g_roi`, `g_color_stats`, `g_sat`, `gan_scale_mean`, `gan_scale_max`, `D_grad_norm`, `r1_main`,
     `band_tmin_rev`, `band_tmax_rev`, `gan_enabled`.
- `save`, `_load_discriminators_and_opts_if_present`: generator, EMA and optimiser as in ADM, plus `D_main<step>.pt`,
  `D_small<step>.pt` and their optimiser states next to them. Without discriminator files on resume,
  `_g_adv_warmup_factor` ramps the generator's adversarial term over `g_adv_warmup_steps_if_no_d_ckpt` steps.
- `_save_debug_pairs`: every `image_save_every` steps, writes the perturbed input and the clean-image prediction of a
  few in-band samples with their boxes drawn, for eyeballing a run.

Two copies of the loop sit next to it. `train_util_coco_fdgan_late_band_lsun.py` is the one the COCO jobs import; it
differs from the main file in three lines: when a 3-channel pretrained checkpoint is loaded into the 4-channel model,
the mask-channel weights get the copy-mean initialisation (0.1 times the mean of the loaded RGB filters) instead of
zeros, which matters only when a run starts from a checkpoint, as the COCO runs do from an LSUN-trained ADM.
`train_util_coco_fdgan_late_band_globalnoise.py` is byte-identical to the main file; the Cityscapes global-noise job
imports it through its own copy of the training script.

## guided_diffusion/image_datasets_Run2_3_10_COCO.py

Ours. It replaces ADM's `image_datasets.py`. `load_data` lists the images, finds the box folder (`bb_dir`, or an
`_bb` folder next to the image folder) and returns an infinite generator of batches, sharded by rank without mpi4py.
`ImageDataset.__getitem__` loads the image, reads the box JSON of the same stem, converts it to exclusive corner
coordinates (Cityscapes and COCO store corners; Traffic Signs stores x, y, width, height, hence `bb_is_xywh`),
mirrors the box together with the image on a random flip, and returns the image in [-1, 1] with
`{"image_filenames", "flip", "bb"}`.

## scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py

Ours. The training entry point.

- `SpectralNLayerDiscriminator`: the main PatchGAN. `n_layers` 4 x 4 convolutions with spectral norm and LeakyReLU
  (stride 2, 2, 1 for three layers; 128, 256, 512 channels), then a 3 x 3 convolution to one channel. No batch norm.
- `create_argparser`: every default of the run, including the ADM model and diffusion flags from `script_util`.
- `main`: builds the ADM model and diffusion (`create_model_and_diffusion`), and with `--use_bb_mask_cond True`
  calls `_inflate_unet_first_conv_to_4ch` (3 to 4 input channels, mask weights = `mask_init_scale` times the RGB mean)
  and wraps the U-Net in `_BB4ChWrapper`, which concatenates `bb_mask * mask_gain` to the image and forwards fp16
  conversion. Then builds the discriminator, the loader and `TrainLoop`.

The `_lsun` and `_globalnoise` copies differ in one line each: the import of the matching copy of the loop.

## scripts/image_sample_fdgan_4ch.py

Ours. It replaces ADM's `image_sample.py`.

- `load_bbs`, `bb_to_xyxy`, `xyxy_to_mask`, `pad_xyxy`: boxes from the dataset's JSON files, in sorted file order,
  cycled; mask and padded-mask tensors.
- `inflate_first_conv_to_4ch`, `BB4ChWrapper`, `load_weights`: the 4-channel model, same as in training.
- `mask_gain_at`: gain of the mask channel per step (`--mask_mode always` in our sets).
- `sample_batch`: the reverse chain. Ancestral sampling with the fixed posterior variance (`--sigma_mode small`) or
  ADM's learned variance, or DDIM when the respacing starts with `ddim`. At in-band steps with
  `--roi_bump_mode predict` the network input is `x + sqrt(1 - alpha_bar) * k_focus * M * noise`, the prediction
  from it drives the update, and the chain itself is never perturbed. `--roi_bump_region global` spreads the same
  perturbation energy over the whole image (the global-noise control); `--mask_cfg_scale S` applies classifier-free
  guidance on the mask channel in clean-image space: x0 = x0_box + (S - 1)(x0_box - x0_nobox), with the no-box pass on
  the unperturbed state; `--cfg_rescale` is the rescaling of Lin et al. against over-saturation. `--color_guide_mode`
  pulls the per-channel means of the clean estimate towards real-image colour statistics inside a window of steps.
- `guidance_step`: optional gradient step from the saved discriminators (`--guide_mode`); off by default and not used
  for the paper's sets.
- `main`: writes `sample_000000.png` onwards and `meta.json` with every box and the full argument list.

`image_sample_fdgan_4ch_globalnoise.py` is byte-identical; the Cityscapes global-noise sampling job calls it by name.

## tools/

- `make_sbatch_files.py`: writes the job files. `DATASETS`, `ROWS`, `MODEL_FLAGS`, `DIFF_FLAGS` and `TRAIN_FLAGS` hold
  everything the runs share; a row is its line in `ROWS`.
- `objeval_core.py`: square crops around the conditioned box of generated and matching real images, person detection
  inside the box with a COCO-pretrained YOLO, IoU and the summary used by the object-level evaluation.
- `real_color_stats.py`: per-channel mean colour of a training set and the per-image means, written to the JSON files
  in `color_stats/` that the sampler's colour guidance reads.

## jobs/

See `jobs/README.md`.

## evaluation/

See `evaluation/README.md`.

## figures/

`build_figures.py` builds the sample grids of the paper from the extracted sample sets (Figure 3 and the supplementary
grids: samples paired by index so that each column shows the same box across models). `build_guidance_figure.py`
plots the guidance curves. `figure2/` holds the architecture diagram as an editable draw.io file, the PDF used in the
paper and the scripts that export it.

## downstream_yolo/

`build_fdgan2026_sets.py` builds the training sets of the downstream study: the real training images plus a fraction
of FDGAN images, each FDGAN image labelled with its conditioning box after a detector trained on real images confirms an
object in that box. `fdgan2026_val.py` evaluates the trained detectors. The runs are in progress.
