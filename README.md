# Focused Diffusion-GAN (FDGAN)

Code and evaluation record for

> Vasile Marian, Yong-Bin Kang, Alexander Buddery.
> Focused Diffusion-GAN: Object-Centric Image Generation Using Integrated GAN and Diffusion Frameworks.
> Image and Vision Computing, under revision (manuscript IMAVIS-D-26-03725), 2026.

FDGAN generates images of a given object class from a bounding box. It is built on OpenAI's
[guided-diffusion](https://github.com/openai/guided-diffusion) (ADM). We kept the ADM U-Net, the linear noise
schedule, the EMA bookkeeping and the mixed-precision trainer, and wrote the rest ourselves: the training loop, the
sampler, the dataset loader and the discriminators are our own files. We do not edit the U-Net file; our training
script widens its first convolution at run time so that it takes the box mask. The ADM modules are in the package as
we ran them; four of them carry small leftovers from our earlier experiments that the FDGAN runs never call (listed in
`docs/CODE_GUIDE.md`). The sections below say exactly what is ours and what is ADM's.

## What FDGAN adds to ADM

1. **A box mask as a fourth input channel.** The loader reads one bounding box per image and the training loop
   rasterises it into a binary mask. The first convolution of the U-Net is widened from 3 to 4 input channels; the new
   weights start at 0.1 times the mean of the RGB weights and the mask is multiplied by 50 before it is concatenated
   with the image. Nothing else in the U-Net changes. Flag: `--use_bb_mask_cond`.

2. **An adversarial stage on a band of timesteps.** For the first 40,000 iterations the model trains exactly like ADM.
   After that, a sample whose timestep falls inside a band of 120 reverse-time steps also goes through the adversarial
   and reconstruction terms. The band opens at reverse steps 820 to 940 and slides down to 700 to 820 by iteration
   50,000, where it stays. Outside the band nothing adversarial happens and the discriminator is not updated.

3. **ANPM, the box-localised perturbation.** For an in-band sample we draw the forward noise as
   eps' = eps + k_focus * M * eps_focus, with k_focus = 1.75 and M the box mask, form x_t from eps', and take the
   model's clean-image prediction from that perturbed input. The diffusion loss itself still uses the ordinary x_t.
   With `--noise_option extra_gan_only_global` the same total noise energy is spread over the whole image instead
   of the box (the global-noise control of the paper); with `--k_focus 0` there is no extra noise at all.

4. **A PatchGAN discriminator that also sees the mask.** Spectral-norm convolutions, no batch norm, 128 base channels,
   three blocks, one map of patch logits per image (63 x 63 for a 256 px input). The mask enters through a 1 x 1
   convolution that maps RGB plus mask to three channels in front of the first block. Real pairs are (training image,
   mask), fake pairs are (clean-image prediction, mask). Least-squares loss, R1 on the real images every 16 updates.

5. **Per-sample weighting of the adversarial term by timestep.** lambda_GAN rises exponentially from 1e-4 at the noisy
   end to 1e-2 at the clean end of the trajectory, so the cleaner states of the band get the stronger push. The same
   weight multiplies the discriminator loss on that sample. There is no spatial weighting of the adversarial loss; the
   box reaches the adversarial stage through the mask input of the discriminator and through ANPM.

6. **Reconstruction terms on the clean-image prediction, in-band only.** L1 against the real image (weight 0.05), the
   same L1 restricted to the background at 0.2 of that weight, a mean-squared error restricted to the box for boxes
   smaller than 15 % of the image (weight 0.05), a colour-statistics term on the per-channel mean and standard
   deviation (0.05) and a saturation penalty above 0.95 (0.01). The in-band diffusion loss is scaled by 0.7.

7. **Discriminator checkpoints.** The discriminator and its optimiser are saved and resumed with the generator. When a
   run resumes without them, the generator's adversarial term is ramped in over 2,000 iterations.

8. **A sampler with the same box input.** At in-band steps the network predicts from a box-perturbed copy of the state,
   exactly as in training, and the chain continues from the unperturbed state (`--roi_bump_mode predict`).
   Classifier-free guidance on the mask channel (`--mask_cfg_scale`), a global colour guidance towards real-image
   colour statistics and an optional discriminator guidance are in the same file. The set names in the evaluation
   sheets record which of these each reported set used.

The matched control **FDGAN\*** is the same code with `--gan_warmup_steps 999999999`: the band never opens, only the
diffusion loss trains, and the mask channel stays. **ADM** is FDGAN\* with `--use_bb_mask_cond False`.

## Layout

| Folder | What is in it |
|---|---|
| `guided_diffusion/` | Our training loop `train_util_coco_fdgan_late_band.py` and our loader `image_datasets_Run2_3_10_COCO.py`, next to the ADM modules of guided-diffusion as we ran them (`unet.py`, `gaussian_diffusion.py`, `respace.py`, `script_util.py`, `fp16_util.py`, `nn.py`, `resample.py`, `dist_util.py`, `logger.py`, `losses.py`). The `_lsun` and `_globalnoise` copies of the loop are explained in the code guide. |
| `scripts/` | `image_train_Run2_3_10_FDGAN_COCO_LateBand.py` (training entry point: discriminator, 4-channel U-Net, loop) and `image_sample_fdgan_4ch.py` (sampler), with their `_lsun` and `_globalnoise` copies. |
| `jobs/` | The Slurm job files of every reported run as submitted on OzSTAR, including the second training seed and the wide-band run, and `jobs/README.md` on how to read them. Paths are the cluster's. |
| `tools/` | `make_sbatch_files.py` (writes the job files; all shared flags live there), `objeval_core.py` (object-level metrics), `real_color_stats.py` (colour statistics of the training images), `color_stats/` (the statistics we used). |
| `evaluation/` | Notebooks, Colab cells, the metric sheets with every number in the paper, the bootstrap replicates, the script that builds the tables and intervals from the sheets, and the audit scripts. See `evaluation/README.md`. |
| `figures/` | The scripts that build the sample grids, the final figure files and the editable architecture diagram. |
| `downstream_yolo/` | Scripts of the downstream YOLO11 study (real images plus FDGAN images); the runs are in progress. |
| `docs/CODE_GUIDE.md` | File-by-file guide: what each function does and where it departs from ADM. |
| `setup.py`, `requirements.txt`, `environment_cluster.txt` | Package install, the pinned versions we used, and the full package list of the cluster environment. |

## Installing

```bash
git clone https://github.com/Latzi/focused-diffusion-gan.git
cd focused-diffusion-gan
pip install -r requirements.txt
pip install -e .
```

The runs were made with Python 3.10.8, torch 2.2.1, torchvision 0.17.1, numpy 1.24.2, Pillow 9.4.0, blobfile 2.1.1
and mpi4py 3.1.4 on one A100 per run; `environment_cluster.txt` is the complete package list of that environment.

## Training and sampling as we ran them

Every reported run is one job file in `jobs/`. Training:

```bash
sbatch jobs/train_cityscapes_fdgan_mask.sh
```

The file name is `train_<dataset>_<row>.sh`. The rows differ only in the flags listed in `tools/make_sbatch_files.py`:

| Row suffix | Paper row | Flags that differ |
|---|---|---|
| `adm` | ADM | `--use_bb_mask_cond False --gan_warmup_steps 999999999` |
| `adm_mask` | FDGAN\* (no adversarial stage) | `--use_bb_mask_cond True --gan_warmup_steps 999999999` |
| `fdgan_mask` | FDGAN | `--use_bb_mask_cond True --gan_warmup_steps 40000` |
| `fdgan_nomask` | without box mask | `--use_bb_mask_cond False --gan_warmup_steps 40000` |
| `fdgan_nonoise` | without localised noise | `--k_focus 0` |
| `fdgan_globalnoise` | global noise, same energy | `--noise_option extra_gan_only_global` |
| `fdgan_mask_smallD` | plus small ROI discriminator | `--enable_small_roi_d True` |
| `fdgan_mask_rep2`, `adm_mask_rep2` | second training seed of FDGAN and FDGAN\* | same flags, a second independent run |
| `fdgan_wideband35` | wide-band run | `--tmin_end 600 --band_width 350 --max_gan_per_microbatch 8` |

Everything else is shared: 256 px, 128 base channels, 2 residual blocks, attention at 16, learned sigma, 1,000 linear
diffusion steps, batch 16 in micro-batches of 8, fp16, lr 1e-4 annealed to zero at 200,000 iterations, EMA 0.9999.
Traffic Signs boxes are stored as x, y, width, height, so the Traffic jobs are the `traffic2_*` files, which add
`--bb_is_xywh True`. The COCO jobs start from an LSUN-pretrained ADM checkpoint (`*_lsun.sh`).

Sampling:

```bash
sbatch --export=ALL,STEP=200000,NUM=3000,RESP=500,EXTRA="--mask_cfg_scale 2.0" jobs/sample_cityscapes_fdgan_mask.sh
```

`STEP` picks the checkpoint (EMA weights by default), `NUM` the number of images, `RESP` the respacing, `COLOR` the
colour guidance mode and `EXTRA` any further sampler flag. Boxes are taken from the dataset's box files in sorted order,
so sample i of every set is conditioned on the same box. The sampler writes `sample_000000.png` onwards plus a
`meta.json` with the box of every image.

## Evaluation

Distributional metrics come from the public [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) implementation with
the DINOv2 encoder, on 3,000 images per set against the training set, with the held-out set as the test reference.
Differences between rows are paired bootstrap intervals over 200 replicates that share the same resample of the real
set. Object-level metrics score a square crop around the conditioned box against the real crop of the same box and
measure object presence and box adherence with detectors. Every number in the paper traces back to a row of the
sheets in `evaluation/sheets/`, and the audit scripts compare each sheet value with its raw dgm-eval output file.

## Data

The three datasets are released separately:
[object-centric-low-data-datasets](https://github.com/Latzi/object-centric-low-data-datasets) (Traffic Signs with
images and YOLO labels; reconstruction assets for Cityscapes-Pedestrian and COCO potted plant). The loader here reads
one JSON box file per image; a script that writes those files from the release's box tables is on its way.

## Checkpoints and sample sets

The EMA checkpoints and the generated sets behind every reported row will be archived separately; the link will be
added here.

## Citing

See `CITATION.cff`. Until the article appears, please cite the manuscript as given there.

## Licence

MIT, see `LICENSE`. guided-diffusion is MIT-licensed by OpenAI; our additions are MIT-licensed as well.

## Acknowledgement

Training, sampling and the memory measurements ran on the OzSTAR supercomputing facility at Swinburne University of
Technology.
