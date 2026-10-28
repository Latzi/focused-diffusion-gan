# Job files

These are the Slurm scripts as we submitted them on OzSTAR (one A100 each). We keep them unchanged, cluster paths
included, because they are the record of the flags behind every reported run. To run them elsewhere, change the
`REPO`, `DATA_DIR`, `BB_DIR` and `source` lines at the top.

## Names

`train_<dataset>_<row>.sh` trains a model, `sample_<dataset>_<row>.sh` samples from it.

Datasets: `cityscapes` (Cityscapes-Pedestrian), `traffic2` (Traffic Signs), `coco` (COCO potted plant).

The `traffic_*` files read the Traffic Signs boxes as corner coordinates. Those boxes are stored as x, y, width,
height, so the box-conditioned Traffic rows of the paper come from the `traffic2_*` files, which add
`--bb_is_xywh True`. The `traffic_*` files are kept because they document that earlier state.

Rows (the flags that differ are listed in `tools/make_sbatch_files.py`):

| Suffix | Paper row |
|---|---|
| `adm` | ADM |
| `adm_mask` | FDGAN\* (adversarial stage disabled, mask channel kept) |
| `fdgan_mask` | FDGAN |
| `fdgan_nomask` | FDGAN without the box mask |
| `fdgan_nonoise` | FDGAN without the localised noise |
| `fdgan_globalnoise` | FDGAN with the same noise energy spread over the whole image |
| `fdgan_mask_smallD` | FDGAN plus the small ROI discriminator |
| `fdgan_mask_rep2`, `adm_mask_rep2` | second training seed of FDGAN and FDGAN\* on Cityscapes: the same files with a new tag, a second independent run |
| `fdgan_wideband35` | the wide-band run on Cityscapes: `--tmin_end 600 --band_width 350 --max_gan_per_microbatch 8`, sampled with `--bump_tmin_rev 600 --bump_tmax_rev 950` |
| `fdgan_advbox` | the box-gradient run on Cityscapes: `--adv_grad_in_box_only True` with the `_advbox` copies of the training loop and the training script, written by `tools/make_advbox_files.py`; the generator's adversarial gradient reaches only the box pixels while the discriminator still sees the whole image |

`*_lsun.sh` are the COCO runs; they start from an LSUN-pretrained ADM checkpoint instead of random weights. The
Cityscapes global-noise pair calls the `_globalnoise` copies of the training script and the sampler.

## Training

```bash
sbatch jobs/train_cityscapes_fdgan_mask.sh
```

A job resumes from the newest `model*.pt` in its checkpoint folder, so re-submitting the same file continues the run.
Checkpoints, EMA weights and discriminator files are written every 10,000 iterations; the run stops at 200,000.

## Sampling

```bash
sbatch --export=ALL,STEP=200000,NUM=3000,RESP=500 jobs/sample_cityscapes_fdgan_mask.sh
```

Variables: `STEP` (checkpoint iteration), `KIND` (`ema` or `model`), `NUM` (images), `RESP` (respacing: `250`,
`500`, `ddim250`), `COLOR` (`none`, `mean`, `sample`: colour guidance from the dataset's colour statistics), `EXTRA`
(further sampler flags, for example `--mask_cfg_scale 2.0`), `LABEL` (suffix for the output folder). The output folder
name records these settings, and so do the set names in the evaluation sheets.

## Memory probe

`memory_probe/` holds the job that measured peak GPU memory of training and sampling for the paper's cost table.
