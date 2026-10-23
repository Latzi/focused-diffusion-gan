# make_sbatch_files.py - writes the Slurm job files (training and sampling, every dataset and row) into ./sbatch/.
# Re-run it after changing a path or a setting; everything the runs share lives in the blocks below.
import io, os, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sbatch")
os.makedirs(OUT, exist_ok=True)

REPO = "/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion"
VENV = "/fred/oz097/latzi/Diffusion_ProjectedGAN/Diffusion_ProjectedGAN/bin/activate"
TOTAL_STEPS = 200000                 # same --lr_anneal_steps as the COCO run (LR decays linearly to 0 and the run stops)

DATASETS = {
    "cityscapes": dict(data=f"{REPO}/datasets/Cityscapes_3009_img",
                       bb_candidates=[f"{REPO}/datasets/Cityscapes_3009_bb"]),
    "traffic":    dict(data=f"{REPO}/datasets/TrafficSigns_img",
                       bb_candidates=[f"{REPO}/datasets/TrafficSigns_bb"]),   # 14 Oct 2026: TrafficSigns_bb1 holds a full-image box [0,0,256,256] for every file (dummy set); the real boxes are in TrafficSigns_bb
    "coco":       dict(data=f"{REPO}/datasets/COCO_PottedPlant_256sq_img",            # 2,363 images, checked 12 Oct 2026
                       bb_candidates=[f"{REPO}/datasets/COCO_PottedPlant_256sq_bb"]),
    # 17 Oct 2026: the TrafficSigns_bb JSON files store x, y, WIDTH, HEIGHT (verified by drawing them on the images), whereas the
    # Cityscapes and COCO files store corner coordinates. The "traffic" rows above read the Traffic boxes as corners, so their masks
    # were wrong; every box-conditioned Traffic model must come from "traffic2", which appends --bb_is_xywh True to training and sampling.
    "traffic2":   dict(data=f"{REPO}/datasets/TrafficSigns_img",
                       bb_candidates=[f"{REPO}/datasets/TrafficSigns_bb"],
                       extra="--bb_is_xywh True"),
}
# the three rows of the base-dataset matrix; everything not listed here is identical across rows
ROWS = {
    "adm":          "--use_bb_mask_cond False --gan_warmup_steps 999999999",   # GAN, ANPM and all auxiliary terms never switch on
    "adm_mask":     "--use_bb_mask_cond True  --gan_warmup_steps 999999999",   # FDGAN* : mask channel only, the adversarial stage never opens
    "fdgan_mask":   "--use_bb_mask_cond True  --gan_warmup_steps 40000",       # the paper's method, mask channel ON  (COCO recipe)
    "fdgan_nomask": "--use_bb_mask_cond False --gan_warmup_steps 40000",       # same, mask channel OFF ("the setting that was off")
    "fdgan_nonoise": "--use_bb_mask_cond True  --gan_warmup_steps 40000 --k_focus 0",   # adversarial + reconstruction losses but NO extra box noise
    "fdgan_globalnoise": "--use_bb_mask_cond True  --gan_warmup_steps 40000 --noise_option extra_gan_only_global",   # same extra-noise energy spread over the whole image (matched total strength)
    "fdgan_mask_smallD": "--use_bb_mask_cond True  --gan_warmup_steps 40000 --enable_small_roi_d True",  # full FDGAN + the small ROI discriminator (Sept 2025 component); enables main-D and ROI-D guidance at sampling later
}

# the COCO ADM backbone (train_ADM_Original.sh) = the matched baseline architecture, at the base-dataset width 128
MODEL_FLAGS = ("--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16 "
               "--use_scale_shift_norm False --resblock_updown False --learn_sigma True --class_cond False --dropout 0.0")
DIFF_FLAGS = "--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
# the COCO four-channel run's training flags (stdout_tee.txt, 7 Mar 2026), width/backbone excepted
TRAIN_FLAGS = (f"--lr 1e-4 --batch_size 16 --microbatch 8 --use_fp16 True --fp16_scale_growth 1e-3 --ema_rate 0.9999 "
               f"--log_interval 10 --save_interval 10000 --lr_anneal_steps {TOTAL_STEPS} "
               "--bb_mask_key bb_mask --mask_gain 50.0 --mask_init_mode copy_mean --mask_init_scale 0.10 "
               "--tmin_start 820 --tmin_end 700 --band_width 120 --tmax_cap 950 --band_ramp_steps 50000 --diffusion_in_band_weight 0.7 "
               "--gan_initial_scale 1e-4 --gan_final_scale 1e-2 --l1_initial_scale 5e-2 --l1_final_scale 5e-2 --lambda_enhanced 5e-2 "
               "--color_stats_scale 0.05 --sat_penalty_scale 0.01 --sat_threshold 0.95 "
               "--noise_option extra_gan_only --k_focus 1.75 --max_gan_per_microbatch 4 "
               "--use_mask_adapter True --ndf 128 --n_layers_D 3 --use_sigmoid False --lr_D 1e-5 --beta1_D 0.5 --beta2_D 0.999 "
               "--save_discriminators True --resume_discriminators True --g_adv_warmup_steps_if_no_d_ckpt 2000 "
               "--bb_is_xywh False --bb_xyxy_end_inclusive False --image_save_every 1000")

HEADER = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={repo}/logs_Step11/{tag}_%j.txt
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time={time}
set -euo pipefail
module load gcc/12.2.0
module load python-scientific/3.10.8-foss-2022b
source {venv}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
REPO="{repo}"
cd "$REPO"
mkdir -p "$REPO/logs_Step11"
DATA_DIR="{data}"
BB_DIR=""
for c in {bb_candidates}; do if [ -d "$c" ]; then BB_DIR="$c"; break; fi; done
[ -n "$BB_DIR" ] || {{ echo "no box folder found among: {bb_candidates}"; exit 1; }}
[ -d "$DATA_DIR" ] || {{ echo "no image folder: $DATA_DIR"; exit 1; }}
grep -q "bb_is_xywh" guided_diffusion/image_datasets_Run2_3_10_COCO.py || {{ echo "guided_diffusion/image_datasets_Run2_3_10_COCO.py is not the 2026 loader - upload the package file first"; exit 1; }}
grep -q "gan_warmup_remaining" guided_diffusion/train_util_coco_fdgan_late_band.py || {{ echo "guided_diffusion/train_util_coco_fdgan_late_band.py is not the 2026 loop - upload the package file first"; exit 1; }}
echo "DATA_DIR=$DATA_DIR  BB_DIR=$BB_DIR  ($(ls "$DATA_DIR" | wc -l) images, $(ls "$BB_DIR" | wc -l) box files)"
"""

TRAIN_BODY = """TAG="{tag}"
CKPT_DIR="$REPO/checkpoints_Step11/$TAG"
LOG_DIR="$REPO/logs_Step11/$TAG"
IMG_DIR="$REPO/images_debug_Step11/$TAG"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$IMG_DIR"
# resume from the newest model*.pt in CKPT_DIR, if any (so re-submitting this same script continues the run)
RESUME=$(ls -1 "$CKPT_DIR"/model*.pt 2>/dev/null | sed -E 's/.*model([0-9]+)\\.pt/\\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
RESUME_ARG=""
if [ -n "${{RESUME:-}}" ]; then RESUME_ARG="--resume_checkpoint $RESUME"; echo "resuming from $RESUME"; else echo "starting fresh"; fi
echo "dataset extra flags: {extra}"
echo "===== $TAG @ $(date) ====="
srun python -u scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py \\
  --data_dir "$DATA_DIR" --bb_dir "$BB_DIR" \\
  --log_dir "$LOG_DIR" --checkpoint_dir "$CKPT_DIR" --image_save_dir "$IMG_DIR" \\
  {model_flags} \\
  {diff_flags} \\
  {train_flags} \\
  {row_flags} \\
  $RESUME_ARG
"""

SAMPLE_BODY = """TAG="{tag}"
STEP="${{STEP:-{total}}}"                      # which checkpoint:  sbatch --export=ALL,STEP=150000 sample_{tag}.sh
NUM="${{NUM:-3000}}"                         # how many images:   sbatch --export=ALL,STEP=10000,NUM=16 ... for a smoke test
RESP="${{RESP:-250}}"                        # respacing: 250 = ancestral 250 steps (the COCO protocol); ddim250 also works
KIND="${{KIND:-ema}}"                        # ema (default) = averaged weights; model = the raw weights of the same step
case "$KIND" in
  ema)   CKPT="$REPO/checkpoints_Step11/$TAG/ema_0.9999_$(printf '%06d' "$STEP").pt" ;;
  model) CKPT="$REPO/checkpoints_Step11/$TAG/model$(printf '%06d' "$STEP").pt" ;;
  *) echo "KIND must be ema or model"; exit 1 ;;
esac
[ -f "$CKPT" ] || {{ echo "checkpoint not found: $CKPT"; exit 1; }}
EXTRA="${{EXTRA:-}}"                        # extra sampler flags, appended last so they override the defaults below
COLOR="${{COLOR:-none}}"                    # none | mean | sample : global colour guidance from datasets/color_stats_<dataset>.json
case "$TAG" in cityscapes*) CSTATS="$REPO/datasets/color_stats_cityscapes.json" ;; coco*) CSTATS="$REPO/datasets/color_stats_coco.json" ;; *) CSTATS="$REPO/datasets/color_stats_traffic.json" ;; esac
COLORFLAGS=""; COLTAG=""
if [ "$COLOR" != "none" ]; then [ -f "$CSTATS" ] || {{ echo "colour stats not found: $CSTATS"; exit 1; }}; COLORFLAGS="--color_guide_mode $COLOR --color_stats_json $CSTATS"; COLTAG="_col$COLOR"; fi
LABEL="${{LABEL:-}}"                        # short label for the output folder when EXTRA is used
OUT="$REPO/samples_Step11/${{TAG}}_${{KIND}}${{STEP}}_n${{NUM}}_r${{RESP}}${{COLTAG}}${{LABEL:+_$LABEL}}"
mkdir -p "$OUT"
echo "===== sampling $TAG step $STEP -> $OUT @ $(date) ====="
srun python -u scripts/image_sample_fdgan_4ch.py \\
  --model_path "$CKPT" --bb_dir "$BB_DIR" --out_dir "$OUT" \\
  --num_samples "$NUM" --batch_size 16 --seed 17 --timestep_respacing "$RESP" \\
  {model_flags} \\
  {diff_flags} \\
  {sample_flags} \
  $COLORFLAGS $EXTRA
N_PNG=$(ls "$OUT"/*.png 2>/dev/null | wc -l)
[ "$N_PNG" -ge "$NUM" ] || {{ echo "only $N_PNG png written, expected $NUM - keeping the folder for inspection"; exit 1; }}
cd "$REPO/samples_Step11" && rm -f "$(basename "$OUT").zip" && zip -q -r "$(basename "$OUT").zip" "$(basename "$OUT")" -x '*.npz' 'log.txt' 'progress.csv'
# the group file quota is nearly full: keep ONE zip per sample set (png + meta.json) and drop the loose files
unzip -tq "$(basename "$OUT").zip" >/dev/null && rm -rf "$OUT" && echo "zip ready, loose files removed: $REPO/samples_Step11/$(basename "$OUT").zip  ($N_PNG png)"
"""
# sampling flags per row: the mask channel must match training; ANPM bump as in the paper (once, in the terminal band)
SAMPLE_FLAGS = {
    "adm":          "--use_bb_mask_cond False --roi_bump_mode none --k_focus 0",
    "adm_mask":     "--use_bb_mask_cond True  --mask_gain 50.0 --mask_mode always --roi_bump_mode none --k_focus 0",
    "fdgan_mask":   "--use_bb_mask_cond True  --mask_gain 50.0 --mask_mode always --roi_bump_mode predict --k_focus 1.75 --bump_tmin_rev 700 --bump_tmax_rev 820",
    "fdgan_nomask": "--use_bb_mask_cond False --roi_bump_mode predict --k_focus 1.75 --bump_tmin_rev 700 --bump_tmax_rev 820",
    "fdgan_nonoise": "--use_bb_mask_cond True  --mask_gain 50.0 --mask_mode always --roi_bump_mode none --k_focus 0",
    "fdgan_globalnoise": "--use_bb_mask_cond True  --mask_gain 50.0 --mask_mode always --roi_bump_mode predict --roi_bump_region global --k_focus 1.75 --bump_tmin_rev 700 --bump_tmax_rev 820",
    "fdgan_mask_smallD": "--use_bb_mask_cond True  --mask_gain 50.0 --mask_mode always --roi_bump_mode predict --k_focus 1.75 --bump_tmin_rev 700 --bump_tmax_rev 820",
}

written = []
for ds, row in itertools.product(DATASETS, ROWS):
    tag = f"{ds}_{row}"
    d = DATASETS[ds]
    extra = d.get("extra", "")          # dataset-specific flags, appended LAST so they override the shared defaults (argparse keeps the last value)
    common = dict(repo=REPO, venv=VENV, tag=tag, data=d["data"], bb_candidates=" ".join(d["bb_candidates"]))
    train = HEADER.format(job=f"s11_{tag}", time="3-00:00:00", **common) + TRAIN_BODY.format(
        tag=tag, extra=extra or "(none)", model_flags=MODEL_FLAGS, diff_flags=DIFF_FLAGS, train_flags=TRAIN_FLAGS, row_flags=(ROWS[row] + " " + extra).strip())
    sample = HEADER.format(job=f"s11smp_{tag}", time="06:00:00", **common) + SAMPLE_BODY.format(
        tag=tag, total=TOTAL_STEPS, model_flags=MODEL_FLAGS, diff_flags=DIFF_FLAGS, sample_flags=(SAMPLE_FLAGS[row] + " " + extra).strip())
    for name, text in ((f"train_{tag}.sh", train), (f"sample_{tag}.sh", sample)):
        io.open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n").write(text)
        written.append(name)
print("\n".join(written))
