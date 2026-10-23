#!/bin/bash
#SBATCH --job-name=s11_guided
#SBATCH --output=/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/logs_Step11/guided_%j.txt
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
# =====================================================================================================================
#  Discriminator-GUIDED sampling (the September 2025 plug-and-play step, x0 space) from the discriminators saved with
#  the checkpoint: GMODE=main (whole-image D, every FDGAN run), GMODE=small (the ROI discriminator, only runs trained
#  with --enable_small_roi_d, i.e. the *_smallD tags) or GMODE=both. Every knob is an environment variable passed with
#  --export. Examples:
#     sbatch --export=ALL,STEP=160000,NUM=16,RESP=250,ETA=0.1 sample_guided.sh
#     sbatch --export=ALL,TAG=cityscapes_fdgan_mask_smallD,STEP=160000,NUM=16,RESP=250,GMODE=small,ETA=0.1 sample_guided.sh
#  STRENGTH: the mean push per guided step inside the box is ETA x TARGET in image units (range 2). The September values
#  ETA=0.006 TARGET=0.45 give 0.003 per step, which changes nothing visible (verified 10 Oct 2026: ETA 0.006 and 0.012
#  differ by at most 2 grey levels). Use ETA 0.05 .. 0.3. The log prints the realised push and whether the cap bit.
#
#   TAG    run to sample (default cityscapes_fdgan_mask; traffic_* works too)
#   STEP   checkpoint step (default 200000)          NUM    number of images (default 16)
#   RESP   chain length: 250 | 400 | 1000 | ddim250   (default 250)
#   KF     k_focus, extra box noise strength (default 1.75; 0 = none)
#   BUMP   predict | once | band | none   (default predict = perturb the network input only, training-consistent)
#   BTMIN  BTMAX  box-noise window in reverse-t units of the 1,000-step horizon (default 700 820)
#   GTMIN  GTMAX  guidance window, same units (default 225 300 = steps 280..310 of the September 400-step chain)
#   GMODE  main | small | both (default main)        WMAIN  weight of the main-D loss when GMODE=both (default 1.0; September used 0.25)
#   ETA    guidance step size (default 0.006 = September, too weak; see STRENGTH)   TARGET normalised step target (default 0.45)
#   MAXS   cap on the step scale (default 1e9 = off; September's 60000 could silently weaken the push)   PAD  gradient region padding (default 0.5)
#   GAIN   mask-channel gain (default 50)             MMODE  always | band | ramp (default always)
#   TEMP   noise temperature for ancestral steps (default 1.0)
#   SIG    ancestral variance: small (default, fixed posterior = September) | learned (ADM's own sampler)
#   CFG    classifier-free guidance scale on the mask channel (default 1.0 = off; try 1.5 .. 3)
#   CFGR   where the CFG extrapolation applies: all (default) | roi (padded box only)
#   KIND   ema (default) | model = the raw weights of the same step
#   COLOR  none (default) | mean | sample : global colour guidance from datasets/color_stats_<dataset>.json (see README)
#   LABEL  optional extra label; a label is always built from the values above so folders never collide
# =====================================================================================================================
set -euo pipefail
module load gcc/12.2.0
module load python-scientific/3.10.8-foss-2022b
source /fred/oz097/latzi/Diffusion_ProjectedGAN/Diffusion_ProjectedGAN/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
REPO="/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion"
cd "$REPO"
mkdir -p "$REPO/logs_Step11" "$REPO/samples_Step11"

TAG="${TAG:-cityscapes_fdgan_mask}"
STEP="${STEP:-200000}";  NUM="${NUM:-16}";  RESP="${RESP:-250}"
KF="${KF:-1.75}";  BUMP="${BUMP:-predict}";  BTMIN="${BTMIN:-700}";  BTMAX="${BTMAX:-820}"
GTMIN="${GTMIN:-225}";  GTMAX="${GTMAX:-300}";  ETA="${ETA:-0.006}";  TARGET="${TARGET:-0.45}";  MAXS="${MAXS:-1e9}";  PAD="${PAD:-0.5}"
GMODE="${GMODE:-main}";  WMAIN="${WMAIN:-1.0}"
GAIN="${GAIN:-50}";  MMODE="${MMODE:-always}";  TEMP="${TEMP:-1.0}";  LABEL="${LABEL:-}"
SIG="${SIG:-small}";  CFG="${CFG:-1.0}";  CFGR="${CFGR:-all}";  KIND="${KIND:-ema}";  COLOR="${COLOR:-none}"

case "$TAG" in
  cityscapes*) DATA_DIR="$REPO/datasets/Cityscapes_3009_img"; BB_CANDS="$REPO/datasets/Cityscapes_3009_bb" ;;
  traffic*)    DATA_DIR="$REPO/datasets/TrafficSigns_img";    BB_CANDS="$REPO/datasets/TrafficSigns_bb" ;;
  coco*)       DATA_DIR="$REPO/datasets/COCO_PottedPlant_256sq_img"; BB_CANDS="$REPO/datasets/COCO_PottedPlant_256sq_bb" ;;
  *) echo "TAG must start with cityscapes, traffic or coco"; exit 1 ;;
esac
BB_DIR=""; for c in $BB_CANDS; do if [ -d "$c" ]; then BB_DIR="$c"; break; fi; done
[ -n "$BB_DIR" ] || { echo "no box folder found among: $BB_CANDS"; exit 1; }
case "$TAG" in cityscapes*) CSTATS="$REPO/datasets/color_stats_cityscapes.json" ;; coco*) CSTATS="$REPO/datasets/color_stats_coco.json" ;; *) CSTATS="$REPO/datasets/color_stats_traffic.json" ;; esac
COLORFLAGS=""; COLTAG=""
if [ "$COLOR" != "none" ]; then [ -f "$CSTATS" ] || { echo "colour stats not found: $CSTATS"; exit 1; }; COLORFLAGS="--color_guide_mode $COLOR --color_stats_json $CSTATS"; COLTAG="_col$COLOR"; fi

case "$KIND" in
  ema)   CKPT="$REPO/checkpoints_Step11/$TAG/ema_0.9999_$(printf '%06d' "$STEP").pt" ;;
  model) CKPT="$REPO/checkpoints_Step11/$TAG/model$(printf '%06d' "$STEP").pt" ;;
  *) echo "KIND must be ema or model"; exit 1 ;;
esac
DMAIN="$REPO/checkpoints_Step11/$TAG/D_main$(printf '%06d' "$STEP").pt"
DSMALL="$REPO/checkpoints_Step11/$TAG/D_small$(printf '%06d' "$STEP").pt"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
case "$GMODE" in
  main)  [ -f "$DMAIN" ]  || { echo "main discriminator checkpoint not found: $DMAIN"; exit 1; } ;;
  small) [ -f "$DSMALL" ] || { echo "small ROI discriminator checkpoint not found: $DSMALL (only *_smallD runs save it)"; exit 1; } ;;
  both)  [ -f "$DMAIN" ] && [ -f "$DSMALL" ] || { echo "need both $DMAIN and $DSMALL"; exit 1; } ;;
  *) echo "GMODE must be main, small or both"; exit 1 ;;
esac

# the mask channel must match how the run was trained
case "$TAG" in *_adm|*_nomask) MASKFLAG="--use_bb_mask_cond False" ;; *) MASKFLAG="--use_bb_mask_cond True" ;; esac

# short folder name (long names broke unzipping on Windows): step, chain, bump, guidance window + a 6-char code for the rest
SETTINGS="RESP=$RESP KF=$KF BUMP=$BUMP BTMIN=$BTMIN BTMAX=$BTMAX GMODE=$GMODE WMAIN=$WMAIN GTMIN=$GTMIN GTMAX=$GTMAX ETA=$ETA TARGET=$TARGET MAXS=$MAXS PAD=$PAD GAIN=$GAIN MMODE=$MMODE TEMP=$TEMP SIG=$SIG CFG=$CFG CFGR=$CFGR KIND=$KIND COLOR=$COLOR"
CODE=$(printf '%s' "$SETTINGS" | md5sum | cut -c1-6)
OUT="$REPO/samples_Step11/${TAG}_${KIND}${STEP}_G${GMODE}_${RESP}_k${KF}${BUMP}${BTMIN}-${BTMAX}_w${GTMIN}-${GTMAX}_e${ETA}_c${CFG}${COLTAG}_${CODE}${LABEL:+_$LABEL}"
mkdir -p "$OUT"
printf 'TAG=%s STEP=%s NUM=%s %s\n' "$TAG" "$STEP" "$NUM" "$SETTINGS" > "$OUT/settings.txt"
echo "===== GUIDED sampling $TAG step $STEP -> $OUT @ $(date) ====="
echo "settings: $SETTINGS"
srun python -u scripts/image_sample_fdgan_4ch.py \
  --model_path "$CKPT" --d_main_path "$DMAIN" --d_small_path "$DSMALL" --bb_dir "$BB_DIR" --out_dir "$OUT" \
  --num_samples "$NUM" --batch_size 16 --seed 17 --timestep_respacing "$RESP" --noise_temp "$TEMP" --sigma_mode "$SIG" \
  --mask_cfg_scale "$CFG" --mask_cfg_region "$CFGR" $COLORFLAGS \
  --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16 --use_scale_shift_norm False --resblock_updown False --learn_sigma True --class_cond False --dropout 0.0 \
  --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False \
  $MASKFLAG --mask_gain "$GAIN" --mask_mode "$MMODE" \
  --roi_bump_mode "$BUMP" --k_focus "$KF" --bump_tmin_rev "$BTMIN" --bump_tmax_rev "$BTMAX" \
  --guide_mode "$GMODE" --guide_main_w "$WMAIN" --guide_small_w 1.0 --roi_size 128 --roi_pad_ratio 0.03 \
  --guide_tmin_rev "$GTMIN" --guide_tmax_rev "$GTMAX" --pnp_eta "$ETA" --pnp_norm mean_roi --pnp_target "$TARGET" --max_step_scale "$MAXS" --grad_pad_ratio "$PAD" \
  --ndf 128 --n_layers_D 3 --use_mask_adapter True

N_PNG=$(ls "$OUT"/*.png 2>/dev/null | wc -l)
[ "$N_PNG" -ge "$NUM" ] || { echo "only $N_PNG png written, expected $NUM - keeping the folder for inspection"; exit 1; }
cd "$REPO/samples_Step11" && rm -f "$(basename "$OUT").zip" && zip -q -r "$(basename "$OUT").zip" "$(basename "$OUT")" -x '*.npz' 'log.txt' 'progress.csv'
unzip -tq "$(basename "$OUT").zip" >/dev/null && rm -rf "$OUT" && echo "zip ready, loose files removed: $REPO/samples_Step11/$(basename "$OUT").zip  ($N_PNG png)"
