#!/bin/bash
#SBATCH --job-name=s11smp_coco_adm
#SBATCH --output=/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/logs_Step11/coco_adm_%j.txt
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
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
mkdir -p "$REPO/logs_Step11"
DATA_DIR="/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/datasets/COCO_PottedPlant_256sq_img"
BB_DIR=""
for c in /fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/datasets/COCO_PottedPlant_256sq_bb; do if [ -d "$c" ]; then BB_DIR="$c"; break; fi; done
[ -n "$BB_DIR" ] || { echo "no box folder found among: /fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/datasets/COCO_PottedPlant_256sq_bb"; exit 1; }
[ -d "$DATA_DIR" ] || { echo "no image folder: $DATA_DIR"; exit 1; }
grep -q "bb_is_xywh" guided_diffusion/image_datasets_Run2_3_10_COCO.py || { echo "guided_diffusion/image_datasets_Run2_3_10_COCO.py is not the 2026 loader - upload the package file first"; exit 1; }
grep -q "gan_warmup_remaining" guided_diffusion/train_util_coco_fdgan_late_band.py || { echo "guided_diffusion/train_util_coco_fdgan_late_band.py is not the 2026 loop - upload the package file first"; exit 1; }
echo "DATA_DIR=$DATA_DIR  BB_DIR=$BB_DIR  ($(ls "$DATA_DIR" | wc -l) images, $(ls "$BB_DIR" | wc -l) box files)"
TAG="coco_adm"
STEP="${STEP:-200000}"                      # which checkpoint:  sbatch --export=ALL,STEP=150000 sample_coco_adm.sh
NUM="${NUM:-3000}"                         # how many images:   sbatch --export=ALL,STEP=10000,NUM=16 ... for a smoke test
RESP="${RESP:-250}"                        # respacing: 250 = ancestral 250 steps (the COCO protocol); ddim250 also works
KIND="${KIND:-ema}"                        # ema (default) = averaged weights; model = the raw weights of the same step
case "$KIND" in
  ema)   CKPT="$REPO/checkpoints_Step11/$TAG/ema_0.9999_$(printf '%06d' "$STEP").pt" ;;
  model) CKPT="$REPO/checkpoints_Step11/$TAG/model$(printf '%06d' "$STEP").pt" ;;
  *) echo "KIND must be ema or model"; exit 1 ;;
esac
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
EXTRA="${EXTRA:-}"                        # extra sampler flags, appended last so they override the defaults below
COLOR="${COLOR:-none}"                    # none | mean | sample : global colour guidance from datasets/color_stats_<dataset>.json
case "$TAG" in cityscapes*) CSTATS="$REPO/datasets/color_stats_cityscapes.json" ;; coco*) CSTATS="$REPO/datasets/color_stats_coco.json" ;; *) CSTATS="$REPO/datasets/color_stats_traffic.json" ;; esac
COLORFLAGS=""; COLTAG=""
if [ "$COLOR" != "none" ]; then [ -f "$CSTATS" ] || { echo "colour stats not found: $CSTATS"; exit 1; }; COLORFLAGS="--color_guide_mode $COLOR --color_stats_json $CSTATS"; COLTAG="_col$COLOR"; fi
LABEL="${LABEL:-}"                        # short label for the output folder when EXTRA is used
OUT="$REPO/samples_Step11/${TAG}_${KIND}${STEP}_n${NUM}_r${RESP}${COLTAG}${LABEL:+_$LABEL}"
mkdir -p "$OUT"
echo "===== sampling $TAG step $STEP -> $OUT @ $(date) ====="
srun python -u scripts/image_sample_fdgan_4ch.py \
  --model_path "$CKPT" --bb_dir "$BB_DIR" --out_dir "$OUT" \
  --num_samples "$NUM" --batch_size 16 --seed 17 --timestep_respacing "$RESP" \
  --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16 --use_scale_shift_norm False --resblock_updown False --learn_sigma True --class_cond False --dropout 0.0 \
  --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False \
  --use_bb_mask_cond False --roi_bump_mode none --k_focus 0   $COLORFLAGS $EXTRA
N_PNG=$(ls "$OUT"/*.png 2>/dev/null | wc -l)
[ "$N_PNG" -ge "$NUM" ] || { echo "only $N_PNG png written, expected $NUM - keeping the folder for inspection"; exit 1; }
cd "$REPO/samples_Step11" && rm -f "$(basename "$OUT").zip" && zip -q -r "$(basename "$OUT").zip" "$(basename "$OUT")" -x '*.npz' 'log.txt' 'progress.csv'
# the group file quota is nearly full: keep ONE zip per sample set (png + meta.json) and drop the loose files
unzip -tq "$(basename "$OUT").zip" >/dev/null && rm -rf "$OUT" && echo "zip ready, loose files removed: $REPO/samples_Step11/$(basename "$OUT").zip  ($N_PNG png)"
