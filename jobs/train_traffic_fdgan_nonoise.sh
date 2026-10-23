#!/bin/bash
#SBATCH --job-name=s11_traffic_fdgan_nonoise
#SBATCH --output=/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/logs_Step11/traffic_fdgan_nonoise_%j.txt
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
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
DATA_DIR="/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/datasets/TrafficSigns_img"
BB_DIR=""
for c in /fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/datasets/TrafficSigns_bb; do if [ -d "$c" ]; then BB_DIR="$c"; break; fi; done
[ -n "$BB_DIR" ] || { echo "no box folder found among: /fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/datasets/TrafficSigns_bb"; exit 1; }
[ -d "$DATA_DIR" ] || { echo "no image folder: $DATA_DIR"; exit 1; }
grep -q "bb_is_xywh" guided_diffusion/image_datasets_Run2_3_10_COCO.py || { echo "guided_diffusion/image_datasets_Run2_3_10_COCO.py is not the 2026 loader - upload the package file first"; exit 1; }
grep -q "gan_warmup_remaining" guided_diffusion/train_util_coco_fdgan_late_band.py || { echo "guided_diffusion/train_util_coco_fdgan_late_band.py is not the 2026 loop - upload the package file first"; exit 1; }
echo "DATA_DIR=$DATA_DIR  BB_DIR=$BB_DIR  ($(ls "$DATA_DIR" | wc -l) images, $(ls "$BB_DIR" | wc -l) box files)"
TAG="traffic_fdgan_nonoise"
CKPT_DIR="$REPO/checkpoints_Step11/$TAG"
LOG_DIR="$REPO/logs_Step11/$TAG"
IMG_DIR="$REPO/images_debug_Step11/$TAG"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$IMG_DIR"
# resume from the newest model*.pt in CKPT_DIR, if any (so re-submitting this same script continues the run)
RESUME=$(ls -1 "$CKPT_DIR"/model*.pt 2>/dev/null | sed -E 's/.*model([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then RESUME_ARG="--resume_checkpoint $RESUME"; echo "resuming from $RESUME"; else echo "starting fresh"; fi
echo "dataset extra flags: (none)"
echo "===== $TAG @ $(date) ====="
srun python -u scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py \
  --data_dir "$DATA_DIR" --bb_dir "$BB_DIR" \
  --log_dir "$LOG_DIR" --checkpoint_dir "$CKPT_DIR" --image_save_dir "$IMG_DIR" \
  --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16 --use_scale_shift_norm False --resblock_updown False --learn_sigma True --class_cond False --dropout 0.0 \
  --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False \
  --lr 1e-4 --batch_size 16 --microbatch 8 --use_fp16 True --fp16_scale_growth 1e-3 --ema_rate 0.9999 --log_interval 10 --save_interval 10000 --lr_anneal_steps 200000 --bb_mask_key bb_mask --mask_gain 50.0 --mask_init_mode copy_mean --mask_init_scale 0.10 --tmin_start 820 --tmin_end 700 --band_width 120 --tmax_cap 950 --band_ramp_steps 50000 --diffusion_in_band_weight 0.7 --gan_initial_scale 1e-4 --gan_final_scale 1e-2 --l1_initial_scale 5e-2 --l1_final_scale 5e-2 --lambda_enhanced 5e-2 --color_stats_scale 0.05 --sat_penalty_scale 0.01 --sat_threshold 0.95 --noise_option extra_gan_only --k_focus 1.75 --max_gan_per_microbatch 4 --use_mask_adapter True --ndf 128 --n_layers_D 3 --use_sigmoid False --lr_D 1e-5 --beta1_D 0.5 --beta2_D 0.999 --save_discriminators True --resume_discriminators True --g_adv_warmup_steps_if_no_d_ckpt 2000 --bb_is_xywh False --bb_xyxy_end_inclusive False --image_save_every 1000 \
  --use_bb_mask_cond True  --gan_warmup_steps 40000 --k_focus 0 \
  $RESUME_ARG
