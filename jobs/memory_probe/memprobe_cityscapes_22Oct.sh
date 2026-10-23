#!/bin/bash
#SBATCH --job-name=memprobe_cityscapes
#SBATCH --output=/fred/oz097/latzi/Diffusion_ProjectedGAN/guided-diffusion/logs_Step11/memprobe_cityscapes_%j.txt
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=01:30:00
# ---------------------------------------------------------------------------------------------
# Peak GPU memory of the Table-1 protocol on Cityscapes-Pedestrian (AE point 10), one A100.
# Five short probes, each with the exact flags of the Step11 job files:
#   train ADM            (500 iterations, batch 16, micro-batch 8)
#   train FDGAN*         (500 iterations; mask channel, adversarial stage disabled)
#   train FDGAN          (500 iterations; adversarial stage active from iteration 0 so that the
#                         GAN path, the discriminator and the extra noise draw are exercised)
#   sample FDGAN         (16 images, 500 steps, mask guidance 2.0, colour guidance = Table 1 row)
#   sample ADM           (16 images, 250 steps, colour guidance = Table 1 row)
# Two measurements per probe: PyTorch max_memory_allocated / max_memory_reserved (memprobe_run.py)
# and the nvidia-smi memory.used peak sampled once per second.
# Everything is written under a NEW folder memprobe_22Oct/; the probe checkpoints it writes are
# removed at the end of each probe. No existing checkpoint, sample set or log is touched.
# ---------------------------------------------------------------------------------------------
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
DATA_DIR="$REPO/datasets/Cityscapes_3009_img"
BB_DIR="$REPO/datasets/Cityscapes_3009_bb"
[ -d "$DATA_DIR" ] || { echo "no image folder: $DATA_DIR"; exit 1; }
[ -d "$BB_DIR" ] || { echo "no box folder: $BB_DIR"; exit 1; }
[ -f "$REPO/memprobe_run.py" ] || { echo "memprobe_run.py not found in $REPO - upload it first"; exit 1; }
grep -q "bb_is_xywh" guided_diffusion/image_datasets_Run2_3_10_COCO.py || { echo "loader is not the 2026 version"; exit 1; }
grep -q "gan_warmup_remaining" guided_diffusion/train_util_coco_fdgan_late_band.py || { echo "train loop is not the 2026 version"; exit 1; }
PROBE="$REPO/memprobe_22Oct"
mkdir -p "$PROBE"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
SUMMARY="$PROBE/summary_${SLURM_JOB_ID}.txt"
: > "$SUMMARY"
echo "===== memprobe job $SLURM_JOB_ID on $(hostname), GPU $GPU_ID @ $(date) =====" | tee -a "$SUMMARY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$SUMMARY"

# common flags, copied from train_cityscapes_*.sh (Step11); only the output folders, the step
# count (500), the save interval and the warm-up differ per probe
TRAIN_COMMON="--data_dir $DATA_DIR --bb_dir $BB_DIR \
  --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16 --use_scale_shift_norm False --resblock_updown False --learn_sigma True --class_cond False --dropout 0.0 \
  --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False \
  --lr 1e-4 --batch_size 16 --microbatch 8 --use_fp16 True --fp16_scale_growth 1e-3 --ema_rate 0.9999 --log_interval 50 --save_interval 1000000 --lr_anneal_steps 500 --bb_mask_key bb_mask --mask_gain 50.0 --mask_init_mode copy_mean --mask_init_scale 0.10 --tmin_start 820 --tmin_end 700 --band_width 120 --tmax_cap 950 --band_ramp_steps 50000 --diffusion_in_band_weight 0.7 --gan_initial_scale 1e-4 --gan_final_scale 1e-2 --l1_initial_scale 5e-2 --l1_final_scale 5e-2 --lambda_enhanced 5e-2 --color_stats_scale 0.05 --sat_penalty_scale 0.01 --sat_threshold 0.95 --noise_option extra_gan_only --k_focus 1.75 --max_gan_per_microbatch 4 --use_mask_adapter True --ndf 128 --n_layers_D 3 --use_sigmoid False --lr_D 1e-5 --beta1_D 0.5 --beta2_D 0.999 --save_discriminators True --resume_discriminators True --g_adv_warmup_steps_if_no_d_ckpt 2000 --bb_is_xywh False --bb_xyxy_end_inclusive False --image_save_every 1000"
SAMPLE_COMMON="--bb_dir $BB_DIR --batch_size 16 --seed 17 \
  --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16 --use_scale_shift_norm False --resblock_updown False --learn_sigma True --class_cond False --dropout 0.0 \
  --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False \
  --color_guide_mode sample --color_stats_json $REPO/datasets/color_stats_cityscapes.json --sigma_mode learned"
CKPT_FDGAN="$REPO/checkpoints_Step11/cityscapes_fdgan_mask/ema_0.9999_200000.pt"
CKPT_ADM="$REPO/checkpoints_Step11/cityscapes_adm/ema_0.9999_200000.pt"
[ -f "$CKPT_FDGAN" ] || { echo "checkpoint not found: $CKPT_FDGAN"; exit 1; }
[ -f "$CKPT_ADM" ] || { echo "checkpoint not found: $CKPT_ADM"; exit 1; }
[ -f "$REPO/datasets/color_stats_cityscapes.json" ] || { echo "colour stats not found: $REPO/datasets/color_stats_cityscapes.json"; exit 1; }

run_probe () {   # $1 = probe name, $2.. = python script and its arguments
  local NAME="$1"; shift
  local OUTLOG="$PROBE/${NAME}_${SLURM_JOB_ID}.log"
  local NVLOG="$PROBE/${NAME}_${SLURM_JOB_ID}_nvsmi.csv"
  echo "----- probe $NAME @ $(date) -----" | tee -a "$SUMMARY"
  nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 1 > "$NVLOG" &
  local NVPID=$!
  set +e
  srun python -u memprobe_run.py "$@" > "$OUTLOG" 2>&1
  local RC=$?
  set -e
  kill "$NVPID" 2>/dev/null || true
  sleep 1
  local PEAK_MIB
  PEAK_MIB=$( (cut -d, -f2 "$NVLOG" | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1) || true)
  local TORCH_LINE
  TORCH_LINE=$( (grep -F "[memprobe]" "$OUTLOG" | tail -1) || true)
  echo "$NAME | exit $RC | nvidia-smi peak memory.used ${PEAK_MIB:-?} MiB | ${TORCH_LINE:-no torch counters printed}" | tee -a "$SUMMARY"
  [ "$RC" -eq 0 ] || { echo "probe $NAME failed, see $OUTLOG"; tail -30 "$OUTLOG"; }
}

# 1) ADM (no mask channel, adversarial stage never active)
run_probe train_adm scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py $TRAIN_COMMON \
  --log_dir "$PROBE/train_adm" --checkpoint_dir "$PROBE/ckpt_train_adm" --image_save_dir "$PROBE/img_train_adm" \
  --use_bb_mask_cond False --gan_warmup_steps 999999999
rm -rf "$PROBE/ckpt_train_adm" "$PROBE/img_train_adm"

# 2) FDGAN* (mask channel, adversarial stage disabled)
run_probe train_fdgan_star scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py $TRAIN_COMMON \
  --log_dir "$PROBE/train_fdgan_star" --checkpoint_dir "$PROBE/ckpt_train_fdgan_star" --image_save_dir "$PROBE/img_train_fdgan_star" \
  --use_bb_mask_cond True --gan_warmup_steps 999999999
rm -rf "$PROBE/ckpt_train_fdgan_star" "$PROBE/img_train_fdgan_star"

# 3) FDGAN (mask channel, adversarial stage active from iteration 0 for this probe only)
run_probe train_fdgan scripts/image_train_Run2_3_10_FDGAN_COCO_LateBand.py $TRAIN_COMMON \
  --log_dir "$PROBE/train_fdgan" --checkpoint_dir "$PROBE/ckpt_train_fdgan" --image_save_dir "$PROBE/img_train_fdgan" \
  --use_bb_mask_cond True --gan_warmup_steps 0
rm -rf "$PROBE/ckpt_train_fdgan" "$PROBE/img_train_fdgan"

# 4) sampling FDGAN, Table-1 row: 500 ancestral steps, mask guidance 2.0, in-band perturbation, colour guidance
run_probe sample_fdgan_cfg20 scripts/image_sample_fdgan_4ch.py $SAMPLE_COMMON \
  --model_path "$CKPT_FDGAN" --out_dir "$PROBE/sample_fdgan_cfg20" --num_samples 16 --timestep_respacing 500 \
  --use_bb_mask_cond True --mask_gain 50.0 --mask_mode always --roi_bump_mode predict --k_focus 1.75 --bump_tmin_rev 700 --bump_tmax_rev 820 --mask_cfg_scale 2.0

# 5) sampling ADM, Table-1 row: 250 ancestral steps, learned variance, colour guidance
run_probe sample_adm_r250 scripts/image_sample_fdgan_4ch.py $SAMPLE_COMMON \
  --model_path "$CKPT_ADM" --out_dir "$PROBE/sample_adm_r250" --num_samples 16 --timestep_respacing 250 \
  --use_bb_mask_cond False --roi_bump_mode none --k_focus 0

echo "===== done @ $(date) =====" | tee -a "$SUMMARY"
echo "summary: $SUMMARY"
cat "$SUMMARY"
