#!/bin/bash
#
#SBATCH --job-name=FDGAN2026_evalall
#SBATCH --output=YOLOv11_log_%j.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:a100:1
# Evaluates every trained run of every generator on the shared test split, conf 0.001 (see eval_all_generators.py).
module load gcc/12.2.0
module load openmpi/4.1.4
module load python-scientific/3.10.8-foss-2022b
export PSM2_CUDA=1
source /fred/oz097/latzi/YOLOv11/yolov11/bin/activate
cd /fred/oz097/latzi/YOLOv11/ultralytics
srun python eval_all_generators.py
