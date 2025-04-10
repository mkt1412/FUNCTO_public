#!/bin/bash
#SBATCH --job-name=SAM
#SBATCH --output=result_sam.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anxingxiao@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane5
nvidia-smi
source activate learning
python /data/home/anxing/service/sam/server.py