#!/bin/bash
#
#SBATCH --job-name=sh_interventions
#SBATCH --output=/mnt/beegfs/work/tiblias/time-stuff/logs/record_interventions.txt
#SBATCH --mail-user=federico.tiblias@tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -q gpu
#SBATCH -p gpu
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_model:a100


export PATH=$PATH:/ukp-storage-1/tiblias/miniconda/envs/td/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/tiblias/miniconda/envs/td/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/tiblias/miniconda/lib
# export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=true
export CC=/ukp-storage-1/tiblias/miniconda/envs/gcc/bin/gcc
export CXX=/ukp-storage-1/tiblias/miniconda/envs/gcc/bin/g++
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=0
# export NCCL_DEBUG=INFO
export SCRATCH=/ukp-storage-1/tiblias/.cache
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_home
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export HF_MODULES_CACHE=$SCRATCH/hf_modules

source /ukp-storage-1/tiblias/miniconda/bin/activate td

python record_interventions.py --config configs/interventions.yaml