#!/bin/bash
#
#SBATCH --job-name=time_dynamics
#SBATCH --output=/mnt/beegfs/work/tiblias/time-stuff/logs/eval_time.txt
#SBATCH --mail-user=federico.tiblias@tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -q gpu
#SBATCH -p gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1


# export PATH=$PATH:/ukp-storage-1/tiblias/miniconda/envs/td/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/tiblias/miniconda/envs/td/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/tiblias/miniconda/lib
# export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=true
# export CC=/ukp-storage-1/tiblias/miniconda/envs/gcc/bin/gcc
# export CXX=/ukp-storage-1/tiblias/miniconda/envs/gcc/bin/g++
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=0
# export NCCL_DEBUG=INFO

source /ukp-storage-1/tiblias/miniconda/bin/activate td

python run_experiments.py