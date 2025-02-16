#!/bin/bash -i
#SBATCH --partition=gpu-opteron
#SBATCH --job-name=my_first_job
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --qos=normal

module load miniconda/conda-22.11.1

echo "This is my first job in $(hostname -s)"

source activate ~/erichuangpython
source ./env

python Tentative.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
