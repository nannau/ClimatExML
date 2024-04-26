#!/bin/bash
#SBATCH --account=def-acannon
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=05-16:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=6
#SBATCH --output=log_lightning_%j.out

echo "CONFIG OPTIONS"
export PROJECT_DIR=$SLURM_TMPDIR # code uses project dir as base.
export DATA_DIR=$SLURM_TMPDIR/data
export OUTPUT_COMET_ZIP=$HOME/scratch

export CODE_PATH=/home/nannau/scratch/comet/
export HOST_DATA_PATH=/home/nannau/scratch/data
echo "END CONFIG OPTIONS"

echo "INSTALL SOFTWARE"
rsync -av --progress $CODE_PATH $SLURM_TMPDIR/

module load python
unset KUBERNETES_PORT # mysterious, needs to be here. Not sure why.

virtualenv --no-download ${PROJECT_DIR}/mlenv
source ${PROJECT_DIR}/mlenv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index ${PROJECT_DIR}/ClimatExML/
echo "END INSTALL"

echo "SET RANDOM NODE VARS"
export NCCL_BLOCKING_WAIT=1
unset OMP_NUM_THREADS # mysterious, needs to be here. Not sure why.
unset COMET_API_KEY # force offline mode

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo "END SET"

echo "COPY AND EXTRACT DATA"
rsync -av --progress $HOST_DATA_PATH $SLURM_TMPDIR/

for var in uas vas tas RH pr; do
    echo
    echo "Extracting validation $var"
    echo
    tar -xf $SLURM_TMPDIR/data/validation/$var.tar.gz -C $SLURM_TMPDIR/data --checkpoint=.50000
done

for var in uas vas tas RH pr; do
    echo
    echo "Extracting train $var"
    echo
    tar -xf $SLURM_TMPDIR/data/train/$var.tar.gz -C $SLURM_TMPDIR/data --checkpoint=.50000
done

echo "END COPY"

echo "RUN TRAINING"
unset OMP_NUM_THREADS
python ${PROJECT_DIR}/ClimatExML/ClimatExML/train.py 
