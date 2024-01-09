#!/bin/bash
#SBATCH --account=def-acannon
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=03-16:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=12
#SBATCH --output=log_kurtosis.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# srun --cpus-per-task=$SLURM_CPUS_PER_TASK ./program

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

rsync -av --progress /home/nannau/scratch/marvin_light_container/ $SLURM_TMPDIR/
rsync -av --progress /home/nannau/scratch/data $SLURM_TMPDIR/
# tar -xf $SLURM_TMPDIR/light_container/data.tar.gz -C $SLURM_TMPDIR/light_container --checkpoint=.50000

for var in uas vas tas Q2 pr; do
    echo
    echo "Extracting Train $var"
    echo
    tar -xf $SLURM_TMPDIR/data/train/$var.tar.gz -C $SLURM_TMPDIR/data/train --checkpoint=.50000
    echo
    echo "Extracting Validation $var"
    echo
    tar -xf $SLURM_TMPDIR/data/validation/$var.tar.gz -C $SLURM_TMPDIR/data/validation --checkpoint=.50000
done


module load apptainer
module load cuda
unset KUBERNETES_PORT

# This command should make it so you can edit outside of $SLURM_TMPDIR and have changes appear in container
# apptainer shell --home /home/nannau --nv --bind $HOME/scratch/marvin_light_container/:/home/nannau/,$SLURM_TMPDIR:/home/nannau/,/home/nannau/scratch/marvin_light_container/light_container/mlflow/:/home/nannau/light_container/mlflow/, --overlay $SLURM_TMPDIR/light_container/ $SLURM_TMPDIR/light_container/lightning.sif
####
# This command works well for mounting changes and the new data:
# apptainer shell --home /home/nannau --nv --bind $HOME/scratch/marvin_light_container/:/home/nannau/,$SLURM_TMPDIR/data:/home/nannau/data,/home/nannau/scratch/marvin_light_container/light_container/mlflow/:/home/nannau/light_container/mlflow/, --overlay $SLURM_TMPDIR/light_container/ $SLURM_TMPDIR/light_container/lightning.sif
##

# okay for dumb technical reasons we have to match the output artifact directory to the exact same directory in the apptainer container as where it's bing
# stored and served in mlflow server
# apptainer shell --home /home/nannau --nv --bind $HOME/scratch/marvin_light_container/:/home/nannau/,$SLURM_TMPDIR/data:/home/nannau/data,/home/nannau/scratch/marvin_light_container/light_container/mlflow/:/home/nannau/scratch/marvin_light_container/light_container/mlflow/, --overlay $SLURM_TMPDIR/light_container/ $SLURM_TMPDIR/light_container/lightning.sif 
# associated sserver command:
# mlflow ui --backend-store-uri sqlite:////home/nannau/scratch/marvin_light_container/light_container/mlflow/climatexdb.sqlite
# srun apptainer exec --home /home/nannau --nv --bind $SLURM_TMPDIR:/home/nannau/,/home/nannau/scratch/marvin_light_container/light_container/mlflow/:/home/nannau/light_container/mlflow/, --overlay $SLURM_TMPDIR/light_container/ $SLURM_TMPDIR/light_container/lightning.sif bash "/home/nannau/apptainer_cmd.sh"

srun apptainer exec --home /home/nannau --nv --bind $HOME/scratch/marvin_light_container/:/home/nannau/,$SLURM_TMPDIR/data:/home/nannau/data,/home/nannau/scratch/marvin_light_container/light_container/mlflow/:/home/nannau/scratch/marvin_light_container/light_container/mlflow/, --overlay $SLURM_TMPDIR/light_container/ $SLURM_TMPDIR/light_container/lightning.sif  bash "/home/nannau/apptainer_cmd.sh"
