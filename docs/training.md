# Training

ClimatExML has a container with the environment pre-configured so that dependencies do not need to be installed. Users can also choose to install these dependencies normally on their host system to suit their needs.

## Containers

### Setting up Binds

Binding allows directories in the host to be replicated in the container so that changes are represented inside of the container. This makes it easier to edit and debug code within the container framework, rather than copying all data and code into the container. It also allows for the container to change files and have them represented on the host system which is useful for tracking ML runs with MLflow.

### Apptainer (Recommended on HPC systems)

Simply execute 

```
srun apptainer exec --home /home/nannau --nv --bind $HOME/scratch/marvin_light_container/:/home/nannau/,$SLURM_TMPDIR/data:/home/nannau/data,/home/nannau/scratch/marvin_light_container/light_container/mlflow/:/home/nannau/scratch/marvin_light_container/light_container/mlflow/, --overlay $SLURM_TMPDIR/light_container/ $SLURM_TMPDIR/light_container/lightning.sif  bash "/home/nannau/apptainer_cmd.sh"
```

```
# this works (just for my own notes)
apptainer shell --fakeroot --home /project --bind $HOME/scratch/apptainer:/project/ --overlay $HOME/scratch/apptainer/ $HOME/scratch/apptainer/ClimatExML/sr.sif
```

### Docker

This runs a bash script but doesn't work just yet
```
docker run -v /home/nannau/dockerize/:/project/ nannau/sr  bash container_cmd.sh
```

```
docker run -it --rm --runtime=nvidia --gpus all -v $PROJECT_DIR:/project/ -v $DATA_DIR:/project/data/ nannau/sr
```

## Without Containers

Installing ClimatExML is just like any other python package. Start by creating a virtual environment:

```bash
python -m venv climatexenv
source climatexenv/bin/activate
```

Install ClimatExML in virtual environment, you have several options. For development, it is recommended that you clone the main branch of the ClimatExML github repository and install.

```bash
git clone git@github.com:nannau/ClimatExML.git
cd ClimatExML

pip install -e .
```

This will install an editable version of ClimatExML in your environment so that changes you make do not require you to rebuild the package.