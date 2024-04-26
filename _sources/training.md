# Training

ClimatExML has a container with the environment pre-configured so that dependencies do not need to be installed. However, at this stage, it is recommended that users install ClimatEx using the Python installation instructions.

## Without Containers (Recommended)

In a virtual environment with ClimatExML installed, and after testing that the GPU(s) are properly configured (refere to [installing](./installing.md)) to run with PyTorch, simply run

```python
python ClimatExML/ClimatExML/train.py
```

## On DRAC Machines (Narval)

To submit a SLURM job is a bit finicky but as of writing this documentation, the following method works as provided in `hpc/train_job.sh`. On DRAC machines you can submit a job with `sbatch train_job.sh` once you modified its contents to match your configuration.

## Containers (Experimental)

### Setting up Volumes/Binds

Binding allows directories in the host to be replicated in the container so that changes are represented inside of the container. This makes it easier to edit and debug code within the container framework, rather than copying all data and code into the container. It also allows for the container to change files and have them represented on the host system which is useful for tracking ML runs with MLflow.

### Docker (untested)

To enter an environment with the necessary environment (for example if you have limited access to linux machines or are having troubles getting the environment installed manually) 
```
docker run -it --rm --runtime=nvidia --gpus all -v $PROJECT_DIR:/project/ -v $DATA_DIR:/project/data/ nannau/sr
```
