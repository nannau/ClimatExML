import torch
import lightning as pl
from ClimatExML.wgan_gp import SuperResolutionWGANGP
from ClimatExML.loader import ClimatExLightning
from ClimatExML.mlclasses import HyperParameters, ClimatExMlFlow, ClimateExMLTraining
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import logging
import hydra
from hydra.utils import instantiate
import os


@hydra.main(config_path="conf", config_name="config")
def main(cfg: dict):
    os.environ["SLURM_JOB_NAME"]="bash"

    hyperparameters = instantiate(cfg.hyperparameters)
    tracking = cfg.tracking
    hardware = instantiate(cfg.hardware)

    mlflow.autolog(log_models=True, exclusive=False)

    mlflow_logger = MLFlowLogger(
        tracking_uri=tracking.tracking_uri,
        experiment_name=tracking.experiment_name,
        run_name=tracking.run_name,
        log_model = tracking.log_model,
        artifact_location=tracking.default_artifact_root,
    )

    train_data = instantiate(cfg.train_data)
    validation_data = instantiate(cfg.validation_data)
    invariant = instantiate(cfg.invariant)

    clim_data = ClimatExLightning(train_data, validation_data, invariant, hyperparameters.batch_size, num_workers=hardware.num_workers)

    srmodel = SuperResolutionWGANGP(
        tracking,
        hardware,
        hyperparameters,
        invariant,
        log_every_n_steps=tracking.log_every_n_steps,
    )

    trainer = pl.Trainer(
        precision=hardware.precision,
        accelerator=hardware.accelerator,
        max_epochs=hyperparameters.max_epochs,
        logger=mlflow_logger,
        detect_anomaly=False,
        devices=-1,
        strategy=hardware.strategy,
        check_val_every_n_epoch=1,
        log_every_n_steps=tracking.log_every_n_steps,
    )

    trainer.fit(srmodel, datamodule=clim_data)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()
