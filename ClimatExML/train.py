import torch
import glob
import lightning as pl
from ClimatExML.wgan_gp import SuperResolutionWGANGP
from ClimatExML.loader import ClimatExMLDataHRCov
from ClimatExML.mlclasses import HyperParameters, ClimatExMlFlow, ClimateExMLTraining
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import logging
import hydra
from hydra.utils import instantiate


def start_mlflow_run(hardware: HyperParameters):
    mlflow.pytorch.autolog(log_models=hardware.log_model)
    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    if mlflow.get_experiment_by_name(hardware.experiment_name) is None:
        logging.info(
            f"Creating experiment: {hardware.experiment_name} with artifact location: {hardware.default_artifact_root}"
        )
        mlflow.create_experiment(
            hardware.experiment_name,
        )

    experiment = mlflow.get_experiment_by_name(hardware.experiment_name)
    mlflow.set_experiment(hardware.experiment_name)
    return experiment


@hydra.main(config_path="conf", config_name="config")
def main(cfg: dict):
    hyperparameters = instantiate(cfg.hyperparameters)
    hardware = instantiate(cfg.hardware)
    experiment = start_mlflow_run(hardware)

    logging.info(f"Experiment ID: {experiment.experiment_id}")
    with mlflow.start_run() as run:
        logging.info(f"Artifact Location: {run.info.artifact_uri}")

        data = {
            "lr_train": [sorted(glob.glob(path)) for path in cfg.data.files.lr_train],
            "hr_train": [sorted(glob.glob(path)) for path in cfg.data.files.hr_train],
            "hr_cov": cfg.data.files.hr_cov,
            "lr_invariant": cfg.data.files.lr_invariant,
        }

        lr_shape = cfg.data.lr_shape
        lr_shape.insert(
            0, len(cfg.data.files.lr_train) + len(cfg.data.files.lr_invariant)
        )

        hr_shape = cfg.data.hr_shape
        hr_shape.insert(0, len(cfg.data.files.hr_train))

        clim_data = ClimatExMLDataHRCov(
            data_glob=data,
            batch_size=cfg.hyperparameters.batch_size,
            num_workers=cfg.training.num_workers,
        )

        mlflow_logger = MLFlowLogger(
            experiment_name=hardware.experiment_name,
            run_id=run.info.run_id,
        )

        srmodel = SuperResolutionWGANGP(
            batch_size=cfg.hyperparameters.batch_size,
            num_workers=cfg.training.num_workers,
            learning_rate=cfg.hyperparameters.learning_rate,
            b1=cfg.hyperparameters.b1,
            b2=cfg.hyperparameters.b2,
            n_critic=cfg.hyperparameters.n_critic,
            gp_lambda=cfg.hyperparameters.gp_lambda,
            alpha=cfg.hyperparameters.alpha,
            lr_shape=lr_shape,
            hr_shape=hr_shape,
            log_every_n_steps=hardware.log_every_n_steps,
        )

        trainer = pl.Trainer(
            precision=cfg.training.precision,
            accelerator=cfg.training.accelerator,
            max_epochs=cfg.hyperparameters.max_epochs,
            logger=mlflow_logger,
            detect_anomaly=False,
            devices=-1,
            strategy=cfg.training.strategy,
        )
        trainer.fit(srmodel, datamodule=clim_data)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()
