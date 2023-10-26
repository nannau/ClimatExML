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


def start_mlflow_run(tracking: ClimatExMlFlow):
    mlflow.pytorch.autolog(log_models=tracking.log_model)
    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    if mlflow.get_experiment_by_name(tracking.experiment_name) is None:
        logging.info(
            f"Creating experiment: {tracking.experiment_name} with artifact location: {tracking.default_artifact_root}"
        )
        mlflow.create_experiment(
            tracking.experiment_name,
        )

    experiment = mlflow.get_experiment_by_name(tracking.experiment_name)
    mlflow.set_experiment(tracking.experiment_name)
    return experiment


@hydra.main(config_path="conf", config_name="config")
def main(cfg: dict):
    hyperparameters = instantiate(cfg.hyperparameters)
    tracking = instantiate(cfg.tracking)
    experiment = start_mlflow_run(tracking)
    hardware = instantiate(cfg.hardware)

    with mlflow.start_run() as run:
        logging.info(f"Experiment ID: {experiment.experiment_id}")
        logging.info(f"Artifact Location: {run.info.artifact_uri}")

        train_data = instantiate(cfg.train_data)
        test_data = instantiate(cfg.test_data)
        invariant = instantiate(cfg.invariant)

        print(train_data)

        lr_shape = invariant.lr_shape
        hr_shape = invariant.hr_shape
        hr_cov_shape = invariant.hr_cov_shape

        clim_data = ClimatExLightning(train_data, test_data, invariant)

        mlflow_logger = MLFlowLogger(
            experiment_name=tracking.experiment_name,
            run_id=run.info.run_id,
        )

        srmodel = SuperResolutionWGANGP(
            num_workers=hardware.num_workers,
            learning_rate=hyperparameters.learning_rate,
            b1=hyperparameters.b1,
            b2=hyperparameters.b2,
            n_critic=hyperparameters.n_critic,
            gp_lambda=hyperparameters.gp_lambda,
            alpha=hyperparameters.alpha,
            lr_shape=lr_shape,
            hr_shape=hr_shape,
            hr_cov_shape=hr_cov_shape,
            is_noise=cfg.stochasticity.noise_injection,
            log_every_n_steps=cfg.tracking.log_every_n_steps,
        )

        trainer = pl.Trainer(
            precision=hardware.precision,
            accelerator=hardware.accelerator,
            max_epochs=hyperparameters.max_epochs,
            logger=mlflow_logger,
            detect_anomaly=False,
            devices=-1,
            strategy=hardware.strategy,
        )
        trainer.fit(srmodel, datamodule=clim_data)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()
