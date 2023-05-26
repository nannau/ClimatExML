import torch
import glob
import pytorch_lightning as pl
from ClimatExML.wgan_gp import SuperResolutionWGANGP
from ClimatExML.loader import ClimatExMLData
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import logging
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: dict):
    mlflow.pytorch.autolog(log_every_n_step=100, log_models=True)
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    artifact_path = f"{cfg.tracking.default_artifact_root}/{cfg.tracking.run_name}"

    # check if experiment name already exists
    if mlflow.get_experiment_by_name(cfg.tracking.experiment_name) is None:
        logging.info(f"Creating experiment: {cfg.tracking.experiment_name} with artifact location: {cfg.tracking.default_artifact_root}")
        mlflow.create_experiment(cfg.tracking.experiment_name, artifact_location=cfg.tracking.default_artifact_root)

    experiment = mlflow.get_experiment_by_name(cfg.tracking.experiment_name)
    mlflow.set_experiment(cfg.tracking.experiment_name)
    logging.info(f"Experiment ID: {experiment.experiment_id}")

    with mlflow.start_run() as run:
        data = {
            "lr_train": [glob.glob(path) for path in cfg.data.files.lr_train],
            "hr_train": [glob.glob(path) for path in cfg.data.files.hr_train],
            "lr_test": [glob.glob(path) for path in cfg.data.files.lr_test],
            "hr_test": [glob.glob(path) for path in cfg.data.files.hr_test] ## if hrcov, load here
        }

        clim_data = ClimatExMLData(
            data_glob=data,
            batch_size=cfg.hyperparameters.batch_size,
            num_workers=cfg.training.num_workers
        )

        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.tracking.experiment_name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            artifact_location=artifact_path,
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
            lr_shape=cfg.data.lr_shape,
            hr_shape=cfg.data.hr_shape,
            artifact_path=artifact_path,
            log_every_n_steps=cfg.tracking.log_every_n_steps
        )

        trainer = pl.Trainer(
            precision=cfg.training.precision,
            accelerator=cfg.training.accelerator,
            max_epochs=cfg.hyperparameters.max_epochs,
            logger=mlflow_logger,
            default_root_dir=artifact_path,
            detect_anomaly=False,
        )
        trainer.fit(srmodel, datamodule=clim_data)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()