import torch
import glob
import pytorch_lightning as pl
from ClimatExML.wgan_gp import SuperResolutionWGANGP
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import logging
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: dict):
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # check if experiment name already exists
    if mlflow.get_experiment_by_name(cfg.tracking.experiment_name) is None:
        mlflow.create_experiment(cfg.tracking.experiment_name)

    experiment = mlflow.get_experiment_by_name(cfg.tracking.experiment_name)

    mlflow.set_experiment(cfg.tracking.experiment_name)
    data = {
        "lr_train": [glob.glob(path) for path in cfg.data.files.lr_train],
        "hr_train": [glob.glob(path) for path in cfg.data.files.hr_train],
        "lr_test": [glob.glob(path) for path in cfg.data.files.lr_test],
        "hr_test": [glob.glob(path) for path in cfg.data.files.hr_test]
    }

    # with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=cfg.tracking.run_name):
    # with MLFlowLogger(experiment_name=experiment.name, run_name=cfg.tracking.run_name) as mlf_logger:
    mlf_logger = MLFlowLogger(
        experiment_name=experiment.name,
        run_name=cfg.tracking.run_name
    )
    srmodel = SuperResolutionWGANGP(
        data_glob=data,
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
    )
    trainer = pl.Trainer(
        precision=cfg.training.precision,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.hyperparameters.max_epochs,
        logger=mlf_logger
    )
    trainer.fit(srmodel)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()