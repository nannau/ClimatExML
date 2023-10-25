from pydantic import BaseModel
import os


class HyperParameters(BaseModel):
    learning_rate: float
    b1: float
    b2: float
    gp_lambda: float
    alpha: float
    n_critic: int
    max_epochs: int


class ClimatExMlFlow(BaseModel):
    host: str = "http://206.12.93.183/"
    port: int = 5000
    tracking_uri: str# = os.environ("MLFLOW_TRACKING_URI")
    default_artifact_root: str# = os.environ("MLFLOW_ARTIFACT_ROOT")


class ClimateExMLTraining:
    num_workers: int = 24
    precision: int = 32
    accelerator: str = "gpu"
    strategy: str = "ddp_find_unused_parameters_true"


class CliamtExMLdata:
    lr_shape: list
    hr_shape: list
    files: dict
