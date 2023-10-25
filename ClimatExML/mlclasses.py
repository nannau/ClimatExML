from pydantic.dataclasses import dataclass
import os
import glob


@dataclass
class HyperParameters():
    learning_rate: float
    b1: float
    b2: float
    gp_lambda: float
    alpha: float
    n_critic: int
    max_epochs: int


@dataclass
class ClimatExMlFlow():
    host: str = "http://206.12.93.183/"
    port: int = 5000
    tracking_uri: str# = os.environ("MLFLOW_TRACKING_URI")
    default_artifact_root: str# = os.environ("MLFLOW_ARTIFACT_ROOT")


@dataclass
class ClimateExMLTraining:
    num_workers: int = 24
    precision: int = 32
    accelerator: str = "gpu"
    strategy: str = "ddp_find_unused_parameters_true"

@dataclass
class StochasticityParameters:
    noise_injection: bool = True


@dataclass
class SuperResolutionData:
    lr_shape: list
    hr_shape: list
    lr_train_paths: list
    hr_train_paths: list
    hr_cov_paths: str
    lr_invariant_paths: list
    lr_train_glob: list[list[str]]
    hr_train_glob: list[list[str]]
    hr_cov_glob: str
    lr_invariant_glob: list[list[str]]

    def __post_init__(self):
        self.lr_train_glob = [sorted(glob.glob(path)) for path in self.lr_train_paths]
        self.hr_train_glob = [sorted(glob.glob(path)) for path in self.hr_train_paths]
        self.lr_invariant_glob = [sorted(glob.glob(path)) for path in self.lr_invariant_paths]
        self.hr_cov_glob = sorted(glob.glob(self.hr_cov))