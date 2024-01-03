# from pydantic.dataclasses import dataclass
from pydantic import Field
import os
import glob
from dataclasses import dataclass
import numpy as np


@dataclass
class HyperParameters:
    learning_rate: float
    b1: float
    b2: float
    gp_lambda: float
    alpha: float
    n_critic: int
    max_epochs: int
    noise_injection: bool = True
    batch_size: int = 3


@dataclass
class ClimatExMlFlow:
    host: str = Field(default="http://206.12.93.183/")
    port: int = Field(default=5000)
    tracking_uri: str = None
    default_artifact_root: str = None
    log_model: bool = True
    experiment_name: str = Field(default="ClimatExML")
    log_every_n_steps: int = Field(default=100)
    validation_log_every_n_steps: int = Field(default=500)


@dataclass
class ClimatExMLTraining:
    num_workers: int = Field(default=24)
    precision: int = Field(default=32)
    accelerator: str = Field(default="gpu")
    strategy: str = Field(default="ddp_find_unused_parameters_true")


@dataclass
class InvariantData:
    lr_shape: list
    hr_shape: list
    hr_invariant_shape: list
    hr_invariant_paths: list
    lr_invariant_paths: list


@dataclass
class InputVariables:
    lr_paths: list
    hr_paths: list

    def __post_init__(self):
        self.lr_files = np.array([sorted(glob.glob(path)) for path in self.lr_paths])
        self.hr_files = np.array([sorted(glob.glob(path)) for path in self.hr_paths])
