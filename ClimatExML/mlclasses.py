from pydantic.dataclasses import dataclass
from pydantic import Field
import os
import glob


@dataclass
class HyperParameters:
    learning_rate: float
    b1: float
    b2: float
    gp_lambda: float
    alpha: float
    n_critic: int
    max_epochs: int


@dataclass
class ClimatExMlFlow:
    host: str = Field(default="http://206.12.93.183/")
    port: int = Field(default=5000)
    tracking_uri: str = None
    default_artifact_root: str = None
    log_model: bool = True
    experiment_name: str = Field(default="ClimatExML")
    log_every_n_steps: int = Field(default=100)


@dataclass
class ClimateExMLTraining:
    num_workers: int = Field(default=24)
    precision: int = Field(default=32)
    accelerator: str = Field(default="gpu")
    strategy: str = Field(default="ddp_find_unused_parameters_true")

@dataclass
class StochasticityParameters:
    noise_injection: bool = True


@dataclass
class InvariantData:
    lr_shape: list
    hr_shape: list
    hr_cov_shape: list
    hr_cov_paths: list
    lr_invariant_paths: list

    # def __post_init__(self):
    #     self.lr_invariant_list = [
    #         sorted(glob.glob(path)) for path in self.lr_invariant_paths
    #     ]
    # #     self.hr_cov_list = [sorted(glob.glob(path)) for path in self.hr_cov_paths]
    #     print("LIST" + 80 * "-")
    #     print(self.lr_invariant_list)
    #     print(self.hr_cov_list)
    #     print("LIST" + 80 * "-")


@dataclass
class InputVariables:
    lr_paths: list
    hr_paths: list
    # lr_files: list
    # hr_files: list

    def __post_init__(self):
        self.lr_files = [sorted(glob.glob(path)) for path in self.lr_paths]
        self.hr_files = [sorted(glob.glob(path)) for path in self.hr_paths]


# @dataclass
# class InputVariables:
#     lr_shape: list
#     hr_shape: list
#     lr_paths: list
#     hr_paths: list

#     def __post_init__(self):
#         self.lr_paths = [sorted(glob.glob(path)) for path in self.lr_paths]
#         self.hr_paths = [sorted(glob.glob(path)) for path in self.hr_paths]


# @dataclass
# class DataModule:
#     invariant: InvariantData
#     train_data: InputVariables
#     test_data: InputVariables
