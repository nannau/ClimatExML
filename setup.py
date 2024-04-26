from setuptools import setup

setup(
    name="ClimatExML",
    version="0.1.0",
    author="Nic Annau",
    author_email="nicannau@gmail.com ",
    packages=["ClimatExML"],
    license="LICENSE",
    description="Python module for deep learning WGAN-GPs.",
    long_description=open("README.md").read(),
    install_requires=[
        #"pytorch-lightning==2.1.1",
        "lightning",
        #"torch==2.1.1",
        "hydra-core",
        "pytorch_msssim",
        "torchvision",
        "matplotlib",
        "pydantic",
        "comet-ml"
    ],
)
