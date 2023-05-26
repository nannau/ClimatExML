# Plots matplotlib grids and saves to file
import torch
import matplotlib.pyplot as plt
import torchvision
import mlflow
import torch.nn.functional as F


def log_pytorch_model(model: torch.nn.Module, path: str) -> None:
    """
    Logs a pytorch model to mlflow
    Args:
        model (torch.nn.Module): The model to log.
        name (str): The name of the model.
    Returns:
        None
    """
    mlflow.pytorch.log_model(model, path)


def log_metrics_every_n_steps(
    metrics: dict,
    sr: torch.Tensor,
    hr: torch.Tensor,
    batch_idx: int,
    n_steps: int = 100,
) -> None:
    """
    Logs metrics to mlflow every n steps
    Args:
        metrics (dict): The metrics to log.
        batch_idx (int): The current batch index.
        n_steps (int): The number of steps to log after.
    Returns:
        None
    """
    # Evaluate the metrics in the metrics dictionary

    if (batch_idx + 1) % n_steps == 0:
        metrics = {
            "MSE": F.mse_loss(sr, hr).item(),
            "MAE": F.l1_loss(sr, hr).item(),
            "Wasserstein": torch.mean(),
        }
        mlflow.log_metrics(metrics)


def gen_grid_images(
    var,
    fig,
    G,
    lr: torch.Tensor,
    hr: torch.Tensor,
    hr_cov: torch.Tensor,
    batch_size: int,
    use_hr_cov: bool,
    n_examples: int = 3,
    cmap="viridis",
) -> None:
    """
    Plots a grid of images and saves them to file
    Args:
        coarse (torch.Tensor): The coarse input.
        fake (torch.Tensor): The fake input.
        real (torch.Tensor): The real input.
    Returns:
        None
    """
    torch.manual_seed(0)
    random = torch.randint(0, batch_size, (n_examples,))
    
    if(use_hr_cov):
        sr = G(lr[random, ...],hr_cov[random,...])
    else:
        sr = G(lr[random, ...])
        
    lr_grid = torchvision.utils.make_grid(lr[random, ...], nrow=n_examples, padding=5)[
        var, ...
    ]

    sr_grid = torchvision.utils.make_grid(sr, nrow=n_examples)[var, ...]

    hr_grid = torchvision.utils.make_grid(hr[random, ...], nrow=n_examples)[var, ...]

    fig.suptitle("Training Samples")

    # Plot the coarse and fake samples
    subfigs = fig.subfigures(nrows=n_examples, ncols=1)

    ax = make_subfig(
        subfigs,
        0,
        f"Low Resolution Fields Min: {lr_grid.min()} Max: {lr_grid.max()}",
        lr_grid,
        cmap,
    )
    ax = make_subfig(
        subfigs,
        1,
        f"Super Resolved Fields Min: {sr_grid.min()} Max: {sr_grid.max()}",
        sr_grid,
        cmap,
    )
    ax = make_subfig(
        subfigs,
        2,
        f"Ground Truth Fields Min: {hr_grid.min()} Max: {hr_grid.max()}",
        hr_grid,
        cmap,
    )

    return fig


def make_subfig(subfigs, idx, title, grid, cmap):
    """Function to plot a subfigure of the tensor grid
    Args:
        subfigs (matplotlib.figure.SubFigure): The subfigure to plot on.
        idx (int): The index of the subfigure.
        title (str): The title of the subfigure.
        grid (torch.Tensor): The grid to plot.
        cmap (str): The colormap to use.

    Returns:
        matplotlib.axes.Axes: The axes of the subfigure.
    """
    subfigs[idx].suptitle(title)
    result = subfigs[idx].subplots(1, 1)
    result.imshow(grid.cpu().detach(), origin="lower", cmap=cmap)

    return result
