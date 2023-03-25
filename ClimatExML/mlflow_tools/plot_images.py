# Plots matplotlib grids and saves to file
import torch
import matplotlib.pyplot as plt
import torchvision
import mlflow


def gen_grid_images(G, lr: torch.Tensor, hr: torch.Tensor, batch_size: int, n_examples: int=3, cmap="viridis") -> None:
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
    random = torch.randint(0, batch_size, (n_examples, ))

    sr = G(lr[random, ...])
    lr_grid = torchvision.utils.make_grid(
        lr[random, ...],
        nrow=n_examples,
        padding=5
    )[0, ...]

    sr_grid = torchvision.utils.make_grid(
        sr,
        nrow=n_examples
    )[0, ...]

    hr_grid = torchvision.utils.make_grid(
        hr[random, ...],
        nrow=n_examples
    )[0, ...]

    fig = plt.figure(figsize=(30, 10))
    fig.suptitle("Training Samples")

    # Plot the coarse and fake samples
    subfigs = fig.subfigures(nrows=n_examples, ncols=1)

    ax = make_subfig(subfigs, 0, "Low Resolution Fields", lr_grid, cmap)
    ax = make_subfig(subfigs, 1, "Super Resolved Fields", sr_grid, cmap)
    ax = make_subfig(subfigs, 2, "Ground Truth Fields", hr_grid, cmap)

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
