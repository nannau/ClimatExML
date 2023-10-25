import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


def wass_loss(real, fake, device):
    return real - fake


def content_loss(hr: torch.Tensor, fake: torch.Tensor) -> float:
    """Calculates the L1 loss (pixel wise error) between both
    samples. Note that this is done on the high resolution
    (or super resolved fields)
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        content_loss (float): Single value corresponding to L1.
    """
    criterion_pixelwise = nn.L1Loss()
    # content_loss = criterion_pixelwise(hr/hr.std(), fake/fake.std())
    content_loss = criterion_pixelwise(hr, fake)

    return content_loss


def content_MSELoss(
    hr: torch.Tensor, fake: torch.Tensor, device: torch.device
) -> float:
    """Calculates the L1 loss (pixel wise error) between both
    samples. Note that this is done on the high resolution (or super resolved fields)
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        content_loss (float): Single value corresponding to L1.
    """
    criterion_pixelwise = nn.MSELoss().to(device)
    content_loss = criterion_pixelwise(hr, fake)
    return content_loss
