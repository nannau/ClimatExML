import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from pytorch_msssim import MS_SSIM


def wass_loss(real, fake, device):
    return real - fake


def SSIM_Loss(x, y, device, reduction="mean", window_size=11):
    """Return MS_SSIM
    """
    maxu = x[:, 0, ...].max()
    minu = x[:, 0, ...].min()
    maxv = x[:, 1, ...].max()
    minv = x[:, 1, ...].min()

    x[:, 0, ...] = (x[:, 0, ...] - minu)/(maxu-minu)
    x[:, 1, ...] = (x[:, 1, ...] - minv)/(maxv-minv)

    maxu = y[:, 0, ...].max()
    minu = y[:, 0, ...].min()
    maxv = y[:, 1, ...].max()
    minv = y[:, 1, ...].min()

    y[:, 0, ...] = (y[:, 0, ...] - minu)/(maxu-minu)
    y[:, 1, ...] = (y[:, 1, ...] - minv)/(maxv-minv)

    assert float(x.max()) == 1.0
    assert float(y.max()) == 1.0
    assert float(x.min()) == 0.0
    assert float(x.min()) == 0.0

    # return ssim(x, y, reduction=reduction, window_size=window_size)
    ms_ssim_mod =  MS_SSIM(win_size=7, data_range=1,  channel=2)
    return ms_ssim_mod(x, y)

def content_loss(hr: torch.Tensor, fake: torch.Tensor, device: torch.device) -> float:
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
    criterion_pixelwise = nn.L1Loss().to(device)
    # content_loss = criterion_pixelwise(hr/hr.std(), fake/fake.std())
    content_loss = criterion_pixelwise(hr, fake)

    return content_loss


def content_MSELoss(hr: torch.Tensor, fake: torch.Tensor, device: torch.device) -> float:
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
