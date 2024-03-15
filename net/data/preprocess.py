import numpy as np
import torch
import torchvision.transforms.functional as F


def randomCrop(img: torch.FloatTensor, size: tuple):
    _, h, w = F.get_dimensions(img)
    th, tw = size

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()

    return F.crop(img, i, j, th, tw)


def randomHorizontalFlip(img: torch.FloatTensor, p=0.5):
    if torch.rand(1) < p:
        return F.hflip(img)
    return img


def randomVerticalFlip(img: torch.FloatTensor, p=0.5):
    if torch.rand(1) < p:
        return F.vflip(img)
    return img


def randomTranspose(img: torch.FloatTensor, p=0.5):
    dim = len(img.shape)
    if torch.rand(1) < p:
        return torch.transpose(img, dim-2, dim-1)
    return img

def randomProcess(img, size, p):
    img = randomCrop(img, size)
    img = randomTranspose(img, p)
    img = randomHorizontalFlip(img, p)
    img = randomVerticalFlip(img, p)

    return img


from torch_mando import *
def sgm_roi(sgm, cfg):
    if len(sgm.shape) == 3: sgm = sgm[None, :]
    B, C, H_, W_ = sgm.shape
    h_pad = (H_ - cfg.views) // 2
    w_pad = (W_ - cfg.detEltCount) // 2
    sgm_roi = sgm[:, :, h_pad: h_pad + cfg.views, w_pad: w_pad + cfg.detEltCount]

    return sgm_roi


def batch_recon(sgm, cfg):
    return MandoFanbeamFbp(sgm_roi(sgm, cfg), cfg)


def sgmPadding(sgm: np.array, targetHW: tuple, style='wrap'):
    H, W = sgm.shape
    H_, W_ = targetHW

    sgm_pad = np.pad(sgm, (((H_ - H) // 2, H_ - H - (H_ - H) // 2),
                           ((W_ - W) // 2, W_ - W - (W_ - W) // 2)), style)

    return sgm_pad
