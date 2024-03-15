import torch
from torch_mando import *
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from crip.physics import *


operation_seed_counter = 0


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)

    return g_cuda_generator


def img_alignment(img, cfg, type=8):  # mask1 <= mask2
    center_x, center_y = (cfg.imgDim-1)/2, (cfg.sid/cfg.pixelSize*2 + cfg.imgDim-1)/2
    img_ = F2.rotate(img, np.degrees(np.arctan2((type-0.5)*cfg.detEltSize, cfg.sdd)), center=(center_x, center_y))
    return img_


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0, high=8, size=(n * h // 2 * w // 2,), generator=get_generator(), out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1

    return mask1, mask2


def space_to_depth(x, block_size):
    n, c, h, w = x.shape
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)

    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)

    return subimage


def generate_mask_pair4(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    mask3 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    mask4 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], 
         [1, 3, 0, 2], [1, 3, 2, 0], [2, 3, 0, 1], [2, 3, 1, 0], 
         [1, 0, 3, 2], [1, 0, 2, 3], [2, 0, 1, 3], [2, 0, 3, 1], 
         [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0, high=idx_pair.shape[0], size=(n * h // 2 * w // 2,), generator=get_generator(), out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    mask3[rd_pair_idx[:, 2]] = 1
    mask4[rd_pair_idx[:, 3]] = 1

    return mask1, mask2, mask3, mask4


def sgm_roi(sgm, cfg):
    B, C, H_, W_ = sgm.shape
    h_pad = (H_ - cfg.views) // 2
    w_pad = (W_ - cfg.detEltCount) // 2
    sgm_roi = sgm[:, :, h_pad: h_pad + cfg.views, w_pad: w_pad + cfg.detEltCount]

    return sgm_roi


def batch_recon(sgm, cfg):
    return MandoFanbeamFbp(sgm_roi(sgm, cfg), cfg)


def adjustParams(cfg, zoom_h=2, zoom_w=2, zoom_I=2):
    cfg_new = MandoFanBeamConfig(imgDim=cfg.imgDim // zoom_I, pixelSize=cfg.pixelSize * zoom_I, sid=cfg.sid, sdd=cfg.sdd,
                                 detEltCount=cfg.detEltCount // zoom_w, detEltSize=cfg.detEltSize * zoom_w,
                                 views=cfg.views // zoom_h, reconKernelEnum=cfg.reconKernelEnum,
                                 reconKernelParam=cfg.reconKernelParam, fpjStepSize=cfg.fpjStepSize)

    return cfg_new


def batch_crop(img, radius, fill=0):

    N, M = img.shape[-2:]
    x = torch.FloatTensor(range(N)) - N / 2 - 0.5
    y = torch.FloatTensor(range(M)) - M / 2 - 0.5
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    outside = xx**2 + yy**2 > radius**2
    cropped = img.clone()
    cropped[:, :, outside] = fill

    return cropped


def sgmToMaterial(sgm, water_coef, bone_coef, A=None):
    P_LE, P_HE = sgm[:, 0], sgm[:, 1]  # (B, 736, 800)
    if A is None:
        SS = torch.stack([P_LE, P_HE, P_LE ** 2, P_LE * P_HE, P_HE ** 2])  # (5, B, 736, 800)
        pred_water = torch.einsum('i,ibjk->bjk', torch.FloatTensor(water_coef).cuda(), SS)
        pred_bone = torch.einsum('i,ibjk->bjk', torch.FloatTensor(bone_coef).cuda(), SS)
    else:
        pred_water, pred_bone = torch.einsum('ij,jxyz->ixyz',
                                             torch.linalg.pinv(A),
                                             torch.stack([P_LE, P_HE]))

    return torch.stack([pred_water, pred_bone], dim=1)


def T(img: torch.FloatTensor):
    # return img.permute(0, 1, 3, 2)
    return F2.hflip(img)


def randomCrop(img: torch.FloatTensor, size: tuple):
    _, h, w = F2.get_dimensions(img)
    th, tw = size

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()

    return F2.crop(img, i, j, th, tw)


def randomHorizontalFlip(img: torch.FloatTensor, p=0.5):
    if torch.rand(1) < p:
        return F2.hflip(img)
    return img


def randomVerticalFlip(img: torch.FloatTensor, p=0.5):
    if torch.rand(1) < p:
        return F2.vflip(img)
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


def sgmPadding(sgm: torch.FloatTensor, targetHW: tuple, style='circular'):
    _, H, W = F2.get_dimensions(sgm)
    H_, W_ = targetHW

    sgm_pad = F.pad(sgm, ((W_ - W) // 2, W_ - W - (W_ - W) // 2,
                          (H_ - H) // 2, H_ - H - (H_ - H) // 2), style)

    return sgm_pad


def take(x):
    return x.squeeze().detach().cpu().numpy()