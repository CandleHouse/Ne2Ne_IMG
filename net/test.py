import logging
import tifffile
from tqdm import tqdm
from models import Ne2Ne_UNet
import torch
import numpy as np
from crip.io import *
from torch_mando import *
from data.dataset import DatasetLoad
from torch.utils.data import DataLoader
from crip.postprocess import *
from utils import *
from kornia.losses import *
from crip.physics import *
from itertools import islice

input_dir = './sgm'
label_dir = r'./L'

raw_dir = './basis_raw'
pred_dir = './basis_pred'
gt_dir = './basis_gt'


dir_checkpoint = f'./checkpoints'  # every epoch
checkpoint_name = 'epoch_199_self_supervised.pth'


if __name__ == '__main__':
    ### Model
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    cfg = MandoFanBeamConfig(imgDim=512, pixelSize=0.8, sid=800, sdd=1200, detEltCount=800, detEltSize=0.8,
                             views=720, reconKernelEnum=KERNEL_GAUSSIAN_RAMP, reconKernelParam=0.75, fpjStepSize=0.2)
    
    atten_water = Atten.fromBuiltIn('Water', 1.0)
    atten_bone = Atten.fromBuiltIn('I', 1.0)
    ratio = atten_bone.mu[67] / atten_water.mu[67]

    model1 = Ne2Ne_UNet(in_channels=2, out_channels=2, kernel_size=(3, 3), padding=(1, 1)).to(device).eval()
    model1.load_state_dict(torch.load(dir_checkpoint + checkpoint_name, map_location={'cuda:0': 'cuda:0'})['net1'])

    test_set = DatasetLoad(input_dir, label_dir, mode='test', kVps=(80, 140), dose=1e5, material_basis=('w', 'i'))
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    raw = np.zeros((len(test_set), 2, 512, 512), dtype=np.float32)
    pred = np.zeros_like(raw)
    gt = np.zeros_like(raw)
    step = 0
    global_step = 0

    # ###### Testing ######
    for batch in tqdm(test_loader, ncols=100, desc='[Testing]'):
        ### jump
        batch = next(islice(test_loader, 22, None))

        full_noisy_sgm = batch['full_noisy_sgm'].to(device)
        full_noisy_material = batch['full_noisy_material'].to(device)
        full_clean_sgm = batch['full_clean_sgm'].to(device)

        A = torch.FloatTensor([[atten_water.mu[54], atten_bone.mu[54]],
                                [atten_water.mu[73], atten_bone.mu[73]]]).to(device)
        with torch.no_grad():
            ### GT
            clean_rec = batch_recon(full_clean_sgm, cfg)
            clean_map = sgmToMaterial(clean_rec, None, None, A)
            
            noisy_rec = batch_recon(full_noisy_sgm, cfg)
            noisy_map = sgmToMaterial(noisy_rec, None, None, A)
                                    
            pred_rec = model1(noisy_map, noisy_rec)
            pred_map = sgmToMaterial(pred_rec, None, None, A)


        raw[step: step+full_noisy_sgm.shape[0], :] = noisy_map.detach().cpu().numpy()
        pred[step: step+full_noisy_sgm.shape[0], :] = pred_map.detach().cpu().numpy()
        gt[step: step+full_noisy_sgm.shape[0], :] = clean_map.detach().cpu().numpy()
        
        step += full_noisy_sgm.shape[0]

        ### Save
        for index in range(full_noisy_sgm.shape[0]):
            tifffile.imwrite(f'{pred_dir}/water/w_{global_step}.tif', pred[global_step, 0])
            tifffile.imwrite(f'{pred_dir}/bone/b_{global_step}.tif', pred[global_step, 1])

            tifffile.imwrite(f'{raw_dir}/water/w_{global_step}.tif', raw[global_step, 0])
            tifffile.imwrite(f'{raw_dir}/bone/b_{global_step}.tif', raw[global_step, 1])

            tifffile.imwrite(f'{gt_dir}/water/w_{global_step}.tif', gt[global_step, 0])
            tifffile.imwrite(f'{gt_dir}/bone/b_{global_step}.tif', gt[global_step, 1])
            global_step += 1

        break
