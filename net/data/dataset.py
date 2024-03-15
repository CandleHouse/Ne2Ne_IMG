import numpy as np
from torch.utils.data import Dataset
import natsort
from os.path import splitext
from os import listdir
import logging
import tifffile
import torch
from crip.preprocess import *
from crip.physics import *
from .preprocess import *


class DatasetLoad(Dataset):
    def __init__(self, input_dir, label_dir, num_layers=5, H_=None, W_=None,
                 use_ratio=1.0, dose=5e5, mode='train', kVps=(80, 140), A=None,
                 prior_dir=None, fixed_noise=False, material_basis=('w', 'b')):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.num_layers = num_layers
        self.H_, self.W_ = H_, W_
        self.use_ratio = use_ratio
        self.dose = dose  # int or list
        self.mode = mode
        self.A = A  # img decompose coef
        self.prior_dir = prior_dir
        self.fixed_noise = fixed_noise  # not use fixed noisy sgm
        if fixed_noise is True:
            print('[Warning] Use noisy sgm offline')

        self.le_ids = natsort.natsorted([splitext(file)[0] for file in listdir(f'{input_dir}/le') if not file.startswith('.')])
        self.he_ids = natsort.natsorted([splitext(file)[0] for file in listdir(f'{input_dir}/he') if not file.startswith('.')])
        if label_dir != None:
            self.water_ids = natsort.natsorted([splitext(file)[0] for file in listdir(f'{label_dir}/water') if not file.startswith('.')])
            self.bone_ids = natsort.natsorted([splitext(file)[0] for file in listdir(f'{label_dir}/bone') if not file.startswith('.')])

        logging.info(f'Creating dataset with {len(self.he_ids)} examples')

        if kVps == (80, 120):
            spectrum_path = r'.'
            with open(f'{spectrum_path}/80kVp_0.4mmCu_1.0mmAl.spc', 'r') as f:
                spec1 = Spectrum.fromText(f.read(), 'eV')
            with open(f'{spectrum_path}/120kVp_0.4mmCu_1.0mmAl.spc', 'r') as f:
                spec2 = Spectrum.fromText(f.read(), 'eV')
        elif kVps == (80, 140):
            spectrum_path = r'./spektr'
            with open(f'{spectrum_path}/0.4mmCu_1.0mmAl_80kVp.txt', 'r') as f:
                spec1 = Spectrum.fromText(f.read(), 'keV')
            with open(f'{spectrum_path}/0.4mmCu_1.0mmAl_140kVp.txt', 'r') as f:
                spec2 = Spectrum.fromText(f.read(), 'keV')

        omega1, omega2 = spec1.omega[kVps[0]//10: kVps[0]], spec2.omega[kVps[1]//10: kVps[1]]
        energy1, energy2 = np.arange(kVps[0]//10, kVps[0]), np.arange(kVps[1]//10, kVps[1])

        if material_basis == ('w', 'b'):
            self.atten_water = Atten.fromBuiltIn('Water', 1.0)
            self.atten_bone = Atten.fromBuiltIn('Bone', 1.92)
            L_water = np.arange(0, 300, 5)  # mm
            L_bone = np.arange(0, 110, 5)  # mm
        elif material_basis == ('w', 'i'):
            self.atten_water = Atten.fromBuiltIn('Water', 1.0)  # VNC g/mL
            self.atten_bone = Atten.fromBuiltIn('I', 1.0)  # pure I g/mL
            L_water = np.arange(0, 300, 5)  # mm
            L_bone = np.arange(0, 2, 0.1)  # mm

        water_list, bone_list, P_LE_list, P_HE_list = [], [], [], []

        for l_water in L_water:
            for l_bone in L_bone:
                water_list.append(l_water), bone_list.append(l_bone)

                energy_inf = (omega1 * energy1).reshape(-1, 1) * \
                             np.exp(-self.atten_water.mu[kVps[0]//10: kVps[0]].reshape(-1, 1) * l_water -
                                    self.atten_bone.mu[kVps[0]//10: kVps[0]].reshape(-1, 1) * l_bone)
                P_L = np.log(sum(omega1 * energy1)) - np.log(sum(energy_inf) + 1e-5)
                P_LE_list.append(P_L)

                energy_inf = (omega2 * energy2).reshape(-1, 1) * \
                             np.exp(-self.atten_water.mu[kVps[1]//10: kVps[1]].reshape(-1, 1) * l_water -
                                    self.atten_bone.mu[kVps[1]//10: kVps[1]].reshape(-1, 1) * l_bone)
                P_L = np.log(sum(omega2 * energy2)) - np.log(sum(energy_inf) + 1e-5)
                P_HE_list.append(P_L)

        Water, Bone, P_LE, P_HE = np.array(water_list), np.array(bone_list), np.array(P_LE_list), np.array(P_HE_list)

        S = np.array([P_LE, P_HE, P_LE ** 2, P_LE * P_HE, P_HE ** 2]).T  # (x, 5)
        self.water_coef = np.squeeze(np.linalg.pinv(S) @ water_list)
        self.bone_coef = np.squeeze(np.linalg.pinv(S) @ bone_list)

    def __len__(self):
        return int(len(self.he_ids) * self.use_ratio)

    def __getitem__(self, i):
        sgm_le = tifffile.imread(f'{self.input_dir}/le/{self.le_ids[i]}.tif')
        sgm_he = tifffile.imread(f'{self.input_dir}/he/{self.he_ids[i]}.tif')
        if self.label_dir is not None:
            sgm_water = tifffile.imread(f'{self.label_dir}/water/{self.water_ids[i]}.tif')
            sgm_bone = tifffile.imread(f'{self.label_dir}/bone/{self.bone_ids[i]}.tif')
            

        noisy_sgm_list, clean_sgm_list = [], []
        for postlog_energy, postlog_sgm in zip(["le", "he"], [sgm_le, sgm_he]):
            postlog_sgm[postlog_sgm > 9] = 9  # AAPM abdomen
            # postlog_sgm[postlog_sgm > 7.5] = 7.5  # UIH abdomen
            postlog_sgm[postlog_sgm < 0] = 0
            clean_postlog_sgm = postlog_sgm.copy()

            if self.dose == -1:  # no noise
                postlog_sgm = clean_postlog_sgm
                real_dose = self.dose

            else:  # one or mixed noise level
                if isinstance(self.dose, list):
                    real_dose = self.dose[np.random.randint(0, len(self.dose))]
                elif isinstance(self.dose, (int, float)):
                    real_dose = self.dose
                else:
                    raise Exception('Wrong dose input!')

                if self.fixed_noise is False:  # generate noisy sgm online
                    proj = np.random.poisson(real_dose * np.exp(-postlog_sgm))
                    if np.min(proj) <= 0:
                        proj[proj <= 0] = real_dose * np.exp(-postlog_sgm)[proj <= 0]   # no Poisson and add 1

                    air = np.ones_like(proj) * real_dose
                    postlog_sgm = flatDarkFieldCorrection(proj, air)

                else:  # generate noisy sgm offline
                    dose = format(real_dose, '.0e').replace('+0', '')
                    if postlog_energy == "le":
                        postlog_sgm = tifffile.imread(f'{self.input_dir}_noise/{dose}/le/{self.le_ids[i]}.tif')
                    elif postlog_energy == "he":
                        postlog_sgm = tifffile.imread(f'{self.input_dir}_noise/{dose}/he/{self.he_ids[i]}.tif')

            H, W = postlog_sgm.shape
            scale = 2 ** self.num_layers

            H_ = (H + scale-1) // scale * scale
            W_ = (W + scale-1) // scale * scale
            if self.H_ is not None:
                H_, W_ = self.H_, self.W_

            noisy_postlog_pad = sgmPadding(postlog_sgm, targetHW=(H_, W_), style='wrap')
            clean_postlog_pad = sgmPadding(clean_postlog_sgm, targetHW=(H_, W_), style='wrap')

            noisy_sgm_list.append(noisy_postlog_pad)
            clean_sgm_list.append(clean_postlog_pad)

        P_LE, P_HE = noisy_sgm_list
        if self.A is None:
            SS = np.array([P_LE, P_HE, P_LE ** 2, P_LE * P_HE, P_HE ** 2])  # (5, 736, 800)
            noisy_water = np.einsum('i,ijk->jk', self.water_coef, SS)
            noisy_bone = np.einsum('i,ijk->jk', self.bone_coef, SS)
        else:
            noisy_water, noisy_bone = np.einsum('ij,jxy->ixy', np.linalg.pinv(self.A), np.array([P_LE, P_HE]))

        # print(f'Input image ({H}, {W}) padding to ({H_}, {W_})')

        ### Input
        full_noisy_sgm = torch.FloatTensor(np.array(noisy_sgm_list, dtype=np.float32))
        full_clean_sgm = torch.FloatTensor(np.array(clean_sgm_list, dtype=np.float32))
        full_noisy_material = torch.FloatTensor(np.array([noisy_water, noisy_bone], dtype=np.float32))

        noisy_sgm, noisy_material = full_noisy_sgm.clone(), full_noisy_material.clone()

        if self.mode == 'train':
            combine = torch.vstack([noisy_sgm, noisy_material])
            combine = randomProcess(combine, size=(512, 512), p=0.5)
            noisy_sgm, noisy_material = combine.reshape(2, 2, 512, 512)

        return_dict = {
            'noisy_sgm': noisy_sgm,
            'noisy_material': noisy_material,
            'full_clean_sgm': full_clean_sgm,
            # 'full_clean_material': full_clean_material,  # => rat
            'full_noisy_sgm': full_noisy_sgm,
            'full_noisy_material': full_noisy_material,
            'real_dose': real_dose
        }

        if self.prior_dir is not None:
            dose = format(real_dose, '.0e').replace('+0', '')
            water_prior = tifffile.imread(f'{self.prior_dir}/{dose}/{self.water_ids[i]}.tif')
            return_dict['water_prior'] = torch.FloatTensor(water_prior)
        # if self.label_dir is not None:
        #     clean_material_list = [sgmPadding(clean_material_pad, targetHW=(H_, W_), style='wrap')
        #                            for clean_material_pad in [sgm_water, sgm_bone]]
        #     full_clean_material = torch.FloatTensor(np.array(clean_material_list, dtype=np.float32))
        #     return_dict['full_clean_material'] = full_clean_material

        return return_dict
