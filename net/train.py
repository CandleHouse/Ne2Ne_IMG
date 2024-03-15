import logging
import torch
from loss import FIRELoss
from data import DatasetLoad
from models import Ne2Ne_UNet
from torch.utils.data import DataLoader, random_split
import tqdm
import logging
from tqdm import tqdm
import torch
from utils import *
from torch_mando import *
from kornia.losses import *
from crip.physics import *
import random

# new dataset
input_dir = r'./sgm'
label_dir = r'./L'
prior_dir = r'./prior'
new_dir = r'.'
dir_checkpoint = f'{new_dir}/'  # every epoch
# final_checkpoint = './checkpoints/'  # the last epoch


def eval_net(net1, loader, criterion, water_coef, bone_coef, device):
    """Evaluation without the densecrf with the dice coefficient"""
    criterion_fire, criterion_similar = criterion
    net1.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:

            full_noisy_sgm = batch['full_noisy_sgm'].to(device)
            # full_noisy_material = batch['full_noisy_material'].to(device)

            A = torch.FloatTensor([[atten_water.mu[54], atten_bone.mu[54]],
                                    [atten_water.mu[73], atten_bone.mu[73]]]).to(device)
            cfg.pixelSize = 0.8
            cfg.reconKernelParam = 0.25
            
            with torch.no_grad():
                noisy_rec = batch_recon(full_noisy_sgm, cfg)
                noisy_map = sgmToMaterial(noisy_rec, water_coef, bone_coef, A)
                # noisy_map = batch_recon(full_noisy_material, cfg)

                # random process of train dataset
                # full_maskHoles_sgm = torch.ones_like(full_noisy_sgm).to(device)
                combine = torch.stack([noisy_rec, noisy_map])
                combine = randomProcess(combine, size=(256, 256), p=0.5)
                noisy_rec, noisy_map = combine.reshape(2, 16, 2, 256, 256)
                
                ### fire
                mask1_img, mask2_img, mask3_img, mask4_img = generate_mask_pair4(noisy_map)
                map1 = generate_subimages(noisy_map, mask1_img)
                map2 = generate_subimages(noisy_map, mask2_img)
                map3 = generate_subimages(noisy_map, mask3_img)
                map4 = generate_subimages(noisy_map, mask4_img)
                
                rec1 = generate_subimages(noisy_rec, mask1_img)
                rec2 = generate_subimages(noisy_rec, mask2_img)
                rec3 = generate_subimages(noisy_rec, mask3_img)
                rec4 = generate_subimages(noisy_rec, mask4_img)

                f_rec1 = net1(map1, rec1)
                f_map1 = sgmToMaterial(f_rec1, water_coef, bone_coef, A)
                with torch.no_grad():
                    X_rec = net1(noisy_map, noisy_rec)
                    X_map = sgmToMaterial(X_rec, water_coef, bone_coef, A)

                x_map1 = generate_subimages(X_map, mask1_img)
                x_map2 = generate_subimages(X_map, mask2_img)
                x_map3 = generate_subimages(X_map, mask3_img)
                x_map4 = generate_subimages(X_map, mask4_img)
                
                # x_rec1 = generate_subimages(X_rec, mask1_img)
                # x_rec2 = generate_subimages(X_rec, mask2_img)
                # x_rec3 = generate_subimages(X_rec, mask3_img)
                # x_rec4 = generate_subimages(X_rec, mask4_img)
                
                ### losses
                loss1 = criterion_fire(f_map1, map2, map3, map4,
                                       x_map1, x_map2, x_map3, x_map4, epoch_ratio=1)
                # loss1 = criterion_sgm(f_rec1, rec2, x_rec1, x_rec2, epoch_ratio=1)

                loss = loss1

            tot += loss.item()

            pbar.update(full_noisy_sgm.shape[0])

    # net.train()
    return None, tot / n_val


def train_net(net1, device, epochs=5, batch_size=16, lr=1e-4, val_percent=0.2, save_cp=True, use_ratio=1.0):
    # 1. Data prepare and divide
    dataset = DatasetLoad(input_dir, label_dir, mode='train_', use_ratio=use_ratio, kVps=(80, 140),
                          dose=1e5, prior_dir=None, fixed_noise=False, material_basis=('w', 'i'))
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    # - prepare tensorboard
    # writer = SummaryWriter(log_dir=f'{new_dir}/runs', comment=f'_BS_{batch_size}_LR_{lr}_EPOCH_{epochs}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # 2. Select optimizer and criterion
    optimizer = torch.optim.Adam([
        {'params': net1.parameters(), 'lr': lr, 'weight_decay': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8},
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 80, 120, 160], gamma=0.5)

    criterion_fire = FIRELoss(beta=1, gamma=1)
    criterion_similar = torch.nn.MSELoss()

    # 3. Train
    for epoch in range(epochs):
        net1.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='proj') as pbar:
            for batch in train_loader:
                ### load sgm
                full_noisy_sgm = batch['full_noisy_sgm'].to(device)

                A = torch.FloatTensor([[atten_water.mu[54], atten_bone.mu[54]],
                                       [atten_water.mu[73], atten_bone.mu[73]]]).to(device)
                cfg.pixelSize = random.choice([0.4, 0.5, 0.6, 0.7, 0.8])
                
                with torch.no_grad():
                    noisy_rec = batch_recon(full_noisy_sgm, cfg)
                    noisy_map = sgmToMaterial(noisy_rec, dataset.water_coef, dataset.bone_coef, A)
                
                # random process of train dataset
                combine = torch.stack([noisy_rec, noisy_map])
                combine = randomProcess(combine, size=(256, 256), p=0.5)
                noisy_rec, noisy_map = combine.reshape(2, batch_size, 2, 256, 256)
                    
                ### fire
                mask1_img, mask2_img, mask3_img, mask4_img = generate_mask_pair4(noisy_map)
                map1 = generate_subimages(noisy_map, mask1_img)
                map2 = generate_subimages(noisy_map, mask2_img)
                map3 = generate_subimages(noisy_map, mask3_img)
                map4 = generate_subimages(noisy_map, mask4_img)
                
                rec1 = generate_subimages(noisy_rec, mask1_img)
                rec2 = generate_subimages(noisy_rec, mask2_img)
                rec3 = generate_subimages(noisy_rec, mask3_img)
                rec4 = generate_subimages(noisy_rec, mask4_img)

                f_rec1 = net1(map1, rec1)
                f_map1 = sgmToMaterial(f_rec1, dataset.water_coef, dataset.bone_coef, A)
                with torch.no_grad():
                    X_rec = net1(noisy_map, noisy_rec)
                    X_map = sgmToMaterial(X_rec, dataset.water_coef, dataset.bone_coef, A)

                x_map1 = generate_subimages(X_map, mask1_img)
                x_map2 = generate_subimages(X_map, mask2_img)
                x_map3 = generate_subimages(X_map, mask3_img)
                x_map4 = generate_subimages(X_map, mask4_img)
                
                # x_rec1 = generate_subimages(X_rec, mask1_img)
                # x_rec2 = generate_subimages(X_rec, mask2_img)
                # x_rec3 = generate_subimages(X_rec, mask3_img)
                # x_rec4 = generate_subimages(X_rec, mask4_img)

                ### losses
                # loss1 = criterion_fire(f_map1[:, 0], map2[:, 0], map3[:, 0], map4[:, 0],
                #                        x_map1[:, 0], x_map2[:, 0], x_map3[:, 0], x_map4[:, 0], epoch_ratio=epoch/epochs)
                loss2 = criterion_fire(f_map1[:, 1], map2[:, 1], map3[:, 1], map4[:, 1],
                                       x_map1[:, 1], x_map2[:, 1], x_map3[:, 1], x_map4[:, 1], epoch_ratio=epoch/epochs)
                
                loss1 = criterion_fire(f_map1, map2, map3, map4,
                                       x_map1, x_map2, x_map3, x_map4, epoch_ratio=1)
                # loss = loss1 + loss2 * 100
                loss = loss1

                epoch_loss += loss.item()

                # if epoch > 0:
                #     writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss1 ': loss1.item(), 'loss2 ': (loss2*1000).item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(full_noisy_sgm.shape[0])  # update batch size forward
                global_step += 1

        # Save checkpoint for every epoch
        save_model = {
            'net1': net1.state_dict(),
        }
        torch.save(save_model, dir_checkpoint + f'epoch_{epoch}.pth')

        # epoch loss average
        # writer.add_scalar('Epoch Loss/train', epoch_loss / len(train_loader), epoch)
        
        # 4. Validation
        val_labels_pred, val_score = eval_net(net1, val_loader,
                                              [criterion_fire, criterion_similar],
                                              dataset.water_coef, dataset.bone_coef, device)
        scheduler.step()
        logging.info('Validation Loss: {}'.format(val_score))

        # - tensorboard record
        # for tag, value in net.named_parameters():
        #     tag = tag.replace('.', '/')
        #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
        #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
        # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # writer.add_scalar('Loss/test', val_score, epoch)
        # writer.add_images('val label/true', val_labels_true, epoch)
        # writer.add_images('val label/pred', val_labels_pred, epoch)

    # 5. Save checkpoints
    # if save_cp:
    #     torch.save(net.state_dict(), final_checkpoint + f'CP_BS_{batch_size}_LR_{lr}_EPOCH_{epochs}_0222.pth')
    #     logging.info(f'Checkpoint saved !')


if __name__ == '__main__':
    # -1. Select device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    cfg = MandoFanBeamConfig(imgDim=512, pixelSize=0.8, sid=800, sdd=1200, detEltCount=800, detEltSize=0.8,
                             views=720, reconKernelEnum=KERNEL_GAUSSIAN_RAMP, reconKernelParam=0.25, fpjStepSize=0.2)  # UIH
    atten_water = Atten.fromBuiltIn('Water', 1.0)
    atten_bone = Atten.fromBuiltIn('I', 1.0)
    # 0. Select model and initialize
    model1 = Ne2Ne_UNet(in_channels=2, out_channels=2, init_features=32, kernel_size=(3, 3), padding=(1, 1)).to(device)

    # 1-5 steps in train_net function
    train_net(net1=model1, epochs=500, batch_size=16, lr=.001, device=device, val_percent=0.3, use_ratio=1.0)
