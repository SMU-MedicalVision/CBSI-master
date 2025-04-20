import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.utils import make_grid
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Nii_utils import dice_coefficient, NiiDataWrite, NiiDataRead, compute_mae



def Model_Validation_multitask(args, epoch, trainer, dataloader, dataset, save_dir=None, writer=None, ref_timestep=1000, inf_bs=24, train=True, save_image=True):
    """
    save_image:

    """

    trainer.model.eval()
    count = 0
    Metric_df = pd.DataFrame({'ID': [], 'modal': [], 'MAE': [], 'ROI_MAE':[], 'SSIM': [], 'PSNR': [], 'DICE': []})
    start_time = time.time()
    with torch.no_grad():
        for DATA in tqdm(dataloader[dataset]):
            count += 1
            ID = DATA['ID']
            target_img, cond_img = np.array(DATA['B']), np.array(DATA['A'])
            label = DATA['label']
            target_seg = np.array(DATA['tumor_mask'])
            brain_mask = np.array(DATA['brain_mask'])

            brain_mask[brain_mask != 0] = 1
            if args.data_dim == '2D':
                cond_img = cond_img.squeeze(0)
                ID = ID[0]
                label = label[0]
                target_img = target_img.squeeze(0)
                brain_mask = brain_mask.squeeze(0)
                brain_mask[brain_mask != 0] = 1
                target_seg = target_seg.squeeze(0)

                original_shape = DATA['B'].squeeze(0).shape
                final_syn_volume = np.zeros(original_shape)
                final_seg_volume = np.zeros(original_shape)
                if len(original_shape) == 4:
                    brain_mask = np.repeat(brain_mask[np.newaxis, ...], 3, axis=0)

                # =============== Multi-slice inference ===============
                Val_bs = inf_bs
                n_num = original_shape[-3] // Val_bs
                n_num = n_num + 0 if original_shape[-3] % Val_bs == 0 else n_num + 1  # Calculate the number of inferences
                for n in range(n_num):
                    print(f'{n + 1}/{n_num}', end=' || ')
                    if n == n_num - 1:
                        cond_img_one = cond_img[n * Val_bs:, ...]
                    else:
                        cond_img_one = cond_img[n * Val_bs: (n + 1) * Val_bs, ...]
                    outputs, outputs_seg = trainer.pred(torch.from_numpy(cond_img_one).float().cuda(), T=ref_timestep,
                                                 seg_cond=None, label=np.repeat(label, cond_img_one.shape[0], axis=0).int().cuda())
                    if len(original_shape) == 3:
                        if n == n_num - 1:
                            final_syn_volume[n * Val_bs:] = outputs[:, 0, :, :].cpu().numpy()
                            final_seg_volume[n * Val_bs:] = outputs_seg[:, 0, :, :].cpu().numpy()
                        else:
                            final_syn_volume[n * Val_bs: (n + 1) * Val_bs] = outputs[:, 0, :, :].cpu().numpy()
                            final_seg_volume[n * Val_bs: (n + 1) * Val_bs] = outputs_seg[:, 0, :, :].cpu().numpy()

            final_syn_volume[final_syn_volume > 1] = 1
            final_syn_volume[final_syn_volume < -1] = -1

            # =============== visualize ===============
            if train:
                if count == 1:  # visualize the first case as a sample
                    if len(original_shape) == 3:
                        writer[dataset].add_image('synthetic', make_grid(torch.tensor((final_syn_volume[0::4] + 1) / 2 * 255).unsqueeze(1), 2, normalize=True), epoch)
                        writer[dataset].add_image('seg', make_grid(torch.tensor(final_seg_volume[0::4] * 255).unsqueeze(1), 2, normalize=True), epoch)
                    if epoch == 0:  # visualize the real T1Gd and tumor mask once
                        if len(original_shape) == 3:
                            writer[dataset].add_image('real', make_grid(torch.tensor((target_img[0::4] + 1) / 2 * 255).unsqueeze(1), 2, normalize=True), epoch)
                            writer[dataset].add_image('mask', make_grid(torch.tensor(target_seg[0::4] * 255).to(torch.float32).unsqueeze(1), 2, normalize=True), epoch)
            target_img = np.clip((target_img+1)*255/2, 0, 255)
            final_syn_volume = np.clip((final_syn_volume+1)*255/2, 0, 255)
            final_syn_volume[brain_mask == 0] = args.MR_min
            final_seg_volume[brain_mask == 0] = 0
            try:
                data_range = args.MR_max - args.MR_min
                MAE = compute_mae(final_syn_volume, target_img, mask=brain_mask)
                ROI_MAE = compute_mae(final_syn_volume, target_img, mask=target_seg)
                SSIM = compare_ssim(final_syn_volume[brain_mask > 0], target_img[brain_mask > 0], data_range=data_range)
                PSNR = compare_psnr(final_syn_volume[brain_mask > 0], target_img[brain_mask > 0], data_range=data_range)
                final_seg_volume[final_seg_volume >= 0.5] = 1
                final_seg_volume[final_seg_volume < 0.5] = 0
                DICE = dice_coefficient(final_seg_volume, target_seg)
                Metric_df = Metric_df._append({'ID': ID, 'modal': int(label), 'MAE': MAE, 'ROI_MAE': ROI_MAE, 'SSIM': SSIM, 'PSNR': PSNR, 'DICE': DICE}, ignore_index=True)
            except:
                print('ID:', ID, 'Attention! Error Metrics !')

            if not train and save_image:
                print(f"MAE {MAE} save image")
                samlple_parameter = np.array(DATA['image_parameter'][0][0]), np.array(torch.cat(DATA['image_parameter'][1], dim=0)), np.array(torch.cat(DATA['image_parameter'][2], dim=0))
                os.makedirs(os.path.join(save_dir, dataset, f'ET-{int(label)}'), exist_ok=True)
                NiiDataWrite(os.path.join(save_dir, dataset, f'ET-{int(label)}', f'{ID}.nii.gz'), final_syn_volume, *samlple_parameter)
                NiiDataWrite(os.path.join(save_dir, dataset, f'ET-{int(label)}', f'{ID}_seg.nii.gz'), final_seg_volume, *samlple_parameter)

        mMAE = np.mean(Metric_df['MAE'])
        mROI_MAE = np.mean(Metric_df['ROI_MAE'])
        mSSIM = np.mean(Metric_df['SSIM'])
        mPSNR = np.mean(Metric_df['PSNR'])
        mDICE = np.mean(Metric_df['DICE'])
        Metric_df = Metric_df._append({'ID': 'mean', 'modal': '01', 'MAE': mMAE, 'ROI_MAE': mROI_MAE, 'SSIM': mSSIM, 'PSNR': mPSNR, 'DICE': mDICE}, ignore_index=True)

        mMAE = torch.tensor(mMAE).cuda()
        mROI_MAE = torch.tensor(mROI_MAE).cuda()
        mSSIM = torch.tensor(mSSIM).cuda()
        mPSNR = torch.tensor(mPSNR).cuda()
        mDICE = torch.tensor(mDICE).cuda()

        if train:
            writer[dataset].add_scalar('MAE', mMAE, epoch)
            writer[dataset].add_scalar('ROI_MAE', mROI_MAE, epoch)
            writer[dataset].add_scalar('SSIM', mSSIM, epoch)
            writer[dataset].add_scalar('PSNR', mPSNR, epoch)
            writer[dataset].add_scalar('DICE', mDICE, epoch)
    if train:
        writer[dataset].add_scalar('time', (time.time() - start_time) / 60, epoch)
    return mMAE, ROI_MAE, mSSIM, mPSNR, mDICE, Metric_df
