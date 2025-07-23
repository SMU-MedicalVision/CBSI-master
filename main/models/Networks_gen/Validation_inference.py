import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from torchvision.utils import make_grid
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Nii_utils import harmonize_mr_reverse, NiiDataWrite, compute_mae



def Model_Validation_multitask(opt, epoch, trainer, loader, dataset, save_dir=None, writer=None, ref_timestep=1000, train=True, save_image=True):
    """
    Perform validation for a multi-task diffusion model that outputs both synthetic images and segmentation maps.

    Args:
        args: Namespace of training/testing parameters.
        epoch: Current epoch index.
        trainer: The DDPM trainer object containing model and inference logic.
        loader: Dictionary of dataset-specific dataloaders.
        dataset: Name/key of the dataset to validate on.
        save_dir: Path to save validation results (only used in test mode).
        writer: TensorBoard SummaryWriter object (used for logging images and metrics).
        ref_timestep: Number of diffusion timesteps used during sampling.
        inf_bs: Inference batch size for slicing long image volumes.
        train: Whether this is a training-time validation (enables visualization).
        save_image: Whether to save output images (only enabled when train=False).

    Returns:
        mMAE, mROI_MAE, mSSIM, mPSNR: Mean evaluation metrics over the validation set.
        Metric_df: DataFrame storing per-sample metrics.
    """

    count = 0
    Metric_df = pd.DataFrame({'ID': [], 'modal': [], 'MAE': [], 'ROI_MAE':[], 'SSIM': [], 'PSNR': []})
    start_time = time.time()
    trainer.model.eval()
    with torch.no_grad():
        for DATA in tqdm(loader[dataset]):
            count += 1
            ID = DATA['ID']
            target_img, cond_img = np.array(DATA['B']), np.array(DATA['A'])
            label = DATA['label']
            target_seg = np.array(DATA['tumor_mask'])
            brain_mask = np.array(DATA['brain_mask'])

            brain_mask[brain_mask != 0] = 1

            # Processing for 2D inputs
            if opt.data_dim == '2D':
                cond_img = cond_img.squeeze(0)
                ID = ID[0]
                label = label[0]
                target_img = target_img.squeeze(0)
                brain_mask = brain_mask.squeeze(0)
                brain_mask[brain_mask != 0] = 1
                target_seg = target_seg.squeeze(0)

                original_shape = DATA['B'].squeeze(0).shape
                final_syn_volume = np.zeros(original_shape)
                if len(original_shape) == 4:
                    brain_mask = np.repeat(brain_mask[np.newaxis, ...], 3, axis=0)

                # =============== Multi-slice inference ===============
                n_num = original_shape[-3] // opt.val_bs
                n_num = n_num + 0 if original_shape[-3] % opt.val_bs == 0 else n_num + 1  # Calculate the number of inferences
                for n in range(n_num):
                    print(f'{n + 1}/{n_num}', end=' || ')
                    if n == n_num - 1:
                        cond_img_one = cond_img[n * opt.val_bs:, ...]
                    else:
                        cond_img_one = cond_img[n * opt.val_bs: (n + 1) * opt.val_bs, ...]

                    # Generate synthetic image and segmentation
                    outputs, _ = trainer.pred(torch.from_numpy(cond_img_one).float().to(opt.device), T=ref_timestep,
                                                 seg_cond=None, label=np.repeat(label, cond_img_one.shape[0], axis=0).int().to(opt.device))

                    # Assemble full volume from slices
                    if len(original_shape) == 3:
                        if n == n_num - 1:
                            final_syn_volume[n * opt.val_bs:] = outputs[:, 0, :, :].cpu().numpy()
                        else:
                            final_syn_volume[n * opt.val_bs: (n + 1) * opt.val_bs] = outputs[:, 0, :, :].cpu().numpy()

            # Clip the intensity of synthetic images to valid range
            final_syn_volume[final_syn_volume > 1] = 1
            final_syn_volume[final_syn_volume < -1] = -1

            # ========== Visualization for the first sample ==========
            if train:
                if count == 1:  # visualize the first case as a sample
                    nrow = target_seg[0::4].shape[0]
                    if len(original_shape) == 3:
                        writer[dataset].add_image('synthetic', make_grid(torch.tensor((final_syn_volume[0::4] + 1) / 2 * 255).unsqueeze(1), nrow, normalize=True), epoch)
                    if epoch == 0:  # visualize the real T1Gd and tumor mask once
                        if len(original_shape) == 3:
                            writer[dataset].add_image('real', make_grid(torch.tensor((target_img[0::4] + 1) / 2 * 255).unsqueeze(1), nrow, normalize=True), epoch)

            # Denormalize images to [0, 255] for evaluation
            target_img = np.clip((target_img+1)*255/2, 0, 255)
            final_syn_volume = np.clip((final_syn_volume+1)*255/2, 0, 255)
            final_syn_volume[brain_mask == 0] = opt.MR_min

            # ========== Compute evaluation metrics ==========
            try:
                data_range = opt.MR_max - opt.MR_min
                MAE = compute_mae(final_syn_volume, target_img, mask=brain_mask)
                ROI_MAE = compute_mae(final_syn_volume, target_img, mask=target_seg)
                SSIM = compare_ssim(final_syn_volume[brain_mask > 0], target_img[brain_mask > 0], data_range=data_range)
                PSNR = compare_psnr(final_syn_volume[brain_mask > 0], target_img[brain_mask > 0], data_range=data_range)


                # Store metrics for the current case
                Metric_df = Metric_df._append({'ID': ID, 'modal': int(label),
                                               'MAE': MAE, 'ROI_MAE': ROI_MAE,
                                               'SSIM': SSIM, 'PSNR': PSNR}, ignore_index=True)
            except:
                print('ID:', ID, 'Attention! Error Metrics !')

            # ========== Save results if in test mode ==========
            if not train and save_image:
                print(f"MAE {MAE} save image")
                samlple_parameter = np.array(DATA['image_parameter'][0][0]), np.array(torch.cat(DATA['image_parameter'][1], dim=0)), np.array(torch.cat(DATA['image_parameter'][2], dim=0))
                os.makedirs(os.path.join(save_dir, dataset, f'ET-{int(label)}'), exist_ok=True)
                NiiDataWrite(os.path.join(save_dir, dataset, f'ET-{int(label)}', f'{ID}.nii.gz'), final_syn_volume, *samlple_parameter)

        # ========== Compute and log mean metrics ==========
        mMAE = np.mean(Metric_df['MAE'])
        mROI_MAE = np.mean(Metric_df['ROI_MAE'])
        mSSIM = np.mean(Metric_df['SSIM'])
        mPSNR = np.mean(Metric_df['PSNR'])
        Metric_df = Metric_df._append({'ID': 'mean', 'modal': '01', 'MAE': mMAE, 'ROI_MAE': mROI_MAE, 'SSIM': mSSIM, 'PSNR': mPSNR}, ignore_index=True)

        mMAE = torch.tensor(mMAE).to(opt.device)
        mROI_MAE = torch.tensor(mROI_MAE).to(opt.device)
        mSSIM = torch.tensor(mSSIM).to(opt.device)
        mPSNR = torch.tensor(mPSNR).to(opt.device)

        # ========== Write to TensorBoard ==========
        if train:
            writer[dataset].add_scalar('gen_MAE', mMAE, epoch)
            writer[dataset].add_scalar('gen_ROI_MAE', mROI_MAE, epoch)
            writer[dataset].add_scalar('gen_SSIM', mSSIM, epoch)
            writer[dataset].add_scalar('gen_PSNR', mPSNR, epoch)
    if train:
        writer[dataset].add_scalar('gen_time', (time.time() - start_time) / 60, epoch)

    return mMAE, ROI_MAE, mSSIM, mPSNR, Metric_df



def Model_Inference_multitask(opt, loader, trainer, dataset, label_known=False):
    if label_known:
        label_list = []
    for i, DATA in enumerate(tqdm(loader)):
        ID = DATA['ID'][0]
        print(f'Processing {ID}')
        for syn_label in torch.tensor([[0], [1]]).long():
            brain_mask = DATA['brain_mask']
            cond_img = DATA['A']
            syn_label = syn_label[0]
            if label_known:
                label = DATA['label'][0]
                label_list.append({'synET_ID': f'ET-{syn_label.item()}/{ID}', 'ID':ID, 'et_label':label.item(), 'cls_label': (syn_label.item() == label.item()) * 1})
            # Processing for 2D inputs
            if os.path.exists(join(opt.save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn', f'ET-{int(syn_label)}', f'{ID}.nii.gz')):
                continue
            cond_img = cond_img.squeeze(0)
            brain_mask = brain_mask.squeeze(0)
            brain_mask[brain_mask != 0] = 1
            original_shape = brain_mask.shape
            final_syn_volume = np.zeros(original_shape)
            # final_seg_volume = np.zeros(original_shape)
            if len(original_shape) == 4:
                brain_mask = np.repeat(brain_mask[np.newaxis, ...], 3, axis=0)

            # =============== Multi-slice inference ===============
            n_num = original_shape[-3] // opt.val_bs
            n_num = n_num + 0 if original_shape[-3] % opt.val_bs == 0 else n_num + 1  # Calculate the number of inferences
            for n in range(n_num):
                print(f'{n + 1}/{n_num}', end=' || ')
                if n == n_num - 1:
                    cond_img_one = cond_img[n * opt.val_bs:, ...]
                else:
                    cond_img_one = cond_img[n * opt.val_bs: (n + 1) * opt.val_bs, ...]

                # Generate synthetic image and segmentation
                outputs, _ = trainer.pred(cond_img_one.to(opt.device),
                                          T=opt.ref_timestep,
                                          seg_cond=None,
                                          label=np.repeat(syn_label, cond_img_one.shape[0], axis=0).int().to(
                                              opt.device))

                # Assemble full volume from slices
                if len(original_shape) == 3:
                    if n == n_num - 1:
                        final_syn_volume[n * opt.val_bs:] = outputs[:, 0, :, :].cpu().numpy()
                        # final_seg_volume[n * opt.val_bs:] = outputs_seg[:, 0, :, :].cpu().numpy()
                    else:
                        final_syn_volume[n * opt.val_bs: (n + 1) * opt.val_bs] = outputs[:, 0, :, :].cpu().numpy()
                        # final_seg_volume[n * opt.val_bs: (n + 1) * opt.val_bs] = outputs_seg[:, 0, :, :].cpu().numpy()

            # Clip the intensity of synthetic images to valid range
            final_syn_volume[final_syn_volume > 1] = 1
            final_syn_volume[final_syn_volume < -1] = -1

            # Denormalize images to [0, 255] for evaluation
            final_syn_volume = harmonize_mr_reverse(final_syn_volume)
            final_syn_volume[brain_mask == 0] = opt.MR_min

            # ========== Save results if in test mode ==========
            samlple_parameter = np.array(DATA['image_parameter'][0][0]), np.array(
                torch.cat(DATA['image_parameter'][1], dim=0)), np.array(torch.cat(DATA['image_parameter'][2], dim=0))
            os.makedirs(join(opt.save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn', f'ET-{int(syn_label)}'), exist_ok=True)
            NiiDataWrite(join(opt.save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn', f'ET-{int(syn_label)}', f'{ID}.nii.gz'), final_syn_volume, *samlple_parameter)
            # NiiDataWrite(join(opt.save_dir, 'syn',  f'ET-{int(label)}', f'{ID}_seg.nii.gz'), final_seg_volume, *samlple_parameter)

    if label_known:
        df = pd.DataFrame(label_list).astype(str)
        df.to_csv(join(opt.save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn', 'label_data.csv'), index=False)