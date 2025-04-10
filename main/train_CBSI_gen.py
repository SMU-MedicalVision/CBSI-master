import os
import time
import torch
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# ======== Import custom model, training utilities, dataset loader, and helpers ========
from Networks_UNet_DDPM import Unet_class
from Networks_DDPM_trainer import GaussianDiffusion, DDPM_Trainer
from dataset import CA_Dataset_harmonize_2D
from Nii_utils import NiiDataRead, NiiDataWrite, harmonize_mr, compute_mae



def main(args):
    # ======== Setup saving path and logging ========
    save_name = 'bs{}_epoch{}_gae{}_seed{}_class'.format(args.bs, args.epoch, args.gae, args.seed)
    save_dir = os.path.join(
        'trained_models/DDPM/{}_{}_{}_condition_act_{}_pretrain_{}'.format(args.objective, args.net, args.loss,
                                                                           args.activate, args.pretrain), save_name)

    os.makedirs(save_dir, exist_ok=True)
    train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
    # val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
    print(save_dir)
    # ======== Initialize model and trainer ========
    if args.net.lower() == 'unet_ddpm_64_class':
        net = Unet_class(dim=64, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                         condition_channels=args.cc,
                         condition=True, class_cond=2, resnet_block_groups=8).cuda()
    elif args.net.lower() == 'unet_ddpm_32_class':
        net = Unet_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                         condition_channels=args.cc,
                         condition=True, class_cond=2, resnet_block_groups=8).cuda()
    diffusion_model = GaussianDiffusion(net, image_size=args.ImageSize, timesteps=1000, loss_type=args.loss,
                                        objective=args.objective, activate=args.activate).cuda()
    trainer = DDPM_Trainer(diffusion_model, train_batch_size=args.bs, train_lr=args.lr_max)
    # ======== Learning rate scheduler setup ========
    scheduler_lr = MultiStepLR(trainer.opt, milestones=[int((5 / 10) * args.epoch),
                                                        int((8 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
    # ======== Load pretrained weights if necessary ========
    if args.continue_train or args.pretrain:
        for i in range(args.continue_epoch + 1):
            scheduler_lr.step()
        trainer.load("./trained_models/latest.pth")

    # ======== Load dataset and dataloader ========
    root_dir = './glioma_data/Train_val'
    # test_dir = './glioma_data/Test'
    train_set =  CA_Dataset_harmonize_2D(args, root_dir=root_dir)  # get training options
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=False, num_workers=args.num_threads,
                                  pin_memory=True)

    # ======== Load test subject for validation and saving visualization ========
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    t1, spacing, origin, direction = NiiDataRead('./glioma_data/ID_001/T1.nii.gz')
    t2f, _, _, _ = NiiDataRead('./glioma_data/ID_001/T2F.nii.gz')
    t1c, _, _, _ = NiiDataRead('./glioma_data/ID_001/T1C.nii.gz')
    ROI, _, _, _ = NiiDataRead('./glioma_data/ID_001/ROI.nii.gz', as_type=np.uint8)
    Brain, _, _, _ = NiiDataRead('./glioma_data/ID_001/Brain_mask.nii.gz', as_type=np.uint8)
    t1 = harmonize_mr(t1, args.MR_min, args.MR_max)
    t2f = harmonize_mr(t2f, args.MR_min, args.MR_max)
    t1c = np.clip(t1c, 0, 255)
    NiiDataWrite(os.path.join(save_dir, 'predictions', 'real.nii.gz'), t1c, spacing, origin, direction)

    # ======== Start training loop ========
    for epoch in range(args.epoch):
        start_time = time.time()
        for param_group in trainer.opt.param_groups:
            lr = param_group['lr']
            break
        epoch_train_loss = []
        for i, data in enumerate(train_dataloader):
            # 目标， 原域
            x, x_cond = data['B'].cuda(), data['A'].cuda()
            label = data['label'].cuda()  # 标签
            trainer.opt.zero_grad()
            trainer.model.train()
            loss = trainer.calculate_loss(x, x_cond, label)
            loss_back = loss / args.gae
            loss_back.backward()
            if (i + 1) % args.gae == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1)
                trainer.opt.step()
            epoch_train_loss.append(loss.item())
            if i % 100 == 0:
                print('[%d/%d, %5d/%d] train_loss: %.3f ' % (
                epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
        # ======== Epoch metrics logging and checkpointing ========
        epoch_train_loss = np.mean(epoch_train_loss)
        scheduler_lr.step()
        print('[%d/%d] train_loss: %.3f' % (epoch + 1, args.epoch, epoch_train_loss))
        train_writer.add_scalar('loss', epoch_train_loss, epoch)
        train_writer.add_scalar('lr', lr, epoch)
        print('Train One Epoch Time Taken: %.1f min' % ((time.time() - start_time) / 60))

        # Save model and generate predictions every 10% of epochs
        if (epoch + 1) % (args.epoch // 10) == 0 or epoch == 2:
            trainer.save(os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))
            original_shape = t1.shape
            final_syn_volume = np.zeros(original_shape)

            n_num = original_shape[0] // 24
            n_num = n_num + 0 if original_shape[0] % 24 == 0 else n_num + 1
            for n in range(n_num):
                print(f'{n + 1}/{n_num}', end=' || ')
                if n == n_num - 1:
                    one_t1 = t1[n * 24:, :, :]
                    one_t2f = t2f[n * 24:, :, :]
                else:
                    one_t1 = t1[n * 24: (n + 1) * 24, :, :]
                    one_t2f = t2f[n * 24: (n + 1) * 24, :, :]
                one_t1 = torch.from_numpy(one_t1).unsqueeze(1).float().cuda()
                one_t2f = torch.from_numpy(one_t2f).unsqueeze(1).float().cuda()
                inputs = torch.cat((one_t1, one_t2f), dim=1)
                # label = torch.tensor([1] * inputs.shape[0]).unsqueeze(-1).cuda()
                label = torch.tensor([1] * inputs.shape[0]).cuda()
                outputs = trainer.pred(inputs, T=1000, label=label)
                if n == n_num - 1:
                    final_syn_volume[n * 24:] = outputs[:, 0, :, :].cpu().numpy()
                else:
                    final_syn_volume[n * 24: (n + 1) * 24] = outputs[:, 0, :, :].cpu().numpy()
            syn_pred = final_syn_volume
            syn_pred = (syn_pred + 1) / 2 * 255  # Post-processing
            syn_pred[Brain == 0] = 0
            # syn_pred = np.clip(syn_pred, 0, 255)
            # ======== Evaluate results ========
            MAE = compute_mae(syn_pred, t1c, mask=Brain)
            ROI_MAE = compute_mae(syn_pred, t1c, mask=ROI)

            SSIM = compare_ssim(syn_pred[Brain > 0], t1c[Brain > 0], data_range=255)
            PSNR = compare_psnr(t1c[Brain > 0], syn_pred[Brain > 0], data_range=255)

            NiiDataWrite(os.path.join(save_dir, 'predictions', f'epoch{epoch}_{MAE}.nii.gz'), syn_pred, spacing, origin,direction)
            print(epoch, 'MAE:', MAE, ROI_MAE, 'SSIM:', SSIM, 'PSNR:', PSNR)
    # ======== Final save ========
    trainer.save(os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))
    train_writer.close()



if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
    parser.add_argument('--pretrain', type=bool, default=True, help='continue_train')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue_train')
    parser.add_argument('--continue_epoch', type=int, default=0, help='start from')
    parser.add_argument('--objective', type=str, default='pred_noise', help='pred_noise/pred_x0')
    parser.add_argument('--activate', type=str, default='none', help='none/tanh')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--gae', type=int, default=1, help='gradient_accumulate_every')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr_max', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--num_threads', default=5, type=int, help='# threads for loading data')

    # -------------------- Data settings
    parser.add_argument('--bs', type=int, default=5, help='batch size')
    parser.add_argument('--test_bs', type=int, default=24, help='batch size')
    parser.add_argument('--ImageSize', type=int, default=424, help='then crop to this size')  # 对于整图160不如256
    parser.add_argument('--MR_max', type=int, default=255, help='MR_max')
    parser.add_argument('--MR_min', type=int, default=0, help='MR_min')

    # -------------------- Model settings
    parser.add_argument('--net', type=str, default='simple_unet_Improved_in_relu_32',
                        help='simple_unet_Improved_in_relu_32# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--cc', type=int, default=2, help='condition_channels')
    parser.add_argument('--class_cond', default=2, help='None # of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')

    # -------------------- Loss function
    parser.add_argument('--loss', type=str, default='l1', help='l1/l2')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # -------------------- Seed setup for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)