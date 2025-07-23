import os
import time
import glob
import torch
import argparse
import numpy as np
from os.path import join
from fileinput import close
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

from Nii_utils import setup_seed, Save_Parameter
from models.Networks_gen.Networks_UNet_DDPM import Unet_class
from models.Networks_gen.Networks_DDPM_trainer import GaussianDiffusion, DDPM_Trainer
from models.Networks_gen.Networks_simple_UNet_DDPM import simple_Unet_for_Improved_DDPM_class
from models.Networks_gen.Validation_inference import Model_Validation_multitask, Model_Inference_multitask
from dataset.Dataset_gen import Dataset_harmonize_2D, Dataset_harmonize, Dataset_harmonize_inference



def main(opt):
    train_writer = SummaryWriter(join(opt.save_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(join(opt.save_dir, 'log/val'), flush_secs=2)
    print(opt.save_dir)
    # ======== Initialize model and trainer ========
    if opt.model_name.lower() == 'unet_ddpm_64_class':
        net = Unet_class(dim=64, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, condition_channels=opt.cc,
                   condition=True, class_cond=opt.class_cond, resnet_block_groups=8, seg_head=opt.seg).to(opt.device)
    elif opt.model_name.lower() == 'unet_ddpm_32_class':
        net = Unet_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, condition_channels=opt.cc,
                   condition=True, class_cond=opt.class_cond, resnet_block_groups=8, seg_head=opt.seg).to(opt.device)
    elif opt.model_name == 'simple_unet_Improved_32_class':
        net = simple_Unet_for_Improved_DDPM_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                                              condition_channels=opt.cc, condition=True, class_cond=opt.class_cond, seg_head=opt.seg).to(opt.device)

    diffusion_model = GaussianDiffusion(net, image_size=opt.ImageSize, timesteps=1000, loss_type=opt.loss, seg_weight=opt.seg_weight,
                                        objective=opt.objective, activate=opt.activate).to(opt.device)
    trainer = DDPM_Trainer(diffusion_model, train_batch_size=opt.bs, train_lr=opt.lr_max)

    # ======== Learning rate scheduler setup ========
    scheduler_lr = MultiStepLR(trainer.opt, milestones=[int(0.5 * opt.max_epoch), int(0.8 * opt.max_epoch)], gamma=0.1, last_epoch=-1)

    # ======== Load dataset and dataloader ========
    root_dir = './Glioma_DATA/Preprocessing_DATA/Train'
    val_dir = './Glioma_DATA/Preprocessing_DATA/Val'

    train_set =  Dataset_harmonize_2D(opt, root_dir=root_dir)
    val_set =  Dataset_harmonize(opt, root_dir=val_dir)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.bs, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=True)

    best_MAE = 1000
    best_epoch = 0
    Save_Parameter(opt)  # save the training parameter
    for epoch in range(opt.max_epoch):
        train_start_time = time.time()
        for param_group in trainer.opt.param_groups:
            lr = param_group['lr']
            break
        epoch_train_loss = []
        for i, DATA in enumerate(train_loader):
            x, x_cond = DATA['B'].to(opt.device), DATA['A'].to(opt.device)  # x is target (T1Gd) and x_cond is source (T1 and T2-FLAIR)
            label = DATA['label'].to(opt.device)  # label
            mask = DATA['ROI'].to(opt.device)
            trainer.opt.zero_grad()
            trainer.model.train()
            loss = trainer.calculate_loss_multitask(x, x_cond, mask, seg_cond=None, label=label)
            loss_back = loss / opt.gae
            loss_back.backward()
            if (i + 1) % opt.gae == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1)
                trainer.opt.step()
            epoch_train_loss.append(loss.item())
            if i % 100 == 0:
                print('[%d/%d, %5d/%d] train_loss: %.3f ' % (epoch + 1, opt.max_epoch, i + 1, len(train_loader), loss.item()))
        epoch_train_loss = np.mean(epoch_train_loss)
        scheduler_lr.step()
        print('[%d/%d] train_loss: %.3f' % (epoch + 1, opt.max_epoch, epoch_train_loss))
        train_writer.add_scalar('gen_loss', epoch_train_loss, epoch)
        train_writer.add_scalar('gen_lr', lr, epoch)
        train_writer.add_scalar('gen_time', (time.time() - train_start_time) / 60, epoch)
        print('Train One Epoch Time Taken: %.1f min' % ((time.time() - train_start_time) / 60))


        first_stage = (epoch <= opt.max_epoch // 3 and epoch % 10 == 0)
        second_stage = (epoch > opt.max_epoch // 3 and epoch <= opt.max_epoch // 3 * 2 and epoch % 5 == 0)
        third_stage = (epoch > opt.max_epoch // 3 * 2 and epoch % 5 == 0)

        if (first_stage + second_stage + third_stage) or (epoch == opt.max_epoch - 1):
            ref_timestep = max(1, int(max(10 * first_stage + 50 * second_stage + 100 * third_stage, 10) * opt.val_timestep_scale))
            print('epoch:', epoch, 'ref_timestep:', ref_timestep)
            mMAE_val, mROI_MAE_val, _, _, Metric_df_val = Model_Validation_multitask(opt, epoch, trainer, {'val':val_loader}, dataset='val', save_dir=opt.save_dir, writer={'val':val_writer}, ref_timestep=ref_timestep, train=True, save_image=False)
            if mMAE_val < best_MAE:
                try:
                    os.remove(join(opt.save_dir, 'train_model', f'best_MAE_epoch{best_epoch}.pth'))
                except:
                    pass
                best_epoch = epoch
                if epoch > 0:
                    best_MAE = mMAE_val
                trainer.save(join(opt.save_dir, 'train_model', f'best_MAE_epoch{best_epoch}.pth'))

        trainer.save(join(opt.save_dir, 'train_model', f'latest_epoch{epoch}.pth'))
        try:
            os.remove(join(opt.save_dir, 'train_model', f'latest_epoch{epoch-1}.pth'))
        except:
            pass
    trainer.save(join(opt.save_dir, 'train_model', 'final' + '.pth'))
    train_writer.close()


def pred(opt, trainer=None):
    if not trainer:
        if opt.model_name.lower() == 'unet_ddpm_64_class':
            net = Unet_class(dim=64, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                             condition_channels=opt.cc,
                             condition=True, class_cond=opt.class_cond, resnet_block_groups=8, seg_head=opt.seg).to(opt.device)
        elif opt.model_name.lower() == 'unet_ddpm_32_class':
            net = Unet_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                             condition_channels=opt.cc,
                             condition=True, class_cond=opt.class_cond, resnet_block_groups=8, seg_head=opt.seg).to(opt.device)
        elif opt.model_name == 'simple_unet_Improved_32_class':
            net = simple_Unet_for_Improved_DDPM_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8),
                                                      channels=1,
                                                      condition_channels=opt.cc, condition=True,
                                                      class_cond=opt.class_cond, seg_head=opt.seg).to(opt.device)
        diffusion_model = GaussianDiffusion(net, image_size=opt.ImageSize, timesteps=1000, objective=opt.objective, activate=opt.activate, ddim_sampling_eta=0).cuda()
        trainer = DDPM_Trainer(diffusion_model, train_batch_size=1, train_lr=0.0001)
    trainer.load(sorted(glob.glob(join(opt.save_dir, 'train_model', 'best_MAE_epoch*.pth')))[-1])

    # ======== Load dataset and dataloader ========
    train_dir = './Glioma_DATA/Preprocessing_DATA/Train'
    val_dir = './Glioma_DATA/Preprocessing_DATA/Val'
    test_dir = './Glioma_DATA/Preprocessing_DATA/Test'
    train_datatset = Dataset_harmonize_inference(opt, root_dir=train_dir, label_known=True)
    val_datatset = Dataset_harmonize_inference(opt, root_dir=val_dir, label_known=True)
    test_datatset = Dataset_harmonize_inference(opt, root_dir=test_dir, label_known=False)
    train_loader = DataLoader(dataset=train_datatset, batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
    val_loader = DataLoader(dataset=val_datatset, batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
    test_loader = DataLoader(dataset=test_datatset, batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
    trainer.model.eval()
    with torch.no_grad():
        Model_Inference_multitask(opt, train_loader, trainer, dataset='Train', label_known=True)
        Model_Inference_multitask(opt, val_loader, trainer, dataset='Val', label_known=True)
        Model_Inference_multitask(opt, test_loader, trainer, dataset='Test', label_known=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_epoch', type=int, default=200, help='all_epochs')
    parser.add_argument('--lr_max', type=float, default=1e-5, help='max learning rate')
    parser.add_argument('--gae', type=int, default=1, help='gradient_accumulate_every')
    parser.add_argument('--bs', type=int, default=2, help='training input batch size')
    parser.add_argument('--num_threads', type=int, default=8, help='# threads for loading dataset')

    # -------------------- Inference settings
    parser.add_argument('--val_bs', type=int, default=4, help='Val/Test batch size')
    parser.add_argument('--ref_timestep', type=int, default=100, help='<=1000, 1000 is time-consuming but of higher quality')
    parser.add_argument('--val_timestep_scale', type=float, default=1, help='>=100 AND <=1000, 1000 is time-consuming but of higher quality, at least 100 to ensure image quality')
    parser.add_argument('--save_dir', type=str, default='', help='./main/trained_models/CBSI_gen/{pred_*_...class_seg_time}')  # Path for saving model parameters

    # -------------------- Data settings
    parser.add_argument('--data_dim', type=str, default='2D')
    parser.add_argument('--ImageSize', type=int, default=424, help='Spatial dimension cropped to 424 * 424')
    parser.add_argument('--MR_max', type=int, default=255, help='max value of preprocessed MR image')
    parser.add_argument('--MR_min', type=int, default=0, help='min value of preprocessed MR image')
    parser.add_argument('--preloading', type=bool, default=True, help='preloading the image')

    # -------------------- Model settings
    parser.add_argument('--model_name', type=str, default='unet_ddpm_32_class', choices=['unet_ddpm_64_class', 'unet_ddpm_32_class', 'simple_unet_Improved_32_class'],)
    parser.add_argument('--seg', type=bool, default=True, help='Introduce the auxiliary segmentation task')
    parser.add_argument('--objective', type=str, default='pred_x0', choices=['pred_x0', 'pred_noise'], help='The output target of the generator')
    parser.add_argument('--activate', type=str, default='tanh', choices=['none', 'tanh'], help='tanh can only be used when pred_x0')
    parser.add_argument('--cc', type=int, default=2, help='condition_channels: non-contrast MR (T1 and T2-FLAIR)')
    parser.add_argument('--class_cond', type=int, default=2, help='ET label types (enhancing and non-enhancing)')
    parser.add_argument('--output_nc', type=int, default=1, help='output T1Gd image only has one channel')

    # -------------------- Loss function
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2', 'mix'])
    parser.add_argument('--seg_weight', type=float, default=0.1, help='weight of segmentation loss')

    # -------------------- Quick test settings
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')

    opt = parser.parse_args()
    # torch.cuda.is_available = lambda: False
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(opt.seed)
    print("CBSI_gen Start")
    if opt.quick_test:
        opt.max_epoch = 10
        opt.ref_timestep = 10
        opt.val_timestep_scale = 0.1
        opt.model_name = 'simple_unet_Improved_32_class'
    # -------------- Experiment naming & directory setup --------------
    if not opt.save_dir or not opt.inference_only:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_name = 'bs{}_epoch{}_gae{}_seed{}_class_seg'.format(opt.bs, opt.max_epoch, opt.gae, opt.seed)
        opt.save_dir = join(
            './main/trained_models/CBSI_gen/{}_{}_{}_condition_act_{}_{}_{}'.format(opt.objective, opt.model_name, opt.loss,
                                                                              opt.activate, save_name, current_time))
    os.makedirs(join(opt.save_dir, 'train_model'), exist_ok=True)

    if not opt.inference_only:
        trainer = main(opt)
        pred(opt, trainer)
    else:
        pred(opt)
    if not opt.inference_only:
        print("CBSI_gen Training Done")
        print("-------------------------------------------")
        if opt.quick_test:
            print(f"Attention !! Please use this command to carry out the next stage of the quick test:\n python ./main/train_CBSI_ide.py --quick_test --gen_save_dir {opt.save_dir}")
        else:
            print(f"Attention !! If you want to use the model trained in this session, Please use this command to carry out the next stage of training:\n python ./main/train_CBSI_ide.py --gen_save_dir {opt.save_dir}")
        print("-------------------------------------------")
    else:
        print("CBSI_gen Inference Done")