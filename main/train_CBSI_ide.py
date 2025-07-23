import os
import time
import glob
import math
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from Nii_utils import *
from dataset.Dataset_ide import Dataset_harmonize_2D_t1_t2f_t1c
from models.Networks_ide.model import EfficientNet
from models.Networks_ide.Validation_inference import Model_Validation, Model_Inference



def main(opt):
    train_writer = SummaryWriter(join(opt.save_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(join(opt.save_dir, 'log/val'), flush_secs=2)
    print(opt.save_dir)
    # -------------- Identification Model Setup ----------------
    net = EfficientNet.from_name(model_name=f'efficientnet-{opt.model_name[-2:]}', in_channels=opt.inchannel, num_classes=opt.classes, dropout_rate=opt.drop).to(opt.device)
    # Load pretrained weights for EfficientNet
    net_weights = net.state_dict()
    pre_weights = torch.load(f'./main/models/Networks_ide/pretrain/efficientnet-b0-355c32eb.pth')  # Download from the official website
    pre_dict = {k: v for k, v in pre_weights.items() if net_weights[k].numel() == v.numel()}
    net.load_state_dict(pre_dict, strict=False)
    print(f'The model is 2D {opt.model_name}')

    # ----------------------loss & optimizer------------------------
    criterion = nn.BCEWithLogitsLoss().to(opt.device)  # Sigmoid-BCELoss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr_max, weight_decay=5e-4)

    # -------------- Learning Rate Scheduler ----------------
    if opt.warmup:
        # warm_up_with_cosine_lr
        int_decay = opt.lr_min / opt.lr_max
        zoom = 1 - int_decay

        warm_up_with_cosine_lr = lambda \
            epoch: int_decay + epoch / opt.warm_up_epochs * zoom if epoch <= opt.warm_up_epochs else int_decay + zoom * 0.5 * (
                    math.cos((epoch - opt.warm_up_epochs) / (opt.max_epoch - opt.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[int((1 / 3) * opt.max_epoch), int((2 / 3) * opt.max_epoch)], gamma=0.1, last_epoch=-1)

    # -------------- Dataset Setup ----------------
    ckpt_dir = join(opt.save_dir, 'train_model')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ======== Load dataset and dataloader ========
    root_dir = './Glioma_DATA/Preprocessing_DATA/Train'
    val_dir = './Glioma_DATA/Preprocessing_DATA/Val'

    train_set = Dataset_harmonize_2D_t1_t2f_t1c(opt, root_dir=root_dir, dataset='Train')
    val_set = Dataset_harmonize_2D_t1_t2f_t1c(opt, root_dir=val_dir, dataset='Val')

    train_loader = DataLoader(train_set, batch_size=opt.bs, shuffle=True, num_workers=opt.num_threads, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_threads, drop_last=False)
    print(f'Size of train_dataset:{len(train_set)}.')
    # print(f'Size of val_dataset:{len(val_set)}.')
    print('Data prepared.')

    # -------------- Metrics Init ----------------
    threshold = 0.5
    best_AUC, best_epoch = 0, 0
    Save_Parameter(opt)
    # ==================== Training Loop ====================
    print('Start training.')
    for epoch in tqdm(range(opt.max_epoch)):
        train_start_time = time.time()
        net.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        train_acc_list = []
        train_loss_list = []
        y_true = torch.tensor([]).to(opt.device)
        y_pred = torch.tensor([]).to(opt.device)
        y_binary = torch.tensor([]).to(opt.device)
        # ------------------ Batch Training ------------------
        for i, DATA in enumerate(train_loader):
            image = DATA['image'].to(opt.device)
            label = DATA['label'].to(opt.device)
            net.zero_grad()
            y = net(image)
            loss = criterion(y, label)
            # Optional flood loss
            if opt.do_flood:
                flood_loss = (loss - opt.flood).abs() + opt.flood
                flood_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # Metrics logging
            y = torch.sigmoid(y)
            y_binary = torch.cat([y_binary, (y.detach() > threshold)])
            y_true = torch.cat([y_true, label.detach()])
            y_pred = torch.cat([y_pred, y.detach()])

            hit = ((y.detach() < threshold) ^ label.bool()).sum()
            train_acc_list.append(np.array(hit.cpu()))
            train_loss_list.append(np.array(loss.detach().cpu()))

        scheduler.step()
        # Calculate epoch-level metrics
        train_loss = np.array(train_loss_list).mean()
        train_acc = np.array(train_acc_list).sum() / len(train_set)
        train_auc = roc_auc_score(y_true.cpu(), y_pred.cpu())

        train_precision, train_recall, train_F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(),
                                                                                           y_binary.int().cpu(),
                                                                                           average='binary')
        # TensorBoard Logging
        train_writer.add_scalar('ide_lr', lr, epoch)
        train_writer.add_scalar('ide_loss', train_loss, epoch)
        train_writer.add_scalar('ide_AUC', train_auc, epoch)
        train_writer.add_scalar('ide_ACC', train_acc, epoch)
        train_writer.add_scalar('ide_F1_score', train_F1_score, epoch)
        train_writer.add_scalar('ide_time', (time.time() - train_start_time) / 60, epoch)
        train_writer.close()

        # ------------------ Validation Phase ------------------
        AUC_val, _ = Model_Validation(opt, epoch, net, {'val':val_loader}, dataset='val', save_dir=opt.save_dir, writer={'val':val_writer}, train=True, criterion=criterion)

        if AUC_val > best_AUC:
            try:
                os.remove(join(opt.save_dir, 'train_model', f'best_AUC_epoch{best_epoch}.pth'))
            except:
                pass
            best_epoch = epoch
            if epoch > 10:
                best_AUC = AUC_val
            torch.save(net.state_dict(),join(opt.save_dir, 'train_model', f'best_AUC_epoch{best_epoch}.pth'))
        torch.save(net.state_dict(),join(opt.save_dir, 'train_model', f'latest_epoch{epoch}.pth'))
        try:
            os.remove(join(opt.save_dir, 'train_model', f'latest_epoch{epoch - 1}.pth'))
        except:
            pass
    torch.save(net.state_dict(),join(opt.save_dir, 'train_model', 'final' + '.pth'))

    return net


def pred(opt, net=None):
    if not net:
        net = EfficientNet.from_name(model_name=f'efficientnet-{opt.model_name[-2:]}', in_channels=opt.inchannel, num_classes=opt.classes, dropout_rate=opt.drop).to(opt.device)
    net.load_state_dict(torch.load(sorted(glob.glob(join(opt.save_dir, 'train_model', 'best_AUC_epoch*.pth')))[-1]), strict=True)

    val_dir = './Glioma_DATA/Preprocessing_DATA/Val'
    test_dir = './Glioma_DATA/Preprocessing_DATA/Test'
    with torch.no_grad():
        Model_Inference(opt, val_dir, net, dataset='Val', label_known=True)
        Model_Inference(opt, test_dir, net, dataset='Test', label_known=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_epoch', type=int, default=100, help='all_epochs')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--lr_max', type=float, default=5e-4, help='max learning rate')
    parser.add_argument('--bs', type=int, default=2, help='training input batch size')
    parser.add_argument('--num_threads', type=int, default=1, help='# threads for loading dataset')

    # -------------------- Inference settings
    parser.add_argument('--val_bs', type=int, default=4, help='Val/Test batch size')
    parser.add_argument('--ref_timestep', type=int, default=100, help='<=1000, 1000 is time-consuming but of higher quality')
    parser.add_argument('--save_dir', type=str, default='', help="./main/trained_models/CBSI_ide/{bs*_ImageSize*_epoch*_seed*_time}/")  # Path for saving model parameters

    # -------------------- Data settings
    parser.add_argument('--data_dim', type=str, default='2D')
    parser.add_argument('--ImageSize', type=int, default=424, help='Spatial dimension cropped to 424 * 424')
    parser.add_argument("--gen_save_dir", type=str, default='', help="./main/trained_models/CBSI_gen/{pred_*_...class_seg_time}/")
    parser.add_argument('--MR_max', type=int, default=255, help='max value of preprocessed MR image')
    parser.add_argument('--MR_min', type=int, default=0, help='min value of preprocessed MR image')
    parser.add_argument('--preloading', type=bool, default=True, help='preloading the image')

    # -------------------- Model settings
    parser.add_argument('--model_name', type=str, default='EfficientNet_b0')
    parser.add_argument('--inchannel', type=int, default=3, help='input channel (T1, T2-FLAIR, and synthetic T1Gd)')
    parser.add_argument('--classes', type=int, default=1, help='')
    parser.add_argument('--drop', type=float, default=0.2, help='dropout rate 0~1 ')

    # -------------------- Loss function
    parser.add_argument('--do_flood', type=bool, default=True, help='do flood loss')
    parser.add_argument('--flood', type=float, default=0.1, help='flood loss threshold')
    parser.add_argument('--warmup', action='store_false')
    parser.add_argument('--warm_up_epochs', type=int, default=5, help='warm_up_epochs')

    # -------------------- Quick test settings
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--inference_only', action='store_true')
    opt = parser.parse_args()
    # torch.cuda.is_available = lambda: False
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    setup_seed(opt.seed)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CBSI_ide Start")
    if opt.quick_test:
        opt.max_epoch = 10
        opt.ref_timestep = 10

    # -------------- Experiment naming & directory setup --------------
    if not opt.save_dir or not opt.inference_only:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        opt.save_dir = './main/trained_models/CBSI_ide/bs{}_ImageSize{}_epoch{}_seed{}_{}'.format(opt.bs, opt.ImageSize, opt.max_epoch, opt.seed, current_time)
        os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)

    if not opt.inference_only:
        net = main(opt)
        pred(opt, net)
    else:
        pred(opt)

    if not opt.inference_only:
        print("CBSI_ide Training Done")
    else:
        print("CBSI_ide Inference Done")
    print("-------------------------------------------")
    print(f"Attention !! Results can be viewed here :\n {opt.save_dir}")
    print("-------------------------------------------")