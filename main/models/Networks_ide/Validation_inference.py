import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from Nii_utils import harmonize_mr, NiiDataRead



def Model_Validation(opt, epoch, net, loader, dataset, save_dir=None, writer=None, train=True, criterion=None):
    count = 0
    start_time = time.time()
    loss_list = []
    y_true = torch.tensor([]).to(opt.device)
    y_pred = torch.tensor([]).to(opt.device)

    net.eval()
    with torch.no_grad():
        for i, DATA in enumerate(loader[dataset]):
            count += 1
            label = DATA['label'].to(opt.device)
            image = DATA['image'].to(opt.device)
            y = torch.sigmoid(net(image))
            loss = criterion(y, label)
            loss_list.append(np.array(loss.cpu()))
            y_true = torch.cat([y_true, label.detach()])
            y_pred = torch.cat([y_pred, y.detach()])

        loss = np.array(loss_list).mean()
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())

        # ========== Write to TensorBoard ==========
        if train:
            writer[dataset].add_scalar('ide_loss', loss, epoch)
            writer[dataset].add_scalar('ide_AUC', auc, epoch)
        if train:
            writer[dataset].add_scalar('ide_time', (time.time() - start_time) / 60, epoch)
    return auc, loss


def Model_Inference(opt, dataset_dir, net, dataset, label_known=False):
    pred_all = {'ID': [], "argmax_pred": [], "softmax_pred": []}
    # ------------------ Validation Phase ------------------
    net.eval()
    if label_known:
        # Load filenames from both classes
        files_a_0 = [ID for ID in os.listdir(join(dataset_dir, 'ET-0')) if os.path.isdir(os.path.join(join(dataset_dir, 'ET-0'), ID))]
        files_a_1 = [ID for ID in os.listdir(join(dataset_dir, 'ET-1')) if os.path.isdir(os.path.join(join(dataset_dir, 'ET-1'), ID))]

        all_ID_list = files_a_0 + files_a_1
        labels_list = [0] * len(files_a_0) + [1] * len(files_a_1)
    else:
        all_ID_list = [ID for ID in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, ID))]

    syn_T1Gd_dir = join(opt.gen_save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn')

    for i, ID in enumerate(tqdm(all_ID_list)):
        if label_known:
            T1, _, _, _ = NiiDataRead(join(dataset_dir, f'ET-{labels_list[i]}/{ID}/T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(dataset_dir, f'ET-{labels_list[i]}/{ID}/T2F.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(dataset_dir, f'ET-{labels_list[i]}/{ID}/ROI.nii.gz'))
        else:
            T1, _, _, _ = NiiDataRead(join(dataset_dir, f'{ID}/T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(dataset_dir, f'{ID}/T2F.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(dataset_dir, f'{ID}/ROI.nii.gz'))
        non_T1Gd, _, _, _ = NiiDataRead(join(syn_T1Gd_dir, f'ET-0/{ID}.nii.gz'))
        en_T1Gd, _, _, _ = NiiDataRead(join(syn_T1Gd_dir, f'ET-1/{ID}.nii.gz'))
        # Get model predictions and labels
        preds = Test_pred(net, {'t1': T1, 't2f': T2F, 'en_T1Gd': en_T1Gd, 'non_T1Gd': non_T1Gd}, ROI, opt)
        pred_all[f"ID"].append(ID)
        pred_all[f"softmax_pred"].append(torch.softmax(preds, dim=0)[0].item())
        pred_all[f"argmax_pred"].append(torch.argmax(preds).item())
    os.makedirs(join(opt.save_dir, 'prediction'), exist_ok=True)
    pd.DataFrame(pred_all).to_csv(join(opt.save_dir, 'prediction', f'pred_{dataset}.csv'), index=False)


def Test_pred(net, Image, ROI, opt):
    if isinstance(Image, dict):
        for image_name in Image.keys():
            Image[image_name] = harmonize_mr(Image[image_name])
    else:
        Image = harmonize_mr(Image)
    pred_list_1 = torch.zeros(1).cuda()
    pred_list_0 = torch.zeros(1).cuda()


    image_list_1 = []
    image_list_0 = []
    count = 0
    for slice in np.unique(ROI.nonzero()[0]):
        if isinstance(Image, dict):
            pred_1 = np.stack((Image['t1'][slice], Image['t2f'][slice], Image['en_T1Gd'][slice]), axis=0)
            pred_0 = np.stack((Image['t1'][slice], Image['t2f'][slice], Image['non_T1Gd'][slice]), axis=0)
        else:
            pred_1 = np.stack((Image[slice], Image['en_T1Gd'][slice]), axis=0)
            pred_0 = np.stack((Image[slice], Image['non_T1Gd'][slice]), axis=0)
        image_list_1.append(pred_1)
        image_list_0.append(pred_0)
        count += 1
    image_1 = np.array(image_list_1)
    image_0 = np.array(image_list_0)

    n_num = image_1.shape[0] // opt.val_bs
    n_num = n_num + 0 if image_1.shape[0] % opt.val_bs == 0 else n_num + 1

    for n in range(n_num):
        # print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image_1[n * opt.val_bs:, :, :, :]
        else:
            one_image = image_1[n * opt.val_bs: (n + 1) * opt.val_bs, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""
        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_list_1 += torch.sum(k.detach(), dim=0)

    for n in range(n_num):
        print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image_0[n * opt.val_bs:, :, :, :]
        else:
            one_image = image_0[n * opt.val_bs: (n + 1) * opt.val_bs, :, :, :]
        one_image = torch.from_numpy(one_image).float().to(opt.device)
        """one_image = torch.from_numpy(one_image).float()"""

        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_list_0 += torch.sum(k.detach(), dim=0)


    pred_list1 = pred_list_1 / count
    pred_list0 = pred_list_0 / count
    return torch.cat((pred_list0, pred_list1), dim=0).cpu()

