import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip, Resize)

from Nii_utils import NiiDataRead, harmonize_mr


class CA_Dataset_harmonize_2D(Dataset):
    def __init__(self, opt, root_dir):
        self.image_labels = []
        self.image_filenames = []
        self.root_dir = root_dir
        self.a_t1 = []
        self.a_t2f = []
        self.b_t1c = []
        self.ROI = []
        self.Brain = []
        self.image_labels = []
        self.MR_min = opt.MR_min
        self.MR_max = opt.MR_max

        for c in os.listdir(root_dir):
            if '_after' not in c:
                continue
            if c == 'flair1_after':
                label = 1
            else:
                label = 0
            class_dir = join(root_dir, c)
            for patient in tqdm(os.listdir(class_dir)):
                image_filenames = join(c, patient)
                a_t1, spacing, origin, direction = NiiDataRead(join(self.root_dir, image_filenames, 'T1.nii.gz'))
                a_t2f, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'T2F.nii.gz'))
                b_t1c, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'T1C.nii.gz'))
                ROI, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'ROI.nii.gz'))
                Brain, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'Brain_mask.nii.gz'))
                ROI[ROI > 0] = 1
                Brain[Brain > 0] = 1

                for slice in range(b_t1c.shape[0]):
                    self.a_t1.append(a_t1[slice])
                    self.a_t2f.append(a_t2f[slice])
                    self.b_t1c.append(b_t1c[slice])
                    self.ROI.append(ROI[slice])
                    self.Brain.append(Brain[slice])
                    self.image_labels.append(label)

        self.trans_train = Compose([ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                                                     rotate_limit=180, p=0.5,
                                                     border_mode=cv2.BORDER_CONSTANT, value=0,
                                                     interpolation=cv2.INTER_NEAREST),
                                    HorizontalFlip(p=0.3), VerticalFlip(p=0.3)])
        self.all_patch_num = len(self.b_t1c)

    def __getitem__(self, index):

        label = self.image_labels[index]
        a_t1 = self.a_t1[index]
        a_t2f = self.a_t2f[index]
        b_t1c = self.b_t1c[index]
        ROI = self.ROI[index]
        Brain = self.Brain[index]
        a_t1 = torch.tensor(harmonize_mr(a_t1, self.MR_min, self.MR_max)).float()
        a_t2f = torch.tensor(harmonize_mr(a_t2f, self.MR_min, self.MR_max)).float()
        b_t1c = torch.tensor(harmonize_mr(b_t1c, self.MR_min, self.MR_max)).float()
        ROI = torch.tensor(ROI).float()
        Brain = torch.tensor(Brain).float()
        label = torch.tensor(label).long()

        # random Horizontal and Vertical Flip to both image and mask
        image_3 = np.concatenate((a_t1[:, :, np.newaxis], a_t2f[:, :, np.newaxis], b_t1c[:, :, np.newaxis]), axis=2)
        mask_2 = np.concatenate((ROI[:, :, np.newaxis], Brain[:, :, np.newaxis]), axis=2)


        augmented = self.trans_train(image=image_3, mask=mask_2)
        image = augmented['image']
        mask = augmented['mask']

        a_t1 = torch.tensor(image[:, :, 0]).unsqueeze(0)
        a_t2f = torch.tensor(image[:, :, 1]).unsqueeze(0)
        b_t1c = torch.tensor(image[:, :, 2]).unsqueeze(0)
        ROI = torch.tensor(mask[:, :, 0]).unsqueeze(0)
        Brain = torch.tensor(mask[:, :, 1]).unsqueeze(0)
        label = torch.tensor(label).long()

        return {
            'A': torch.cat((a_t1, a_t2f), dim=0),
            'B': b_t1c,
            'ROI': ROI,
            'Brain': Brain,
            'label': label
        }


    def __len__(self):
        return self.all_patch_num

class Dataset_train(Dataset):
    def __init__(self, data_root='data', size=(256, 256), txt_path='train_set.txt'):
        self.data_root = data_root
        with open(txt_path, 'r') as f:
            name_list = f.readlines()
        name_list = [n.strip('\n') for n in name_list]
        self.T2_all = []
        self.CT_all = []
        self.seg_all = []
        self.mask_all = []
        print('loading train data!')
        for name in name_list:
            print(name)
            t2, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'T2_final_norm.nii.gz'))
            ct, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'CT_final.nii.gz'))
            seg, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'OARs.nii.gz'), as_type=np.uint8)
            mask, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'mask_CT.nii.gz'), as_type=np.uint8)
            ct[ct <= -1000] = -1000
            ct[ct > 2500] = 2500
            ct = ((ct + 1000) / 1750 - 1) * mask
            self.T2_all.append(t2)
            self.CT_all.append(ct)
            self.seg_all.append(seg)
            self.mask_all.append(mask)
        self.T2_all = np.concatenate(self.T2_all, axis=0)
        self.CT_all = np.concatenate(self.CT_all, axis=0)
        self.seg_all = np.concatenate(self.seg_all, axis=0)
        self.mask_all = np.concatenate(self.mask_all, axis=0)
        self.len = self.T2_all.shape[0]

    def __getitem__(self, idx):
        T2 = self.T2_all[idx]
        CT = self.CT_all[idx]
        seg = self.seg_all[idx]
        mask = self.mask_all[idx]

        T2_CT_origianl = np.concatenate((T2[:, :, np.newaxis], CT[:, :, np.newaxis]), axis=2)
        seg_mask_origianl = np.concatenate((seg[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=T2_CT_origianl, mask=seg_mask_origianl)
        T2_CT = augmented['image']
        seg_mask = augmented['mask']
        T2 = T2_CT[:, :, 0]
        CT = T2_CT[:, :, 1]
        seg = seg_mask[:, :, 0]
        mask = seg_mask[:, :, 1]
        a = 0
        while (mask.sum() == 0 and a < 3):
            augmented = self.transforms(image=T2_CT_origianl, mask=seg_mask_origianl)
            T2_CT = augmented['image']
            seg_mask = augmented['mask']
            T2 = T2_CT[:, :, 0]
            CT = T2_CT[:, :, 1]
            seg = seg_mask[:, :, 0]
            mask = seg_mask[:, :, 1]
            a += 1
        if a == 3:
            T2 = T2_CT_origianl[:, :, 0]
            CT = T2_CT_origianl[:, :, 1]
            seg = seg_mask_origianl[:, :, 0]
            mask = seg_mask_origianl[:, :, 1]
        T2 = torch.from_numpy(T2).float().unsqueeze(0)
        CT = torch.from_numpy(CT).float().unsqueeze(0)
        seg = torch.from_numpy(seg).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return T2, CT, seg, mask

    def __len__(self):
        return self.len

class Dataset_val(Dataset):
    def __init__(self, data_root='data', size=(256, 256), txt_path='val_set.txt'):
        self.data_root = data_root
        with open(txt_path, 'r') as f:
            name_list = f.readlines()
        name_list = [n.strip('\n') for n in name_list]
        self.T2_all = []
        self.CT_all = []
        self.seg_all = []
        self.mask_all = []
        print('loading val data!')
        for name in name_list:
            print(name)
            t2, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'T2_final_norm.nii.gz'))
            ct, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'CT_final.nii.gz'))
            seg, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'OARs.nii.gz'), as_type=np.uint8)
            mask, _, _, _ = NiiDataRead(os.path.join(data_root, name, 'mask_CT.nii.gz'), as_type=np.uint8)
            ct[ct <= -1000] = -1000
            ct[ct > 2500] = 2500
            ct = ((ct + 1000) / 1750 - 1) * mask  # mistake
            self.T2_all.append(t2)
            self.CT_all.append(ct)
            self.seg_all.append(seg)
            self.mask_all.append(mask)
        self.T2_all = np.concatenate(self.T2_all, axis=0)
        self.CT_all = np.concatenate(self.CT_all, axis=0)
        self.seg_all = np.concatenate(self.seg_all, axis=0)
        self.mask_all = np.concatenate(self.mask_all, axis=0)
        self.len = self.T2_all.shape[0]
        self.transforms = Compose([Resize(size[1], size[0], interpolation=cv2.INTER_NEAREST)])

    def __getitem__(self, idx):
        T2 = self.T2_all[idx]
        CT = self.CT_all[idx]
        seg = self.seg_all[idx]
        mask = self.mask_all[idx]

        T2_CT_origianl = np.concatenate((T2[:, :, np.newaxis], CT[:, :, np.newaxis]), axis=2)
        seg_mask_origianl = np.concatenate((seg[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=T2_CT_origianl, mask=seg_mask_origianl)
        T2_CT = augmented['image']
        seg_mask = augmented['mask']
        T2 = T2_CT[:, :, 0]
        CT = T2_CT[:, :, 1]
        seg = seg_mask[:, :, 0]
        mask = seg_mask[:, :, 1]

        T2 = torch.from_numpy(T2).float().unsqueeze(0)
        CT = torch.from_numpy(CT).float().unsqueeze(0)
        seg = torch.from_numpy(seg).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return T2, CT, seg, mask

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass