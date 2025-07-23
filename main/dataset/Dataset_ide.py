import torch
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Nii_utils import harmonize_mr, NiiDataRead



class Dataset_harmonize_2D_t1_t2f_t1c(Dataset):
    def __init__(self, opt, root_dir='', dataset=''):
        self.aug = dataset.lower() == 'train'
        syn_T1Gd_dir = join(opt.gen_save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn')
        label_data = pd.read_csv(join(opt.gen_save_dir, f'prediction_ddim_{opt.ref_timestep}', f'{dataset}_syn', 'label_data.csv'))
        if opt.quick_test:
            label_data = label_data[:opt.val_bs*2]
        self.synET_IDs_list = label_data['synET_ID'].values.tolist()
        self.IDs_list = label_data['ID'].values.tolist()
        self.cls_label_list = label_data['cls_label'].values.tolist()
        self.et_label_list = label_data['et_label'].values.tolist()

        self.real_t1s = []
        self.real_t2fs = []
        self.real_t1cs = []
        self.syn_t1cs = []
        self.label_list = []

        for synET_ID, ID, cls_label, et_label in zip(self.synET_IDs_list, self.IDs_list, self.cls_label_list, self.et_label_list):
            T1, _, _, _ = NiiDataRead(join(root_dir, f'ET-{et_label}', ID, 'T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(root_dir, f'ET-{et_label}', ID, 'T2F.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(root_dir, f'ET-{et_label}', ID, 'ROI.nii.gz'))
            syn_T1C, _, _, _ = NiiDataRead(join(syn_T1Gd_dir, synET_ID+'.nii.gz'))

            T1 = harmonize_mr(T1)
            T2F = harmonize_mr(T2F)
            syn_T1C = harmonize_mr(syn_T1C)

            for slice in np.unique(ROI.nonzero()[0]):
                self.real_t1s.append(T1[slice])
                self.real_t2fs.append(T2F[slice])
                self.syn_t1cs.append(syn_T1C[slice])
                self.label_list.append(cls_label)

    def __len__(self):
        return len(self.real_t1s)

    def __getitem__(self, idx):
        T1 = self.real_t1s[idx]
        T2F = self.real_t2fs[idx]
        syn_t1c = self.syn_t1cs[idx]
        label = self.label_list[idx]

        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T2F = torch.from_numpy(T2F).unsqueeze(0).float()
        syn_t1c = torch.from_numpy(syn_t1c).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T2F = self.trans(T2F)
            syn_t1c = self.trans(syn_t1c)
        # Convert data to PyTorch tensor

        data = torch.cat((T1, T2F, syn_t1c), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return {
            'image': data,
            'label': label}


