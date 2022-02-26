import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2

class DataSet(tordata.Dataset):
    def __init__(self, dataset, seq_dir_list, seq_label_list, seq_label_clean_list, seq_type_list, index_dict, resolution, cut_padding):
        self.seq_dir_list = seq_dir_list
        self.seq_label_list = seq_label_list
        self.seq_label_clean_list = seq_label_clean_list
        self.index_dict = index_dict
        self.resolution = int(resolution)
        self.cut_padding = int(cut_padding)
        self.data_size = len(self.seq_label_list)
        self.label_set = sorted(list(set(self.seq_label_list)))
        self.seq_type_list = seq_type_list
        self.seq_type_set = sorted(list(set(self.seq_type_list)))
        self.dataset = dataset

    def __loader__(self, path):
        if self.cut_padding > 0:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0
        else: 
            return self.img2xarray(
                path).astype(
                'float32') / 255.0

    def __getitem__(self, index):
        seq_path = self.seq_dir_list[index]
        seq_imgs = self.__loader__(seq_path)
        seq_label = self.seq_label_list[index]
        seq_label_clean = self.seq_label_clean_list[index]
        seq_type = self.seq_type_list[index]
        return seq_imgs, seq_label, seq_label_clean, seq_type, index

    def img2xarray(self, file_path):
        if self.dataset == 'oumvlp':
            pkl_name = 'seqs.pkl'
        elif self.dataset == 'Outdoor':
            pkl_name = os.listdir(file_path)[0]
        else:
            pkl_name = '{}.pkl'.format(os.path.basename(file_path))
        all_imgs = pickle.load(open(osp.join(file_path, pkl_name), 'rb'))
        return all_imgs

    def __len__(self):
        return self.data_size
