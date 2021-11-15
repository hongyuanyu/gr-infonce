import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2

class DataSet(tordata.Dataset):
    def __init__(self, seq_dir_list, seq_label_list, index_dict, resolution, cut_padding):
        self.seq_dir_list = seq_dir_list
        self.seq_label_list = seq_label_list
        self.index_dict = index_dict
        self.resolution = int(resolution)
        self.cut_padding = int(cut_padding)
        self.data_size = len(self.seq_label_list)
        self.label_set = sorted(list(set(self.seq_label_list)))

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
        return seq_imgs, seq_label

    def img2xarray(self, file_path):
        pkl_name = '{}.pkl'.format(os.path.basename(file_path))
        all_imgs = pickle.load(open(osp.join(file_path, pkl_name), 'rb'))
        return all_imgs

    def __len__(self):
        return self.data_size
