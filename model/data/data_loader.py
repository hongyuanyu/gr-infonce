import os
import os.path as osp

import numpy as np
import pickle

from .data_set import DataSet
from pdb import set_trace

def load_data(config):
    print("############################")
    dataset, dataset_path, resolution, pid_num, pid_shuffle = \
        config['dataset'], config['dataset_path'], config['resolution'], config['pid_num'], config['pid_shuffle']
    print("dataset={}, dataset_path={}, resolution={}".format(dataset, dataset_path, resolution))
    print("############################")
    seq_dir_list = list()
    seq_id_list = list()
    seq_type_list = list()
    seq_id_clean_list = list()

    cut_padding = int(float(resolution)/64*10)
    check_frames = config['check_frames']
    for i, dataset_path_i in enumerate(dataset_path):
        prefix = "ds{}-".format(i+1) if i > 0 else ''
        print("dataset_path={}, prefix={}".format(dataset_path_i, prefix))
        check_resolution = config['check_resolution']
        for _id in sorted(list(os.listdir(dataset_path_i))):
            # In CASIA-B, data of subject #5 is incomplete. Thus, we ignore it in training.
            if dataset == 'CASIA-B' and _id.split('_')[0] == '005':
                continue
            id_path = osp.join(dataset_path_i, _id)
            for _type in sorted(list(os.listdir(id_path))):
                type_path = osp.join(id_path, _type)
                for _view in sorted(list(os.listdir(type_path))):
                    view_path = osp.join(type_path, _view)
                    if check_frames:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        if all_imgs.shape[0] < 15:
                            continue
                    seq_dir_list.append(view_path)
                    seq_id_list.append(prefix+_id)
                    seq_type_list.append(_type)
                    seq_id_clean_list.append(prefix+_id[:3])
                    #############################################################
                    if check_resolution:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        assert(all_imgs.shape[1]==resolution)
                        assert(all_imgs.shape[2]==resolution or all_imgs.shape[2]==(resolution-2*cut_padding))
                        check_resolution = False
                        print("Check Resolution: view_path={}, resolution={}, cut_padding={}, img_shape={}".format(\
                                view_path, resolution, cut_padding, all_imgs.shape))
                    #############################################################
    if all_imgs.shape[2]==(resolution-2*cut_padding):
        cut_padding = 0

    total_id = len(list(set(seq_id_list)))
    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(seq_id_list)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        assert(pid_num <= len(pid_list))
        if pid_num == 0:
            #the first three for training (only for convenience) and all for test
            pid_list = [pid_list[0:3], pid_list[:]]
        elif pid_num == -1: 
            #all for training and the last three for test (only for convenience)
            pid_list = [pid_list[:], pid_list[-3:]]
        else:
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)

    pid_list = np.load(pid_fname, allow_pickle=True)
    train_id_list = pid_list[0]
    test_id_list = pid_list[1]

    print("############################")
    print("pid_fname:", pid_fname)
    print("resolution={}, cut_padding={}".format(resolution, cut_padding))

    print("number of ids for train:", len(pid_list[0]))
    print_num = min(30, len(pid_list[0]))
    print("example ids for train:", pid_list[0][0:print_num])

    print("number of ids for test:", len(pid_list[1]))
    print_num = min(30, len(pid_list[1]))
    print("example ids for test:", pid_list[1][0:print_num])
    print("############################")

    # train source
    train_seq_dir_list = [seq_dir_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in train_id_list]
    train_seq_id_list = [seq_id_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in train_id_list]
    train_seq_type_list = [seq_type_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in train_id_list]
    train_seq_id_clean_list = [seq_id_clean_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in train_id_list]
    
    train_index_info = {}
    for i, l in enumerate(train_seq_id_list):
        if l not in train_index_info.keys():
            train_index_info[l] = []
        train_index_info[l].append(i)
    train_source = DataSet(train_seq_dir_list, train_seq_id_list, train_seq_id_clean_list, train_seq_type_list, train_index_info, resolution, cut_padding)

    # test source
    test_seq_dir_list = [seq_dir_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in test_id_list]
    test_seq_id_list = [seq_id_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in test_id_list]
    test_seq_type_list = [seq_type_list[i] for i, l in enumerate(seq_id_list) if l.split('_')[0] in test_id_list]

    test_index_info = {}
    for i, l in enumerate(test_seq_id_list):
        if l not in test_index_info.keys():
            test_index_info[l] = []
        test_index_info[l].append(i)
    test_source = DataSet(test_seq_dir_list, test_seq_id_list, test_seq_id_list, test_seq_type_list, test_index_info, resolution, cut_padding)

    print('train label set={}, total={}'.format(sorted(list(set(train_seq_id_list))), len(list(set(train_seq_id_list)))))
    print('test label set={}, total={}'.format(sorted(list(set(test_seq_id_list))), len(list(set(test_seq_id_list)))))   

    return train_source, test_source
