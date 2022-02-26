import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from .re_ranking import re_ranking
from .metric import compute_CMC_mAP, compute_PR

def cuda_euc_dist(x, y):
    x = x.permute(1, 0, 2).contiguous() # num_parts * num_probe * part_dim
    y = y.permute(1, 0, 2).contiguous() # num_parts * num_gallery * part_dim
    dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
        2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2)) # num_parts * num_probe * num_gallery
    dist = torch.sqrt(F.relu(dist)) # num_parts * num_probe * num_gallery
    dist = torch.mean(dist, 0) # num_probe * num_gallery
    return dist

def cuda_cos_dist(x, y):
    x = F.normalize(x, p=2, dim=2).permute(1, 0, 2) # num_parts * num_probe * part_dim
    y = F.normalize(y, p=2, dim=2).permute(1, 0, 2) # num_parts * num_gallery * part_dim
    dist = 1 - torch.mean(torch.matmul(x, y.transpose(1, 2)), 0) # num_probe * num_gallery
    return dist

def evaluation(data, config):
    print("############################")
    if config['euc_or_cos_dist'] == 'euc':
        print("Compute Euclidean Distance")
    elif config['euc_or_cos_dist'] == 'cos':
        print("Compute Cosine Distance")
    else:
        print('Illegal Distance Type')
        os._exit(0)
    print("############################")

    dataset = config['dataset']
    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'CASIAC': [['H_scene1_nm_1', 'H_scene1_nm_2', 'H_scene1_nm_3', 'H_scene1_nm_4', 'H_scene2_nm_1', 'H_scene2_nm_2', 'H_scene2_nm_3', 'H_scene2_nm_4',
                                    'L_scene1_nm_1', 'L_scene1_nm_2', 'L_scene1_nm_3', 'L_scene1_nm_4', 'L_scene2_nm_1', 'L_scene2_nm_2', 'L_scene2_nm_3', 'L_scene2_nm_4',], 
                                  [ 'H_scene1_bg_1', 'H_scene1_bg_2', 'H_scene1_bg_3', 'H_scene1_bg_4', 'H_scene2_bg_1', 'H_scene2_bg_2', 'H_scene2_bg_3', 'H_scene2_bg_4',
                                    'L_scene1_bg_1', 'L_scene1_bg_2', 'L_scene1_bg_3', 'L_scene1_bg_4', 'L_scene2_bg_1', 'L_scene2_bg_2', 'L_scene2_bg_3', 'L_scene2_bg_4', ],
                                  [ 'H_scene1_cl_1', 'H_scene1_cl_2', 'H_scene1_cl_3', 'H_scene1_cl_4', 'H_scene2_cl_1', 'H_scene2_cl_2', 'H_scene2_cl_3', 'H_scene2_cl_4',
                                    'L_scene1_cl_1', 'L_scene1_cl_2', 'L_scene1_cl_3', 'L_scene1_cl_4', 'L_scene2_cl_1', 'L_scene2_cl_2', 'L_scene2_cl_3', 'L_scene2_cl_4', ]],
                    'CASIA-EN': [['H_scene1_nm_1', 'H_scene1_nm_2', 'H_scene2_nm_1', 'H_scene2_nm_2', 'L_scene1_nm_1', 'L_scene1_nm_2', 'L_scene2_nm_1', 'L_scene2_nm_2'], 
                                  ['H_scene1_bg_1', 'H_scene1_bg_2', 'H_scene2_bg_1', 'H_scene2_bg_2', 'L_scene1_bg_1', 'L_scene1_bg_2', 'L_scene2_bg_1', 'L_scene2_bg_2'],
                                  ['H_scene1_cl_1', 'H_scene1_cl_2', 'H_scene2_cl_1', 'H_scene2_cl_2', 'L_scene1_cl_1', 'L_scene1_cl_2', 'L_scene2_cl_1', 'L_scene2_cl_2']],
                      }
    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'CASIA-EN': [['H_scene3_nm_1', 'H_scene3_nm_2', 'L_scene3_nm_1', 'L_scene3_nm_2']],
                        'CASIAC': [['H_scene3_nm_1', 'H_scene3_nm_2', 'H_scene3_nm_3', 'H_scene3_nm_4', 'L_scene3_nm_1', 'L_scene3_nm_2','L_scene3_nm_2', 'L_scene3_nm_4']]
                        }
    if dataset not in probe_seq_dict.keys():
        evaluation_real(data, config)
        os._exit(0)

    feature, view, seq_type, label = data
    label = np.asarray(label)
    view_list = sorted(list(set(view)))
    view_num = len(view_list)
    sample_num = len(feature)

    print("############################")
    print("Feature Shape: ", feature.shape)
    print("############################")

    CMC = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, config['max_rank']])
    mAP = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    P_thres = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    R_thres = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_y = label[gseq_mask]
                    gseq_mask = torch.from_numpy(np.asarray(gseq_mask, dtype=np.uint8))
                    gallery_x = feature[gseq_mask, :, :]  #取出特定type和view的gallary的x和y

                    if config['remove_no_gallery']:
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)
                    else:
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_y = label[pseq_mask]
                    pseq_mask = torch.from_numpy(np.asarray(pseq_mask, dtype=np.uint8))
                    probe_x = feature[pseq_mask, :, :]  #取出特定type和view的probe的x和y

                    if config['reranking']:
                        #assert(config['euc_or_cos_dist'] == 'cos')
                        if config['euc_or_cos_dist'] == 'euc':
                            dist_p_p = 1 - cuda_euc_dist(probe_x, probe_x).cpu().numpy()
                            dist_p_g = 1 - cuda_euc_dist(probe_x, gallery_x).cpu().numpy()
                            dist_g_g = 1 - cuda_euc_dist(gallery_x, gallery_x).cpu().numpy()
                            dist = re_ranking(dist_p_g, dist_p_p, dist_g_g, lambda_value=config['relambda'])
                        elif config['euc_or_cos_dist'] == 'cos':
                            dist_p_p = 1 - cuda_cos_dist(probe_x, probe_x).cpu().numpy()
                            dist_p_g = 1 - cuda_cos_dist(probe_x, gallery_x).cpu().numpy()
                            dist_g_g = 1 - cuda_cos_dist(gallery_x, gallery_x).cpu().numpy()
                            dist = re_ranking(dist_p_g, dist_p_p, dist_g_g, lambda_value=config['relambda'])
                    else:
                        if config['euc_or_cos_dist'] == 'euc':
                            dist = cuda_euc_dist(probe_x, gallery_x)
                        elif config['euc_or_cos_dist'] == 'cos':
                            dist = cuda_cos_dist(probe_x, gallery_x)
                        dist = dist.cpu().numpy()
                    eval_results = compute_CMC_mAP(dist, probe_y, gallery_y, config['max_rank'])
                    CMC[p, v1, v2, :] = np.round(eval_results[0] * 100, 2)
                    mAP[p, v1, v2] = np.round(eval_results[1] * 100, 2)
                    if config['euc_or_cos_dist'] == 'cos' and config['cos_sim_thres'] > -1:
                        eval_results = compute_PR(dist, probe_y, gallery_y, config['cos_sim_thres'])
                        P_thres[p, v1, v2] = np.round(eval_results[0] * 100, 2)
                        R_thres[p, v1, v2] = np.round(eval_results[1] * 100, 2)

    return CMC, mAP, [P_thres, R_thres]

def evaluation_real(data, config):
    dataset = config['dataset']
    gallery_seq_dict = {'WATRIX_A': ['type-01'+'view-000']
                        }
    gallery_seq = gallery_seq_dict[dataset]

    feature, view, seq_type, label = data
    label = np.asarray(label)
    view_list = sorted(list(set(view)))
    type_list = sorted(list(set(seq_type)))
    view_num = len(view_list)
    type_num = len(type_list)
    sample_num = len(feature)
    assert( len(seq_type) == len(view) )
    seq_type_view = [ seq_type[i]+view[i] for i in range(len(seq_type)) ]

    print("############################")
    print("Feature Shape: ", feature.shape)
    print("############################")

    CMC = np.zeros([type_num, config['max_rank']])
    mAP = np.zeros(type_num)
    P_thres = np.zeros(type_num)
    R_thres = np.zeros(type_num)
    for _idx, _type in enumerate(type_list):
        gseq_mask = np.isin(seq_type_view, gallery_seq)
        gallery_y = label[gseq_mask]
        gseq_mask = torch.from_numpy(np.asarray(gseq_mask, dtype=np.uint8))
        gallery_x = feature[gseq_mask, :, :]

        if config['remove_no_gallery']:
            pseq_mask = np.isin(seq_type, [_type]) & np.isin(seq_type_view, gallery_seq, invert=True) & np.isin(label, gallery_y)
        else:
            pseq_mask = np.isin(seq_type, [_type]) & np.isin(seq_type_view, gallery_seq, invert=True)
        probe_y = label[pseq_mask]
        pseq_mask = torch.from_numpy(np.asarray(pseq_mask, dtype=np.uint8))
        probe_x = feature[pseq_mask, :, :]

        if config['reranking']:
            assert(config['euc_or_cos_dist'] == 'cos')
            dist_p_p = 1 - cuda_cos_dist(probe_x, probe_x).cpu().numpy()
            dist_p_g = 1 - cuda_cos_dist(probe_x, gallery_x).cpu().numpy()
            dist_g_g = 1 - cuda_cos_dist(gallery_x, gallery_x).cpu().numpy()
            dist = re_ranking(dist_p_g, dist_p_p, dist_g_g, lambda_value=config['relambda'])
        else:
            if config['euc_or_cos_dist'] == 'euc':
                dist = cuda_euc_dist(probe_x, gallery_x)
            elif config['euc_or_cos_dist'] == 'cos':
                dist = cuda_cos_dist(probe_x, gallery_x)
            dist = dist.cpu().numpy()
        print('type={}, dist={}'.format(_type, dist.shape))
        eval_results = compute_CMC_mAP(dist, probe_y, gallery_y, config['max_rank'])
        CMC[_idx, :] = np.round(eval_results[0] * 100, 2)
        mAP[_idx] = np.round(eval_results[1] * 100, 2)
        if config['euc_or_cos_dist'] == 'cos' and config['cos_sim_thres'] > -1:
            eval_results = compute_PR(dist, probe_y, gallery_y, config['cos_sim_thres'])
            P_thres[_idx] = np.round(eval_results[0] * 100, 2)
            R_thres[_idx] = np.round(eval_results[1] * 100, 2)

    rank_list = np.asarray(config['rank']) - 1
    for _idx, _type in enumerate(type_list):
        for i in rank_list:
            print("Rank-{} Acc for {}: {}".format(i+1, _type, CMC[_idx, i]))
        print("mAP for {}: {}".format(_type, mAP[_idx]))
        if config['euc_or_cos_dist'] == 'cos' and config['cos_sim_thres'] > -1:
            print("Precision@COS_SIM_THRES={} for {}: {}".format(config['cos_sim_thres'], _type, P_thres[_idx]))
            print("Recall@COS_SIM_THRES={} for {}: {}".format(config['cos_sim_thres'], _type, R_thres[_idx]))

    os._exit(0)
