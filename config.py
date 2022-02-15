import argparse
import torch
import numpy as np
import random

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser()
##$
parser.add_argument('--mem_bank', default=False, type=boolean_string, help='whether to use mem_bank for statistic')
parser.add_argument('--triplet_type', default='full', type=str, help='checkpoint name for saving')
parser.add_argument('--encoder_contrastive_weight', default=0.0, type=float, help='weight for contrastive after encoder') 
parser.add_argument('--encoder_contrastive_margin', default=0.2, type=float, help='margin for contrastive after encoder') 
parser.add_argument('--contrastivelearning', default=False, type=boolean_string, help='whether to use cl')
parser.add_argument('--clean_subset', default=False, type=boolean_string, help='whether to use mem_bank for statistic')
parser.add_argument('--dataset_path_clean', default=['/home/yuweichen/workspace/dataset_clean_noisy/clean_subset'], type=str, nargs='+', help='path to dataset')
parser.add_argument('--dataset_path_noisy', default=['/home/yuweichen/workspace/dataset_clean_noisy/noisy_subset'], type=str, nargs='+', help='path to dataset')
parser.add_argument('--model_mix', default=False, type=boolean_string, help='whether to use mixmodel')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--cloth_loss_weight', default=0.0, type=float, help='weight for warmup_cloth_loss') 
parser.add_argument('--cloth_loss_margin', default=0.2, type=float, help='margin for warmup_cloth_loss') 
parser.add_argument('--mixup_weight', default=0.0, type=float, help='weight for warmup_cloth_loss') 
parser.add_argument('--warmup_iter', default=0, type=int, help='model_mix warm_up iteration')
parser.add_argument('--static_iter', default=0, type=int, help='model_mix for loss_stat iteration')
parser.add_argument('--reranking_graph', default=False, type=boolean_string, help='reranking or not')
parser.add_argument('--hiar_mem_bank', default=False, type=boolean_string, help='whether to use hiar_mem_bank or not')
parser.add_argument('--log_name', default='', type=str, help='for writing logs')
parser.add_argument('--spcloss_weight', default=0.0, type=float, help='weight for spcloss') 
parser.add_argument('--self_supervised_weight', default=0.0, type=float, help='weight for infonce') 
parser.add_argument('--da_iter', default=20000, type=int, help='after this iter starts da iteration')
parser.add_argument('--restore_name', default='', type=str, help='for writing logs')
parser.add_argument('--infonce_git_weight', default=0.0, type=float, help='weight for infonce loss from github') 
parser.add_argument('--model_usl', default=False, type=boolean_string, help='whether to use ModelUSL as Model')


parser.add_argument('--gpu', default='0,1,2,3', type=str, help='gpu id')
parser.add_argument('--model_name', default='GaitSet', type=str, help='checkpoint name for saving')
parser.add_argument('--random_seed', default=2020, type=int, help='random_seed')
#data
parser.add_argument('--dataset', default='CASIA-B', type=str, help='name of dataset')
parser.add_argument('--dataset_path', default=['/home2/ywc/workspace/noisy_output_pkl_6444'], type=str, nargs='+', help='path to dataset')
parser.add_argument('--dataset_augment', default=False, type=boolean_string, help='dataset augmentation')
parser.add_argument('--check_frames', default=True, type=boolean_string, help='check minimum frames for each seq')
parser.add_argument('--check_resolution', default=True, type=boolean_string, help='check resolution for each dataset')
parser.add_argument('--resolution', default=64, type=int, help='image resolution')
parser.add_argument('--pid_num', default=73, type=int, help='split train and test')
parser.add_argument('--pid_shuffle', default=False, type=boolean_string, help='shuffle dataset or not')
parser.add_argument('--num_workers', default=48, type=int, help='workers to load data')
parser.add_argument('--frame_num', default=30, type=int, help='frames per sequence')
parser.add_argument('--batch_size', default=[8, 16], type=int, nargs='+', help='batch size')
parser.add_argument('--sample_type', default='random', type=str, choices=['random', 'random_fn', 'all'], help='sample type')
parser.add_argument('--min_frame_num', default=20, type=int, help='min frame_num for random_fn')
parser.add_argument('--max_frame_num', default=40, type=int, help='max frame_num for random_fn')
parser.add_argument('--label_smooth', default=True, type=boolean_string, help='label smooth')
#optimizer
parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'ADAM'], type=str, help='SGD or ADAM')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for SGD')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--milestones', default=[10000, 20000, 30000], type=int, nargs='+', help='milestones for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for SGD')
parser.add_argument('--init_model', default=None, type=str, help='checkpoint name for initialization')
parser.add_argument('--restore_iter', default=0, type=int, help='restore iteration')
parser.add_argument('--total_iter', default=40000, type=int, help='total iteration')
parser.add_argument('--warmup', default=True, type=boolean_string, help='warm up')
parser.add_argument('--AMP', default=False, type=boolean_string, help='automatic mixed precision')
parser.add_argument('--DDP', default=False, type=boolean_string, help='distributed data parallel')
parser.add_argument('--local_rank', default=0, type=int, help='local rank for DDP')
#encoder
parser.add_argument('--bin_num', default=[1,2,4,8,16], type=int, nargs='+', help='bin num')
parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim')
parser.add_argument('--more_channels', default=False, type=boolean_string, help='more channels for large datasets')
parser.add_argument('--encoder_triplet_margin', default=0.2, type=float, help='margin for triplet after encoder') # triplet
parser.add_argument('--encoder_triplet_weight', default=1, type=float, help='weight for triplet after encoder') # triplet
parser.add_argument('--encoder_entropy_weight', default=0.1, type=float, help='weight for entropy after encoder') # entropy
###################################################################################################
#test
parser.add_argument('--test_set', default='test', choices=['train', 'test'], type=str, help='train or test set for eval')
parser.add_argument('--ckp_prefix', default=None, type=str, help='ckp_prefix: prefix of the checkpoint to load')
parser.add_argument('--feat_idx', default=0, type=int, help='feat index')
parser.add_argument('--euc_or_cos_dist', default='euc', type=str, help='euclidean or cosine distance for test')
parser.add_argument('--cos_sim_thres', default=0.75, type=float, help='cosine distance threshold')
parser.add_argument('--rank', default=[1], type=int, nargs='+', help='rank list for show')
parser.add_argument('--max_rank', default=20, type=int, help='max rank for CMC')
parser.add_argument('--reranking', default=False, type=boolean_string, help='reranking or not')
parser.add_argument('--relambda', default=0.7, type=float, help='lambda for re-ranking')
parser.add_argument('--remove_no_gallery', default=False, type=boolean_string, help='remove those thave have no gallery')
parser.add_argument('--resume', default=False, type=boolean_string, help='resume or not')
###################################################################################################
config = vars(parser.parse_args())
print('#######################################')
print("Config:", config)
print('#######################################')

SEED=config['random_seed']
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False