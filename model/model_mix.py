import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import TripletSampler, DistributedTripletSampler, build_data_transforms, SeqSampler
from .loss import DistributedLossWrapper, CenterLoss, CrossEntropyLabelSmooth, TripletLoss, ClothLoss, SPCLoss #RerankingGraph  #PartTripletLoss,
from .solver import WarmupMultiStepLR
from .network import GaitSet
from .network.sync_batchnorm import DataParallelWithCallback
from pdb import set_trace
import torch.nn.functional as F
import pickle 
from sklearn.cluster import DBSCAN

class ModelMix:
    def __init__(self, config):
        self.config = deepcopy(config)
        if self.config['DDP']:
            torch.cuda.set_device(self.config['local_rank'])
            dist.init_process_group(backend='nccl')
            self.config['encoder_entropy_weight'] *= dist.get_world_size()
            self.config['encoder_triplet_weight'] *= dist.get_world_size()
            self.random_seed = self.config['random_seed'] + dist.get_rank()
        else:
            self.random_seed = self.config['random_seed']
        
        self.config.update({'num_id_clean': len(self.config['train_source_clean'].label_set)})
        self.config.update({'num_id_noisy': len(self.config['train_source_noisy'].label_set)})
        self.config.update({'num_id_all': len(self.config['train_source_all'].label_set)})
        self.config.update({'k': 5})
        self.alpha = 0.99

        self.encoder = GaitSet(self.config).float().cuda()
        if self.config['DDP']:
            self.encoder = DDP(self.encoder, device_ids=[self.config['local_rank']], output_device=self.config['local_rank'], find_unused_parameters=True)
        else:
            self.encoder = DataParallelWithCallback(self.encoder)
        self.build_data()
        self.build_loss()
        self.build_loss_metric()
        self.build_optimizer()

        if self.config['DDP']:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        
        if self.config['mem_bank']:
            num_of_id = self.config['num_id_all']  #117
            self.mem_bank = torch.zeros(num_of_id, sum(self.config['bin_num']), self.config['hidden_dim']+4).cuda()  #min, max, mean, sigma  116*16*260+3
            print('self.mem_bank.shape', self.mem_bank.shape)
            self.mem_statistic = list()
            if self.config['hiar_mem_bank']:
                self.clean_flag = torch.cat((torch.zeros(self.config['num_id_clean']),torch.ones(self.config['num_id_noisy'])),dim=0).cuda()
                self.M = 3
                self.hiar_mem_bank = torch.zeros(num_of_id, sum(self.config['bin_num']), self.config['hidden_dim']+self.M).cuda()  #3层的label
                self.hiar_mem_bank[:,:,-3] = torch.tensor(range(num_of_id)).unsqueeze(1).repeat(1, sum(self.config['bin_num']))
                #self.clean_label = torch.cat((torch.tensor(range(43)).unsqueeze(1).repeat(1,2).view(-1),torch.tensor(range(43,73)))).cuda()
                self.clean_label = torch.cat((torch.tensor(range(0,86,2)).unsqueeze(1).repeat(1,2).view(-1),torch.tensor(range(86,116)))).cuda()

    def build_data(self):
        # data augment
        if self.config['dataset_augment']:
            self.data_transforms = build_data_transforms(random_erasing=False, random_rotate=False, \
                                        random_horizontal_flip=False, random_pad_crop=False, cloth_dilate=True,\
                                        resolution=self.config['resolution'], random_seed=self.random_seed) 
        
        #triplet sampler
        if self.config['DDP']:
            self.triplet_sampler = DistributedTripletSampler(self.config['train_source'], self.config['batch_size'], random_seed=self.random_seed)
        else:
            self.triplet_sampler = TripletSampler(self.config['train_source_clean'], self.config['batch_size'])
            self.seq_sampler = SeqSampler(self.config['train_source_clean'], self.config['batch_size'])

    def build_loss(self):
        if self.config['encoder_entropy_weight'] > 0:
            if self.config['label_smooth']:
                self.encoder_entropy_loss = CrossEntropyLabelSmooth(self.config['num_id_all']).float().cuda()
            else:
                self.encoder_entropy_loss = nn.CrossEntropyLoss().float().cuda()
            if self.config['DDP']:
                self.encoder_entropy_loss = DistributedLossWrapper(self.encoder_entropy_loss, dim=0)

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss = TripletLoss(self.config['encoder_triplet_margin'], self.config['triplet_type']).float().cuda()
            if self.config['DDP']:
                self.encoder_triplet_loss = DistributedLossWrapper(self.encoder_triplet_loss, dim=1)
        if self.config['cloth_loss_weight'] > 0:
            self.cloth_loss = ClothLoss(self.config['cloth_loss_margin'], )
        if self.config['reranking_graph']:
            self.spcloss = SPCLoss(temperature=0.05)
    
    def build_loss_metric(self):
        if self.config['encoder_entropy_weight'] > 0:
            self.encoder_entropy_loss_metric = [[], []]

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss_metric = [[], []]

        if self.config['cloth_loss_weight'] > 0:
            self.cloth_loss_metric = [[]]
        if self.config['spcloss_weight'] > 0:
            self.spcloss_metric = [[],[],[],[]]
        self.total_loss_metric = []
    
    def build_optimizer(self):
        #params
        tg_params = self.encoder.parameters()
 
        #optimizer
        if self.config['optimizer_type'] == 'SGD':
            self.optimizer = optim.SGD(tg_params, lr=self.config['lr'], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        elif self.config['optimizer_type'] == 'ADAM': #if ADAM set the first stepsize equal to total_iter
            self.optimizer = optim.Adam(tg_params, lr=self.config['lr'])
        if self.config['warmup']:
            self.scheduler = WarmupMultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])

        #AMP
        if self.config['AMP']:
            self.scaler = GradScaler()

    def renew_mem_bank_all(self, mem_bank, batch_size=1, flag='train'):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source_all']
        #self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        #view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        #seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()
        id_num_statistic = torch.zeros(mem_bank.shape[0]).cuda()
        for i, x in enumerate(data_loader):
            seq, label, label_clean, seq_type, index, batch_frame = x
            seq = torch.from_numpy(seq).float().cuda()
            if batch_frame is not None:
                batch_frame = torch.from_numpy(batch_frame).int().cuda()
            output = self.encoder(seq, batch_frame)
            feature = output[0]
            feature_list.append(feature.detach())             
            label_list += label
            #index = source.label_set.index(label[0])
            #mem_bank[index,:,:-4] += feature[0]
            #id_num_statistic[index] += 1
        #mem_bank[:,:,:-4] = mem_bank[:,:,:-4] / id_num_statistic.unsqueeze(1).unsqueeze(1)
        feature_list = torch.cat(feature_list, 0)
        label_list = np.array(label_list)
        #更新mem_bank
        for i in range(len(source.label_set)):  #label和顺序的对应关系
            feature_i_all = feature_list[np.argwhere(label_list == source.label_set[i])[:,0],:,:]  #nums*16*256
            mem_bank[i,:,:-4] = 0.1*mem_bank[i,:,:-4] + 0.9*torch.mean(feature_i_all, dim=0)  #16*256
            dist_inclass = self.cuda_euc_dist(feature_i_all, mem_bank[i,:,:-4].unsqueeze(0))  #16*88(nums)
            mem_bank[i,:,-4] = torch.mean(dist_inclass, dim=1)[:,0]  # 16
            mem_bank[i,:,-3] = torch.min(dist_inclass, dim=1)[0][:,0]
            mem_bank[i,:,-2] = torch.max(dist_inclass, dim=1)[0][:,0]
            mem_bank[i,:,-1] = torch.var(dist_inclass, dim=1)[:,0]

        return mem_bank

    def dist_statistic(self, mem_statistic, iteration, mem_bank):
        interclass_dist = self.cuda_euc_dist(mem_bank[:,:,:-4], mem_bank[:,:,:-4])  #116*31*256   16*116*116
        mem_statistic.append(torch.cat((   # 16*116
            torch.max(interclass_dist, dim=2)[0].unsqueeze(2), torch.sort(interclass_dist, dim=2)[0][:,:,1:4], torch.sort(interclass_dist, dim=2)[1][:,:,1:4], torch.mean(interclass_dist, dim=2).unsqueeze(2),
            mem_bank[:,:,-4:].permute(1,0,2)), dim=2).unsqueeze(0))  #116*16*4  output=1*16*116*8
        if iteration == self.config['warmup_iter']:
            statistic_path = '/home/yuweichen/workspace/noisy_gait/visualize_feature'
            if not osp.exists(statistic_path):
                os.makedirs(statistic_path)
            statistic_file_path = statistic_path+'statistic_file_dist_100.pkl'
            mem_statistic = torch.cat(mem_statistic, dim=0)
            with open(statistic_file_path, 'wb') as f:
                pickle.dump(mem_statistic, f)
            if mem_statistic.shape[0] == 1:
                _ = range(86)
                nearest_right_index = torch.tensor([[_[2*i]+1,_[2*i+1]-1] for i in range(43)]).view(-1).unsqueeze(1).repeat(1,3).cuda()
                nearest_right_all = (mem_statistic[0,0,0:86,4:7] == nearest_right_index).sum()
                nearest_right_top1 = (mem_statistic[0,0,0:86,4] == nearest_right_index[:,0]).sum()
                print('nearest_right_all', nearest_right_all)
                print('nearest_right_top1', nearest_right_top1)
        return mem_statistic

    def cuda_euc_dist(self, x, y):
        x = x.permute(1, 0, 2).contiguous() # num_parts * num_probe * part_dim
        y = y.permute(1, 0, 2).contiguous() # num_parts * num_gallery * part_dim
        #set_trace()
        dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2)) # num_parts * num_probe * num_gallery
        dist = torch.sqrt(F.relu(dist)) # num_parts * num_probe * num_gallery
        #dist = torch.mean(dist, 0) # num_probe * num_gallery
        return dist
    
    def print_info(self):
        print('iter {}:'.format(self.config['restore_iter']))

        def print_loss_info(loss_name, loss_metric, loss_weight, loss_info):
            print('{:#^30}: loss_metric={:.6f}, loss_weight={:.6f}, {}'.format(loss_name, np.mean(loss_metric), loss_weight, loss_info))

        if self.config['encoder_entropy_weight'] > 0:
            loss_name = 'Encoder Entropy'
            loss_metric = self.encoder_entropy_loss_metric[0]
            loss_weight = self.config['encoder_entropy_weight']
            loss_info = 'label_smooth={}'.format(self.config['label_smooth'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['encoder_triplet_weight'] > 0:
            loss_name = 'Encoder Triplet'
            loss_metric = self.encoder_triplet_loss_metric[0]
            loss_weight = self.config['encoder_triplet_weight']
            loss_info = 'nonzero_num={:.6f}, margin={}'.format(np.mean(self.encoder_triplet_loss_metric[1]), self.config['encoder_triplet_margin'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['mixup_weight'] > 0:
            loss_name = 'mixup_loss'
            loss_metric = self.encoder_entropy_loss_metric[1]
            loss_weight = self.config['encoder_entropy_weight']
            loss_info = 'label_smooth={}'.format(self.config['label_smooth'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['cloth_loss_weight'] > 0:
            loss_name = 'Cloth_loss'
            loss_metric = self.cloth_loss_metric[0]
            loss_weight = self.config['cloth_loss_weight']
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info=None)

        if self.config['spcloss_weight'] > 0:
            loss_name = 'spcloss'
            loss_metric = self.spcloss_metric
            loss_weight = self.config['spcloss_weight']
            #print_loss_info(loss_name, loss_metric, loss_weight, loss_info=None)
            print(loss_name,loss_metric)

        print('{:#^30}: total_loss_metric={:.6f}'.format('Total Loss', np.mean(self.total_loss_metric)))
        
        #optimizer
        print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}'.format( \
            'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay']))            
        sys.stdout.flush()

    def transform(self, flag, batch_size=1, feat_idx=0):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source_noisy']
        self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, label, label_clean, seq_type, index, batch_frame = x
            seq = self.np2var(seq).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            output = self.encoder(seq, batch_frame)
            feature = output[feat_idx]
            feature_list.append(feature.detach())             
            label_list += label
        set_trace()
        return torch.cat(feature_list, 0), view_list, seq_type_list, label_list

    def collate_fn(self, batch):
        batch_size = len(batch)
        seqs = [batch[i][0] for i in range(batch_size)]
        label = [batch[i][1] for i in range(batch_size)]
        label_clean = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        index = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, label, label_clean, seq_type, index, None]
        batch_frames = []
        if self.config['DDP']:
            gpu_num = 1
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)

        # generate batch_frames for next step
        for gpu_id in range(gpu_num):
            batch_frames_sub = []
            for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                if i < batch_size:
                    if self.config['sample_type'] == 'random':
                        batch_frames_sub.append(self.config['frame_num'])
                    elif self.config['sample_type'] == 'all':
                        batch_frames_sub.append(seqs[i].shape[0])
                    elif self.config['sample_type'] == 'random_fn':
                        frame_num = np.random.randint(self.config['min_frame_num'], self.config['max_frame_num'])
                        batch_frames_sub.append(frame_num)
            batch_frames.append(batch_frames_sub)
        if len(batch_frames[-1]) != batch_per_gpu:
            for i in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        # select frames from each seq 
        def select_frame(index):
            sample = seqs[index]
            frame_set = np.arange(sample.shape[0])
            frame_num = batch_frames[int(index / batch_per_gpu)][int(index % batch_per_gpu)]
            if len(frame_set) >= frame_num:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=False))
            else:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=True))
            return sample[frame_id_list, :, :]
        seqs = list(map(select_frame, range(len(seqs))))        

        # data augmentation
        def transform_seq(index):
            sample = seqs[index]
            return self.data_transforms(sample)
        if self.config['dataset_augment']:
            seqs = list(map(transform_seq, range(len(seqs))))  

        

        # concatenate seqs for each gpu if necessary
        if self.config['sample_type'] == 'random':
            seqs = np.asarray(seqs)                      
        elif self.config['sample_type'] == 'all' or self.config['sample_type'] == 'random_fn':
            max_sum_frames = np.max([np.sum(batch_frames[gpu_id]) for gpu_id in range(gpu_num)])
            new_seqs = []
            for gpu_id in range(gpu_num):
                tmp = []
                for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                    if i < batch_size:
                        tmp.append(seqs[i])
                tmp = np.concatenate(tmp, 0)
                tmp = np.pad(tmp, \
                    ((0, max_sum_frames - tmp.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                new_seqs.append(np.asarray(tmp))
            seqs = np.asarray(new_seqs)  

        batch[0] = seqs
        if self.config['sample_type'] == 'all' or self.config['sample_type'] == 'random_fn':
            batch[-1] = np.asarray(batch_frames)
        
        return batch
   
    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x)) 

    def save(self):
        os.makedirs(osp.join('checkpoint', self.config['model_name']), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], self.config['restore_iter'])))
        torch.save([self.optimizer.state_dict(), self.scheduler.state_dict()],
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], self.config['restore_iter'])))

    def load(self, restore_iter):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.encoder.load_state_dict(encoder_ckp)
        optimizer_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.optimizer.load_state_dict(optimizer_ckp[0])
        self.scheduler.load_state_dict(optimizer_ckp[1])  

    def init_model(self, init_model):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_state_dict = self.encoder.state_dict()
        ckp_state_dict = torch.load(init_model, map_location=map_location)
        init_state_dict = {k: v for k, v in ckp_state_dict.items() if k in encoder_state_dict}
        drop_state_dict = {k: v for k, v in ckp_state_dict.items() if k not in encoder_state_dict}
        print('#######################################')
        if init_state_dict:
            print("Useful Layers in Init_model for Initializaiton:\n", init_state_dict.keys())
        else:
            print("None of Layers in Init_model is Used for Initializaiton.")
        print('#######################################')
        if drop_state_dict:
            print("Useless Layers in Init_model for Initializaiton:\n", drop_state_dict.keys())
        else:
            print("All Layers in Init_model are Used for Initialization.")
        encoder_state_dict.update(init_state_dict)
        none_init_state_dict = {k: v for k, v in encoder_state_dict.items() if k not in init_state_dict}
        print('#######################################')
        if none_init_state_dict:
            print("The Layers in Target_model that Are *Not* Initialized:\n", none_init_state_dict.keys())
        else:
            print("All Layers in Target_model are Initialized")  
        print('#######################################')      
        self.encoder.load_state_dict(encoder_state_dict)    

    def warmup(self, warmup_iter):
        warmup_loader = tordata.DataLoader(
            dataset=self.config['train_source_clean'],
            batch_sampler=self.triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])
        self.encoder.train()
        train_label_set = list(self.config['train_source_all'].label_set)
        train_label_set.sort()
        seq_type_set = list(self.config['train_source_clean'].seq_type_set)
        seq_type_set.sort()

        _time1 = datetime.now()

        for seq, label, label_clean, seq_type, index, batch_frame in warmup_loader:
            #set_trace()
            #############################################################
            if self.config['DDP'] and self.config['restore_iter'] > 0 and \
                self.config['restore_iter'] % self.triplet_sampler.total_batch_per_world == 0:
                self.triplet_sampler.set_random_seed(self.triplet_sampler.random_seed+1)
            #############################################################
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            target_label_clean = [train_label_set.index(l) for l in label_clean]
            target_label_clean = self.np2var(np.asarray(target_label_clean)).long()
            train_seq_type =  [seq_type_set.index(s) for s in seq_type]  #from str to int
            train_seq_type = self.np2var(np.asarray(train_seq_type)).long()
            
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            with autocast(enabled=self.config['AMP']):
                encoder_feature, encoder_bn_feature, encoder_cls_score \
                = self.encoder(seq, self.config['restore_iter'], batch_frame, target_label)

            loss = torch.zeros(1).to(encoder_feature.device)
            if self.config['encoder_entropy_weight'] > 0:
                entropy_loss_metric = 0
                for i in range(encoder_cls_score.size(1)):
                    entropy_loss_metric += self.encoder_entropy_loss(encoder_cls_score[:, i, :].float(), target_label, target_label_clean, l=0, loss_statistic=False)
                entropy_loss_metric = entropy_loss_metric / encoder_cls_score.size(1)
                loss += entropy_loss_metric * self.config['encoder_entropy_weight']
                self.encoder_entropy_loss_metric[0].append(entropy_loss_metric.mean().data.cpu().numpy())

            if self.config['encoder_triplet_weight'] > 0:
                encoder_triplet_feature = encoder_feature.float().permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                loss_statistic = False
                triplet_loss_metric, nonzero_num = self.encoder_triplet_loss(encoder_triplet_feature, triplet_label, label_clean, train_seq_type, loss_statistic)
                loss += triplet_loss_metric.mean() * self.config['encoder_triplet_weight']
                self.encoder_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.encoder_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())
            
            if self.config['cloth_loss_weight'] > 0:
                cloth_loss = self.cloth_loss(encoder_feature, target_label, train_seq_type)
                loss += cloth_loss.mean() * self.config['cloth_loss_weight']
                self.cloth_loss_metric[0].append(cloth_loss.mean().data.cpu().numpy())

            # mixmatch
            l = np.random.beta(self.config['alpha'], self.config['alpha'])        
            l = max(l, 1-l)
            all_inputs = seq
            all_targets = target_label

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b        
            mixed_target = [target_a, target_b]
            
            mix_feature, mix_bn_feature, mix_cls_score \
                = self.encoder(mixed_input, self.config['restore_iter'], batch_frame, mixed_target)
            
            if self.config['mixup_weight'] > 0:
                entropy_loss_mix = 0
                for i in range(mix_cls_score.size(1)):
                    entropy_loss_mix += self.encoder_entropy_loss(mix_cls_score[:, i, :].float(), mixed_target, l)
                entropy_loss_mix = entropy_loss_mix / mix_cls_score.size(1)
                loss += entropy_loss_mix * self.config['encoder_entropy_weight']
                self.encoder_entropy_loss_metric[1].append(entropy_loss_mix.mean().data.cpu().numpy())

            self.total_loss_metric.append(loss.data.cpu().numpy())

            if loss > 1e-9:
                if self.config['AMP']:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:  
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
            
            if self.config['restore_iter'] % 100 == 0:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    print(datetime.now() - _time1)
                    _time1 = datetime.now()
                    self.print_info()
                self.build_loss_metric()
            if self.config['restore_iter'] % 10000 == 0 or self.config['restore_iter'] == self.config['total_iter']:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    self.save()
            if self.config['restore_iter'] == warmup_iter:
                break
            self.config['restore_iter'] += 1

    def fit(self,):
        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0

        if self.config['restore_iter'] < self.config['warmup_iter']:
            self.warmup(self.config['warmup_iter'])
        if self.config['restore_iter'] >= self.config['warmup_iter'] and self.config['mem_bank']:
            
                #print('starting mem_bank and dist_statistic')
                #with torch.no_grad():
                #self.mem_bank = self.renew_mem_bank_all(self.mem_bank)
                #print('ending mem_bank and dist_statistic')
                #self.mem_statistic = self.dist_statistic(self.mem_statistic, self.config['restore_iter'], self.mem_bank)    #49,32
            if self.config['hiar_mem_bank']: 
                print('start hiar_mem_bank init')   
                num_of_id, bins, hidden_dim = self.hiar_mem_bank.shape
                self.hiar_mem_bank[:,:,:-3] = self.renew_mem_bank_all(self.mem_bank)[:,:,:-4]
                db_1 = DBSCAN(eps=0.8, min_samples=2).fit(self.hiar_mem_bank[:,0,:-3].cpu())
                feature_label_1 = deepcopy(self.hiar_mem_bank[:,:,:-1])
                feature_label_1[:,:,-1] = feature_label_1[:,:,-2]
                for i in set(db_1.labels_):
                    if i >=0:
                        feature_label_1[db_1.labels_==i,:,-1] = self.hiar_mem_bank[:,:,-3][np.argwhere(db_1.labels_==i)[0]]   #.repeat(((db_1.labels_==i).sum(), 1))
                        feature_label_1[db_1.labels_==i,:,:-2] = self.hiar_mem_bank[:,:,:-3][[np.argwhere(db_1.labels_==i)]].mean(0)  
                        
                self.hiar_mem_bank[:,:,-2] = feature_label_1[:,:,-1]
                
                db_2 = DBSCAN(eps=0.9, min_samples=2).fit(feature_label_1[:,0,:-2].cpu())
                self.hiar_mem_bank[:,:,-1] = feature_label_1[:,:,-1]
                for i in set(db_2.labels_):
                    if i >=0:
                        self.hiar_mem_bank[db_2.labels_==i,:,-1] = self.hiar_mem_bank[:,:,-2][np.argwhere(db_2.labels_==i)[0]]   #.repeat(((db_1.labels_==i).sum(), 1))
                acc1 = self.onezero_accuracy(self.hiar_mem_bank[:,0,-3], self.clean_label)
                acc2 = self.onezero_accuracy(self.hiar_mem_bank[:,0,-2], self.clean_label)
                acc3 = self.onezero_accuracy(self.hiar_mem_bank[:,0,-1], self.clean_label)
                del feature_label_1
                print('end hiar_mem_bank init, time = ', acc1, acc2, acc3) 
        
        if self.config['restore_iter'] >= self.config['warmup_iter'] and self.config['restore_iter'] < self.config['warmup_iter']+self.config['static_iter']+1:
            print('loss_statistic starts')
            #self.loss_statistic(self.config['warmup_iter']+self.config['static_iter'])
        if self.config['restore_iter'] >= self.config['warmup_iter'] and self.config['reranking_graph']:
            print('graph_train starts')
            self.graph_train(self.hiar_mem_bank, graph_train_iter=40000)
            print('graph_train ends')
    
    def gmm_divide(self, model, all_loss):    

        model.eval()
        losses = torch.zeros(self.config['num_id'])    
        eval_loader = tordata.DataLoader(
            dataset=self.config['train_source_noisy'],
            batch_sampler=self.triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        train_label_set = list(self.config['train_source'].label_set)
        train_label_set.sort()
        with torch.no_grad():
            for seq, label, seq_type, batch_frame in enumerate(eval_loader):
                seq = self.np2var(seq).float()
                target_label = [train_label_set.index(l) for l in label]
                target_label = self.np2var(np.asarray(target_label)).long()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                with autocast(enabled=self.config['AMP']):
                    encoder_feature, encoder_bn_feature, encoder_cls_score \
                    = self.encoder(seq, self.config['restore_iter'], batch_frame, target_label)

                loss = CE(outputs, targets)  
                for b in range(inputs.size(0)):
                    losses[index[b]]=loss[b]         
        losses = (losses-losses.min())/(losses.max()-losses.min())    
        all_loss.append(losses)

        if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1,1)
        else:
            input_loss = losses.reshape(-1,1)
        
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss) 
        prob = prob[:,gmm.means_.argmin()]         
        return prob,all_loss

    def loss_statistic(self, stat_iter):
                
        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0

        stat_loader = tordata.DataLoader(
            dataset=self.config['train_source_noisy'],
            batch_sampler=TripletSampler(self.config['train_source_noisy'], self.config['batch_size']),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])
        
        train_label_set = list(self.config['train_source_noisy'].label_set)
        train_label_set.sort()
        seq_type_set = list(self.config['train_source_noisy'].seq_type_set)
        seq_type_set.sort()

        _time1 = datetime.now()

        for seq, label, label_clean, seq_type, index, batch_frame in stat_loader:
            #set_trace()
            #############################################################
            if self.config['DDP'] and self.config['restore_iter'] > 0 and \
                self.config['restore_iter'] % self.triplet_sampler.total_batch_per_world == 0:
                self.triplet_sampler.set_random_seed(self.triplet_sampler.random_seed+1)
            #############################################################
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            index = self.np2var(np.array(index)).float()
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            target_label_clean = [train_label_set.index(l) for l in label_clean]
            target_label_clean = self.np2var(np.asarray(target_label_clean)).long()
            train_seq_type =  [seq_type_set.index(s) for s in seq_type]  #from str to int
            train_seq_type = self.np2var(np.asarray(train_seq_type)).long()

            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            with autocast(enabled=self.config['AMP']):
                encoder_feature, encoder_bn_feature, encoder_cls_score \
                = self.encoder(seq, self.config['restore_iter'], batch_frame, target_label)
            
            loss = torch.zeros(1).to(encoder_feature.device)
            if self.config['encoder_entropy_weight'] > 0:
                entropy_loss_metric = 0
                for i in range(encoder_cls_score.size(1)):
                    entropy_loss_metric += self.encoder_entropy_loss(encoder_cls_score[:, i, :].float(), target_label, target_label_clean, index, l=0, loss_statistic=True)
                entropy_loss_metric = entropy_loss_metric / encoder_cls_score.size(1)
                loss += entropy_loss_metric * self.config['encoder_entropy_weight']
                self.encoder_entropy_loss_metric[0].append(entropy_loss_metric.mean().data.cpu().numpy())

            if self.config['encoder_triplet_weight'] > 0:
                encoder_triplet_feature = encoder_feature.float().permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                ###statistic
                loss_statistic = False
                triplet_loss_metric, nonzero_num = self.encoder_triplet_loss(encoder_triplet_feature, triplet_label, target_label_clean, train_seq_type, loss_statistic)
                loss += triplet_loss_metric.mean() * self.config['encoder_triplet_weight']
                self.encoder_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.encoder_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())
            
            if self.config['cloth_loss_weight'] > 0:
                cloth_loss = self.cloth_loss(encoder_feature, target_label, train_seq_type)
                loss += cloth_loss.mean() * self.config['cloth_loss_weight']
                self.cloth_loss_metric[0].append(cloth_loss.mean().data.cpu().numpy())

            # mixmatch
            l = np.random.beta(self.config['alpha'], self.config['alpha'])        
            l = max(l, 1-l)
            all_inputs = seq
            all_targets = target_label

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b        
            mixed_target = [target_a, target_b]
            
            mix_feature, mix_bn_feature, mix_cls_score \
                = self.encoder(mixed_input, self.config['restore_iter'], batch_frame, mixed_target)
            
            if self.config['mixup_weight'] > 0:
                entropy_loss_mix = 0
                for i in range(mix_cls_score.size(1)):
                    entropy_loss_mix += self.encoder_entropy_loss(mix_cls_score[:, i, :].float(), mixed_target, l)
                entropy_loss_mix = entropy_loss_mix / mix_cls_score.size(1)
                loss += entropy_loss_mix * self.config['encoder_entropy_weight']
                self.encoder_entropy_loss_metric[1].append(entropy_loss_mix.mean().data.cpu().numpy())

            self.total_loss_metric.append(loss.data.cpu().numpy())

            self.build_loss_metric()
            
            if self.config['restore_iter'] == stat_iter:
                break
            self.config['restore_iter'] += 1

    def graph_accuracy(self, flag, batch_size=1, feat_idx=0):
    
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source']
        self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, label, batch_frame = x
            seq = self.np2var(seq).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            output = self.encoder(seq, batch_frame)
            feature = output[feat_idx]
            feature_list.append(feature.detach())             
            label_list += label

        return torch.cat(feature_list, 0), view_list, seq_type_list, label_list

    def graph_train(self, hiar_mem_bank, graph_train_iter):

        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0
        #batch_size = [self.config['num_id_all'],2]
        batch_size = [64,2]
        stat_loader = tordata.DataLoader(
            dataset=self.config['train_source_all'],
            batch_sampler=TripletSampler(self.config['train_source_all'], batch_size),  #116*2
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])
        
        train_label_set = list(self.config['train_source_all'].label_set)
        train_label_set.sort()
        seq_type_set = list(self.config['train_source_all'].seq_type_set)
        seq_type_set.sort()

        _time1 = datetime.now()

        for seq, label, label_clean, seq_type, index, batch_frame in stat_loader:
            #set_trace()
            #############################################################
            if self.config['DDP'] and self.config['restore_iter'] > 0 and \
                self.config['restore_iter'] % self.triplet_sampler.total_batch_per_world == 0:
                self.triplet_sampler.set_random_seed(self.triplet_sampler.random_seed+1)
            #############################################################
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            index = self.np2var(np.array(index)).float()
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            #target_label_clean = [train_label_set.index(l) for l in label_clean]
            #target_label_clean = self.np2var(np.asarray(target_label_clean)).long()
            #train_seq_type =  [seq_type_set.index(s) for s in seq_type]  #from str to int
            #train_seq_type = self.np2var(np.asarray(train_seq_type)).long()

            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            with autocast(enabled=self.config['AMP']):
                encoder_feature, encoder_bn_feature, encoder_cls_score \
                = self.encoder(seq, self.config['restore_iter'], batch_frame, target_label)  #232*16*256

            _, bins, hidden_d= encoder_feature.shape
            #mean of samples from dataset
            encoder_feature_mean = encoder_feature.view(batch_size[0], batch_size[1], bins, hidden_d).mean(1)
            target_label_mean = target_label[0:-1:batch_size[1]]
            label_predicted = list()
            method = 'merge'
            #method = 'seperately'
            #with torch.no_grad():
            if method == 'all':
                for i in range(self.M):
                    #graph construction
                    if i == 0:
                        #set_trace()
                        Y = torch.zeros(batch_size[0], self.config['num_id_all']).cuda()
                        #index_Y = hiar_mem_bank[target_label_mean,0,-3+i].unsqueeze(1).long()
                        index_Y = target_label_mean.unsqueeze(1)
                        Y.scatter_(dim=1, index=index_Y, src=torch.ones(batch_size[0],1).cuda())
                        Y = Y.unsqueeze(0).repeat(bins,1,1)
                        emb1 = encoder_feature_mean.permute(1,0,2).unsqueeze(1)
                        emb2 = encoder_feature_mean.permute(1,0,2).unsqueeze(2)
                        W = ((emb1-emb2)**2).mean(3)   # N*N*d -> N*N
                        W = torch.exp(-W/2)
                        #select the top k
                        if self.config['k']>0:
                            topk, indices = torch.topk(W, self.config['k'], dim=-1)
                            mask = torch.zeros_like(W)
                            mask = mask.scatter(2, indices, 1)
                            
                            mask = ((mask+(mask.transpose(2,1)))>0).type(torch.float32)      # union, kNN graph
                            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
                            W_mask = W * mask

                        ## normalize
                        D = W_mask.sum(2)
                        eps = np.finfo(float).eps
                        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
                        D1 = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,batch_size[0])
                        D2 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,batch_size[0],1)
                        S = D1*W_mask*D2

                        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
                        F  = torch.matmul(torch.inverse(torch.eye(batch_size[0]).cuda(0)-self.alpha*S+eps), Y)
                        predq = torch.max(F,2)[1]  #16*232
                        #if all bins agrees, change the value
                        predq[:,((predq == predq[0,:]).sum(0) != 16)] = index_Y[((predq == predq[0,:]).sum(0) != 16),0]
                        #if i > 0:
                        for j in range(batch_size[0]):
                            hiar_mem_bank[target_label_mean[j],:,-self.M+i] = predq[:,j]
                        label_predicted.append(predq[0].repeat(batch_size[1]).view(batch_size[1],batch_size[0]).permute(1,0).contiguous().view(-1).unsqueeze(0))
                    else:
                        #set_trace()
                        Y = torch.zeros(len(set(predq[i-1])), self.config['num_id_all']).cuda()
                        feature1 = list()
                        for _ in set(predq[i-1]):   #set中的会排序，不要直接用索引
                            rank_ = np.where(predq[0].cpu().numpy() == _.cpu().numpy())
                            Y[rank_,_] = 1
                            feature1.append(encoder_feature_mean[predq[0]==_].mean(0).unsqueeze(0))
                        feature1 = torch.cat(feature1,dim=0)
                        Y = Y.unsqueeze(0).repeat(bins,1,1)
                        emb1 = feature1.permute(1,0,2).unsqueeze(1)
                        emb2 = feature1.permute(1,0,2).unsqueeze(2)
                        W = ((emb1-emb2)**2).mean(3)   # N*N*d -> N*N
                        W = torch.exp(-W/2)
                        #select the top k
                        if self.config['k']>0:
                            topk, indices = torch.topk(W, self.config['k'], dim=-1) 
                            mask = torch.zeros_like(W)
                            mask = mask.scatter(2, indices, 1)
                            
                            mask = ((mask+(mask.transpose(2,1)))>0).type(torch.float32)      # union, kNN graph
                            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
                            W_mask = W * mask

                        ## normalize
                        D = W_mask.sum(2)
                        eps = np.finfo(float).eps
                        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
                        D1 = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,batch_size[0])
                        D2 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,batch_size[0],1)
                        S = D1*W_mask*D2

                        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
                        F  = torch.matmul(torch.inverse(torch.eye(batch_size[0]).cuda(0)-self.alpha*S+eps), Y)
                        predq = torch.max(F,2)[1]  #16*232
                        #if all bins agrees, change the value
                        predq[:,((predq == predq[0,:]).sum(0) != 16)] = index_Y[((predq == predq[0,:]).sum(0) != 16),0]
                        #if i > 0:
                        for j in range(batch_size[0]):
                            hiar_mem_bank[target_label_mean[j],:,-self.M+i] = predq[:,j]
                        label_predicted.append(predq[0].repeat(batch_size[1]).view(batch_size[1],batch_size[0]).permute(1,0).contiguous().view(-1).unsqueeze(0))
                #loss
                label_predicted = torch.cat(label_predicted, dim=0)
            elif method == 'seperately':
                loss=0
            elif method == 'merge':
                #graph construction
                #split
                for i in range(self.M):
                    #graph construction
                    if i == 0:
                        set_trace()
                        Y = torch.zeros(batch_size[0], self.config['num_id_all']).cuda()
                        #index_Y = hiar_mem_bank[target_label_mean,0,-3+i].unsqueeze(1).long()
                        index_Y = target_label_mean.unsqueeze(1)
                        Y.scatter_(dim=1, index=index_Y, src=torch.ones(batch_size[0],1).cuda())
                        Y = Y.unsqueeze(0).repeat(bins,1,1)
                        emb1 = encoder_feature_mean.permute(1,0,2).unsqueeze(1)  #bins,
                        emb2 = encoder_feature_mean.permute(1,0,2).unsqueeze(2)
                        #W = ((emb1-emb2)**2).mean(3)   # N*N*d -> N*N
                        #W = torch.exp(-W/2)
                        W = torch.matmul(emb1,emb2)
                        #select the top k
                        if self.config['k']>0:
                            topk, indices = torch.topk(W, self.config['k'], dim=-1)
                            mask = torch.zeros_like(W)
                            mask = mask.scatter(2, indices, 1)
                            
                            mask = ((mask+(mask.transpose(2,1)))>0).type(torch.float32)      # union, kNN graph
                            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
                            W_mask = W * mask
                        
                        ## normalize
                        D = W_mask.sum(2)
                        eps = np.finfo(float).eps
                        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
                        D1 = torch.unsqueeze(D_sqrt_inv,2).repeat(1,1,batch_size[0])
                        D2 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,batch_size[0],1)
                        S = D1*W_mask*D2
                        
                        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
                        F  = torch.matmul(torch.inverse(torch.eye(batch_size[0]).cuda(0)-self.alpha*S+eps), Y)
                        predq = torch.max(F,2)[1]  #16*232
                        #if all bins agrees, change the value
                        predq[:,((predq == predq[0,:]).sum(0) != 16)] = index_Y[((predq == predq[0,:]).sum(0) != 16),0]
                        #if i > 0:
                        for j in range(batch_size[0]):
                            hiar_mem_bank[target_label_mean[j],:,-self.M+i] = predq[:,j]
                        label_predicted.append(predq[0].repeat(batch_size[1]).view(batch_size[1],batch_size[0]).permute(1,0).contiguous().view(-1).unsqueeze(0))


            loss = torch.zeros(1).to(encoder_feature.device)
            if self.config['spcloss_weight'] > 0:
                
                loss, loss0, loss1, loss2 = self.spcloss(hiar_mem_bank, encoder_feature, label_predicted, batch_size)
                self.spcloss_metric[0].append(loss.data.cpu().numpy())
                self.spcloss_metric[1].append(loss0.data.cpu().numpy())
                self.spcloss_metric[2].append(loss1.data.cpu().numpy())
                self.spcloss_metric[3].append(loss2.data.cpu().numpy())

                acc1 = self.onezero_accuracy(self.hiar_mem_bank[:,0,-3], self.clean_label)
                acc2 = self.onezero_accuracy(self.hiar_mem_bank[:,0,-2], self.clean_label)
                acc3 = self.onezero_accuracy(self.hiar_mem_bank[:,0,-1], self.clean_label)
                #print('acc after batch', acc1, acc2, acc3)
                if self.config['restore_iter'] % 10 == 0:
                    print('not equal to begining:',(label_predicted!=target_label).sum(1))
                    print('loss:',self.config['restore_iter'],loss.data, loss0.data, loss1.data, loss2.data)
            if loss > 1e-9:
                if self.config['AMP']:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:  
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
            #for name,para in self.encoder.named_parameters():
            #    print(name,para.detach())
            
            if self.config['restore_iter'] % 100 == 0:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    print(datetime.now() - _time1)
                    _time1 = datetime.now()
                    #self.print_info()
                self.build_loss_metric()
            if self.config['restore_iter'] % 10000 == 0 or self.config['restore_iter'] == self.config['total_iter']:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    self.save()
            if self.config['restore_iter'] == graph_train_iter:
                break
            self.config['restore_iter'] += 1
            
    def onezero_accuracy(self, label, label_clean):
        num = 1
        for i in label.shape:
            num = num * i
        '''
        label_char_ = set(self.config['train_source_all'].seq_label_list)
        label_char_set_ = list(label_char_)
        _ = np.sort(label_char_set_)
        
        label_char = list()
        for i in  label.cpu():
            label_char.append(_[int(i)])
        #label_char = [_[int(i)] for i in label.cpu()]
        label_clean_char = list()
        for i in  label_clean.cpu():
            label_clean_char.append(_[int(i)])
        '''
        acc = 0
        for i in range(43):
            #if label_char[i][:3] == label_clean_char[i]:
            if label[2*i] == label_clean[2*i] or label[2*i] == label_clean[2*i+1]:
                acc += 1
            if label[2*i] == label[2*i+1]:
                acc += 1
        for i in range(86,116):
            if label[i] == label_clean[i]:
                acc += 1
        acc = acc/num 
        return acc
        


