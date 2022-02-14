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

from .data import TripletSampler, DistributedTripletSampler, build_data_transforms
from .loss import DistributedLossWrapper, CenterLoss, CrossEntropyLabelSmooth, TripletLoss, InfonceLoss   #PartTripletLoss,
from .solver import WarmupMultiStepLR
from .network import GaitSet
from .network.sync_batchnorm import DataParallelWithCallback
from pdb import set_trace
import torch.nn.functional as F
import pickle 
from info_nce import InfoNCE

class Model:
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
        
        self.config.update({'num_id': len(self.config['train_source'].label_set)})
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
            num_of_id = len(self.config['train_source'].label_set)  #117
            self.mem_bank = torch.zeros(num_of_id, sum(self.config['bin_num']), self.config['hidden_dim']+4).cuda()  #min, max, mean, sigma
            print('self.mem_bank.shape', self.mem_bank.shape)
            self.mem_statistic = list()

    def build_data(self):
        # data augment
        if self.config['dataset_augment']:
            self.data_transforms = build_data_transforms(random_erasing=True, random_rotate=False, \
                                        random_horizontal_flip=False, random_pad_crop=False, cloth_dilate=False,\
                                        resolution=self.config['resolution'], random_seed=self.random_seed) 
            self.data_transforms_new = build_data_transforms(random_erasing=False, random_rotate=False, \
                                        random_horizontal_flip=False, random_pad_crop=True, cloth_dilate=False,\
                                        resolution=self.config['resolution'], random_seed=self.random_seed+1) 
        
        #triplet sampler
        if self.config['DDP']:
            self.triplet_sampler = DistributedTripletSampler(self.config['train_source'], self.config['batch_size'], random_seed=self.random_seed)
        else:
            self.triplet_sampler = TripletSampler(self.config['train_source'], self.config['batch_size'])

    def build_loss(self):
        if self.config['encoder_entropy_weight'] > 0:
            if self.config['label_smooth']:
                self.encoder_entropy_loss = CrossEntropyLabelSmooth(self.config['num_id']).float().cuda()
            else:
                self.encoder_entropy_loss = nn.CrossEntropyLoss().float().cuda()
            if self.config['DDP']:
                self.encoder_entropy_loss = DistributedLossWrapper(self.encoder_entropy_loss, dim=0)

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss = TripletLoss(self.config['encoder_triplet_margin'], self.config['triplet_type']).float().cuda()
            if self.config['DDP']:
                self.encoder_triplet_loss = DistributedLossWrapper(self.encoder_triplet_loss, dim=1)
        if self.config['self_supervised_weight'] > 0:
            temperature = 0.07
            self.ap_mode = 'all'  # 'all' 'centor'  'random'
            self.an_mode = 'all'  # 'all' 'centor'  'random'
            self.infonce_loss = InfonceLoss(temperature, self.config['batch_size'], self.ap_mode, self.an_mode).float().cuda()

            self.infonce_loss_git = InfoNCE(negative_mode='paired')

            if self.config['DDP']:
                self.encoder_triplet_loss = DistributedLossWrapper(self.infonce_loss, dim=1)

    def build_loss_metric(self):
        if self.config['encoder_entropy_weight'] > 0:
            self.encoder_entropy_loss_metric = [[]]

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss_metric = [[], []]

        if self.config['self_supervised_weight'] > 0:
            self.infonce_loss_metric = [[]]
        if self.config['infonce_git_weight'] > 0:
            self.infonce_git_loss_metric = [[]]
            
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

    def fit(self):
        self.encoder.train()
        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0

        train_loader = tordata.DataLoader(
            dataset=self.config['train_source'],
            batch_sampler=self.triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        train_label_set = list(self.config['train_source'].label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, label, batch_frame in train_loader:
            #############################################################
            if self.config['DDP'] and self.config['restore_iter'] > 0 and \
                self.config['restore_iter'] % self.triplet_sampler.total_batch_per_world == 0:
                self.triplet_sampler.set_random_seed(self.triplet_sampler.random_seed+1)
            #############################################################
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            with autocast(enabled=self.config['AMP']):
                encoder_feature, encoder_bn_feature, encoder_cls_score \
                = self.encoder(seq, self.config['restore_iter'], batch_frame, target_label)

            loss = torch.zeros(1).to(encoder_feature.device)

            if self.config['encoder_entropy_weight'] > 0:
                entropy_loss_metric = 0
                for i in range(encoder_cls_score.size(1)):
                    entropy_loss_metric += self.encoder_entropy_loss(encoder_cls_score[:, i, :].float(), target_label, target_label, l=0, loss_statistic=False)
                entropy_loss_metric = entropy_loss_metric / encoder_cls_score.size(1)
                loss += entropy_loss_metric * self.config['encoder_entropy_weight']
                self.encoder_entropy_loss_metric[0].append(entropy_loss_metric.mean().data.cpu().numpy())

            if self.config['encoder_triplet_weight'] > 0:
                encoder_triplet_feature = encoder_feature.float().permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                triplet_loss_metric, nonzero_num = self.encoder_triplet_loss(encoder_triplet_feature, triplet_label, triplet_label, seq_type=None, loss_statistic=False)
                loss += triplet_loss_metric.mean() * self.config['encoder_triplet_weight']
                self.encoder_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.encoder_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())
            if self.config['self_supervised_weight'] > 0:

                batch_size = self.config['batch_size'][0] * self.config['batch_size'][1]
                _, bins, dim = encoder_feature.shape
                infonce_loss = self.infonce_loss(encoder_feature[:batch_size,:,:].view(self.config['batch_size'][0],self.config['batch_size'][1],bins, dim),
                    encoder_feature[batch_size:,:,:].view(self.config['batch_size'][0],self.config['batch_size'][1],bins, dim))
                loss += infonce_loss.mean() * self.config['self_supervised_weight']
                self.infonce_loss_metric[0].append(infonce_loss.mean().data.cpu().numpy())
            #######infonce_git#########
            if self.config['infonce_git_weight'] > 0:
                batch_size = self.config['batch_size'][0] * self.config['batch_size'][1]
                batch_size_ = [self.config['batch_size'][0], self.config['batch_size'][1]]
                query = encoder_feature[:batch_size,:,:].view(batch_size, -1)
                positive_key = encoder_feature[batch_size:,:,:].view(batch_size, -1)
                negative_keys = []
                for i in range(batch_size_[0]):
                    negative_keys.append(positive_key[torch.arange(positive_key.size(0))!=i])

                negative_keys = torch.stack(negative_keys, dim=0)  # 8,7,16,4096
                negative_keys = negative_keys.permute(0,2,1,3).contiguous().view(batch_size,batch_size_[0],-1)
                infonce_loss_git = self.infonce_loss_git(query, positive_key, negative_keys)
                loss += infonce_loss_git.mean() * self.config['infonce_git_weight']
                self.infonce_git_loss_metric[0].append(infonce_loss_git.mean().data.cpu().numpy())

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
            time3 = datetime.now()
            if self.config['mem_bank']:
                if self.config['restore_iter'] % 100 == 0:
                    with torch.no_grad():
                        self.mem_bank = self.renew_mem_bank_all(self.mem_bank)
                        self.mem_statistic = self.statistic(self.mem_statistic, self.config['restore_iter'], self.mem_bank)
                    print('time for mem_bank',datetime.now() - time3)

            
            if self.config['restore_iter'] % 100 == 0:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    print(datetime.now() - _time1)
                    _time1 = datetime.now()
                    self.print_info()
                self.build_loss_metric()
            if self.config['restore_iter'] % 100 == 0 or self.config['restore_iter'] == self.config['total_iter']:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    self.save()
            if self.config['restore_iter'] == self.config['total_iter']:
                break
            self.config['restore_iter'] += 1

    def renew_mem_bank_all(self, mem_bank, batch_size=1, flag='train'):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source']
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
            seq, label, batch_frame = x
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

    def statistic(self, mem_statistic, iteration, mem_bank):
        interclass_dist = self.cuda_euc_dist(mem_bank[:,:,:-4], mem_bank[:,:,:-4])  #116*31*256   16*116*116
        mem_statistic.append(torch.cat((   # 16*116
            torch.max(interclass_dist, dim=2)[0].unsqueeze(2), torch.sort(interclass_dist, dim=2)[0][:,:,1:4], torch.sort(interclass_dist, dim=2)[1][:,:,1:4], torch.mean(interclass_dist, dim=2).unsqueeze(2),
            mem_bank[:,:,-4:].permute(1,0,2)), dim=2).unsqueeze(0))  #116*16*4  output=1*16*116*8
        if iteration == self.config['total_iter']:
            statistic_path = '/home2/ywc/workspace/IDNoise_Gait_Full-master/visualization'
            if not osp.exists(statistic_path):
                os.makedirs(statistic_path)
            statistic_file_path = statistic_path+'statistic_file_dist_100.pkl'
            mem_statistic = torch.cat(mem_statistic, dim=0)
            with open(statistic_file_path, 'wb') as f:
                pickle.dump(mem_statistic, f)
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

        if self.config['self_supervised_weight'] > 0:
            loss_name = 'InfoNCE'
            loss_metric = self.infonce_loss_metric[0]
            loss_weight = self.config['self_supervised_weight']
            loss_info = 'ap_mode={},an_mode={}'.format(self.ap_mode, self.an_mode)
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['infonce_git_weight'] > 0:
            loss_name = 'InfoNCE_git'
            loss_metric = self.infonce_git_loss_metric[0]
            loss_weight = self.config['infonce_git_weight']
            loss_info = 'paired'
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        print('{:#^30}: total_loss_metric={:.6f}'.format('Total Loss', np.mean(self.total_loss_metric)))
        
        #optimizer
        print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}'.format( \
            'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay']))            
        sys.stdout.flush()

    def transform(self, flag, batch_size=1, feat_idx=0):
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

    def collate_fn(self, batch):
        batch_size = len(batch)
        seqs = [batch[i][0] for i in range(batch_size)]
        label = [batch[i][1] for i in range(batch_size)]
        batch = [seqs, label, None]
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
        #print('batch_frames:',batch_frames)
        # select frames from each seq 
        def select_frame(index):
            sample = seqs[index]
            frame_set = np.arange(sample.shape[0])
            frame_num = batch_frames[int(index / batch_per_gpu)][int(index % batch_per_gpu)]
            if len(frame_set) >= frame_num:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=False))
            else:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=True))
            #print('frame_id_list',frame_id_list)
            return sample[frame_id_list, :, :]
        seqs = list(map(select_frame, range(len(seqs))))        

        # data augmentation
        def transform_seq(index):
            sample = seqs[index]
            return self.data_transforms(sample)
            ############为了多加一部分的loss###################
        def transform_seq_new(index):
            sample = seqs[index]
            return self.data_transforms_new(sample)
        if self.config['dataset_augment']:
            seqs_original = list(map(transform_seq, range(len(seqs))))  
            seqs_da = list(map(transform_seq_new, range(len(seqs)))) 
            seqs = [seqs_original,seqs_da]
            seqs = np.concatenate(seqs, axis=0)
            label = np.concatenate([label,label], axis=0)
            batch[1] = label

        def NoneView(seqs):
            seqs = np.array(seqs)
            _,frame,h,w = seqs.shape
            seqs = seqs.reshape(self.config['batch_size'][0],-1,h,w)
            none_view_seq = list()
            #random select frame
            '''
            for i in range(self.config['batch_size'][0]):
                frame_list = np.array(np.random.choice(self.config['batch_size'][1]*self.config['frame_num'], self.config['batch_size'][1]*self.config['frame_num'], replace=False))
                for j in range(self.config['batch_size'][1]):
                    index = frame_list[j*self.config['frame_num']:self.config['frame_num']*(j+1)]
                    #print('index:',i,j,index)
                    #print('shape:', seqs[i].shape, _,frame,h,w, seqs[i][index].shape)
                    none_view_seq.append(seqs[i][index])
            '''
            #choose the different part
            for i in range(self.config['batch_size'][0]):
                for j in range(self.config['batch_size'][1]):
                    index = np.array([range(j*self.config['frame_num'],j*self.config['frame_num']+15),range((j-1)*self.config['frame_num']+15,(j-1)*self.config['frame_num']+30)]
                        ).reshape(-1)
                    #print('index:',i,j,index)
                    none_view_seq.append(seqs[i][index])
                    #print('shape:', seqs[i][index].shape, seqs[i][index])
            return none_view_seq
        none_view = False
        if none_view and self.config['phase']=='train':
            seqs = NoneView(seqs)

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
        #print('seqs:', seqs.shape)
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
