import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class TripletLoss(nn.Module):
    def __init__(self, margin, triplet_type, hard_mining=False, nonzero=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.nonzero = nonzero
        self.triplet_type = triplet_type
        self.loss = list()
        self.loss_nonzero = list()
        self.label_stat = list()
        self.label_clean_stat = list()
        self.iteration = 0

    def forward(self, feature, label, label_clean, seq_type, loss_statistic,is_mean=False):
        # feature: [n, m, d], label: [n, m]      16*128*128  16*128
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2))
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2))
        
        dist = self.batch_dist(feature)

        if self.triplet_type == 'hard-random':
            # hard
            hn_mask_sum = hn_mask.sum(2)[0]
            random_hn_dist = list()
            index = list()
            for i in range(len(hn_mask_sum)):  #128
                #hp_index = np.random.choice(range(hp_mask_sum[i]), 1, replace=False)
                hn_index_i = np.random.choice(range(hn_mask_sum[i]), 1, replace=False)
                random_hn_dist.append(dist[:,i,hn_index_i])  #31*1*1 
                index.append(hn_index_i)
            #set_trace()
            hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]  #31*128
            random_hn_dist = torch.cat(random_hn_dist, dim=1)  #16*128
            if self.margin > 0:
                hard_random_loss_metric = F.relu(self.margin + hard_hp_dist - random_hn_dist).view(n, -1)
            else:
                hard_random_loss_metric = F.softplus(hard_hp_dist - random_hn_dist).view(n, -1)
                
            nonzero_num = (hard_random_loss_metric != 0).sum(1).float()

            if self.nonzero:
                hard_random_loss_metric_mean = hard_random_loss_metric.sum(1) / nonzero_num
                hard_random_loss_metric_mean[nonzero_num == 0] = 0
            else:
                hard_random_loss_metric_mean = torch.mean(hard_random_loss_metric, 1)

            return hard_random_loss_metric_mean.mean(), nonzero_num.mean()
        elif self.triplet_type == 'random-random':
            
            np.random.choice()
            full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
            full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
            if self.margin > 0:
                full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
            else:
                full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(n, -1)  

            nonzero_num = (full_loss_metric != 0).sum(1).float()

            if self.nonzero:
                full_loss_metric_mean = full_loss_metric.sum(1) / nonzero_num
                full_loss_metric_mean[nonzero_num == 0] = 0
            else:
                full_loss_metric_mean = full_loss_metric.mean(1)
            
            # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
            return full_loss_metric_mean.mean(), nonzero_num.mean()
        elif self.triplet_type == 'hard':
            hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
            hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
            if self.margin > 0:
                hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
            else:
                hard_loss_metric = F.softplus(hard_hp_dist - hard_hn_dist).view(n, -1)
                
            nonzero_num = (hard_loss_metric != 0).sum(1).float()

            if self.nonzero:
                hard_loss_metric_mean = hard_loss_metric.sum(1) / nonzero_num
                hard_loss_metric_mean[nonzero_num == 0] = 0
            else:
                hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
            
            return hard_loss_metric_mean.mean(), nonzero_num.mean()
        elif self.triplet_type == 'full':
            # full
            
            full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
            full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
            if self.margin > 0:
                full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n,m, -1)  #16*128*16*128
            else:
                full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(n,m, -1)  

            #nonzero_num = (full_loss_metric != 0).sum(1).float()
            nonzero_num = (full_loss_metric != 0).sum(2).float() #16*128

            if self.nonzero:
                #full_loss_metric_mean = full_loss_metric.sum(1) / nonzero_num
                full_loss_metric_mean = full_loss_metric.sum(2) / nonzero_num
                full_loss_metric_mean[nonzero_num == 0] = 0
            else:
                full_loss_metric_mean = full_loss_metric.mean(1)

            if loss_statistic == True :
                with torch.no_grad():
                    #set_trace()
                    nonzero_num = ((full_loss_metric.view(n,m,-1)) != 0).sum(2).float()  #16*128
                    loss_nonzero_mean = full_loss_metric.view(n,m,-1).sum(2)/nonzero_num
                    loss_mean = full_loss_metric.view(n,m,-1).mean(2)
                    self.loss.append(loss_mean.permute(1,0).contiguous())  #128*16*~
                    self.loss_nonzero.append(loss_nonzero_mean.permute(1,0).contiguous())  #128*16*~
                    self.label_stat.append(label[0])
                    self.label_clean_stat.append(label_clean)
                    self.iteration += 1
                    if self.iteration == 100-1:
                        
                        self.loss = torch.cat(self.loss, dim=0)
                        self.loss_nonzero = torch.cat(self.loss_nonzero, dim=0)
                        self.label_stat = torch.cat(self.label_stat, dim=0)
                        self.label_clean_stat = torch.cat(self.label_clean_stat, dim=0)
                        self.loss_statistic(self.loss, self.loss_nonzero, self.label_stat, self.label_clean_stat)

                        self.loss = list()
                        self.loss_nonzero = list()
                        self.label_stat = list()
                        self.label_clean_stat = list()
            if is_mean:
                return full_loss_metric_mean.mean(0),nonzero_num.mean()
            else:
                return full_loss_metric_mean.mean(), nonzero_num.mean()

        elif self.triplet_type == 'full_cloth':
            # full
            index_matrix = (label.unsqueeze(1) == label.unsqueeze(1).transpose(0,1))  #128*128 bool
            index_cl = torch.max(seq_type == 2, seq_type == 3)
            index_cl_matrix = (index_cl.unsqueeze(1) == index_cl.unsqueeze(1).transpose(0,1))
            index = torch.min(index_matrix, index_cl_matrix)
            index = index.unsqueeze(0).repeat(n, 1, 1)
            cloth_loss = 0
            for i in range(m):
                index[:,i,:].sum(1)
                dist_ap_i = torch.masked_select(dist[:,i,:], index[:,i,:]).view(n,-1, 1)
                dist_an_i = torch.masked_select(dist[:,i,:], hn_mask[:,i,:]).view(n, 1,-1)
                cloth_loss_i = F.relu(self.margin + dist_ap_i - dist_an_i).view(n,-1)
                nonzero_num = (cloth_loss_i != 0).sum(1).float()
                if self.nonzero:
                    cloth_loss_i_mean = cloth_loss_i.sum(1) / nonzero_num
                    cloth_loss_i_mean[nonzero_num == 0] = 0
                else:
                    cloth_loss_i_mean = cloth_loss_i_mean.mean(1)
                cloth_loss += cloth_loss_i_mean

            full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)  #16*128*16*1
            full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)  #16*128*1*112
            
            if self.margin > 0:
                full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)  #16*128*16*112
            else:
                full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(n, -1)  

            nonzero_num = (full_loss_metric != 0).sum(1).float()

            if self.nonzero:
                full_loss_metric_mean = full_loss_metric.sum(1) / nonzero_num
                full_loss_metric_mean[nonzero_num == 0] = 0
            else:
                full_loss_metric_mean = full_loss_metric.mean(1)
            
            # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
            return full_loss_metric_mean.mean()+cloth_loss, nonzero_num.mean()

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist

    def loss_statistic(self, loss, loss_nonzero, label_noisy, label_clean):
        batch_size, bins = loss.shape  #1280*16
        #set_trace()
        clean_index = (label_noisy == label_clean).unsqueeze(1).repeat(1,bins)
        clean_loss = torch.masked_select(loss,clean_index).cpu()
        clean_loss_nonzero = torch.masked_select(loss_nonzero, clean_index).cpu()
        noisy_loss = torch.masked_select(loss,~clean_index).cpu()
        noisy_loss_nonzero = torch.masked_select(loss_nonzero, ~clean_index).cpu()

        save_path = '/home/yuweichen/workspace/noisy_gait/visualize_feature'
        img_name = os.path.join(save_path, 'loss_histogram')
        plt.figure(1, figsize=(20, 10))
        plt.subplot(231)
        plt.hist(clean_loss,facecolor='blue',bins=100)
        plt.subplot(232)
        plt.hist(noisy_loss,facecolor='red',bins=100)
        plt.subplot(233)
        plt.hist(clean_loss,facecolor='blue',bins=100)
        plt.hist(noisy_loss,facecolor='red',bins=100)
        plt.subplot(234)
        plt.hist(clean_loss_nonzero,facecolor='blue',bins=100)
        plt.subplot(235)
        plt.hist(noisy_loss_nonzero,facecolor='red',bins=100)
        plt.subplot(236)
        plt.hist(clean_loss_nonzero,facecolor='blue',bins=100)
        plt.hist(noisy_loss_nonzero,facecolor='red',bins=100)
        
        plt.savefig(img_name)
        plt.clf()                                                   
        print('loss statistic process Done')
        return None