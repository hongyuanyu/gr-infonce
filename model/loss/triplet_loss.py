import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace

class TripletLoss(nn.Module):
    def __init__(self, margin, triplet_type, hard_mining=False, nonzero=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.nonzero = nonzero
        self.triplet_type = triplet_type

    def forward(self, feature, label, iteration):
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
    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist

    def loss_statistic(self, feature, label_noisy, label_clean):
        
        return loss_clean, loss_noisy, iteration