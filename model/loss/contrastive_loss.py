import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace
import random
class ContrastiveLoss(nn.Module):
    def __init__(self, margin, batch_size, bins):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch_size = batch_size[1]*batch_size[0]  #128
        self.bins = bins

    def forward(self, feature, feature_aug, iter):

        double = True
        if double == True:
            dist = self.cuda_euc_dist(feature, feature_aug).view(-1)
            #ap_mask = [[torch.zeros(self.batch_size, self.batch_size),torch.eye(self.batch_size, self.batch_size)],
            #           [torch.eye(self.batch_size, self.batch_size),torch.zeros(self.batch_size, self.batch_size)]].bool().repeat(self.bins,1,1).view(-1).cuda()
            ap_mask = torch.eye(self.batch_size, self.batch_size).bool().repeat(self.bins,1,1).view(-1).cuda()
            an_mask = ~ap_mask
            ap = torch.masked_select(dist, ap_mask).view(self.bins, self.batch_size)  #31*256
            an = torch.masked_select(dist, an_mask).view(self.bins, self.batch_size, self.batch_size-1)  #31*256*255
        else:
            feature_double = torch.cat((feature, feature_aug), dim=1) #16*256*256
            dist_double = self.cuda_euc_dist(feature_double, feature_double).view(-1) #16*256*256
            ap_mask = [torch.cat((torch.zeros(self.batch_size, self.batch_size),torch.eye(self.batch_size, self.batch_size)),dim=0),
                       torch.cat((torch.eye(self.batch_size, self.batch_size),torch.zeros(self.batch_size, self.batch_size)),dim=0)]
            ap_mask = torch.cat(ap_mask,dim=1).bool().repeat(self.bins,1,1).view(-1).cuda()  #16*256*256
            an_mask = ~ap_mask
            an_mask -= torch.eye()
            ap = torch.masked_select(dist_double, ap_mask).view(self.bins, self.batch_size)  #31*256
            an = torch.masked_select(dist_double, an_mask).view(self.bins, self.batch_size, self.batch_size-1)  #31*256*255
        loss = F.relu(self.margin + ap.unsqueeze(2) - an) #31 *256*255
        non_zero = (loss != 0).sum()  
        if non_zero != 0:
            loss = loss.sum() / non_zero
        else:
            loss = 0
        return loss

    def cuda_euc_dist(self, x, y):
        dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2)) # num_parts * num_probe * num_gallery
        dist = torch.sqrt(F.relu(dist)) # num_parts * num_probe * num_gallery
        #dist = torch.mean(dist, 0) # num_probe * num_gallery
        return dist  #31*128*128