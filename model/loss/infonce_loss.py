import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace
import random
class InfonceLoss(nn.Module):
    def __init__(self, temperature, batch_size, ap_mode, an_mode):
        super(InfonceLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = [batch_size[0],batch_size[1]]  #16*8
        self.ap_mode = ap_mode
        self.an_mode = an_mode

    def forward(self, feature, feature_aug):
        #认为除了自己的和自己的DA外，别的都是反例
        #feature batch_size[0]*batch_size[1]*bins*dim
        feature = feature.permute(0,2,1,3).contigous()
        feature_aug = feature_aug.permute(0,2,1,3).contiguous()
        
        for i in range(self.batch_size[0]):  
            #正例取随机/中心/所有
            if self.ap_mode == 'all':
                ap = feature[i,:,:,:] * feature_aug[i,:,:,:].permute(0,2,1)  #bins*batch_size[1]*batch_size[1]
                ap_exp = torch.exp(ap).sum(0).sum(0)
            elif self.ap_mode == 'centor':
                ap = feature[i,:,:,:] * feature_aug[i,:,:,:].permute(0,2,1).mean(0)  #bins*batch_size[1]
                ap_exp = torch.exp(ap).sum(0)
            elif self.ap_mode == 'random':
                index = np.random.randint(0,self.batch_size[1])
                ap = feature[i,:,:,:] * feature_aug[i,:,index,:].permute(0,2,1).mean(0)  #bins*batch_size[1]
                ap_exp = torch.exp(ap).sum(0)
            #反例取随机/中心/所有
            if self.an_mode == 'all':
                #有（batch_size[0]-1）*batch_size[1]个an对
                an = [feature[i,:,:,:] * feature_aug[0:i,:,:,:].permute(0,2,1),
                     feature[i,:,:,:] * feature_aug[i+1:,:,:,:].permute(0,2,1)] 
                an = np.concatenate(an, axis=0) #(batch_size[0]-1)*bins*batch_size[1]*batch_size[1]
                an_exp = torch.exp(an).sum(0).sum(0).sum(1) #bins
            elif self.an_mode == 'centor':
                an = [feature[i,:,:,:] * feature_aug[0:i,:,:,:].permute(0,2,1).mean(0),
                    feature[i,:,:,:] * feature_aug[i+1:,:,:,:].permute(0,2,1).mean(0)]  
                feature[i,:,:,:] * feature_aug[0:i,:,:,:].permute(0,2,1).mean(0) #(batch_size[0]-1)*bins*batch_size[1]
                an_exp = torch.exp(an).sum(0).sum(1)
            elif self.an_mode == 'random':
                index = np.random.randint(0,self.batch_size[1],size=self.batch_size[0]-1)
                an = [feature[i,:,:,:] * feature_aug[0:i,:,index[:i],:].permute(0,2,1).mean(0),
                    feature[i,:,:,:] * feature_aug[i+1:,:,index[i:],:].permute(0,2,1).mean(0)]  
                feature[i,:,:,:] * feature_aug[0:i,:,:,:].permute(0,2,1).mean(0) #(batch_size[0]-1)*bins*batch_size[1]
                an_exp = torch.exp(an).sum(0).sum(1)
            set_trace()
            loss = -torch.log(ap_exp/an_exp)
        return loss.mean()
