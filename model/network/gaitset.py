import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import numpy as np
from copy import deepcopy
from .basic_blocks import BasicConv2d
from .sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d

class GaitSet(nn.Module):
    def __init__(self, config):
        super(GaitSet, self).__init__()
        self.config = deepcopy(config)
        if self.config['more_channels']:
            self.config.update({'channels':[64, 128, 256]})
        else:
            self.config.update({'channels':[32, 64, 128]})
        print("############################")
        print("GaitSet: channels={}, bin_num={}, hidden_dim={}".format(\
                self.config['channels'], self.config['bin_num'], self.config['hidden_dim']))       
        print("############################")

        self.phase = self.config['phase']
        self.bin_num = list(self.config['bin_num'])
        self.hidden_dim = self.config['hidden_dim']

        self.config.update({'in_channels':1})
        self.layer1 = BasicConv2d(self.config['in_channels'], self.config['channels'][0], kernel_size=5, stride=1, padding=2)
        self.layer2 = BasicConv2d(self.config['channels'][0], self.config['channels'][0], kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.layer3 = BasicConv2d(self.config['channels'][0], self.config['channels'][1], kernel_size=3, stride=1, padding=1)
        self.layer4 = BasicConv2d(self.config['channels'][1], self.config['channels'][1], kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.layer5 = BasicConv2d(self.config['channels'][1], self.config['channels'][2], kernel_size=3, stride=1, padding=1)
        self.layer6 = BasicConv2d(self.config['channels'][2], self.config['channels'][2], kernel_size=3, stride=1, padding=1)
        if len(self.config['channels']) > 3:
            self.layer7 = BasicConv2d(self.config['channels'][2], self.config['channels'][3], kernel_size=3, stride=1, padding=1)
            self.layer8 = BasicConv2d(self.config['channels'][3], self.config['channels'][3], kernel_size=3, stride=1, padding=1)            

        self.fc_bin = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num), self.config['channels'][-1], self.hidden_dim)))

        if self.config['DDP']:
            self.part_bn = nn.BatchNorm1d(self.hidden_dim*sum(self.bin_num))
            self.part_bn = nn.SyncBatchNorm.convert_sync_batchnorm(self.part_bn)
        else:
            self.part_bn = SynchronizedBatchNorm1d(self.hidden_dim*sum(self.bin_num))
        if self.phase == 'train' and self.config['encoder_entropy_weight'] > 0:
            self.part_cls = nn.Linear(self.hidden_dim*sum(self.bin_num), self.config['num_id'], bias=False)
                
        #initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm1d, \
                        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.LayerNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # initialization for classifier
        self.part_bn.bias.requires_grad_(False)
        if self.phase == 'train' and self.config['encoder_entropy_weight'] > 0:
            nn.init.normal_(self.part_cls.weight, 0, 0.001)

    def set_pool(self, x, n, s, batch_frames=None):
        if batch_frames is None:
            _, c, h, w = x.size()
            return torch.max(x.view(n, s, c, h, w), 1)[0]
        else:
            tmp = []
            for i in range(len(batch_frames) - 1):
                tmp.append(torch.max(x[batch_frames[i]:batch_frames[i+1], :, :, :], 0, keepdim=True)[0])
            return torch.cat(tmp, 0)

    def forward(self, silho, batch_frames=None, label=None):
        with autocast(enabled=self.config['AMP']):
            # n: batch_size, s: frame_num, k: keypoints_num, c: channel
            if batch_frames is not None:
                batch_frames = batch_frames[0].data.cpu().numpy().tolist()
                num_seqs = len(batch_frames)
                for i in range(len(batch_frames)):
                    if batch_frames[-(i + 1)] > 0:
                        break
                    else:
                        num_seqs -= 1
                batch_frames = batch_frames[:num_seqs]
                frame_sum = np.sum(batch_frames)
                if frame_sum < silho.size(1):
                    silho = silho[:, :frame_sum, :, :]
                batch_frames = [0] + np.cumsum(batch_frames).tolist()
            x = silho.unsqueeze(2)
            del silho

            n1, s, c, h, w = x.size()
            x = self.layer1(x.view(-1, c, h, w))
            x = self.layer2(x)
            x = self.max_pool1(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.max_pool2(x)
            x = self.layer5(x)
            x = self.layer6(x)
            if len(self.config['channels']) > 3:
                x = self.layer7(x)
                x = self.layer8(x)
            
            x = self.set_pool(x, n1, s, batch_frames)

            feature = list()
            offset = 0
            for num_bin in self.bin_num:
                n2, c, h, w = x.size()
                z = x.view(n2, c, num_bin, -1).max(-1)[0] + x.view(n2, c, num_bin, -1).mean(-1)
                feature.append(z)

            feature = torch.cat(feature, dim=-1)                # n x c x num_parts
            feature = feature.permute(2, 0, 1).contiguous()     # num_parts x n x c 
            feature = feature.matmul(self.fc_bin)               # num_parts x n x hidden_dim
            feature = feature.permute(1, 0, 2).contiguous()     # n x num_parts x hidden_dim

            part_bn_feature = self.part_bn(feature.view(n2, -1))
            if self.phase == 'train' and self.config['encoder_entropy_weight'] > 0:
                part_cls_score = self.part_cls(part_bn_feature)
            
            bn_feature = part_bn_feature.unsqueeze(1)
            if self.phase == 'train' and self.config['encoder_entropy_weight'] > 0:
                cls_score = part_cls_score.unsqueeze(1)
            else:
                cls_score = torch.empty(1).cuda()

            bn_feature = bn_feature.detach()
            if self.config['encoder_entropy_weight'] <= 0:
                cls_score = cls_score.detach()

            return feature, bn_feature, cls_score
