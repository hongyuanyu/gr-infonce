import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.loss = list()
        self.iteration = 0
        self.label_stat = list()
        self.label_clean_stat = list()
        self.index_bigloss = list()

    def forward(self, inputs, label, label_clean, l, loss_statistic):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        if l > 0:
            targets = l * torch.zeros(log_probs.size()).scatter_(1, label[0].unsqueeze(1).data.cpu(), 1) + (1-l) * torch.zeros(log_probs.size()).scatter_(1, targets[1].unsqueeze(1).data.cpu(), 1)
        else:
            targets = torch.zeros(log_probs.size()).scatter_(1, label.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(1)

        loss_mean = loss.mean(0)
        
        if loss_statistic == True :
            #index_bigloss = index[loss>4.8]

            with torch.no_grad():
                
                n = loss.shape  #128
                self.loss.append(loss)  #128*16*~
                self.index_bigloss.append(index_bigloss)  #128*16*~
                self.label_stat.append(label)
                self.label_clean_stat.append(label_clean)
                self.iteration += 1
                if self.iteration == 10:
                    set_trace()
                    self.index_bigloss = torch.cat(self.index_bigloss, dim=0)
                    save_path = '/home/yuweichen/workspace/noisy_gait/visualize_feature'
                    file_name = os.path.join(save_path, 'bigCEloss.pkl')
                    with open(file_name, 'wb') as f:
                        pickle.dump(self.index_bigloss, f)
                    self.loss = torch.cat(self.loss, dim=0)
                    self.label_stat = torch.cat(self.label_stat, dim=0)
                    self.label_clean_stat = torch.cat(self.label_clean_stat, dim=0)
                    self.loss_statistic(self.loss, self.label_stat, self.label_clean_stat)

                    self.loss = list()
                    self.loss_nonzero = list()
                    self.label_stat = list()
                    self.label_clean_stat = list()
                    self.index_bigloss = list()

        return loss_mean

    def loss_statistic(self, loss, label_noisy, label_clean):
        batch_size = loss.shape  #128
        #set_trace()
        clean_index = (label_noisy == label_clean)
        clean_loss = torch.masked_select(loss,clean_index).cpu()
        noisy_loss = torch.masked_select(loss,~clean_index).cpu()

        save_path = '/home/yuweichen/workspace/noisy_gait/visualize_feature'
        img_name = os.path.join(save_path, 'loss_histogram_CEloss')
        plt.figure(1, figsize=(20, 10))
        plt.subplot(131)
        plt.hist(clean_loss,facecolor='blue',bins=100)
        plt.subplot(132)
        plt.hist(noisy_loss,facecolor='red',bins=100)
        plt.subplot(133)
        plt.hist(clean_loss,facecolor='blue',bins=100)
        plt.hist(noisy_loss,facecolor='red',bins=100)
        
        plt.savefig(img_name)
        plt.clf()                                                   
        print('loss statistic CELoss process Done')
        return None