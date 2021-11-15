#!/usr/bin/env python
# coding=utf-8
import numpy as np

par_list = np.load('CASIA-B_73_False_Total_123.npy')
print(par_list)

prefix = ''
label = []
for i in range(1, 10307):
    label.append(prefix+"{:0>5d}".format(i))
pid_list = sorted(list(set(label)))
pid_list = [pid_list[0:len(pid_list):2],pid_list[1:len(pid_list):2]]
pid_list[1].append(prefix+'10307')
np.save('OUMVLP_5153_False_Total_10307.npy', pid_list)

par_list = np.load('OUMVLP_5153_False_Total_10307.npy')
print(par_list[0][0:5], par_list[0][-5:])
print(par_list[1][0:5], par_list[1][-5:])
print(len(par_list[0]), len(par_list[1]))
