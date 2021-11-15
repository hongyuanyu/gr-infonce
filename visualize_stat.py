import pickle
from matplotlib import pyplot as plt
from config import *
from model.initialization import initialization
from pdb import set_trace
import os

config.update({'phase':'train'})
m = initialization(config)
label_noisy = config['train_source'].label_set
label_clean = [l[:3] for l in label_noisy]
statistic_path = '/home2/ywc/workspace/IDNoise_Gait_Full-master/visualization'
statistic_file_path = statistic_path+'statistic_file_dist_100'
with open(statistic_file_path, 'rb') as f:
    data_statistic = pickle.load(f)
data_statistic = data_statistic.cpu()
iteration, bins, ids, characristics = data_statistic.shape  # 200*16*116*12

#interclass_max, interclass_min_3, interclass_argmin_3, inter_class_mean,  intraclass_mean, intraclass_min, intraclass_max, intraclass_var    
#每个id在训练过程中，类内、类间的变化
plt.figure(1, figsize=(20, 10))

for i in range(ids):
    interclass_max = data_statistic[:,1,i,0]
    interclass_min_3 = data_statistic[:,1,i,1:4]
    interclass_argmin_3 = data_statistic[:,1,i,4:7].int()
    interclass_mean = data_statistic[:,1,i,7]
    intraclass_mean = data_statistic[:,1,i,8]
    intraclass_min = data_statistic[:,1,i,9]
    intraclass_max = data_statistic[:,1,i,10]
    intraclass_var = data_statistic[:,1,i,11]
    is_clean_close_noisy = list()
    for k in range(iteration):
        is_clean_close_noisy.append((label_clean[interclass_argmin_3[k,0].tolist()] == label_clean[i])+(label_clean[interclass_argmin_3[k,1].tolist()] == label_clean[i])+(label_clean[interclass_argmin_3[k,2].tolist()] == label_clean[i])
        + (label_clean[interclass_argmin_3[k,0].tolist()] == label_noisy[i])+(label_clean[interclass_argmin_3[k,1].tolist()] == label_noisy[i])+(label_clean[interclass_argmin_3[k,2].tolist()] == label_noisy[i]))
    
    plt.subplot(131)
    plt.plot(range(iteration), interclass_max, 'r')
    plt.plot(range(iteration), interclass_min_3[:,0], 'b')
    plt.plot(range(iteration), interclass_mean, 'g')
    plt.plot(range(iteration), is_clean_close_noisy, 'c')
    plt.subplot(132)
    plt.plot(range(iteration), intraclass_mean, 'g')
    plt.plot(range(iteration), intraclass_max, 'r')
    plt.plot(range(iteration), intraclass_min, 'b')
    plt.plot(range(iteration), intraclass_var, 'c')
    plt.subplot(133)
    plt.plot(range(iteration), is_clean_close_noisy, 'c')
    


    plt.suptitle(i,fontsize=30)
    save_dir = '/home2/ywc/workspace/IDNoise_Gait_Full-master/visualization'
    img_name = os.path.join(save_dir, 'inter_intra_dist'+str(i)+'part_1')
    plt.savefig(img_name)
    plt.clf()

id_noise_num = 43
plt.figure(2, figsize=(20, 10))

for i in range(id_noise_num):
    interclass_argmin_3_i = data_statistic[:,0,2*i,4:7].int()
    interclass_argmin_3_j = data_statistic[:,0,2*i+1,4:7].int()

    is_clean_close_noisy_i = list()
    is_clean_close_noisy_j = list()

    for k in range(iteration):
        is_clean_close_noisy_i.append((label_clean[interclass_argmin_3_i[k,0].tolist()] == label_clean[2*i])+(label_clean[interclass_argmin_3_i[k,1].tolist()] == label_clean[2*i])+(label_clean[interclass_argmin_3_i[k,2].tolist()] == label_clean[2*i])
        + (label_clean[interclass_argmin_3_i[k,0].tolist()] == label_noisy[2*i])+(label_clean[interclass_argmin_3_i[k,1].tolist()] == label_noisy[2*i])+(label_clean[interclass_argmin_3_i[k,2].tolist()] == label_noisy[2*i]))
        is_clean_close_noisy_j.append((label_clean[interclass_argmin_3_j[k,0].tolist()] == label_clean[2*i+1])+(label_clean[interclass_argmin_3_j[k,1].tolist()] == label_clean[2*i+1])+(label_clean[interclass_argmin_3_j[k,2].tolist()] == label_clean[2*i+1])
        + (label_clean[interclass_argmin_3_j[k,0].tolist()] == label_noisy[2*i+1])+(label_clean[interclass_argmin_3_j[k,1].tolist()] == label_noisy[2*i+1])+(label_clean[interclass_argmin_3_j[k,2].tolist()] == label_noisy[2*i+1]))
    plt.plot(range(iteration), is_clean_close_noisy_i, 'c')
    plt.plot(range(iteration), is_clean_close_noisy_j, 'b')
    plt.suptitle('{}_{}'.format(2*i,2*i+1),fontsize=30)
    save_dir = '/home2/ywc/workspace/IDNoise_Gait_Full-master/visualization'
    img_name = os.path.join(save_dir, 'inter_intra_dist'+str(i)+'part_0_{}_{}'.format(2*i,2*i+1))
    plt.savefig(img_name)
    plt.clf()
    