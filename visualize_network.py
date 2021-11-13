import torch
import numpy as np
import random

from model.initialization import initialization

from config import *
config.update({'phase':'train'})
m = initialization(config)
print('#######################################')
print("Network Structures:", m.encoder.module)
print('#######################################')

from torchviz import make_dot
x = torch.rand(10, config['frame_num'], config['resolution'], config['resolution']-2*int(float(config['resolution'])/64*10)).float().cuda()
print("Input Shape for Network Visualization: ", x.shape)
y = m.encoder.module(x)
g = make_dot(y)
save_name = 'Visualize_Network_{}'.format(config['model_name'])
g.render(save_name, view=False)
print("Visualize Network Saved to {}.pdf".format(save_name))